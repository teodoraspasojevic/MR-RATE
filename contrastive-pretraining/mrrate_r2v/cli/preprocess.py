#!/usr/bin/env python3
"""Stage 1: build a frozen ground-truth cohort.

Selects cases, preprocesses their volumes, and writes them to disk together with a
`cohort.json` contract. Run this ONCE per experiment set; every predict and evaluate run then
consumes the result, so FOV, case list, sample count, and normalization are decided here and
nowhere else.

    python -m mrrate_r2v.cli.preprocess \\
        --manifest-csv  <manifest.csv> \\
        --report-index-csv <report_index.csv> \\
        --split test --sequences T1w T2w FLAIR SWI \\
        --n-per-bucket 200 \\
        --out <workspace>/cohorts/test_v1

Sampling and FOV, the two things you will actually tune:

  --n-per-bucket N       N cases per (modality, plane) bucket -- 10 buckets exist in the real
                         test split, so N=200 gives 2000 cases. Per *bucket* rather than per
                         sequence so every bucket has equal statistical power: the real plane
                         mix is ~81% axial, which would leave coronal buckets too small for a
                         stable per-bucket FID. Deterministic given --seed, and which buckets
                         you request never shifts another bucket's draw.
  --series-selection     which series of a study may be picked. Default
                         one_per_study_per_bucket: one independent observation per (study,
                         modality, plane). `one_per_study_per_sequence` looks equivalent but is
                         not -- it prefers the T1w center-modality series, which collapses the
                         PLANES (measured: T2w SAGITTAL fell to 6 cases). `all`
                         pseudo-replicates near-duplicate series from one session (measured mean
                         1.96, max 13); `one_per_study_deterministic` collapses a 4-sequence
                         request to ~99% T1w.
  --geometry-mode per_modality_plane   (default)
                         Each bucket is resampled to NVIDIA's published recommended FOV for
                         that (modality, plane) *exactly*: shape is the nearest multiple of 32
                         (the diffusion UNet's constraint, verified empirically) and spacing is
                         derived as FOV/shape. Spacing is a real conditioning input to the
                         model, so generation can then be asked for the same geometry -- which
                         is what makes the real and generated populations comparable.
  --geometry-mode fixed  One grid for everything, from `--fixed-shape`/`--fixed-spacing-mm`.
                         Only useful for a deliberate single-geometry study.
  --posterior-shift-mm   Defacing compensation. Default 0 -- see the flag's help.

Volumes are stored as one compressed archive per bucket (see `volumes.py`), which is both ~2.9x
smaller than raw and ~10 files instead of ~2000 -- `/hnvme` has a file-count quota. Expect ~14 MB
per case on disk at the per-bucket FOVs. Write cohorts to a workspace, never to git or `$HOME`.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from ..cohort import (
    CohortCase,
    CohortSpec,
    case_id_for,
    population_bucket_counts,
    select_cohort_buckets,
    sha256_file,
    write_cohort,
)
from ..data import (
    MRReportToVolumeDataset,
    R2VDatasetConfig,
    SentenceJSONLReportStore,
    ShardReportStore,
    StructuredReportStore,
    dhw_to_xyz,
    read_manifest_csv,
    xyz_to_dhw,
)
from ..volumes import VolumeWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("preprocess")

DEFAULT_SEQUENCES = ["T1w", "T2w", "FLAIR", "SWI"]


def build_report_store(args):
    if args.report_index_csv:
        return ShardReportStore(str(args.report_index_csv)), f"ShardReportStore:{args.report_index_csv.name}"
    if args.report_csv:
        return StructuredReportStore(str(args.report_csv)), f"StructuredReportStore:{args.report_csv.name}"
    if args.report_jsonl:
        return SentenceJSONLReportStore(str(args.report_jsonl)), f"SentenceJSONLReportStore:{args.report_jsonl.name}"
    raise SystemExit("give exactly one of --report-index-csv / --report-csv / --report-jsonl")


def resolve_fixed_geometry(args):
    """The fixed target grid, returned in **(X, Y, Z)**.

    Defaults to the NVIDIA generator's own native shape/spacing read from its config -- which is
    already (X, Y, Z), that model's own array order -- so a cohort and that model's output share a
    grid without a hardcoded constant that could drift out of sync. `--fixed-shape` /
    `--fixed-spacing-mm` are (X, Y, Z) too, matching what the Dataset reports and what a cohort
    records.

    `main` converts to (D, H, W) with `xyz_to_dhw` before building `R2VDatasetConfig`, whose
    fixed-target fields are (D, H, W) like every other internal geometry parameter. Do not skip
    that conversion: it is a no-op for a cube at isotropic spacing (the current NVIDIA default,
    256^3 @ 1mm, which is why this was invisible) and scrambles axes for anything else.
    """
    if args.fixed_shape and args.fixed_spacing_mm:
        return tuple(args.fixed_shape), tuple(args.fixed_spacing_mm), "user override"
    from ..models.nvidia import DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_NETWORK_CONFIG, load_config

    cfg = load_config(str(args.env_config or DEFAULT_ENV_CONFIG),
                      str(args.model_config or DEFAULT_MODEL_CONFIG),
                      str(args.network_config or DEFAULT_NETWORK_CONFIG))
    shape = tuple(int(x) for x in cfg.diffusion_unet_inference["dim"])
    spacing = tuple(float(x) for x in cfg.diffusion_unet_inference["spacing"])
    return (tuple(args.fixed_shape) if args.fixed_shape else shape,
            tuple(args.fixed_spacing_mm) if args.fixed_spacing_mm else spacing,
            "NVIDIA model config default")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data source")
    src.add_argument("--manifest-csv", type=Path, required=True,
                     help="built by `python -m mrrate_r2v.cli.build_manifest`")
    src.add_argument("--report-index-csv", type=Path, default=None, help="shard report index (preferred)")
    src.add_argument("--report-csv", type=Path, default=None, help="MR-RATE reports.csv / reports.tar.gz")
    src.add_argument("--report-jsonl", type=Path, default=None, help="findings_sentences.jsonl (fallback)")

    coh = p.add_argument_group("cohort")
    coh.add_argument("--split", default="test", choices=["train", "val", "test"])
    coh.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    coh.add_argument("--n-per-bucket", type=int, default=200,
                     help="cases per (modality, plane) bucket; 0 means every eligible case. "
                          "Sampling per bucket rather than per sequence gives every bucket equal "
                          "statistical power -- the real plane mix is ~81%% axial, which would "
                          "leave coronal buckets too small for a stable per-bucket FID. Because "
                          "the cohort is balanced, a frequency-weighted aggregate is weighted by "
                          "the eligible-population counts recorded in cohort.json, not by these.")
    coh.add_argument("--series-selection", default="one_per_study_per_bucket",
                     choices=["all", "one_per_study_per_bucket", "one_per_study_per_sequence",
                              "one_per_study_deterministic", "one_per_study_random"],
                     help="default one_per_study_per_bucket: one independent observation per "
                          "(study, modality, plane). 'one_per_study_per_sequence' prefers the "
                          "T1w center-modality series and so collapses the planes. "
                          "'all' pseudo-replicates near-duplicate series; "
                          "'one_per_study_deterministic' collapses a multi-sequence request to "
                          "almost entirely T1w -- see MRReportToVolumeDataset._select_series")
    coh.add_argument("--seed", type=int, default=42)

    geo = p.add_argument_group("geometry / FOV")
    geo.add_argument("--geometry-mode", default="per_modality_plane",
                     choices=["per_modality_plane", "fixed"],
                     help="per_modality_plane (default): each bucket gets NVIDIA's published "
                          "recommended FOV exactly, at a shape the diffusion UNet accepts. "
                          "`fixed` forces one grid for everything -- only useful for a "
                          "single-geometry study.")
    geo.add_argument("--fixed-shape", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"))
    geo.add_argument("--fixed-spacing-mm", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    geo.add_argument("--normalizer", default="percentile", choices=["percentile", "zscore", "minmax"])
    geo.add_argument("--posterior-shift-mm", type=float, default=0.0,
                     help="defacing compensation, in mm. Default 0: measured against real "
                          "generated volumes, 8 of 10 buckets align better at 0 than at 15 mm, "
                          "two of them to within 0.1 mm. The 15 mm value exists to protect a "
                          "fixed oversized FOV in the contrastive pipeline and is not what this "
                          "model expects.")
    geo.add_argument("--env-config", type=Path, default=None)
    geo.add_argument("--model-config", type=Path, default=None)
    geo.add_argument("--network-config", type=Path, default=None)

    io = p.add_argument_group("output")
    io.add_argument("--out", type=Path, required=True, help="cohort directory to create")
    io.add_argument("--overwrite", action="store_true")
    io.add_argument("--archive-access-mode", default="stream", choices=["stream", "node_local_cache"])
    io.add_argument("--report-sections", nargs="+", default=["findings", "impression"])
    io.add_argument("--report-format", default=None,
                    help="a single named format from mrrate_r2v.textenc.formats. Must match one of "
                         "the formats the adapter was trained on, or cli.generate_r2v refuses the "
                         "cohort: a cohort stores already-composed text, so the format cannot be "
                         "changed afterwards. Default: --report-sections joined")
    io.add_argument("--dry-run", action="store_true",
                    help="select the cohort and print what would be written, then stop")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} exists and is non-empty; pass --overwrite to replace it")

    if args.geometry_mode == "fixed":
        fixed_shape_xyz, fixed_spacing_xyz, origin = resolve_fixed_geometry(args)
        log.info("geometry_mode=fixed  shape=%s spacing_mm=%s  [X,Y,Z] (%s)",
                 fixed_shape_xyz, fixed_spacing_xyz, origin)
    else:
        fixed_shape_xyz = fixed_spacing_xyz = None
        log.warning("geometry_mode=per_modality_plane: volumes will have different shapes per "
                    "(modality, plane) bucket. Numbers from this cohort are NOT comparable with "
                    "a fixed-geometry cohort.")

    manifest_sha = sha256_file(args.manifest_csv)
    rows = [r for r in read_manifest_csv(str(args.manifest_csv))
            if r.split == args.split and r.modality in args.sequences]
    log.info("manifest: %d rows for split=%s sequences=%s", len(rows), args.split, args.sequences)
    if not rows:
        raise SystemExit("0 manifest rows after split/sequence filtering -- check --split/--sequences")

    report_store, report_source = build_report_store(args)
    if args.report_format is not None:
        # A cohort's text is composed once and frozen, so a sampled multi-name spec has no meaning
        # here -- there would be no single answer to record in `cohort.json`.
        from ..textenc.formats import parse_format_spec

        if len(parse_format_spec(args.report_format)) != 1:
            raise SystemExit(
                f"--report-format {args.report_format!r} names several formats. A cohort freezes one "
                "conditioning string per case; pick the single format the predictions will be "
                "generated under (for a multi-format training run, its first name)."
            )
    config = R2VDatasetConfig(
        split=args.split, geometry_mode=args.geometry_mode,
        series_selection=args.series_selection, seed=args.seed, dtype=torch.float32,
        normalizer=args.normalizer, posterior_shift_mm=args.posterior_shift_mm,
        report_sections=tuple(args.report_sections),
        report_format=args.report_format,
        archive_access_mode=args.archive_access_mode,
        # (X, Y, Z) in, (D, H, W) out: R2VDatasetConfig's fixed-target fields are the internal
        # (D, H, W) order, while the CLI flags and NVIDIA's dim/spacing are (X, Y, Z). See
        # resolve_fixed_geometry's docstring for why this conversion is not optional.
        **({"fixed_target_shape": xyz_to_dhw(fixed_shape_xyz),
            "fixed_target_spacing_mm": xyz_to_dhw(fixed_spacing_xyz)}
           if args.geometry_mode == "fixed" else {}),
    )
    dataset = MRReportToVolumeDataset(manifest=rows, report_store=report_store, config=config)
    if len(dataset) == 0:
        raise SystemExit("0 samples after report filtering + series selection -- check the report source")

    # 0 means "no cap" -- take every eligible case in each bucket.
    n_per_bucket = args.n_per_bucket or None
    selection = select_cohort_buckets(dataset, args.sequences, n_per_bucket, args.seed)
    selection = {k: v for k, v in selection.items() if v}
    total = sum(len(v) for v in selection.values())
    log.info("cohort: %d buckets, %d cases", len(selection), total)
    for (mod, plane), idxs in selection.items():
        spec_g = dataset.geometry.resolve(mod, plane)
        log.info("   %-6s %-9s n=%-5d shape=%s spacing=%s mm",
                 mod, plane, len(idxs), dhw_to_xyz(spec_g.target_shape),
                 tuple(round(x, 4) for x in dhw_to_xyz(spec_g.target_spacing)))
    if not selection:
        raise SystemExit("cohort selection produced 0 cases -- check --sequences against the manifest")

    if args.dry_run:
        log.info("--dry-run: would write %d cases to %s (nothing written)", total, args.out)
        return 0

    # One archive per bucket, so iterate bucket by bucket. Memory stays at one volume: the writer
    # streams each array into its bucket's zip rather than accumulating the bucket.
    cases, reports = [], {}
    n = 0
    t0 = time.time()
    with VolumeWriter(args.out) as writer:
        for (mod, plane), idxs in selection.items():
            for idx in idxs:
                sample = dataset[idx]
                case = CohortCase(
                    case_id=case_id_for(sample["study_key"], sample["series_key"]),
                    study_key=sample["study_key"], series_key=sample["series_key"],
                    sequence=sample["modality"], acquisition_plane=sample["acquisition_plane"],
                    shape=tuple(int(x) for x in sample["target_shape"].tolist()),
                    spacing_mm=tuple(float(x) for x in sample["target_spacing_mm"].tolist()),
                )
                # Guard against a silent bucket/geometry mismatch: the sample's own reported
                # geometry must be the one this bucket's archive is supposed to hold.
                if (case.sequence, case.acquisition_plane) != (mod, plane):
                    raise SystemExit(
                        f"bucket mismatch: selection said ({mod}, {plane}) but the sample reports "
                        f"({case.sequence}, {case.acquisition_plane}) -- refusing to file it wrongly"
                    )
                writer.add(case.bucket, case.case_id, sample["image"][0].numpy())
                reports[case.case_id] = sample["report_text"]
                cases.append(case)
                n += 1
                if n % 50 == 0 or n == total:
                    log.info("[%d/%d] %.1fs elapsed", n, total, time.time() - t0)

    spec = CohortSpec(
        split=args.split, sequences=list(args.sequences),
        series_selection=args.series_selection, n_per_bucket=args.n_per_bucket,
        seed=args.seed, geometry=config.geometry_fingerprint(),
        manifest_csv=str(args.manifest_csv), manifest_sha256=manifest_sha,
        report_source=report_source,
        population_bucket_counts=population_bucket_counts(rows, args.sequences),
    )
    cohort_json = write_cohort(args.out, spec, cases, reports)
    log.info("cohort written: %d cases in %d buckets, cohort_id=%s -> %s",
             len(cases), len(selection), spec.cohort_id(), cohort_json)
    log.info("pass --cohort %s to every predict and evaluate run for this experiment set", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
