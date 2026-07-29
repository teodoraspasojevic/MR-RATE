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
        --n-per-sequence 200 \\
        --out <workspace>/cohorts/test_v1

Sampling and FOV, the two things you will actually tune:

  --n-per-sequence N     N cases per sequence (omit for every eligible case). Selection is
                         deterministic given --seed, and picking one sequence yields the same
                         cases as picking four -- see `cohort.select_cohort`.
  --series-selection     which series of a study may be picked. Default
                         one_per_study_per_sequence: one independent observation per (study,
                         sequence) -- what you want for a multi-sequence evaluation. `all`
                         keeps every series, which pseudo-replicates near-duplicates from one
                         session (measured mean 1.96, max 13 per study-sequence on the real
                         test split) and biases means toward multi-series studies.
                         `one_per_study_deterministic` picks one series per *study* across all
                         sequences, and because the preferred series is the T1w center
                         modality it collapses a 4-sequence request to ~99% T1w -- only use it
                         for a single-sequence cohort.
  --geometry-mode fixed  (default) every volume on one grid -- required for comparing models,
                         and required for `--task generation` to mean anything, since the
                         NVIDIA diffusion model emits ONLY 256^3 @ 1mm. `--fixed-shape` /
                         `--fixed-spacing-mm` set it; the default is read from that model's
                         config so a cohort and its output share a grid.
  --geometry-mode per_modality_plane
                         each (modality, plane) gets its own tighter FOV from NVIDIA's
                         published median training FOVs. Anatomically snugger and fine for
                         training or a VAE-only study, but every bucket differs from the
                         generator's fixed output shape, so a generation evaluation against
                         such a cohort compares populations on different grids. Numbers are
                         also not comparable with a fixed-geometry cohort.

Disk cost is about `prod(shape) * 4` bytes per case: ~67 MB at 256^3, so ~54 GB for 800 cases.
Write cohorts to a workspace, never to git or your home directory.
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
    save_report,
    save_volume,
    select_cohort,
    sha256_file,
    write_cohort,
)
from ..data import (
    MRReportToVolumeDataset,
    R2VDatasetConfig,
    SentenceJSONLReportStore,
    ShardReportStore,
    StructuredReportStore,
    read_manifest_csv,
    xyz_to_dhw,
)

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
    coh.add_argument("--n-per-sequence", type=int, default=None,
                     help="cases per sequence; omit for every eligible case")
    coh.add_argument("--series-selection", default="one_per_study_per_sequence",
                     choices=["all", "one_per_study_per_sequence",
                              "one_per_study_deterministic", "one_per_study_random"],
                     help="default one_per_study_per_sequence: one independent observation per "
                          "(study, sequence). 'all' pseudo-replicates near-duplicate series; "
                          "'one_per_study_deterministic' collapses a multi-sequence request to "
                          "almost entirely T1w -- see MRReportToVolumeDataset._select_series")
    coh.add_argument("--seed", type=int, default=42)

    geo = p.add_argument_group("geometry / FOV")
    geo.add_argument("--geometry-mode", default="fixed", choices=["fixed", "per_modality_plane"])
    geo.add_argument("--fixed-shape", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"))
    geo.add_argument("--fixed-spacing-mm", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    geo.add_argument("--normalizer", default="percentile", choices=["percentile", "zscore", "minmax"])
    geo.add_argument("--posterior-shift-mm", type=float, default=15.0)
    geo.add_argument("--env-config", type=Path, default=None)
    geo.add_argument("--model-config", type=Path, default=None)
    geo.add_argument("--network-config", type=Path, default=None)

    io = p.add_argument_group("output")
    io.add_argument("--out", type=Path, required=True, help="cohort directory to create")
    io.add_argument("--overwrite", action="store_true")
    io.add_argument("--archive-access-mode", default="stream", choices=["stream", "node_local_cache"])
    io.add_argument("--report-sections", nargs="+", default=["findings", "impression"])
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
    config = R2VDatasetConfig(
        split=args.split, geometry_mode=args.geometry_mode,
        series_selection=args.series_selection, seed=args.seed, dtype=torch.float32,
        normalizer=args.normalizer, posterior_shift_mm=args.posterior_shift_mm,
        report_sections=tuple(args.report_sections),
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

    selection = select_cohort(dataset, args.sequences, args.n_per_sequence, args.seed)
    indices = [(seq, i) for seq in args.sequences for i in selection.get(seq, [])]
    log.info("cohort: %s (total %d)", {s: len(v) for s, v in selection.items()}, len(indices))
    if not indices:
        raise SystemExit("cohort selection produced 0 cases -- check --sequences against the manifest")

    if args.dry_run:
        example = dataset.samples[indices[0][1]]
        log.info("--dry-run: would write %d cases to %s", len(indices), args.out)
        log.info("--dry-run: first case modality=%s plane=%s backend=%s",
                 example.modality, example.plane, example.backend)
        return 0

    cases = []
    t0 = time.time()
    for n, (seq, idx) in enumerate(indices, start=1):
        sample = dataset[idx]
        case = CohortCase(
            case_id=case_id_for(sample["study_key"], sample["series_key"]),
            study_key=sample["study_key"], series_key=sample["series_key"],
            sequence=sample["modality"], acquisition_plane=sample["acquisition_plane"],
            shape=tuple(int(x) for x in sample["target_shape"].tolist()),
            spacing_mm=tuple(float(x) for x in sample["target_spacing_mm"].tolist()),
        )
        save_volume(args.out, case.case_id, sample["image"][0].numpy())
        save_report(args.out, case.case_id, sample["report_text"])
        cases.append(case)
        if n % 25 == 0 or n == len(indices):
            log.info("[%d/%d] %.1fs elapsed", n, len(indices), time.time() - t0)

    spec = CohortSpec(
        split=args.split, sequences=list(args.sequences),
        series_selection=args.series_selection, n_per_sequence=args.n_per_sequence,
        seed=args.seed, geometry=config.geometry_fingerprint(),
        manifest_csv=str(args.manifest_csv), manifest_sha256=manifest_sha,
        report_source=report_source,
    )
    cohort_json = write_cohort(args.out, spec, cases)
    log.info("cohort written: %d cases, cohort_id=%s -> %s", len(cases), spec.cohort_id(), cohort_json)
    log.info("pass --cohort %s to every predict and evaluate run for this experiment set", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
