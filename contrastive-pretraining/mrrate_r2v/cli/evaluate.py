#!/usr/bin/env python3
"""The evaluator. Builds the dataset training builds, generates, and scores -- in one command.

    python -m mrrate_r2v.cli.evaluate --task <task> --manifest ... --out <results_dir>

**There is no cohort and no prediction set.** This CLI is `cli.train_r2v` with the backward pass
replaced by a sampler: it calls the same `build_dataset`, constructs the same `R2VDatasetConfig`
from the same flags, and resolves each case's grid through the same `dataset.geometry.resolve`.
The two paths cannot preprocess differently because there is only one description of the
preprocessing, and `run_id` records it.

    cli.train_r2v --split train ...    dataset -> loader -> encode -> UNet -> backward
    cli.evaluate  --split test  ...    dataset -> loader -> encode -> UNet -> sample -> metrics

Three tasks, differing only in how a volume is produced. `eval/tasks.py` -- not this file -- decides
which metrics each one gets:

    report2volume    trained adapter + frozen base UNet, conditioned on the case's report
                     -> fidelity, perceptual, distribution, report_alignment,
                        report_consistency, anatomy
    reconstruction   the frozen NVIDIA autoencoder, encode then decode the real volume
                     -> fidelity, perceptual, distribution, anatomy
    generation       the frozen base UNet from a modality label alone, report-blind
                     -> distribution, anatomy only. No real patient corresponds to a generated
                        volume, so a voxelwise metric would measure "how different are two
                        random brains". It is structurally unreachable, not merely off by default.

**What is evaluated.** Every case in the split, in a deterministic no-RNG order, unless
`--n-per-bucket N` caps it -- and that cap takes the *first* N per bucket in the same order, so a
cheap run is a prefix of the full one rather than a different sample. Per-case sampler noise is
`stable_seed(--seed, case_id)`, a function of the case, so a rerun reproduces every volume.

`--n-per-bucket` is the cost knob. Measured 6.7 s/case at 30 inference steps on one H200:

    --n-per-bucket 8      ~10 min   smoke test, no metric means anything
    --n-per-bucket 200    ~3.7 h    the old cohort scale (2,000 cases over 10 buckets)
    (unset, full split)   ~64 h     34,453 test-split series -- what CTFlow does on CT-RATE

Read `metrics_per_bucket.csv` and `metrics_summary.csv` first; `summary.json` is the machine-
readable mirror and `run_manifest.json` is the record of what actually ran.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("evaluate")

#: NVIDIA's own modality class codes for rflow-mr-brain (unconditional generation).
MODALITY_CODE = {"T1w": 9, "T2w": 10, "FLAIR": 11, "SWI": 20}

#: The MR decoder emits ~[0, 1000]; divide back into the percentile space the Dataset's ground
#: truth lives in. The same constant `cli.predict_generation` used, for the same reason.
NVIDIA_MR_INTENSITY_SCALE = 1000.0


def parse_args(argv=None):
    from ..data import R2VDatasetConfig
    from ..eval.tasks import TASK_NAMES
    from ..eval.wandb_logging import WANDB_MODES

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(TASK_NAMES))
    p.add_argument("--out", type=Path, required=True, help="results directory to create")
    p.add_argument("--overwrite", action="store_true")

    # ---- data: byte-for-byte the same group cli.train_r2v declares, and read off the same
    # dataclass, so a default cannot drift between the two CLIs.
    data = p.add_argument_group("data (identical to cli.train_r2v)")
    data.add_argument("--manifest", type=Path, required=True, help="MR-RATE manifest CSV")
    data.add_argument("--report-index", type=Path, required=True, help="report index CSV")
    data.add_argument("--split", default="test", help="MR-RATE split to evaluate")
    data.add_argument("--report-sections", nargs="+", default=["findings", "impression"])
    data.add_argument("--report-format", default=None,
                      help="a named format from mrrate_r2v.textenc.formats. Must be one the "
                           "adapter was trained on; a multi-name training spec resolves to its "
                           "FIRST name here, exactly as validation does, so the text is composed "
                           "one fixed way rather than sampled")
    data.add_argument("--geometry-mode", default="per_modality_plane",
                      choices=["per_modality_plane", "fixed"])
    data.add_argument("--posterior-shift-mm", type=float,
                      default=R2VDatasetConfig.posterior_shift_mm,
                      help="must match the training run. Checked against the checkpoint's recorded "
                           "value by --task report2volume rather than trusted")
    data.add_argument("--normalizer", default=R2VDatasetConfig.normalizer,
                      choices=["percentile", "zscore", "minmax"])
    data.add_argument("--num-workers", type=int, default=4)

    sel = p.add_argument_group("case selection (deterministic, no RNG)")
    sel.add_argument("--n-per-bucket", type=int, default=None,
                     help="cases per (modality, plane). Default: the ENTIRE split. The cap keeps "
                          "the first N of the same deterministic order, so it is a prefix of the "
                          "full run, never a different sample")
    sel.add_argument("--seed", type=int, default=42,
                     help="seeds sampler noise only -- case selection uses no RNG at all")

    model = p.add_argument_group("model")
    model.add_argument("--checkpoint", type=Path, default=None,
                       help="--task report2volume: the trained adapter (cli.train_r2v)")
    model.add_argument("--base-checkpoint", type=Path, default=None,
                       help="frozen NVIDIA diffusion UNet (report2volume, generation)")
    model.add_argument("--vae-checkpoint", type=Path, default=None,
                       help="frozen NVIDIA autoencoder (all three tasks)")
    model.add_argument("--text-checkpoint", type=Path, default=None,
                       help="text encoder directory; default = whatever the adapter recorded")
    model.add_argument("--network-config", type=Path, default=None)
    model.add_argument("--model-name", default=None, help="recorded in run_manifest.json")
    model.add_argument("--allow-base-mismatch", action="store_true")
    model.add_argument("--allow-preprocessing-mismatch", action="store_true",
                       help="evaluate even when --posterior-shift-mm / --normalizer / "
                            "--geometry-mode differ from the training run's. For a deliberate "
                            "cross-preprocessing ablation only")
    model.add_argument("--allow-report-format-mismatch", action="store_true")
    model.add_argument("--train-world-size", type=int, default=None,
                       help="rank count the adapter was TRAINED on, for the training-sample count "
                            "when the run wrote no train_summary.json")

    samp = p.add_argument_group("sampling")
    samp.add_argument("--num-inference-steps", type=int, default=30)
    samp.add_argument("--report-guidance-scale", type=float, default=4.0)
    samp.add_argument("--modality-guidance-scale", type=float, default=10.0)

    dm = p.add_argument_group("metrics")
    dm.add_argument("--medicalnet-checkpoint", type=Path, default=None,
                    help="MedicalNet ResNet-10 weights. Without it the 3D FID and the blinded "
                         "classifier consistency group are unavailable-with-a-reason, never faked")
    dm.add_argument("--no-distribution-metrics", dest="distribution_metrics",
                    action="store_false",
                    help="skip FID / IS / precision-recall-density-coverage (always on for "
                         "--task generation, which has nothing else)")
    dm.add_argument("--fid-bootstrap", type=int, default=30)
    dm.add_argument("--min-subgroup-n", type=int, default=10)
    dm.add_argument("--diversity-k", type=int, default=5)
    dm.add_argument("--fvd-extractor", default="r3d18", choices=["r3d18", "medicalnet", "none"],
                    help="FVD-family sequence extractor. r3d18 = torchvision Kinetics-400, the "
                         "closest available analogue of standard FVD's I3D; medicalnet = the "
                         "domain 3D classifier, this pipeline's analogue of the VLM3D challenge's "
                         "FVD_CTNet. 'none' skips FVD")
    dm.add_argument("--torch-home", type=Path, default=None,
                    help="where torchvision's r3d_18 Kinetics-400 weights are staged")
    dm.add_argument("--diversity-slices-per-bucket", type=int, default=None,
                    help="mid-slices retained per bucket for intra-set MS-SSIM (0 = no cap). The "
                         "only metric that cannot stream; the cap is logged when it binds")
    dm.add_argument("--skip-metric-groups", nargs="*", default=[],
                    choices=["fidelity", "perceptual", "distribution", "report_alignment",
                             "report_consistency", "anatomy"],
                    help="can only remove groups the task declares, never add one")
    dm.add_argument("--report-classifier", type=Path, default=None,
                    help="blinded pathology classifier from cli.train_report_classifier")
    dm.add_argument("--report-labels-csv", type=Path, default=None)
    dm.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    fig = p.add_argument_group("example figures")
    fig.add_argument("--save-figures", type=int, default=3, metavar="N",
                     help="orthogonal-slice montages per bucket, written to <out>/figures/")
    fig.add_argument("--save-nifti-cases", type=int, default=0, metavar="N")

    wb = p.add_argument_group("weights & biases")
    wb.add_argument("--wandb-mode", default="disabled", choices=list(WANDB_MODES))
    wb.add_argument("--wandb-entity", default=None)
    wb.add_argument("--wandb-project", default=None)
    wb.add_argument("--wandb-group", default=None)
    wb.add_argument("--wandb-name", default=None)
    wb.add_argument("--wandb-panels", type=int, default=6, metavar="N")
    wb.add_argument("--wandb-rank-metric", default="psnr_fg")
    wb.add_argument("--wandb-log-reports", action="store_true",
                    help="allow panels, which embed the conditioning REPORT TEXT. Never set this "
                         "for a public W&B project")

    args = p.parse_args(argv)
    required = {
        "report2volume": ("checkpoint", "base_checkpoint", "vae_checkpoint"),
        "reconstruction": ("vae_checkpoint",),
        "generation": ("base_checkpoint", "vae_checkpoint"),
    }[args.task]
    missing = [n for n in required if getattr(args, n) is None]
    if missing:
        p.error(f"--task {args.task} requires --{', --'.join(m.replace('_', '-') for m in missing)}")
    return args


# --------------------------------------------------------------------------- dataset


def build_dataset(args):
    """The dataset, built exactly as `cli.train_r2v.build_dataset` builds it.

    `series_selection` is the one deliberate difference and it is not a preprocessing difference:
    training uses `"all"` so a study's report is paired with each of its ~7 series (the contrast
    that stops the adapter absorbing modality), while evaluation uses
    `"one_per_study_per_bucket"` so one study contributes one observation per bucket and a
    frequency-weighted aggregate is not silently dominated by studies that happen to carry more
    series. Nothing about the voxels changes.
    """
    import torch

    from ..data import MRReportToVolumeDataset, R2VDatasetConfig
    from ..data.reports import ShardReportStore

    config = R2VDatasetConfig(
        split=args.split,
        report_sections=tuple(args.report_sections),
        report_format=args.report_format,
        geometry_mode=args.geometry_mode,
        series_selection="one_per_study_per_bucket",
        posterior_shift_mm=args.posterior_shift_mm,
        normalizer=args.normalizer,
        dtype=torch.float32,
        seed=args.seed,
    )
    dataset = MRReportToVolumeDataset(
        str(args.manifest), ShardReportStore(str(args.report_index)), config=config
    )
    log.info("dataset: %d (report, volume) pairs in split '%s'", len(dataset), args.split)
    return dataset, config


def population_bucket_counts(dataset) -> dict:
    """Eligible-population counts per bucket, from the dataset's own sample list.

    These are the weights `overall_weighted` uses. When the whole split is evaluated they equal
    the scored counts and the two aggregate rows coincide -- correctly, because there is no
    sampling artefact left to correct for. With `--n-per-bucket` they diverge again, and the
    weighted row is the one that reflects the real split.
    """
    from ..volumes import bucket_name

    counts: dict = {}
    for sample in dataset.samples:
        b = bucket_name(sample.modality or "unknown", sample.plane or "unknown")
        counts[b] = counts.get(b, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- generators


def build_report2volume(args, dataset):
    """Trained adapter + frozen base UNet, conditioned on each case's own report.

    Assembled by `cli.generate_r2v.build_sampler`, the same function the free-form single-report
    script uses, so the evaluated path and the demo path cannot diverge on how a model is built.
    """
    from types import SimpleNamespace

    from ..eval.live import stable_seed
    from ..models.adapter import training_provenance
    from .generate_r2v import assert_report_format_matches, build_sampler

    sampler, _embedder, payload = build_sampler(SimpleNamespace(
        base_checkpoint=args.base_checkpoint, vae_checkpoint=args.vae_checkpoint,
        adapter=args.checkpoint, network_config=args.network_config,
        text_encoder=None, text_checkpoint=args.text_checkpoint, max_report_tokens=None,
        device=args.device, latent_only=False,
        report_guidance_scale=args.report_guidance_scale,
        modality_guidance_scale=args.modality_guidance_scale,
        num_inference_steps=args.num_inference_steps, seed=args.seed,
        batched_guidance=True, allow_base_mismatch=args.allow_base_mismatch,
    ))

    training = training_provenance(payload, args.checkpoint, world_size=args.train_world_size)
    log.info("training provenance: %s optimizer steps, %s epochs, %s samples seen",
             training.get("optimizer_step"), training.get("epochs_completed"),
             training.get("samples_seen"))
    assert_preprocessing_matches_training(training, args,
                                          allow_mismatch=args.allow_preprocessing_mismatch)
    assert_report_format_matches(payload, _FormatView(args.report_format),
                                 allow_mismatch=args.allow_report_format_mismatch,
                                 embedder=sampler.text_embedder)

    needs_sections = bool(getattr(sampler.text_embedder, "needs_sections", False))
    if needs_sections and not args.report_format and not args.report_sections:
        raise SystemExit("this conditioning encodes report sections separately, but no "
                         "--report-sections were given")

    def generate(case, sample):
        # postprocess=False is load-bearing: the Dataset's ground truth is percentile-normalised
        # ~[0, 1] and postprocess_mr would return int16 [0, 1000]. Every metric consumes a
        # 1000x-offset pair without complaint, which is why live.assert_metric_intensity_space
        # checks the first volume rather than trusting this line.
        return sampler.generate(
            sample["report_text"], tuple(case.shape), tuple(case.spacing_mm),
            stable_seed(args.seed, case.case_id), modality=case.sequence,
            report_sections=(dict(sample.get("report_sections_text") or {})
                             if needs_sections else None),
            postprocess=False,
        )

    identity = {"name": args.model_name or "report2volume",
                "adapter": str(args.checkpoint), "training": training,
                "report_guidance_scale": args.report_guidance_scale,
                "modality_guidance_scale": args.modality_guidance_scale,
                "num_inference_steps": args.num_inference_steps}
    return generate, identity


class _FormatView:
    """The two attributes `assert_report_format_matches` reads off a cohort, from CLI flags instead.

    A shim rather than a rewrite of that function: it is the gate that stopped an adapter being
    scored on text composed differently from its training, and it should keep having exactly one
    implementation.
    """

    def __init__(self, report_format) -> None:
        self.geometry = {"report_format": report_format}

    @property
    def has_report_sections(self) -> bool:
        return True


#: Preprocessing settings the evaluation must share with the run it scores. `report_format` is
#: handled separately by `assert_report_format_matches` -- training may sample several formats
#: while an evaluation pins one, so equality is the wrong test there.
PREPROCESSING_KEYS = ("posterior_shift_mm", "normalizer", "geometry_mode")


def assert_preprocessing_matches_training(training: dict, args, allow_mismatch: bool = False) -> None:
    """Refuse to score a model on a grid it was never trained on.

    The durable form of the 2026-08-10 fix. Under the cohort pipeline this compared a checkpoint
    against a frozen directory; it now compares it against the flags this very run is preprocessing
    with, which is strictly tighter -- there is no third artifact that can drift.
    """
    recorded = {k: training.get(k) for k in PREPROCESSING_KEYS if training.get(k) is not None}
    if not recorded:
        log.warning("this adapter's training run recorded no preprocessing settings, so it cannot "
                    "be verified. Runs before 2026-08-10 trained at posterior_shift_mm=15.0 -- "
                    "confirm --posterior-shift-mm matches.")
        return
    current = {"posterior_shift_mm": args.posterior_shift_mm, "normalizer": args.normalizer,
               "geometry_mode": args.geometry_mode}
    differences = {k: (v, current.get(k)) for k, v in recorded.items() if current.get(k) != v}
    if not differences:
        log.info("preprocessing matches training: %s", recorded)
        return
    detail = "\n".join(f"  {k}: training={t!r} evaluation={c!r}"
                       for k, (t, c) in sorted(differences.items()))
    message = (f"this evaluation would preprocess differently from the run it scores:\n{detail}\n"
               f"Every one of these changes the voxels, so paired metrics would penalise the model "
               f"for a difference it did not cause. Pass the training values, or "
               f"--allow-preprocessing-mismatch for a deliberate ablation.")
    if allow_mismatch:
        log.warning("%s (continuing: --allow-preprocessing-mismatch)", message)
    else:
        raise SystemExit(message)


def reconstruct(autoencoder, volume, divisor: int, device: str):
    """One volume in, its reconstruction out, on the identical grid.

    The VAE needs each axis divisible by a model-derived divisor. A per-bucket shape that already
    is (every one is a multiple of 32) passes through untouched; otherwise the volume is
    zero-padded at the end of each axis before encoding and the *exact same* amount is cropped back
    off after decoding, tracked by a `CropPadRecord`. The reconstruction therefore always returns
    on the case's own grid and `check_case_geometry` sees a strict match rather than a resize --
    blind resizing is the bug the geometry contract replaced.
    """
    import numpy as np
    import torch

    from ..eval import geometry_contract as G

    _padded_shape, crop_pad = G.pad_to_divisible(volume.shape, divisor)
    x = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None].to(device)
    if crop_pad is not None:
        pad_width = []
        for a in reversed(crop_pad.per_axis):        # F.pad wants last-dim-first
            pad_width.extend([a["before"], a["after"]])
        x = torch.nn.functional.pad(x, pad_width, mode="constant", value=0.0)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        z_mu, _z_sigma = autoencoder.encode(x)
        recon = autoencoder.decode(z_mu)

    out = recon[0, 0].float().cpu().numpy()
    if crop_pad is not None:
        out = G.crop_using_record(out, crop_pad)
    if out.shape != volume.shape:
        raise RuntimeError(
            f"reconstruction shape {out.shape} != input {volume.shape} after undoing padding -- "
            f"refusing to emit a volume the evaluator would have to guess about"
        )
    return out


def build_reconstruction(args, dataset):
    """The frozen NVIDIA autoencoder: encode the real volume, decode it back, same grid."""
    from ..models.nvidia import (
        DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_NETWORK_CONFIG, load_autoencoder,
    )

    autoencoder, _cfg, divisor = load_autoencoder(
        args.vae_checkpoint, DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG,
        args.network_config or DEFAULT_NETWORK_CONFIG, args.device,
    )
    log.info("autoencoder loaded; required spatial divisor = %d", divisor)

    def generate(case, sample):
        volume = sample["image"].squeeze(0).float().numpy()
        # `reconstruct` pads to the divisor and crops the identical amount back off via a
        # CropPadRecord, so the reconstruction always returns on the case's own grid and
        # check_case_geometry sees a strict match rather than a resize.
        return reconstruct(autoencoder, volume, divisor, args.device)

    from ..cohort import sha256_file

    identity = {"name": args.model_name or "nvidia_maisi_autoencoder",
                "checkpoint": str(args.vae_checkpoint),
                "checkpoint_sha256": sha256_file(args.vae_checkpoint),
                "required_divisor": divisor}
    return generate, identity


def build_generation(args, dataset):
    """The frozen base UNet from a modality label alone -- no report, no image conditioning.

    Each case is generated at **its own** shape and spacing, which is what makes the generated and
    real populations comparable per bucket: `output_size` sizes the latent noise and the spacing
    tensor is a real conditioning input to the UNet.
    """
    import numpy as np
    import torch

    from ..cohort import sha256_file
    from ..data.geometry import UNET_SPATIAL_MULTIPLE
    from ..eval.live import stable_seed
    from ..models.nvidia import (
        DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_NETWORK_CONFIG,
        load_autoencoder_and_unet, prepare_tensors, run_inference, set_random_seed,
    )

    env_config = DEFAULT_ENV_CONFIG
    autoencoder, unet, scale_factor, cfg = load_autoencoder_and_unet(
        env_config, DEFAULT_MODEL_CONFIG, args.network_config or DEFAULT_NETWORK_CONFIG,
        args.device,
        autoencoder_checkpoint_override=args.vae_checkpoint,
        unet_checkpoint_override=args.base_checkpoint,
    )
    cfg.cfg_guidance_scale = cfg.diffusion_unet_inference["cfg_guidance_scale"]
    n_levels = max(1, len(cfg.diffusion_unet_def["num_channels"])
                   if isinstance(cfg.diffusion_unet_def["num_channels"], list)
                   else len(cfg.diffusion_unet_def["attention_levels"]))
    divisor = 2 ** (n_levels - 2)
    top_region, bottom_region, _spacing, _modality = prepare_tensors(cfg, args.device)

    def generate(case, sample):
        if case.sequence not in MODALITY_CODE:
            raise ValueError(f"no NVIDIA modality code for {case.sequence!r}; "
                             f"known: {list(MODALITY_CODE)}")
        bad = [v for v in case.shape if v % UNET_SPATIAL_MULTIPLE]
        if bad:
            raise ValueError(
                f"shape {case.shape} is not a multiple of {UNET_SPATIAL_MULTIPLE} -- the diffusion "
                f"UNet's skip connections require it. Not padded here: that would change the FOV."
            )
        # NVIDIA scales its conditioning tensors by 1e2 (prepare_tensors); match it or the model
        # is conditioned on a spacing 100x off.
        spacing_tensor = torch.from_numpy(
            np.array(case.spacing_mm, dtype=float) * 1e2)[None].half().to(args.device)
        modality_tensor = MODALITY_CODE[case.sequence] * torch.ones(
            (1,), dtype=torch.long).to(args.device)
        set_random_seed(stable_seed(args.seed, case.case_id))
        with torch.no_grad():
            raw = run_inference(cfg, args.device, autoencoder, unet, scale_factor,
                                top_region, bottom_region, spacing_tensor, modality_tensor,
                                tuple(case.shape), divisor, log)
        volume = raw.astype(np.float32) / NVIDIA_MR_INTENSITY_SCALE
        if tuple(volume.shape) != tuple(case.shape):
            raise RuntimeError(f"asked for {tuple(case.shape)} but got {tuple(volume.shape)}")
        return volume

    identity = {"name": args.model_name or "nvidia_maisi_rflow_mr_brain",
                "vae_checkpoint": str(args.vae_checkpoint),
                "unet_checkpoint": str(args.base_checkpoint),
                "unet_checkpoint_sha256": sha256_file(args.base_checkpoint),
                "conditioning": "modality class code + per-case spacing tensor (report-blind)"}
    return generate, identity


BUILDERS = {
    "report2volume": build_report2volume,
    "reconstruction": build_reconstruction,
    "generation": build_generation,
}


# --------------------------------------------------------------------------- output


def _cell(value) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def read_per_case_csv(path: Path) -> list:
    """`per_case_metrics.csv` -> rows with numeric fields parsed, for W&B panel selection.

    Read back off disk rather than threaded out of the evaluator: the file is the deliverable, so
    reading it here means the W&B view and the on-disk results cannot disagree about what was
    scored. Empty when absent or rowless (an unpaired task writes none).
    """
    import csv

    if not path.is_file():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                if value in ("", None):
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def print_table(summary, cohort_view, identity) -> None:
    """The same table W&B receives, printed -- a Slurm log is where these numbers are first read."""
    from ..eval import wandb_evaluation as WB

    print(f"\n{'=' * 100}")
    print(f"EVALUATION  task={summary['task']}  run_id={summary['run_id']}  split={summary['split']}")
    print(f"scored {summary['n_scored']} / {summary['n_cohort_cases']} cases "
          f"({summary['n_excluded']} excluded)"
          + ("  [FULL SPLIT]" if summary["evaluated_full_split"]
             else f"  [--n-per-bucket {summary['n_per_bucket']}]"))
    print("=" * 100)
    columns, rows = WB.metrics_table(summary, cohort_view,
                                     _PredictionsView(summary["task"], identity))
    widths = [min(max(len(str(c)), *(len(_cell(r[i])) for r in rows)) if rows else len(str(c)), 46)
              for i, c in enumerate(columns)]
    header = "  ".join(str(c).ljust(w)[:w] for c, w in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(_cell(v).ljust(w)[:w] for v, w in zip(row, widths)))
    print("=" * 100)


class _PredictionsView:
    """The three attributes the W&B/table code reads off a `PredictionReader`. No such object
    exists any more -- volumes are never written -- so this carries the provenance instead."""

    def __init__(self, task: str, model: dict) -> None:
        self.task = task
        self.model = model
        self.items = []
        self.root = None


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} exists and is non-empty; pass --overwrite to replace it")

    from ..eval.live import (
        DEFAULT_DIVERSITY_SLICES_PER_BUCKET, LiveCohortView, LiveEvalConfig, LiveEvaluator,
        build_cases, run_fingerprint, select_eval_cases,
    )
    from ..eval.tasks import get_task

    task = get_task(args.task)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            log.warning("--device cuda but no GPU visible; falling back to cpu")
            args.device = "cpu"

    distribution = args.distribution_metrics or not task.paired
    if not task.paired and not args.distribution_metrics:
        log.info("--task %s has no paired metrics, so distribution metrics stay on", task.name)

    dataset, dataset_config = build_dataset(args)
    indices = select_eval_cases(dataset, args.n_per_bucket)
    cases = build_cases(dataset, indices)
    if not cases:
        raise SystemExit(f"no cases in split {args.split!r} -- check --manifest and --split")

    generate, identity = BUILDERS[args.task](args, dataset)

    run_id = run_fingerprint(
        split=args.split, cases=cases, geometry=dataset_config.geometry_fingerprint(),
        task=task.name, n_per_bucket=args.n_per_bucket, seed=args.seed, model_identity=identity,
    )
    cohort_view = LiveCohortView(
        cases=cases, split=args.split, geometry=dataset_config.geometry_fingerprint(),
        population_bucket_counts=population_bucket_counts(dataset), run_id=run_id,
    )
    log.info("run_id=%s over %d cases, %d buckets%s", run_id, len(cases),
             len(cohort_view.buckets),
             " (FULL SPLIT)" if args.n_per_bucket is None else f" (first {args.n_per_bucket}/bucket)")
    for bucket in cohort_view.buckets:
        geom = cohort_view.bucket_geometry(bucket)
        log.info("   %-16s n=%-6d shape=%s spacing=%s", bucket, geom["n"],
                 geom["shape_xyz"], [round(v, 4) for v in geom["spacing_mm_xyz"]])

    config = LiveEvalConfig(
        task=task, output_dir=args.out, split=args.split, n_per_bucket=args.n_per_bucket,
        seed=args.seed, device=args.device, distribution_metrics=distribution,
        medicalnet_checkpoint=args.medicalnet_checkpoint, fid_bootstrap=args.fid_bootstrap,
        min_subgroup_n=args.min_subgroup_n, diversity_k=args.diversity_k,
        fvd_extractor=(None if args.fvd_extractor == "none" else args.fvd_extractor),
        torch_home=args.torch_home,
        diversity_slices_per_bucket=(args.diversity_slices_per_bucket
                                     if args.diversity_slices_per_bucket is not None
                                     else DEFAULT_DIVERSITY_SLICES_PER_BUCKET),
        skip_metric_groups=tuple(args.skip_metric_groups),
        save_figures=args.save_figures, save_nifti_cases=args.save_nifti_cases,
        report_classifier=args.report_classifier, report_labels_csv=args.report_labels_csv,
        wandb_panels=(args.wandb_panels if args.wandb_mode != "disabled" else 0),
        wandb_log_reports=args.wandb_log_reports,
        extra_run_metadata={"model": identity, "dataset_config": dataset_config.geometry_fingerprint()},
    )
    evaluator = LiveEvaluator(dataset, cases, config, cohort_view)
    summary = evaluator.run(generate)

    if args.wandb_mode != "disabled":
        from ..eval import wandb_evaluation as WB
        from ..eval.wandb_logging import WandbRun

        run = WandbRun(
            mode=args.wandb_mode, entity=args.wandb_entity, project=args.wandb_project,
            run_name=(args.wandb_name
                      or f"{task.name}-{run_id}-{os.environ.get('SLURM_JOB_ID', 'local')}"),
            group=args.wandb_group or f"mr-rate-{task.name}",
            tags=[task.name, args.split],
            config={"task": task.name, "run_id": run_id, "split": args.split,
                    "n_per_bucket": args.n_per_bucket, "seed": args.seed, "model": identity,
                    "metric_groups_computed": summary.get("metric_groups_computed"),
                    "n_cases": summary.get("n_cohort_cases")},
        )
        # n_panels=0: `log_evaluation` renders panels by re-reading stored volumes, and there are
        # none. The evaluator rendered them inline instead, on the rank that held the volume, and
        # they are logged below.
        logged = WB.log_evaluation(
            run, summary, cohort_view, _PredictionsView(task.name, identity),
            metric_rows=read_per_case_csv(args.out / "per_case_metrics.csv"),
            n_panels=0, log_reports=args.wandb_log_reports,
            rank_metric=args.wandb_rank_metric,
        )
        for panel in evaluator.panels:
            run.log_html(f"examples/{panel['bucket']}/{panel['case_id']}", panel["html"])
        logged["panels"] = [{"case_id": p["case_id"], "bucket": p["bucket"]}
                            for p in evaluator.panels]
        logged["panels_withheld_reason"] = (
            None if evaluator.panels
            else ("--wandb-log-reports not set; panels embed patient report text"
                  if not args.wandb_log_reports else "no panel rendered"))
        log.info("W&B: %d table rows, %d panels", logged.get("table_rows", 0),
                 len(logged["panels"]))
        (args.out / "wandb_run.json").write_text(
            json.dumps({**run.finish(), "logged": logged}, indent=2))

    print_table(summary, cohort_view, identity)
    print(f"full results -> {args.out}/summary.json  (CSVs: metrics_per_bucket, metrics_summary, "
          f"report_consistency_per_label)")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
