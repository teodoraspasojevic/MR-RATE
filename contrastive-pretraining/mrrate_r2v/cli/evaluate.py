#!/usr/bin/env python3
"""The evaluator. Builds the dataset training builds, generates, and scores against the official
VLM3D `mr-volume-generation` challenge metrics -- in one command.

    python -m mrrate_r2v.cli.evaluate --task <task> --manifest ... --out <results_dir>

**There is no cohort and no prediction set.** This CLI is `cli.train_r2v` with the backward pass
replaced by a sampler: it calls the same `build_dataset`, constructs the same `R2VDatasetConfig`
from the same flags, and resolves each case's grid through the same `dataset.geometry.resolve`.

Three tasks, differing only in how a volume is produced -- every task is scored the same way:

    report2volume    trained adapter + frozen base UNet, conditioned on the case's report
    reconstruction   the frozen NVIDIA autoencoder, encode then decode the real volume
    generation       the frozen base UNet from a modality label alone, report-blind

**Metrics are the official challenge's, computed by the vendored code in `eval/challenge/`**
(see `eval/challenge_metrics.py`): `MSE_mean`, `PSNR_mean`, `SSIM_mean`, `FID_2p5D_XY/XZ/YZ/Avg`,
`dice` (a copy of `SSIM_mean` -- the platform's own primary-metric shim), plus
`n_total_files`/`n_scored_files`/`n_missing_outputs`/`n_excluded_out_of_scope_modality`. Only
T1w/T2w/FLAIR/SWI are scored, matching the platform's modality scope.

**What is evaluated.** Every case in the split, in a deterministic no-RNG order, unless
`--n-per-bucket N` caps it -- and that cap takes the *first* N per bucket in the same order, so a
cheap run is a prefix of the full one. Per-case sampler noise is `--seed + <dataset index>`, so a
rerun reproduces every volume.

Results: one `<out>/metrics.json`, shaped `{"metrics": {...}, "per_case": [...], ...}`.
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

TASK_NAMES = ("report2volume", "reconstruction", "generation")


def parse_args(argv=None):
    from ..data import R2VDatasetConfig
    from ..eval.wandb_logging import WANDB_MODES

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(TASK_NAMES))
    p.add_argument("--out", type=Path, required=True, help="results directory to create")
    p.add_argument("--overwrite", action="store_true")

    data = p.add_argument_group("data (identical to cli.train_r2v)")
    data.add_argument("--manifest", type=Path, required=True, help="MR-RATE manifest CSV")
    data.add_argument("--report-index", type=Path, required=True, help="report index CSV")
    data.add_argument("--split", default="test", help="MR-RATE split to evaluate")
    data.add_argument("--report-sections", nargs="+", default=["findings", "impression"])
    data.add_argument("--report-format", default=None,
                      help="a named format from mrrate_r2v.textenc.formats. Must be one the "
                           "adapter was trained on")
    data.add_argument("--geometry-mode", default="per_modality_plane",
                      choices=["per_modality_plane", "fixed"])
    data.add_argument("--posterior-shift-mm", type=float,
                      default=R2VDatasetConfig.posterior_shift_mm,
                      help="must match the training run -- checked, not trusted")
    data.add_argument("--normalizer", default=R2VDatasetConfig.normalizer,
                      choices=["percentile", "zscore", "minmax"])

    sel = p.add_argument_group("case selection (deterministic, no RNG)")
    sel.add_argument("--n-per-bucket", type=int, default=None,
                     help="cases per (modality, plane). Default: the ENTIRE split. The cap keeps "
                          "the first N of the same deterministic order, so it is a prefix of the "
                          "full run, never a different sample")
    sel.add_argument("--seed", type=int, default=42, help="seeds sampler noise only")
    sel.add_argument("--save-volumes", action="store_true",
                     help="keep every generated volume under <out>/volumes/ as one raw float16 "
                          "stack per bucket plus an index.json. ~19 GB for a 2,000-case run, so "
                          "off by default; bundled per bucket because /hnvme's binding limit is a "
                          "file COUNT quota, not space")
    sel.add_argument("--gt-space", default="model", choices=["model", "native"],
                     help="which ground truth to score against. 'model' (default) is the "
                          "preprocessed volume on the model's own bucket grid. 'native' is the "
                          "released volume RAS-reoriented and otherwise untouched -- no resample, "
                          "no normalize, no crop/pad -- which is the geometry the challenge scores "
                          "against. Generation is unaffected either way: the model always samples "
                          "on its bucket grid and the metric resamples the generated volume onto "
                          "the ground truth")

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
    model.add_argument("--model-name", default=None, help="recorded in metrics.json")
    model.add_argument("--allow-base-mismatch", action="store_true")
    model.add_argument("--allow-report-format-mismatch", action="store_true")

    samp = p.add_argument_group("sampling")
    samp.add_argument("--num-inference-steps", type=int, default=30)
    samp.add_argument("--report-guidance-scale", type=float, default=4.0)
    samp.add_argument("--modality-guidance-scale", type=float, default=10.0)
    samp.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    wb = p.add_argument_group("weights & biases")
    wb.add_argument("--wandb-mode", default="disabled", choices=list(WANDB_MODES))
    wb.add_argument("--wandb-entity", default=None)
    wb.add_argument("--wandb-project", default=None)
    wb.add_argument("--wandb-group", default=None)
    wb.add_argument("--wandb-name", default=None)
    wb.add_argument("--wandb-panels", type=int, default=6, metavar="N",
                    help="example ground-truth-vs-generated panels per bucket")
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
    """The dataset, built by the same `data.build_r2v_dataset` `cli.train_r2v` uses.

    `series_selection` is the one deliberate difference and it is not a preprocessing difference:
    training uses `"all"` so a study's report is paired with each of its series, while evaluation
    uses `"one_per_study_per_bucket"` so one study contributes one observation per bucket.
    """
    from ..data import build_r2v_dataset

    dataset = build_r2v_dataset(
        args.manifest, args.report_index, split=args.split, report_format=args.report_format,
        geometry_mode=args.geometry_mode, series_selection="one_per_study_per_bucket",
        posterior_shift_mm=args.posterior_shift_mm, normalizer=args.normalizer, seed=args.seed,
        report_sections=args.report_sections,
        gt_space=("native" if args.gt_space == "native" else "off"),
    )
    log.info("dataset: %d (report, volume) pairs in split %r", len(dataset), args.split)
    return dataset, dataset.config


# --------------------------------------------------------------------------- generators

def build_report2volume(args, dataset):
    """Trained adapter + frozen base UNet, conditioned on each case's own report.

    Assembled by `cli.generate_r2v.build_sampler`, the same function the free-form single-report
    script uses, so the evaluated path and the demo path cannot diverge on how a model is built.
    """
    from types import SimpleNamespace

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

    training = {"optimizer_step": payload.get("optimizer_step") or payload.get("step"),
                "epoch": payload.get("epoch"), "loss": payload.get("loss")}
    log.info("training provenance: %s optimizer steps, epoch %s, loss %s",
             training["optimizer_step"], training["epoch"], training["loss"])
    assert_report_format_matches(payload, _FormatView(args.report_format),
                                 allow_mismatch=args.allow_report_format_mismatch,
                                 embedder=sampler.text_embedder)

    needs_sections = bool(getattr(sampler.text_embedder, "needs_sections", False))
    if needs_sections and not args.report_format and not args.report_sections:
        raise SystemExit("this conditioning encodes report sections separately, but no "
                         "--report-sections were given")

    def generate(case, sample):
        # postprocess=False is load-bearing: the Dataset's ground truth is percentile-normalised
        # ~[0, 1], not sampling.postprocess_mr's int16 [0, 1000].
        return sampler.generate(
            sample["report_text"], tuple(case.shape), tuple(case.spacing_mm),
            args.seed + case.index, modality=case.sequence,
            report_sections=(dict(sample.get("report_sections_text") or {})
                             if needs_sections else None),
            postprocess=False,
        )

    identity = {"name": args.model_name or "report2volume", "adapter": str(args.checkpoint),
               "training": training, "report_guidance_scale": args.report_guidance_scale,
               "modality_guidance_scale": args.modality_guidance_scale,
               "num_inference_steps": args.num_inference_steps}
    return generate, identity


class _FormatView:
    """The two attributes `assert_report_format_matches` reads off a cohort, from CLI flags instead."""

    def __init__(self, report_format) -> None:
        self.geometry = {"report_format": report_format}

    @property
    def has_report_sections(self) -> bool:
        return True


def reconstruct(autoencoder, volume, divisor: int, device: str):
    """One volume in, its reconstruction out, on the identical grid.

    The VAE needs each axis divisible by a model-derived divisor; a per-bucket shape is padded at
    the end of each axis before encoding and the *exact same* amount cropped back off after
    decoding, so the reconstruction always returns on the case's own grid.
    """
    import numpy as np
    import torch

    from ..eval.padding import crop_using_record, pad_to_divisible

    _padded_shape, crop_pad = pad_to_divisible(volume.shape, divisor)
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
        out = crop_using_record(out, crop_pad)
    if out.shape != volume.shape:
        raise RuntimeError(f"reconstruction shape {out.shape} != input {volume.shape} after "
                           f"undoing padding")
    return out


def build_reconstruction(args, dataset):
    """The frozen NVIDIA autoencoder: encode the real volume, decode it back, same grid."""
    from ..models.adapter import sha256_file
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
        return reconstruct(autoencoder, volume, divisor, args.device)

    identity = {"name": args.model_name or "nvidia_maisi_autoencoder",
               "checkpoint": str(args.vae_checkpoint),
               "checkpoint_sha256": sha256_file(args.vae_checkpoint), "required_divisor": divisor}
    return generate, identity


def build_generation(args, dataset):
    """The frozen base UNet from a modality label alone -- no report, no image conditioning."""
    import numpy as np
    import torch

    from ..models.adapter import sha256_file
    from ..data.geometry import UNET_SPATIAL_MULTIPLE
    from ..models.nvidia import (
        DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_NETWORK_CONFIG,
        load_autoencoder_and_unet, prepare_tensors, run_inference, set_random_seed,
    )

    env_config = DEFAULT_ENV_CONFIG
    autoencoder, unet, scale_factor, cfg = load_autoencoder_and_unet(
        env_config, DEFAULT_MODEL_CONFIG, args.network_config or DEFAULT_NETWORK_CONFIG,
        args.device, autoencoder_checkpoint_override=args.vae_checkpoint,
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
            raise ValueError(f"shape {case.shape} is not a multiple of {UNET_SPATIAL_MULTIPLE}")
        spacing_tensor = torch.from_numpy(
            np.array(case.spacing_mm, dtype=float) * 1e2)[None].half().to(args.device)
        modality_tensor = MODALITY_CODE[case.sequence] * torch.ones(
            (1,), dtype=torch.long).to(args.device)
        set_random_seed(args.seed + case.index)
        with torch.no_grad():
            raw = run_inference(cfg, args.device, autoencoder, unet, scale_factor,
                                top_region, bottom_region, spacing_tensor, modality_tensor,
                                tuple(case.shape), divisor, log)
        volume = raw.astype(np.float32) / NVIDIA_MR_INTENSITY_SCALE
        if tuple(volume.shape) != tuple(case.shape):
            raise RuntimeError(f"asked for {tuple(case.shape)} but got {tuple(volume.shape)}")
        return volume

    identity = {"name": args.model_name or "nvidia_maisi_rflow_mr_brain",
               "vae_checkpoint": str(args.vae_checkpoint), "unet_checkpoint": str(args.base_checkpoint),
               "unet_checkpoint_sha256": sha256_file(args.base_checkpoint),
               "conditioning": "modality class code + per-case spacing tensor (report-blind)"}
    return generate, identity


BUILDERS = {
    "report2volume": build_report2volume,
    "reconstruction": build_reconstruction,
    "generation": build_generation,
}


# --------------------------------------------------------------------------- output

def print_metrics(result: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"EVALUATION  task={result['task']}  split={result['split']}")
    print("-" * 70)
    for key, value in result["metrics"].items():
        print(f"  {key:35s} {value}")
    print("=" * 70)


def log_wandb(args, result: dict, panels: list, identity: dict) -> None:
    from ..eval.wandb_logging import WandbRun

    run = WandbRun(
        mode=args.wandb_mode, entity=args.wandb_entity, project=args.wandb_project,
        run_name=args.wandb_name or f"{args.task}-{args.split}-"
                                    f"{os.environ.get('SLURM_JOB_ID', 'local')}",
        group=args.wandb_group or f"mr-rate-{args.task}", tags=[args.task, args.split],
        config={"task": args.task, "split": args.split, "n_per_bucket": args.n_per_bucket,
               "seed": args.seed, "model": identity},
    )
    rows = [[key, value] for key, value in result["metrics"].items()]
    run.log_table("challenge_metrics", ["metric", "value"], rows)
    run.set_summary(result["metrics"])
    run.log(result["metrics"])
    for panel in panels:
        run.log_html(f"examples/{panel['bucket']}/{panel['case_id']}", panel["html"])
    log.info("W&B: %d metric rows, %d panels", len(rows), len(panels))
    (args.out / "wandb_run.json").write_text(json.dumps(run.finish(), indent=2))


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} exists and is non-empty; pass --overwrite to replace it")

    from ..eval.live import LiveEvalConfig, LiveEvaluator, build_cases, select_eval_cases

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            log.warning("--device cuda but no GPU visible; falling back to cpu")
            args.device = "cpu"

    dataset, dataset_config = build_dataset(args)
    indices = select_eval_cases(dataset, args.n_per_bucket)
    cases = build_cases(dataset, indices)
    if not cases:
        raise SystemExit(f"no cases in split {args.split!r} -- check --manifest and --split")

    generate, identity = BUILDERS[args.task](args, dataset)

    bucket_counts: dict = {}
    for case in cases:
        bucket_counts[case.bucket] = bucket_counts.get(case.bucket, 0) + 1
    scale = f"(first {args.n_per_bucket}/bucket)" if args.n_per_bucket else "(FULL SPLIT)"
    log.info("%d cases, %d buckets %s", len(cases), len(bucket_counts), scale)
    for bucket, n in sorted(bucket_counts.items()):
        log.info("   %-16s n=%d", bucket, n)

    config = LiveEvalConfig(
        task_name=args.task, output_dir=args.out, split=args.split, n_per_bucket=args.n_per_bucket,
        device=args.device,
        wandb_panels=(args.wandb_panels if args.wandb_mode != "disabled" else 0),
        wandb_log_reports=args.wandb_log_reports, save_volumes=args.save_volumes,
        extra_run_metadata={"model": identity, "dataset_config": dataset_config.geometry_fingerprint()},
    )
    evaluator = LiveEvaluator(dataset, cases, config)
    result = evaluator.run(generate)

    if args.wandb_mode != "disabled":
        log_wandb(args, result, evaluator.panels, identity)

    print_metrics(result)
    print(f"full results -> {args.out}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
