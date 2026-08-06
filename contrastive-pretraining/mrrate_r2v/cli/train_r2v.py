#!/usr/bin/env python3
"""Train the report-conditioning adapter on MR-RATE. The base NVIDIA denoiser stays frozen.

Follows `NV-Generate-CTMR/scripts/diff_model_train.py` (see `mrrate_r2v/training.py` for the
line-by-line mapping) and reads its geometry, scheduler and modality table from NVIDIA's own configs.

Single-GPU smoke run (2 steps, mock text encoder, no data needed):

    python -m mrrate_r2v.cli.train_r2v --dry-run --max-steps 2 \\
        --out /tmp/r2v_dryrun --text-encoder mock

Real single-GPU run:

    python -m mrrate_r2v.cli.train_r2v \\
        --manifest   <data>/r2v_manifest/manifest_shards_native.csv \\
        --report-index <data>/r2v_manifest/report_index_shards_native.csv \\
        --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \\
        --vae-checkpoint  <ws>/models/autoencoder_v1.pt \\
        --text-encoder radbert \\
        --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \\
        --out <ws>/runs/r2v_adapter_v1 --epochs 1 --batch-size 1

Multi-GPU, official launcher style (`torchrun`, matching diff_model_train.py's DDP setup):

    torchrun --nproc_per_node=4 -m mrrate_r2v.cli.train_r2v --num-gpus 4 ... (same flags)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("train_r2v")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the MR-RATE report-conditioning adapter on a frozen NV-Generate-MR-Brain UNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_argument_group("data")
    data.add_argument("--manifest", type=Path, help="MR-RATE manifest CSV (cli.build_manifest output)")
    data.add_argument("--report-index", type=Path, help="report index CSV for ShardReportStore")
    data.add_argument("--split", default="train")
    data.add_argument("--report-sections", nargs="+", default=["findings", "impression"],
                      help="ignored when --report-format is given")
    data.add_argument("--report-format", default=None,
                      help="a named format from mrrate_r2v.textenc.formats (e.g. "
                           "impression_findings). Default: --report-sections joined, i.e. the "
                           "historical behaviour")
    data.add_argument("--geometry-mode", default="per_modality_plane", choices=["per_modality_plane", "fixed"])
    data.add_argument("--bucket-order", default="interleave", choices=["interleave", "shuffle"],
                      help="'interleave': consecutive batches carry different modalities; "
                           "'shuffle': one flat shuffle. Both use every series exactly once per "
                           "epoch, with no frequency weighting")
    data.add_argument("--num-workers", type=int, default=4)
    data.add_argument("--dry-run", action="store_true",
                     help="synthetic latents and reports; no manifest, VAE or dataset needed")

    model = p.add_argument_group("model")
    model.add_argument("--base-checkpoint", type=Path, help="NVIDIA diffusion UNet checkpoint (frozen)")
    model.add_argument("--vae-checkpoint", type=Path, help="NVIDIA autoencoder checkpoint (frozen)")
    model.add_argument("--network-config", type=Path, default=None, help="NVIDIA config_network_rflow.json")
    model.add_argument("--cross-attention-dim", type=int, default=512,
                       help="adapter/context width fed to the per-block K/V projections")
    model.add_argument("--conditioning-levels", nargs="+", type=int, default=None,
                       help="1/0 per UNet level; default = the base model's attention_levels")
    model.add_argument("--no-condition-mid", dest="condition_mid", action="store_false")
    model.add_argument("--context-hidden-dim", type=int, default=None)

    text = p.add_argument_group("text encoder")
    text.add_argument("--conditioning", default=None, choices=_conditioning_choices(),
                      help="a named conditioning configuration (encoder set + pooling + section "
                           "layout + recommended --report-format). Takes precedence over "
                           "--text-encoder. Default: unset, i.e. the historical token-level path")
    text.add_argument("--text-encoder", default="radbert", choices=_text_encoder_choices(),
                      help="ignored when --conditioning is given. 'radbert' and 'mock' are the "
                           "originals; the rest are the textenc zoo")
    text.add_argument("--text-checkpoint", type=Path, default=None,
                      help="local snapshot directory; zoo encoders default to "
                           "$MRRATE_PRETRAINED_DIR/<their directory>")
    text.add_argument("--text-pooling", default=None, choices=["cls", "mean"],
                      help="override a configuration's pooling. Default: each encoder's own spec "
                           "value, which is what the text-encoder study was run under")
    text.add_argument("--text-trainable", action="store_true",
                      help="unfreeze the text encoder(s). Default frozen, per the study")
    text.add_argument("--max-report-tokens", type=int, default=512)
    text.add_argument("--mock-output-dim", type=int, default=32, help="--text-encoder mock only")

    val = p.add_argument_group("validation")
    val.add_argument("--val-split", default="val")
    val.add_argument("--validate-full-every-steps", type=int, default=None,
                     help="optimizer steps between *full* validation passes; quick otherwise")
    val.add_argument("--val-quick-samples", type=int, default=64,
                     help="fixed, seeded, bucket-stratified subset used at every validation step. "
                          "Must stay >= 16 or the Frechet metrics are withheld as unreliable")
    val.add_argument("--val-metrics", nargs="+", default=["fvd", "fid_2p5d", "ssim"],
                     choices=["fvd", "fid_2p5d", "ssim"],
                     help="fvd and fid_2p5d are distribution-level (lower better); ssim is paired "
                          "(higher better). None measures report-to-volume semantic agreement")
    val.add_argument("--val-fvd-extractor", default="r3d18",
                     choices=["r3d18", "medicalnet"],
                     help="r3d18 = torchvision Kinetics-400 (primary); medicalnet = the staged 3D "
                          "classifier, this pipeline's analogue of GenerateCT's FVD_CT-Net. "
                          "Neither is I3D, so neither is standard FVD")
    val.add_argument("--val-sensitivity-every-steps", type=int, default=None,
                     help="condition-sensitivity diagnostic interval, in optimizer steps. Swaps in "
                          "another study's report at a fixed seed to check the model uses its text "
                          "at all. Costs a second generation for --val-sensitivity-samples cases")
    val.add_argument("--val-sensitivity-samples", type=int, default=8,
                     help="cases the sensitivity diagnostic uses (a small subset of the quick set, "
                          "not the whole validation split)")
    val.add_argument("--val-sequence-frames", type=int, default=16,
                     help="frames per FVD sequence; fixed per volume so a thick volume cannot "
                          "contribute more evidence than a thin one")
    val.add_argument("--val-feature-cache", type=Path, default=None,
                     help="directory for cached real features (real volumes never change)")
    val.add_argument("--validation-reference", type=Path, default=None,
                     help="JSON from cli.validation_reference: real-vs-real noise floors and SSIM "
                          "ceilings, logged as flat reference lines and never recomputed")
    val.add_argument("--torch-home", type=Path, default=None,
                     help="where torchvision's r3d_18 Kinetics-400 weights are staged")
    val.add_argument("--val-full-samples", type=int, default=None,
                     help="the larger set; must be >= --val-quick-samples (the quick set is its prefix)")
    val.add_argument("--val-inference-steps", type=int, default=30,
                     help="sampler steps during validation. Lower than a real generation run on "
                          "purpose: the metric only has to rank checkpoints against each other")
    val.add_argument("--val-visualize", type=int, default=4,
                     help="cases rendered into the interactive W&B panel each validation step")
    val.add_argument("--val-seed", type=int, default=0,
                     help="fixes the validation subset and the sampler noise, so the curve is "
                          "comparable across runs and across resumes")
    val.add_argument("--no-validate-at-end", dest="validate_at_end", action="store_false")
    val.add_argument("--validate-full-at-end", action="store_true")
    val.add_argument("--medicalnet-checkpoint", type=Path, default=None,
                     help="MedicalNet ResNet-10 weights for the FID proxy; default "
                          "$MRRATE_PRETRAINED_DIR/medicalnet/resnet_10_23dataset.pth")

    wb = p.add_argument_group("weights & biases")
    wb.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"],
                    help="training works identically when this is 'disabled' or when wandb is "
                         "not installed")
    wb.add_argument("--wandb-project", default=None)
    wb.add_argument("--wandb-entity", default=None)
    wb.add_argument("--wandb-group", default=None)
    wb.add_argument("--wandb-run-name", default=None)
    wb.add_argument("--wandb-log-reports", action="store_true",
                    help="include report text in the validation panel. Off by default: reports are "
                         "patient data even after anonymisation, so this must be opted into and "
                         "should never point at a public project")

    train = p.add_argument_group("optimisation")
    train.add_argument("--out", type=Path, required=True, help="output directory for checkpoints")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-5, help="NVIDIA's own diffusion_unet_train.lr")
    train.add_argument("--grad-accumulation-steps", type=int, default=1)
    train.add_argument("--grad-clip-norm", type=float, default=None)
    train.add_argument("--no-amp", dest="amp", action="store_false", help="disable mixed precision")
    train.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16"],
                       help="bfloat16 is the default and diverges from NVIDIA deliberately: "
                            "float16 autocast overflowed to NaN at the same step for every LR "
                            "from 1e-5 to 3e-4. bf16 is native on H200 and needs no GradScaler")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--save-every-steps", type=int, default=None,
                       help="optimizer steps between periodic checkpoints")
    train.add_argument("--validate-every-steps", type=int, default=None,
                       help="optimizer steps between validation passes (not micro-steps)")
    train.add_argument("--keep-last-n", type=int, default=3,
                       help="periodic checkpoints to retain; 'last' and 'best_*' are never pruned")
    train.add_argument("--save-format", default="adapter", choices=["adapter", "full", "both"])
    train.add_argument("--resume", type=Path, default=None, help="adapter checkpoint to resume from")
    train.add_argument("--scale-factor", default="auto",
                       help="'auto' = from the base checkpoint (matches official inference), "
                            "'recompute' = 1/std(z) of the first batch (official training), or a literal")
    train.add_argument("--num-gpus", type=int, default=1,
                       help="TOTAL world size (ranks), not GPUs per node. 2 nodes x 4 GPUs = 8. "
                            "Cross-checked against torchrun's WORLD_SIZE, which is the only thing "
                            "that actually knows")
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    cond = p.add_argument_group("conditioning")
    cond.add_argument("--report-dropout-probability", type=float, default=0.1)
    cond.add_argument("--modality-dropout-probability", type=float, default=0.1,
                      help="NVIDIA's augment_modality_label(prob=...); default is theirs")
    args = p.parse_args(argv)

    if not args.dry_run:
        missing = [name for name in ("manifest", "report_index", "base_checkpoint", "vae_checkpoint")
                   if getattr(args, name) is None]
        if missing:
            p.error(f"--{', --'.join(m.replace('_', '-') for m in missing)} required unless --dry-run")
    # `--text-encoder` keeps its "radbert" default even when `--conditioning` supersedes it, so this
    # legacy requirement must not fire in that case -- it predates `--conditioning` and would
    # otherwise reject every named configuration for a checkpoint it never uses.
    if (not args.conditioning and args.text_encoder == "radbert"
            and args.text_checkpoint is None and not args.dry_run):
        p.error("--text-checkpoint is required for --text-encoder radbert "
                "(not needed with --conditioning, which resolves its own checkpoints)")
    return args


def _text_encoder_choices():
    """The originals first, then whatever the zoo has -- so `--help` lists every real option and
    argparse rejects a typo before a queue wait rather than after one."""
    names = ["radbert", "mock"]
    try:
        from ..textenc.encoders import ENCODER_SPECS
    except Exception:  # noqa: BLE001 -- the zoo's deps are optional
        return names
    return names + [n for n in sorted(ENCODER_SPECS) if n not in names]


def _conditioning_choices():
    try:
        from ..textenc.conditioning import CONDITIONING_CONFIGS
    except Exception:  # noqa: BLE001 -- the zoo's deps are optional
        return []
    return sorted(CONDITIONING_CONFIGS)


def resolve_report_format(args):
    """The report format actually used, and where it came from.

    `--report-format` always wins. Otherwise a named `--conditioning` contributes its own
    recommended format, and with neither set the historical `--report-sections` behaviour stands.
    `report2ct_style` deliberately has no format: it encodes sections separately and never joins
    them, so a joined string would be the wrong input entirely.
    """
    if args.report_format is not None:
        return args.report_format, "--report-format"
    if args.conditioning:
        from ..textenc.conditioning import CONDITIONING_CONFIGS

        spec = CONDITIONING_CONFIGS[args.conditioning]
        if spec.get("report_format"):
            return spec["report_format"], f"--conditioning {args.conditioning}"
    return None, "--report-sections (historical default)"


def build_text_embedder_from_args(args):
    from ..text import build_text_embedder

    if args.conditioning:
        from ..textenc.conditioning import build_conditioning

        checkpoints = {}
        if args.text_checkpoint is not None:
            from ..textenc.conditioning import CONDITIONING_CONFIGS

            encoders = CONDITIONING_CONFIGS[args.conditioning]["encoders"]
            if len(encoders) != 1:
                raise SystemExit(
                    f"--text-checkpoint takes a single directory, but --conditioning "
                    f"{args.conditioning} uses {len(encoders)} encoders {list(encoders)}. Stage "
                    "them under MRRATE_PRETRAINED_DIR instead."
                )
            checkpoints[encoders[0]] = str(args.text_checkpoint)
        return build_conditioning(
            args.conditioning, max_length=args.max_report_tokens, pooling=args.text_pooling,
            checkpoints=checkpoints, trainable=args.text_trainable,
        )

    if args.text_encoder == "mock":
        return build_text_embedder("mock", output_dim=args.mock_output_dim, max_length=16)
    if args.text_encoder != "radbert":
        kwargs = {"max_length": args.max_report_tokens, "trainable": args.text_trainable}
        if args.text_checkpoint is not None:
            kwargs["checkpoint"] = str(args.text_checkpoint)
        return build_text_embedder(args.text_encoder, **kwargs)
    return build_text_embedder(
        "radbert", checkpoint=str(args.text_checkpoint), max_length=args.max_report_tokens
    )


# --------------------------------------------------------------------------- distributed


def setup_distributed(args):
    """Initialise DDP from torchrun's environment. Returns `(device, rank, local_rank, world_size)`.

    Reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, which `torchrun` sets -- **not** `--num-gpus`, which
    cannot tell a process what its own rank is. `--num-gpus` is now only a declared expectation and
    is cross-checked against the real world size, so a mismatched launcher fails immediately rather
    than silently training N independent single-GPU models (which is what this CLI did before).

    Device assignment is rank-local (`cuda:LOCAL_RANK`) and `set_device` is called before the
    process group is created, so NCCL binds each rank to its own GPU. This is identical for
    single- and multi-node: `LOCAL_RANK` is per-node (0..gpus_per_node-1) while `RANK` is global,
    which is exactly the split `cuda:LOCAL_RANK` + a global process group needs.
    """
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.num_gpus > 1 and world_size == 1:
        raise SystemExit(
            f"--num-gpus {args.num_gpus} but WORLD_SIZE=1: this process was not launched by "
            f"torchrun, so it would train alone. Use:\n"
            f"    torchrun --nproc_per_node={args.num_gpus} -m mrrate_r2v.cli.train_r2v "
            f"--num-gpus {args.num_gpus} ..."
        )
    if world_size > 1 and args.num_gpus not in (1, world_size):
        raise SystemExit(f"--num-gpus {args.num_gpus} disagrees with WORLD_SIZE={world_size}")

    if world_size == 1:
        return torch.device(args.device), 0, 0, 1

    import torch.distributed as dist

    if args.device == "cpu" or not torch.cuda.is_available():
        backend, device = "gloo", torch.device("cpu")
    else:
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    log.info("DDP rank %d/%d on %s (backend %s)", rank, world_size, device, backend)
    return device, rank, local_rank, world_size


class ShardedBatchSampler:
    """Every `world_size`-th batch of an underlying batch sampler, offset by `rank`.

    Sharding *batches* rather than samples is what keeps one batch inside one (modality, plane)
    bucket -- `GeometryBucketBatchSampler` already guarantees that per batch, and splitting a batch
    across ranks would reintroduce the mixed-shape `collate_fn_r2v` error it exists to prevent.

    Every rank yields exactly `len(underlying) // world_size` batches. The truncation is not
    cosmetic: an uneven split leaves one rank short, and its missing allreduce hangs the others at
    the end of the epoch. Dropped batches are logged, never silent.
    """

    def __init__(self, underlying, rank: int, world_size: int) -> None:
        self.underlying = underlying
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._length = len(underlying) // self.world_size
        dropped = len(underlying) - self._length * self.world_size
        if dropped and self.rank == 0:
            log.info("DDP: %d of %d batches/epoch dropped so every rank runs %d "
                     "(uneven split would deadlock at the epoch boundary)",
                     dropped, len(underlying), self._length)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for i, batch in enumerate(self.underlying):
            if i >= self._length * self.world_size:
                break
            if i % self.world_size == self.rank:
                yield batch

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.underlying, "set_epoch"):
            self.underlying.set_epoch(epoch)
        self._length = len(self.underlying) // self.world_size


def build_dataset(args, split: str, report_format):
    from ..data import MRReportToVolumeDataset, R2VDatasetConfig
    from ..data.reports import ShardReportStore

    config = R2VDatasetConfig(
        split=split,
        report_sections=tuple(args.report_sections),
        report_format=report_format,
        geometry_mode=args.geometry_mode,
        # "all": every eligible series is a training sample, so a study's report is paired with
        # each of its ~7 series. That contrast -- one report, several modalities, distinguished
        # only by class_labels/spacing_tensor -- is what stops the report adapter from absorbing
        # modality. The one-per-study modes exist for cohort construction, not for training.
        series_selection="all",
        dtype=torch.float32,
        seed=args.seed,
    )
    dataset = MRReportToVolumeDataset(
        str(args.manifest), ShardReportStore(str(args.report_index)), config=config
    )
    log.info("dataset: %d (report, volume) pairs in split '%s'", len(dataset), split)
    return dataset


def build_dataloader(args, log, dataset, rank: int = 0, world_size: int = 1):
    from torch.utils.data import DataLoader

    from ..data import GeometryBucketBatchSampler, collate_fn_r2v

    sampler = GeometryBucketBatchSampler(dataset, batch_size=args.batch_size, drop_last=True,
                                         seed=args.seed, bucket_order=args.bucket_order)
    log.info("batching: %d batches/epoch over %d (modality, plane) buckets, order=%s",
             len(sampler), len(sampler.buckets), args.bucket_order)
    if world_size > 1:
        sampler = ShardedBatchSampler(sampler, rank=rank, world_size=world_size)
    return DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn_r2v,
        # Standard input-pipeline hygiene, and measurably worth it here because a training sample
        # is a decompressed multi-MB volume: pinned host memory lets the H2D copy overlap compute,
        # and persistent workers stop the archive readers and their caches being rebuilt every epoch.
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )


def git_state():
    """Commit and dirty flag, for the W&B run config. Never fatal -- a tarball checkout has no git."""
    import subprocess

    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                  cwd=str(Path(__file__).resolve().parent)).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    commit = run("git", "rev-parse", "HEAD")
    return {"git_commit": commit or "unknown",
            "git_dirty": bool(run("git", "status", "--porcelain"))}


def build_wandb_run(args, embedder, training_config, world_size: int):
    """Rank 0 only. Returns a `WandbRun`, which is already a no-op when disabled or when the
    `wandb` package/credentials are missing -- so no call site needs a guard."""
    from ..eval.wandb_logging import WandbRun

    identity = dict(embedder.identity)
    config = {
        "conditioning": args.conditioning,
        "conditioning_kind": identity.get("kind"),
        "conditioning_pooling": identity.get("pooling"),
        "conditioning_sequence_length": identity.get("sequence_length", 1),
        "conditioning_output_dim": embedder.output_dim,
        "conditioning_sections": identity.get("sections"),
        "encoder_order": identity.get("encoder_order"),
        "encoder_dims": identity.get("encoder_dims"),
        # Exact checkpoint identities, so a run is reproducible from its own config page.
        "encoder_checkpoints": [
            {"name": m.get("name"), "hf_repo": m.get("hf_repo"), "checkpoint": m.get("checkpoint"),
             "output_dim": m.get("output_dim"), "max_length": m.get("max_length"),
             "trainable": m.get("trainable")}
            for m in (identity.get("members") or [identity.get("encoder") or identity])
        ],
        "report_format": training_config.report_format,
        "text_trainable": args.text_trainable,
        "max_report_tokens": args.max_report_tokens,
        "cross_attention_dim": args.cross_attention_dim,
        "batch_size_per_gpu": args.batch_size,
        "world_size": world_size,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "effective_batch_size": args.batch_size * args.grad_accumulation_steps * world_size,
        "precision": "bf16/fp16 autocast" if args.amp else "fp32",
        "lr": args.lr,
        "seed": args.seed,
        "validate_every_steps": args.validate_every_steps,
        "val_quick_samples": args.val_quick_samples,
        "val_inference_steps": args.val_inference_steps,
        "geometry_mode": args.geometry_mode,
        "report_dropout_probability": args.report_dropout_probability,
        "modality_dropout_probability": args.modality_dropout_probability,
        "log_reports": args.wandb_log_reports,
        **git_state(),
    }
    return WandbRun(
        mode=args.wandb_mode, entity=args.wandb_entity, project=args.wandb_project,
        run_name=args.wandb_run_name or f"r2v-{args.conditioning or args.text_encoder}",
        group=args.wandb_group, tags=[args.conditioning or args.text_encoder], config=config,
    )


def build_validation_runner(args, dataset, autoencoder, divisor, scale_factor, noise_scheduler,
                            embedder, cfg_args, wandb_run):
    """A `validate(trainer, step, full)` callable for `MRRateAdapterTrainer.fit`.

    The sampler is rebuilt per validation step from the *live* trainer's UNet, so it always
    reflects the current adapter weights rather than a stale reference.
    """
    from ..sampling import ReportToVolumeSampler
    from ..validation import ValidationConfig, ValidationRunner

    config = ValidationConfig(
        every_steps=args.validate_every_steps, at_end=args.validate_at_end,
        n_quick=args.val_quick_samples, n_full=args.val_full_samples,
        full_every_steps=args.validate_full_every_steps, seed=args.val_seed,
        num_inference_steps=args.val_inference_steps, n_visualize=args.val_visualize,
        sequence_frames=args.val_sequence_frames,
        enabled_metrics=tuple(args.val_metrics),
        sensitivity_every_steps=args.val_sensitivity_every_steps,
        n_sensitivity=args.val_sensitivity_samples,
        feature_cache_dir=str(args.val_feature_cache) if args.val_feature_cache else None,
        reference_path=str(args.validation_reference) if args.validation_reference else None,
    )

    # FVD: torchvision r3d_18 (Kinetics-400). An MRI-volume adaptation of FVD, not standard FVD --
    # see eval/video_features.py for exactly how it differs from the I3D reference implementation.
    sequence_extractor = None
    if "fvd" in config.enabled_metrics:
        try:
            from ..eval.video_features import build_sequence_extractor

            sequence_extractor = build_sequence_extractor(
                args.val_fvd_extractor, device=str(args.device),
                torch_home=str(args.torch_home) if args.torch_home else None,
                checkpoint_path=args.medicalnet_checkpoint,
            )
            log.info("FVD extractor: %s", json.dumps(sequence_extractor.configuration(), default=str))
        except Exception as exc:  # noqa: BLE001
            log.warning("FVD extractor unavailable (%s): validation will skip val/fvd. Stage "
                        "r3d_18 into --torch-home first.", exc)

    inception_extractor = None
    if "fid_2p5d" in config.enabled_metrics:
        try:
            from ..eval.distribution import InceptionFeatureExtractor

            inception_extractor = InceptionFeatureExtractor(device=str(args.device))
        except Exception as exc:  # noqa: BLE001
            log.warning("Inception extractor unavailable (%s): validation will skip val/fid_2p5d.", exc)

    from ..sampling import SamplerConfig, official_latent_divisor

    # NOT models.nvidia.required_spatial_divisor: that is the encode-side padding divisor (16),
    # while the sampler needs the output:latent ratio (4). `official_latent_divisor` documents why
    # confusing them yields a valid file that is 4x too small in every axis.
    latent_divisor = official_latent_divisor(cfg_args.diffusion_unet_def["num_channels"])
    sampler_config = SamplerConfig(num_inference_steps=config.num_inference_steps)

    def sampler_factory(trainer):
        model = getattr(trainer.unet, "module", trainer.unet)
        sampler = ReportToVolumeSampler(
            unet=model, autoencoder=autoencoder, text_embedder=embedder,
            noise_scheduler=noise_scheduler, scale_factor=scale_factor, divisor=latent_divisor,
            device=trainer.device, conditioning=trainer.config.conditioning,
            sampler_config=sampler_config,
        )

        def generate(case):
            # Seeded per case, not per step: the same case is sampled from the same noise at every
            # validation step, so the curve reflects the adapter rather than the draw. Sampling is
            # therefore *deterministic given the weights* -- one generation per case, no best-of-N,
            # and no selection of a favourable draw.
            latent = sampler.sample_latent(
                case.report_text, case.modality, case.shape_xyz, case.spacing_xyz,
                seed=config.seed + case.index, report_sections=case.report_sections,
            )
            # `decode`, NOT `generate`: `generate` ends in postprocess_mr, which rescales the
            # decoder's [0, 1] output to int16 [0, 1000]. The ground truth here is the Dataset's
            # percentile-normalised volume, so a postprocessed generation would be 1000x off and
            # every metric would still return a plausible number. See
            # eval/video_features.METRIC_INTENSITY_SPACE.
            return sampler.decode(latent)

        return generate

    runner = ValidationRunner(
        dataset=dataset, sampler_factory=sampler_factory,
        sequence_extractor=sequence_extractor, inception_extractor=inception_extractor,
        config=config, wandb_run=wandb_run if args.wandb_log_reports else None,
        output_dir=args.out,
    )
    if wandb_run is not None:
        wandb_run.log({}, step=0)   # establishes step 0 before any reference line is drawn
    log.info("validation configuration: %s", json.dumps(runner.configuration(), default=str))
    if wandb_run is not None and not args.wandb_log_reports:
        log.info("validation panels disabled: --wandb-log-reports not set (report text is patient "
                 "data). Metrics are still logged.")
    return runner.run


def synthetic_loader(steps: int, latent_channels: int = 4, latent: int = 8):
    """`--dry-run`: latent-space batches and fabricated reports, so the whole trainer can be
    exercised without a manifest, a VAE or a GPU. Shapes and keys match `collate_fn_r2v`."""
    reports = [
        "Findings: No acute infarct. Mild chronic microangiopathic change.",
        "Findings: 12 mm enhancing lesion in the right frontal lobe. Impression: Neoplasm.",
    ]
    sections = [
        {"findings": "No acute infarct. Mild chronic microangiopathic change.",
         "impression": "No acute intracranial abnormality."},
        {"findings": "12 mm enhancing lesion in the right frontal lobe.", "impression": ""},
    ]
    batches = []
    for i in range(steps):
        batches.append({
            "image": torch.randn(1, latent_channels, latent, latent, latent),
            "report_text": [reports[i % len(reports)]],
            # Present so a sectioned-fusion configuration can be dry-run too; `encode_reports`
            # picks whichever the embedder asks for.
            "report_sections_text": [sections[i % len(sections)]],
            "modality": ["T1w"],
            "target_spacing_mm": torch.tensor([[1.0, 1.0, 1.0]]),
        })
    return batches


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from ..conditioning import ConditioningConfig
    from ..models.adapter import sha256_file
    from ..models.report_conditioned_unet import (
        build_report_conditioned_unet,
        load_pretrained_maisi_weights,
    )
    from ..training import (
        LatentEncoder,
        MRRateAdapterTrainer,
        TrainingConfig,
        resolve_scale_factor,
        wrap_distributed,
    )

    device, rank, local_rank, world_size = setup_distributed(args)
    report_format, format_source = resolve_report_format(args)
    embedder = build_text_embedder_from_args(args)
    if rank == 0:
        log.info("conditioning: %s", json.dumps(embedder.identity, indent=None))
        log.info("report format: %s (from %s)", report_format, format_source)

    network_config = args.network_config
    if network_config is None:
        from ..models.nvidia import DEFAULT_NETWORK_CONFIG

        network_config = DEFAULT_NETWORK_CONFIG

    unet_kwargs = dict(
        cross_attention_dim=args.cross_attention_dim,
        condition_mid=args.condition_mid,
        context_hidden_dim=args.context_hidden_dim,
        use_flash_attention=device.type == "cuda",
    )
    if args.conditioning_levels is not None:
        unet_kwargs["conditioning_levels"] = [bool(v) for v in args.conditioning_levels]
    if args.dry_run:
        # A four-level 180M UNet on CPU is pointless for a wiring check; keep the real geometry's
        # shape (spacing input, modality classes, resblock updown) at two levels.
        from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi  # noqa: F401

        from ..models.report_conditioned_unet import ReportConditionedUNetMaisi

        unet = ReportConditionedUNetMaisi(
            context_dim=embedder.output_dim, cross_attention_dim=args.cross_attention_dim,
            spatial_dims=3, in_channels=4, out_channels=4, num_channels=[32, 64],
            attention_levels=[False, True], num_head_channels=[0, 16], num_res_blocks=1,
            norm_num_groups=32, include_spacing_input=True, num_class_embeds=128,
            resblock_updown=True, include_fc=True, use_flash_attention=False,
        ).to(device)
        base_identity, scale_factor, latent_encoder = {"path": None, "sha256": None}, 1.0, None
    else:
        unet = build_report_conditioned_unet(
            context_dim=embedder.output_dim, network_config=network_config, **unet_kwargs
        ).to(device)
        report = load_pretrained_maisi_weights(unet, args.base_checkpoint)
        log.info("base checkpoint loaded:\n%s", report.format())
        base_identity = {"path": str(args.base_checkpoint), "sha256": sha256_file(args.base_checkpoint)}
        scale_factor = resolve_scale_factor(args.base_checkpoint, args.scale_factor)
        log.info("scale_factor = %.6f (mode=%s)", scale_factor, args.scale_factor)

    from ..models.nvidia import DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, define_instance, load_config

    cfg_args = load_config(str(DEFAULT_ENV_CONFIG), str(DEFAULT_MODEL_CONFIG), str(network_config))
    noise_scheduler = define_instance(cfg_args, "noise_scheduler")

    train_loader, validation_dataset, autoencoder, divisor = None, None, None, None
    if args.dry_run:
        train_loader = synthetic_loader(steps=args.max_steps or 2)
    else:
        dataset = build_dataset(args, args.split, report_format)
        train_loader = build_dataloader(args, log, dataset, rank=rank, world_size=world_size)
        from ..models.nvidia import load_autoencoder

        autoencoder, _cfg, divisor = load_autoencoder(
            args.vae_checkpoint, DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, network_config, device=str(device)
        )
        latent_encoder = LatentEncoder(autoencoder, divisor, scale_factor, amp=args.amp,
                                       dtype=getattr(torch, args.amp_dtype))
        log.info("latent encoder ready (divisor=%d, scale_factor=%.6f)", divisor, scale_factor)
        if args.validate_every_steps or args.validate_at_end:
            # A separate Dataset on the val split, with the *same* report_format -- the whole point
            # of resolving the format once above. MR-RATE's splits are patient-isolated (0
            # violations in the release's own check), so no study can appear in both.
            validation_dataset = build_dataset(args, args.val_split, report_format)

    unet = wrap_distributed(unet, local_rank) if world_size > 1 else unet

    training_config = TrainingConfig(
        lr=args.lr, n_epochs=args.epochs, max_steps=args.max_steps, batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps, grad_clip_norm=args.grad_clip_norm,
        amp=args.amp, amp_dtype=args.amp_dtype, seed=args.seed, log_every=args.log_every,
        save_every_steps=args.save_every_steps, validate_every_steps=args.validate_every_steps,
        validate_full_every_steps=args.validate_full_every_steps,
        validate_at_end=args.validate_at_end, validate_full_at_end=args.validate_full_at_end,
        keep_last_n=args.keep_last_n, save_format=args.save_format,
        conditioning_name=args.conditioning, report_format=report_format,
        conditioning=ConditioningConfig(
            modality_dropout_probability=args.modality_dropout_probability,
            report_dropout_probability=args.report_dropout_probability,
        ),
    )
    wandb_run = build_wandb_run(args, embedder, training_config, world_size) if rank == 0 else None

    trainer = MRRateAdapterTrainer(
        unet=unet, text_embedder=embedder, noise_scheduler=noise_scheduler,
        latent_encoder=latent_encoder, config=training_config, device=device,
        output_dir=args.out, base_checkpoint=base_identity,
        num_train_timesteps=int(cfg_args.noise_scheduler["num_train_timesteps"]),
        local_rank=rank, wandb_run=wandb_run,
    )
    if args.resume:
        trainer.load_for_resume(args.resume)

    validate = None
    if validation_dataset is not None:
        validate = build_validation_runner(
            args, validation_dataset, autoencoder, divisor, scale_factor, noise_scheduler,
            embedder, cfg_args, wandb_run,
        )

    summary = trainer.fit(train_loader, validate=validate)
    if rank == 0:
        log.info("done: %d optimizer steps (%d micro) in %.1fs, final loss %s, best %s",
                 summary["optimizer_steps"], summary["steps"], summary["seconds"],
                 summary["final_loss"], summary["best_metrics"])
        (args.out / "train_summary.json").write_text(json.dumps(
            {k: v for k, v in summary.items() if k != "history"}
            | {"trainable_parameters": trainer.freeze_report.trainable_parameters,
               "frozen_parameters": trainer.freeze_report.frozen_parameters,
               "text_encoder": embedder.identity, "base_checkpoint": base_identity,
               "report_format": report_format, "conditioning": args.conditioning,
               "world_size": world_size,
               "effective_batch_size": args.batch_size * args.grad_accumulation_steps * world_size},
            indent=2, default=str,
        ))
        if wandb_run is not None:
            (args.out / "wandb_run.json").write_text(json.dumps(wandb_run.finish(), indent=2))
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()          # rank 0 finishes writing before any rank exits
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
