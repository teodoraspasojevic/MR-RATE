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
    data.add_argument("--report-sections", nargs="+", default=["findings", "impression"])
    data.add_argument("--geometry-mode", default="per_modality_plane", choices=["per_modality_plane", "fixed"])
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
    text.add_argument("--text-encoder", default="radbert", choices=["radbert", "mock"])
    text.add_argument("--text-checkpoint", type=Path, default=None, help="local RadBERT snapshot directory")
    text.add_argument("--max-report-tokens", type=int, default=512)
    text.add_argument("--mock-output-dim", type=int, default=32, help="--text-encoder mock only")

    train = p.add_argument_group("optimisation")
    train.add_argument("--out", type=Path, required=True, help="output directory for checkpoints")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-5, help="NVIDIA's own diffusion_unet_train.lr")
    train.add_argument("--grad-accumulation-steps", type=int, default=1)
    train.add_argument("--grad-clip-norm", type=float, default=None)
    train.add_argument("--no-amp", dest="amp", action="store_false", help="disable mixed precision")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--save-every-steps", type=int, default=None)
    train.add_argument("--validate-every-steps", type=int, default=None)
    train.add_argument("--save-format", default="adapter", choices=["adapter", "full", "both"])
    train.add_argument("--resume", type=Path, default=None, help="adapter checkpoint to resume from")
    train.add_argument("--scale-factor", default="auto",
                       help="'auto' = from the base checkpoint (matches official inference), "
                            "'recompute' = 1/std(z) of the first batch (official training), or a literal")
    train.add_argument("--num-gpus", type=int, default=1)
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
    if args.text_encoder == "radbert" and args.text_checkpoint is None and not args.dry_run:
        p.error("--text-checkpoint is required for --text-encoder radbert")
    return args


def build_text_embedder_from_args(args):
    from ..text import build_text_embedder

    if args.text_encoder == "mock":
        return build_text_embedder("mock", output_dim=args.mock_output_dim, max_length=16)
    return build_text_embedder(
        "radbert", checkpoint=str(args.text_checkpoint), max_length=args.max_report_tokens
    )


def build_dataloader(args, log):
    from torch.utils.data import DataLoader

    from ..data import (
        GeometryBucketBatchSampler,
        MRReportToVolumeDataset,
        R2VDatasetConfig,
        collate_fn_r2v,
    )
    from ..data.reports import ShardReportStore

    config = R2VDatasetConfig(
        split=args.split,
        report_sections=tuple(args.report_sections),
        geometry_mode=args.geometry_mode,
        series_selection="all",
        dtype=torch.float32,
        seed=args.seed,
    )
    dataset = MRReportToVolumeDataset(
        str(args.manifest), ShardReportStore(str(args.report_index)), config=config
    )
    log.info("dataset: %d (report, volume) pairs in split '%s'", len(dataset), args.split)
    sampler = GeometryBucketBatchSampler(dataset, batch_size=args.batch_size, drop_last=True, seed=args.seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn_r2v)


def synthetic_loader(steps: int, latent_channels: int = 4, latent: int = 8):
    """`--dry-run`: latent-space batches and fabricated reports, so the whole trainer can be
    exercised without a manifest, a VAE or a GPU. Shapes and keys match `collate_fn_r2v`."""
    reports = [
        "Findings: No acute infarct. Mild chronic microangiopathic change.",
        "Findings: 12 mm enhancing lesion in the right frontal lobe. Impression: Neoplasm.",
    ]
    batches = []
    for i in range(steps):
        batches.append({
            "image": torch.randn(1, latent_channels, latent, latent, latent),
            "report_text": [reports[i % len(reports)]],
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
    from ..training import LatentEncoder, MRRateAdapterTrainer, TrainingConfig, resolve_scale_factor

    device = torch.device(args.device)
    embedder = build_text_embedder_from_args(args)
    log.info("text encoder: %s", json.dumps(embedder.identity, indent=None))

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

    train_loader = None
    if args.dry_run:
        train_loader = synthetic_loader(steps=args.max_steps or 2)
    else:
        train_loader = build_dataloader(args, log)
        from ..models.nvidia import load_autoencoder

        autoencoder, _cfg, divisor = load_autoencoder(
            args.vae_checkpoint, DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, network_config, device=str(device)
        )
        latent_encoder = LatentEncoder(autoencoder, divisor, scale_factor, amp=args.amp)
        log.info("latent encoder ready (divisor=%d, scale_factor=%.6f)", divisor, scale_factor)

    training_config = TrainingConfig(
        lr=args.lr, n_epochs=args.epochs, max_steps=args.max_steps, batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps, grad_clip_norm=args.grad_clip_norm,
        amp=args.amp, seed=args.seed, log_every=args.log_every,
        save_every_steps=args.save_every_steps, validate_every_steps=args.validate_every_steps,
        save_format=args.save_format,
        conditioning=ConditioningConfig(
            modality_dropout_probability=args.modality_dropout_probability,
            report_dropout_probability=args.report_dropout_probability,
        ),
    )
    trainer = MRRateAdapterTrainer(
        unet=unet, text_embedder=embedder, noise_scheduler=noise_scheduler,
        latent_encoder=latent_encoder, config=training_config, device=device,
        output_dir=args.out, base_checkpoint=base_identity,
        num_train_timesteps=int(cfg_args.noise_scheduler["num_train_timesteps"]),
    )
    if args.resume:
        trainer.load_for_resume(args.resume)

    summary = trainer.fit(train_loader)
    log.info("done: %d steps in %.1fs, final loss %s", summary["steps"], summary["seconds"],
             summary["final_loss"])
    (args.out / "train_summary.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k != "history"}
        | {"trainable_parameters": trainer.freeze_report.trainable_parameters,
           "frozen_parameters": trainer.freeze_report.frozen_parameters,
           "text_encoder": embedder.identity, "base_checkpoint": base_identity},
        indent=2, default=str,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
