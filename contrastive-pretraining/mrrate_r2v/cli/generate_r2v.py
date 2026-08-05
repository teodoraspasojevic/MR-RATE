#!/usr/bin/env python3
"""Report-to-volume inference. `NV-Generate-CTMR/scripts/diff_model_infer.py` with a report.

Loads the frozen NVIDIA base checkpoint strictly, the adapter checkpoint strictly, and the text
encoder named in the adapter checkpoint; then runs NVIDIA's own sampler, guidance point, decoder and
output postprocessing (see `mrrate_r2v/sampling.py` for the line-by-line mapping).

One report from a string:

    python -m mrrate_r2v.cli.generate_r2v \\
        --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \\
        --vae-checkpoint  <ws>/models/autoencoder_v1.pt \\
        --adapter <ws>/runs/r2v_adapter_v1/adapter_last.pt \\
        --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \\
        --report "Findings: 12 mm enhancing lesion in the right frontal lobe." \\
        --modality T1w --dim 256 256 256 --spacing 1 1 1 \\
        --report-guidance-scale 4 --out <ws>/samples/case001.nii.gz

Every case in a cohort's validation split, from its paired MR-RATE report:

    python -m mrrate_r2v.cli.generate_r2v --cohort <ws>/cohorts/val_v1 --out-dir <ws>/samples/val_v1 ...

Modality-only generation (NVIDIA's original behaviour, no report term):

    ... --report "" --report-guidance-scale 0 --modality-guidance-scale 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("generate_r2v")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a 3D brain MRI volume from a radiology report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--vae-checkpoint", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True, help="adapter checkpoint from cli.train_r2v")
    p.add_argument("--network-config", type=Path, default=None)
    p.add_argument("--text-encoder", default=None, choices=[None, "radbert", "mock"],
                   help="default: whatever the adapter checkpoint recorded")
    p.add_argument("--text-checkpoint", type=Path, default=None)
    p.add_argument("--max-report-tokens", type=int, default=None)

    source = p.add_argument_group("report source (exactly one)")
    source.add_argument("--report", default=None, help="report text")
    source.add_argument("--report-file", type=Path, default=None, help="file containing the report text")
    source.add_argument("--cohort", type=Path, default=None,
                        help="generate one volume per case from the cohort's paired reports")
    source.add_argument("--allow-report-format-mismatch", action="store_true",
                        help="generate even when the cohort's report_format differs from the one "
                             "the adapter was trained on. For a deliberate ablation only -- the "
                             "mismatch is otherwise silent")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=None, help="output .nii.gz (single report)")
    out.add_argument("--out-dir", type=Path, default=None, help="output directory (--cohort)")
    out.add_argument("--modality", default="T1w", choices=["T1w", "T2w", "FLAIR", "SWI", "unknown"])
    out.add_argument("--dim", nargs=3, type=int, default=[256, 256, 256], help="output shape (X Y Z)")
    out.add_argument("--spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="mm (X Y Z)")

    sampler = p.add_argument_group("sampler and guidance")
    sampler.add_argument("--num-inference-steps", type=int, default=30, help="NVIDIA's own default")
    sampler.add_argument("--report-guidance-scale", type=float, default=4.0,
                         help="0 disables the report term and reproduces NVIDIA's guidance exactly")
    sampler.add_argument("--modality-guidance-scale", type=float, default=10.0,
                         help="NVIDIA's cfg_guidance_scale for mr-brain; 0 disables modality guidance")
    sampler.add_argument("--no-batched-guidance", dest="batched_guidance", action="store_false",
                         help="run guidance branches as separate forwards (slower, same numbers)")
    sampler.add_argument("--seed", type=int, default=1234, help="NVIDIA's own random_seed default")
    sampler.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sampler.add_argument("--latent-only", action="store_true",
                         help="stop after the diffusion loop; skip the VAE decode (cheap smoke test)")
    sampler.add_argument("--allow-base-mismatch", action="store_true",
                         help="load an adapter trained against a different base checkpoint")

    args = p.parse_args(argv)
    sources = [args.report is not None, args.report_file is not None, args.cohort is not None]
    if sum(sources) != 1:
        p.error("give exactly one of --report, --report-file, --cohort")
    if args.cohort is not None and args.out_dir is None:
        p.error("--cohort needs --out-dir")
    if args.cohort is None and args.out is None:
        p.error("--report/--report-file needs --out")
    return args


def build_sampler(args):
    from ..conditioning import ConditioningConfig
    from ..models.adapter import load_adapter_checkpoint, sha256_file
    from ..models.nvidia import (
        DEFAULT_ENV_CONFIG,
        DEFAULT_MODEL_CONFIG,
        DEFAULT_NETWORK_CONFIG,
        define_instance,
        load_autoencoder,
        load_config,
    )
    from ..models.report_conditioned_unet import build_report_conditioned_unet, load_pretrained_maisi_weights
    from ..sampling import ReportToVolumeSampler, SamplerConfig, official_latent_divisor
    from ..text import rebuild_embedder

    device = torch.device(args.device)
    network_config = args.network_config or DEFAULT_NETWORK_CONFIG

    # The adapter checkpoint is read first: it names the text encoder, the context width and the
    # adapter geometry, so nothing has to be re-specified on the command line and a mismatch is
    # impossible rather than merely unlikely.
    payload = torch.load(str(args.adapter), map_location="cpu", weights_only=False)
    stored_text = payload.get("text_encoder") or {}
    stored_config = payload.get("config") or {}
    # `rebuild_embedder` dispatches on the recorded `kind`, so all three conditioning
    # configurations round-trip. This used to be inline and always built RadBERT regardless of what
    # the checkpoint recorded.
    if args.text_encoder:
        stored_text = dict(stored_text, name=args.text_encoder)
    embedder = rebuild_embedder(
        stored_text,
        conditioning_name=stored_config.get("conditioning_name"),
        checkpoint=str(args.text_checkpoint) if args.text_checkpoint else None,
        max_length=args.max_report_tokens or None,
    )
    if int(stored_text.get("output_dim", embedder.output_dim)) != embedder.output_dim:
        raise SystemExit(
            f"text encoder width {embedder.output_dim} != the {stored_text.get('output_dim')} the "
            "adapter's projection head was built for"
        )

    unet = build_report_conditioned_unet(
        context_dim=embedder.output_dim,
        network_config=network_config,
        cross_attention_dim=int(stored_config.get("cross_attention_dim", 512)),
        conditioning_levels=stored_config.get("conditioning_levels"),
        condition_mid=bool(stored_config.get("condition_mid", True)),
        use_flash_attention=device.type == "cuda",
    ).to(device)
    base_report = load_pretrained_maisi_weights(unet, args.base_checkpoint)
    log.info("base checkpoint:\n%s", base_report.format())
    load_adapter_checkpoint(
        args.adapter, unet,
        base_checkpoint_sha256=sha256_file(args.base_checkpoint),
        allow_base_mismatch=args.allow_base_mismatch,
    )
    unet.eval()

    cfg_args = load_config(str(DEFAULT_ENV_CONFIG), str(DEFAULT_MODEL_CONFIG), str(network_config))
    # The output-size / latent-size ratio, computed the official way from the UNet's depth. Not the
    # autoencoder's padding divisor -- see `sampling.official_latent_divisor`.
    divisor = official_latent_divisor(cfg_args.diffusion_unet_def["num_channels"])
    log.info("latent divisor = %d (dim // %d latent voxels per axis)", divisor, divisor)

    autoencoder = None
    if not args.latent_only:
        autoencoder, _cfg, _pad_divisor = load_autoencoder(
            args.vae_checkpoint, DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, network_config, device=str(device)
        )

    noise_scheduler = define_instance(cfg_args, "noise_scheduler")
    scale_factor = payload.get("scale_factor")
    if scale_factor is None:
        from ..training import resolve_scale_factor

        scale_factor = resolve_scale_factor(args.base_checkpoint, "auto")
    log.info("scale_factor = %.6f", scale_factor)

    sampler = ReportToVolumeSampler(
        unet=unet, autoencoder=autoencoder, text_embedder=embedder, noise_scheduler=noise_scheduler,
        scale_factor=scale_factor, divisor=divisor, device=device,
        conditioning=ConditioningConfig(
            report_guidance_scale=args.report_guidance_scale,
            modality_guidance_scale=args.modality_guidance_scale,
        ),
        sampler_config=SamplerConfig(
            num_inference_steps=args.num_inference_steps, random_seed=args.seed,
            batched_guidance=args.batched_guidance,
        ),
    )
    return sampler, embedder, payload


def manifest_for(args, sampler, embedder, payload, extra: dict) -> dict:
    from ..models.adapter import sha256_file

    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "modality": args.modality,
        "dim_xyz": list(args.dim),
        "spacing_mm_xyz": list(args.spacing),
        "guidance": {
            "report_guidance_scale": args.report_guidance_scale,
            "modality_guidance_scale": args.modality_guidance_scale,
            "batched": args.batched_guidance,
        },
        "sampler": {"num_inference_steps": args.num_inference_steps,
                    "scheduler": type(sampler.noise_scheduler).__name__,
                    "scale_factor": sampler.scale_factor, "divisor": sampler.divisor},
        "checkpoints": {
            "base": {"path": str(args.base_checkpoint), "sha256": sha256_file(args.base_checkpoint)},
            "adapter": {"path": str(args.adapter), "step": payload.get("step")},
            "vae": str(args.vae_checkpoint),
        },
        "text_encoder": embedder.identity,
        **extra,
    }


def assert_report_format_matches(payload: dict, cohort, allow_mismatch: bool = False) -> None:
    """The report text a cohort stores must have been composed the way the adapter was trained.

    A cohort's `reports.json` holds *already-composed* conditioning text, produced at preprocess
    time under that cohort's `report_format`. So this is a metadata comparison, not a re-formatting
    step -- re-composing here is impossible anyway, because the section boundaries are gone by the
    time the text reaches `reports.json`.

    Getting this wrong is silent: an adapter trained on `impression_findings` sampled on
    `findings_impression` text sees the same words in the other order, produces plausible volumes,
    and just scores worse. That is exactly the class of bug this repo's cohort contract exists to
    make impossible, so the default is to refuse.
    """
    trained = (payload.get("config") or {}).get("report_format")
    fingerprint = (getattr(cohort, "spec", None) and getattr(cohort.spec, "geometry_fingerprint", None)) or {}
    if not isinstance(fingerprint, dict):
        fingerprint = {}
    built = fingerprint.get("report_format")
    if trained == built:
        return
    message = (
        f"report-format mismatch: the adapter was trained on report_format={trained!r} but this "
        f"cohort's text was composed with report_format={built!r}. The conditioning strings differ, "
        f"which is silent at generation time and only shows up as a worse score.\n"
        f"  Rebuild the cohort with `cli.preprocess --report-format {trained}`, or pass "
        f"--allow-report-format-mismatch for a deliberate cross-format ablation."
    )
    if allow_mismatch:
        log.warning("%s (continuing: --allow-report-format-mismatch)", message)
    else:
        raise SystemExit(message)


def main(argv=None) -> int:
    args = parse_args(argv)
    sampler, embedder, payload = build_sampler(args)

    if args.cohort is not None:
        from ..cohort import Cohort

        cohort = Cohort(args.cohort)
        assert_report_format_matches(payload, cohort, allow_mismatch=args.allow_report_format_mismatch)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for case in cohort.cases:
            report_text = cohort.load_report(case.case_id)
            if not report_text:
                raise SystemExit(
                    f"cohort case {case.case_id} carries no report text; regenerate the cohort with "
                    "reports, or use --report/--report-file"
                )
            path = args.out_dir / f"{case.case_id}.nii.gz"
            volume = sampler.generate(report_text, tuple(case.shape), tuple(case.spacing_mm),
                                      seed=args.seed, modality=args.modality)
            from ..sampling import save_volume

            save_volume(volume, case.spacing_mm, path)
            written.append(str(path))
            log.info("wrote %s %s", path, volume.shape)
        (args.out_dir / "generation_manifest.json").write_text(json.dumps(
            manifest_for(args, sampler, embedder, payload, {"cohort": str(args.cohort), "outputs": written}),
            indent=2, default=str))
        return 0

    report_text = args.report if args.report is not None else args.report_file.read_text()
    log.info("report: %d characters", len(report_text))
    if args.latent_only:
        latent = sampler.sample_latent(report_text, args.modality, tuple(args.dim), tuple(args.spacing),
                                        seed=args.seed)
        log.info("latent %s finite=%s", tuple(latent.shape), bool(torch.isfinite(latent).all()))
        return 0

    volume = sampler.generate(report_text, tuple(args.dim), tuple(args.spacing), seed=args.seed,
                              modality=args.modality)
    from ..sampling import save_volume

    save_volume(volume, args.spacing, args.out)
    log.info("wrote %s %s dtype=%s range=[%d, %d]", args.out, volume.shape, volume.dtype,
             int(volume.min()), int(volume.max()))
    Path(str(args.out) + ".manifest.json").write_text(json.dumps(
        manifest_for(args, sampler, embedder, payload,
                     {"report_characters": len(report_text), "output": str(args.out)}),
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
