#!/usr/bin/env python3
"""Stage 2: NVIDIA unconditional generation. Computes no metrics.

Generates volumes from a modality label alone (`rflow-mr-brain` has no image, mask, or text
conditioning path) and writes them as a prediction set. Score with
`python -m mrrate_r2v.cli.evaluate --task generation`, which computes only population-level
metrics because no real patient corresponds to any generated volume.

    python -m mrrate_r2v.cli.predict_generation \\
        --cohort <workspace>/cohorts/test_v1 \\
        --checkpoint <workspace>/models/autoencoder_v1.pt \\
        --out <workspace>/predictions/gen_v1

**Each bucket is generated at its own geometry -- the cohort's.** The cohort holds NVIDIA's
published recommended FOV for every (modality, plane), and this script asks the model for exactly
that: `output_size` sizes the latent noise and the spacing tensor is a real conditioning input to
the UNet. So the generated population matches the real one in composition *and* in geometry, which
is what makes a per-bucket FID a comparison of distributions rather than of grids.

Two hard requirements, both checked before any GPU work:

- Every bucket shape must be a multiple of 32. That is the diffusion UNet's constraint (4 levels,
  latent = output/4, so latent must be divisible by 8) -- verified empirically; a div-16-but-not-32
  shape raises a skip-connection size mismatch. A wrong shape is refused here rather than padded,
  because padding would change the FOV the model was asked for.
- Counts default to the cohort's own per-bucket counts, so neither population is larger.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..cohort import Cohort, sha256_file
from ..data.geometry import UNET_SPATIAL_MULTIPLE
from ..predictions import PredictionItem, PredictionSet, write_prediction_set
from ..volumes import VolumeWriter, split_bucket

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict_generation")

# NVIDIA's own modality class codes for rflow-mr-brain.
MODALITY_CODE = {"T1w": 9, "T2w": 10, "FLAIR": 11, "SWI": 20}

# The decoder rescales MR output to an [0, ~1000] int16-like range; divide back so generated
# and real volumes share the cohort's [0, 1]-ish intensity convention.
NVIDIA_MR_INTENSITY_SCALE = 1000.0


def stable_seed(base_seed: int, sequence: str, index: int) -> int:
    """A per-sample seed that depends only on (base_seed, sequence, index), so the same request
    always produces the same volume regardless of run order or how many others were generated.
    """
    h = hashlib.sha256(f"{base_seed}:{sequence}:{index}".encode()).hexdigest()
    return int(h[:8], 16)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-per-bucket", type=int, default=None,
                   help="volumes to generate per (modality, plane) bucket. Default: match the "
                        "cohort's own count for that bucket, so the real and generated populations "
                        "have the same composition -- which is what makes a per-bucket FID a "
                        "comparison of distributions rather than of sample sizes.")
    p.add_argument("--checkpoint", type=Path, required=True, help="autoencoder (VAE) checkpoint")
    p.add_argument("--unet-checkpoint", type=Path, default=None,
                   help="diffusion UNet checkpoint. Defaults to the filename NVIDIA's env config "
                        "names, looked up next to --checkpoint (both ship together). An absolute "
                        "path is required because NVIDIA's config stores it relative to cwd.")
    p.add_argument("--env-config", type=Path, default=None)
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--network-config", type=Path, default=None)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} exists and is non-empty; pass --overwrite to replace it")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    from ..models.nvidia import (
        DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_NETWORK_CONFIG,
        default_unet_filename, load_autoencoder_and_unet, prepare_tensors, run_inference,
        set_random_seed,
    )

    env_config = args.env_config or DEFAULT_ENV_CONFIG
    unet_ckpt = args.unet_checkpoint or (args.checkpoint.parent / default_unet_filename(env_config))
    if not unet_ckpt.is_file():
        raise SystemExit(
            f"diffusion UNet checkpoint not found: {unet_ckpt}\n"
            f"Pass --unet-checkpoint explicitly. (NVIDIA's env config stores this path relative to "
            f"the working directory, so it cannot be relied on.)"
        )

    cohort = Cohort(args.cohort)
    log.info("cohort %s", cohort.summary())
    unknown = [s for s in cohort.sequences if s not in MODALITY_CODE]
    if unknown:
        raise SystemExit(f"no NVIDIA modality code for sequence(s) {unknown}; known: {list(MODALITY_CODE)}")

    log.info("loading autoencoder + diffusion UNet")
    checkpoint_sha = sha256_file(args.checkpoint)
    log.info("autoencoder: %s", args.checkpoint)
    log.info("diffusion UNet: %s", unet_ckpt)
    autoencoder, unet, scale_factor, cfg = load_autoencoder_and_unet(
        env_config, args.model_config or DEFAULT_MODEL_CONFIG,
        args.network_config or DEFAULT_NETWORK_CONFIG, args.device,
        autoencoder_checkpoint_override=args.checkpoint,
        unet_checkpoint_override=unet_ckpt,
    )
    cfg.cfg_guidance_scale = cfg.diffusion_unet_inference["cfg_guidance_scale"]
    n_levels = max(1, len(cfg.diffusion_unet_def["num_channels"])
                   if isinstance(cfg.diffusion_unet_def["num_channels"], list)
                   else len(cfg.diffusion_unet_def["attention_levels"]))
    divisor = 2 ** (n_levels - 2)
    # `prepare_tensors` reads the config's default spacing; only the body-region tensors are reused
    # here. Spacing and modality are rebuilt per bucket below, from the cohort's own geometry.
    top_region, bottom_region, _cfg_spacing, _default_modality = prepare_tensors(cfg, args.device)
    log.info("config default geometry: shape=%s spacing=%s (used only as a fallback); latent divisor=%d",
             cfg.diffusion_unet_inference["dim"], cfg.diffusion_unet_inference["spacing"], divisor)

    # Every bucket is generated at ITS OWN shape and spacing -- the cohort's, which is NVIDIA's
    # published recommended FOV for that (modality, plane). `output_size` sizes the latent noise and
    # `spacing_tensor` is a real conditioning input, so asking for the cohort's geometry is what
    # makes the real and generated populations directly comparable.
    plan = []
    for bucket in cohort.buckets:
        sequence, plane = split_bucket(bucket)
        geom = cohort.bucket_geometry(bucket)
        shape_xyz = tuple(int(x) for x in geom["shape_xyz"])
        bad = [x for x in shape_xyz if x % UNET_SPATIAL_MULTIPLE]
        if bad:
            raise SystemExit(
                f"bucket {bucket} has shape {shape_xyz}, not a multiple of "
                f"{UNET_SPATIAL_MULTIPLE} -- the diffusion UNet's skip connections require it. "
                f"Rebuild the cohort; do not pad here, that would change the FOV."
            )
        n = geom["n"] if args.n_per_bucket is None else min(args.n_per_bucket, geom["n"])
        plan.append((bucket, sequence, plane, shape_xyz,
                     tuple(float(x) for x in geom["spacing_mm_xyz"]), n))

    log.info("generation plan (%d buckets, %d volumes):", len(plan), sum(p[5] for p in plan))
    for bucket, sequence, plane, shape_xyz, spacing_xyz, n in plan:
        log.info("   %-6s %-9s n=%-5d shape=%s spacing=%s mm", sequence, plane, n, shape_xyz,
                 tuple(round(x, 4) for x in spacing_xyz))

    items, failures = [], []
    t0 = time.time()
    with VolumeWriter(args.out) as writer:
        for bucket, sequence, plane, shape_xyz, spacing_xyz, n_wanted in plan:
            # NVIDIA scales its conditioning tensors by 1e2 (see prepare_tensors); match that here
            # or the model is conditioned on a spacing 100x off.
            spacing_tensor = torch.from_numpy(
                np.array(spacing_xyz, dtype=float) * 1e2)[None].half().to(args.device)
            modality_tensor = MODALITY_CODE[sequence] * torch.ones(
                (len(spacing_tensor),), dtype=torch.long).to(args.device)
            for i in range(n_wanted):
                seed = stable_seed(args.seed, bucket, i)
                prediction_id = f"gen_{bucket}_{i:04d}"
                try:
                    set_random_seed(seed)
                    with torch.no_grad():
                        raw = run_inference(cfg, args.device, autoencoder, unet, scale_factor,
                                            top_region, bottom_region, spacing_tensor,
                                            modality_tensor, shape_xyz, divisor, log)
                    volume = raw.astype(np.float32) / NVIDIA_MR_INTENSITY_SCALE
                    if tuple(volume.shape) != shape_xyz:
                        raise RuntimeError(
                            f"asked for {shape_xyz} but got {tuple(volume.shape)}")
                except Exception as e:  # noqa: BLE001 -- one failure must not stop the run
                    failures.append({"prediction_id": prediction_id, "bucket": bucket,
                                     "seed": seed, "error": f"{type(e).__name__}: {e}"})
                    log.warning("%s failed: %s", prediction_id, e)
                    continue
                writer.add(bucket, prediction_id, volume)
                items.append(PredictionItem(
                    prediction_id=prediction_id, case_id=None, sequence=sequence, plane=plane,
                    shape=list(volume.shape), spacing_mm=list(spacing_xyz), seed=seed,
                ))
            log.info("%s done: %d/%d (%.1fs elapsed)", bucket,
                     sum(1 for it in items if it.bucket == bucket), n_wanted, time.time() - t0)

    elapsed = time.time() - t0
    pset = PredictionSet(
        task="generation", cohort_id=cohort.cohort_id, cohort_cases_sha256=cohort.cases_sha256,
        model={"name": "nvidia_maisi_rflow_mr_brain", "checkpoint": str(args.checkpoint),
               "checkpoint_sha256": checkpoint_sha,
               "unet_checkpoint": str(unet_ckpt), "unet_checkpoint_sha256": sha256_file(unet_ckpt),
               "conditioning": "modality class code + per-bucket spacing tensor",
               "requested_geometry_per_bucket": {
                   b: {"shape_xyz": list(s), "spacing_mm_xyz": list(sp), "n": n}
                   for b, _seq, _pl, s, sp, n in plan}},
        items=items, failures=failures, created_by="mrrate_r2v.cli.predict_generation",
        runtime={"device": args.device, "elapsed_sec": round(elapsed, 1), "seed": args.seed,
                 "n_volumes": len(items),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    )
    out = write_prediction_set(args.out, pset)
    log.info("%d generated, %d failures, %.1fs -> %s", len(items), len(failures), elapsed, out)
    log.info("score it: python -m mrrate_r2v.cli.evaluate --task generation "
             "--gt %s --pred %s --out <results>", args.cohort, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
