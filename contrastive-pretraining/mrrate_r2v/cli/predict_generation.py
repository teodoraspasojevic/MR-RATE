#!/usr/bin/env python3
"""Stage 2: NVIDIA unconditional generation. Computes no metrics.

Generates volumes from a modality label alone (the `rflow-mr-brain` model conditions on a
modality class code plus a fixed spacing tensor -- there is no image, mask, or text
conditioning path) and writes them as a prediction set. Score with
`python -m mrrate_r2v.cli.evaluate --task generation`, which computes only population-level
metrics because no real patient corresponds to any generated volume.

    python -m mrrate_r2v.cli.predict_generation \\
        --cohort <workspace>/cohorts/test_v1 \\
        --checkpoint <workspace>/models/autoencoder_v1.pt \\
        --n-per-sequence 100 \\
        --out <workspace>/predictions/generation_v1

The cohort is still required, for two reasons: it names which sequences to generate, and the
evaluator uses its volumes as the real reference population. If the cohort's geometry differs
from the model's native output geometry, this script says so and stops rather than emitting
volumes that cannot be compared against it -- rebuild the cohort with matching
`--fixed-shape`/`--fixed-spacing-mm`, or pass `--allow-geometry-mismatch` if you accept that
the populations sit on different grids.
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
from ..predictions import PredictionItem, PredictionSet, save_prediction_volume, write_prediction_set

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
    p.add_argument("--n-per-sequence", type=int, default=100)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-config", type=Path, default=None)
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--network-config", type=Path, default=None)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-geometry-mismatch", action="store_true",
                   help="proceed even if the cohort's grid differs from the model's native output grid")
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
        load_autoencoder_and_unet, prepare_tensors, run_inference, set_random_seed,
    )

    cohort = Cohort(args.cohort)
    log.info("cohort %s", cohort.summary())
    unknown = [s for s in cohort.sequences if s not in MODALITY_CODE]
    if unknown:
        raise SystemExit(f"no NVIDIA modality code for sequence(s) {unknown}; known: {list(MODALITY_CODE)}")

    log.info("loading autoencoder + diffusion UNet")
    checkpoint_sha = sha256_file(args.checkpoint)
    autoencoder, unet, scale_factor, cfg = load_autoencoder_and_unet(
        args.env_config or DEFAULT_ENV_CONFIG, args.model_config or DEFAULT_MODEL_CONFIG,
        args.network_config or DEFAULT_NETWORK_CONFIG, args.device,
        autoencoder_checkpoint_override=args.checkpoint,
    )
    output_size = tuple(int(x) for x in cfg.diffusion_unet_inference["dim"])
    out_spacing = tuple(float(x) for x in cfg.diffusion_unet_inference["spacing"])
    cfg.cfg_guidance_scale = cfg.diffusion_unet_inference["cfg_guidance_scale"]
    n_levels = max(1, len(cfg.diffusion_unet_def["num_channels"])
                   if isinstance(cfg.diffusion_unet_def["num_channels"], list)
                   else len(cfg.diffusion_unet_def["attention_levels"]))
    divisor = 2 ** (n_levels - 2)
    top_region, bottom_region, spacing_tensor, _default_modality = prepare_tensors(cfg, args.device)
    log.info("native generation geometry: shape=%s spacing_mm=%s divisor=%d", output_size, out_spacing, divisor)

    cohort_shape = tuple(cohort.cases[0].shape) if cohort.cases else None
    cohort_spacing = tuple(cohort.cases[0].spacing_mm) if cohort.cases else None
    if cohort_shape != output_size or cohort_spacing != out_spacing:
        msg = (f"cohort grid (shape={cohort_shape}, spacing={cohort_spacing}) differs from the "
               f"model's native output grid (shape={output_size}, spacing={out_spacing}). The real "
               f"and generated populations would sit on different grids.")
        if not args.allow_geometry_mismatch:
            raise SystemExit(msg + "\nRebuild the cohort with --geometry-mode fixed --fixed-shape "
                                   f"{' '.join(map(str, output_size))} --fixed-spacing-mm "
                                   f"{' '.join(map(str, out_spacing))}, or pass "
                                   "--allow-geometry-mismatch to proceed anyway.")
        log.warning("%s Proceeding because --allow-geometry-mismatch was passed.", msg)

    items, failures = [], []
    t0 = time.time()
    for seq in cohort.sequences:
        log.info("sequence=%s: generating %d volumes", seq, args.n_per_sequence)
        for i in range(args.n_per_sequence):
            seed = stable_seed(args.seed, seq, i)
            prediction_id = f"gen_{seq}_{i:04d}"
            try:
                set_random_seed(seed)
                modality_tensor = MODALITY_CODE[seq] * torch.ones(
                    (len(spacing_tensor),), dtype=torch.long).to(args.device)
                with torch.no_grad():
                    raw = run_inference(cfg, args.device, autoencoder, unet, scale_factor,
                                        top_region, bottom_region, spacing_tensor,
                                        modality_tensor, output_size, divisor, log)
                volume = raw.astype(np.float32) / NVIDIA_MR_INTENSITY_SCALE
            except Exception as e:  # noqa: BLE001 -- one failure must not stop the run
                failures.append({"prediction_id": prediction_id, "sequence": seq, "seed": seed,
                                 "error": f"{type(e).__name__}: {e}"})
                log.warning("%s failed: %s", prediction_id, e)
                continue
            save_prediction_volume(args.out, prediction_id, volume)
            items.append(PredictionItem(
                prediction_id=prediction_id, case_id=None, sequence=seq,
                shape=list(volume.shape), spacing_mm=list(out_spacing), seed=seed,
            ))
        log.info("sequence=%s done (%.1fs elapsed)", seq, time.time() - t0)

    elapsed = time.time() - t0
    pset = PredictionSet(
        task="generation", cohort_id=cohort.cohort_id, cohort_cases_sha256=cohort.cases_sha256,
        model={"name": "nvidia_maisi_rflow_mr_brain", "checkpoint": str(args.checkpoint),
               "checkpoint_sha256": checkpoint_sha, "native_shape": list(output_size),
               "native_spacing_mm": list(out_spacing),
               "conditioning": "modality class code + fixed spacing tensor only"},
        items=items, failures=failures, created_by="mrrate_r2v.cli.predict_generation",
        runtime={"device": args.device, "elapsed_sec": round(elapsed, 1), "seed": args.seed,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    )
    out = write_prediction_set(args.out, pset)
    log.info("%d generated, %d failures, %.1fs -> %s", len(items), len(failures), elapsed, out)
    log.info("score it: python -m mrrate_r2v.cli.evaluate --task generation "
             "--gt %s --pred %s --distribution-metrics --out <results>", args.cohort, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
