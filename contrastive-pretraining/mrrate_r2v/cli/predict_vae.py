#!/usr/bin/env python3
"""Stage 2: VAE reconstruction inference. Computes no metrics.

Encodes and decodes every volume in a cohort with the vendored NVIDIA autoencoder and writes
the reconstructions as a prediction set. Score it with
`python -m mrrate_r2v.cli.evaluate --task reconstruction`.

    python -m mrrate_r2v.cli.predict_vae \\
        --cohort <workspace>/cohorts/test_v1 \\
        --checkpoint <workspace>/models/autoencoder_v1.pt \\
        --out <workspace>/predictions/vae_v1

Padding: the VAE needs each axis divisible by a model-derived divisor. If a cohort's shape
already is (256^3 with num_splits=4 is), nothing happens. If not, the volume is zero-padded at
the end of each axis before encoding and the *exact same* amount is cropped back off after
decoding, tracked by a `CropPadRecord` -- so the reconstruction always comes back on the
cohort's own grid and the evaluator sees a strict geometry match.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..cohort import Cohort, sha256_file
from ..eval import geometry_contract as G
from ..predictions import PredictionItem, PredictionSet, write_prediction_set
from ..volumes import VolumeWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict_vae")


def reconstruct(autoencoder, volume: np.ndarray, divisor: int, device: str) -> np.ndarray:
    """One volume in, its reconstruction out, on the identical grid."""
    padded_shape, crop_pad = G.pad_to_divisible(volume.shape, divisor)
    x = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None].to(device)
    if crop_pad is not None:
        pad_width = []
        for a in reversed(crop_pad.per_axis):  # F.pad wants last-dim-first
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", type=Path, required=True, help="cohort directory from cli.preprocess")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="NVIDIA autoencoder checkpoint")
    p.add_argument("--env-config", type=Path, default=None)
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--network-config", type=Path, default=None)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--limit", type=int, default=None, help="first N cases only, for a smoke test")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} exists and is non-empty; pass --overwrite to replace it")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    from ..models.nvidia import (
        DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_NETWORK_CONFIG, load_autoencoder,
    )

    cohort = Cohort(args.cohort)
    log.info("cohort %s", cohort.summary())

    log.info("loading autoencoder from %s", args.checkpoint)
    checkpoint_sha = sha256_file(args.checkpoint)
    autoencoder, _cfg, divisor = load_autoencoder(
        args.checkpoint, args.env_config or DEFAULT_ENV_CONFIG,
        args.model_config or DEFAULT_MODEL_CONFIG, args.network_config or DEFAULT_NETWORK_CONFIG,
        args.device,
    )
    log.info("required spatial divisor = %d", divisor)

    cases = cohort.cases[:args.limit] if args.limit else cohort.cases
    items, failures = [], []
    t0 = time.time()
    # Cases are visited in cohort order, which is bucket-major, so each bucket's archive is written
    # in one contiguous stretch.
    with VolumeWriter(args.out) as writer:
        for n, case in enumerate(cases, start=1):
            try:
                recon = reconstruct(autoencoder, cohort.load_volume(case.case_id), divisor, args.device)
            except Exception as e:  # noqa: BLE001 -- one bad case must not lose the whole run
                failures.append({"case_id": case.case_id, "error": f"{type(e).__name__}: {e}"})
                log.warning("case %s failed: %s", case.case_id, e)
                continue
            # Same bucket as the ground truth, by construction: a reconstruction is the same
            # (modality, plane) at the same geometry as its input.
            writer.add(case.bucket, case.case_id, recon)
            items.append(PredictionItem(
                prediction_id=case.case_id, case_id=case.case_id, sequence=case.sequence,
                plane=case.acquisition_plane, shape=list(recon.shape),
                spacing_mm=list(case.spacing_mm),
            ))
            if n % 50 == 0 or n == len(cases):
                log.info("[%d/%d] %.1fs elapsed", n, len(cases), time.time() - t0)

    elapsed = time.time() - t0
    pset = PredictionSet(
        task="reconstruction", cohort_id=cohort.cohort_id, cohort_cases_sha256=cohort.cases_sha256,
        model={"name": "nvidia_maisi_autoencoder", "checkpoint": str(args.checkpoint),
               "checkpoint_sha256": checkpoint_sha, "required_divisor": divisor},
        items=items, failures=failures, created_by="mrrate_r2v.cli.predict_vae",
        runtime={"device": args.device, "elapsed_sec": round(elapsed, 1),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "torch_version": torch.__version__},
    )
    out = write_prediction_set(args.out, pset)
    log.info("%d reconstructions, %d failures, %.1fs -> %s", len(items), len(failures), elapsed, out)
    log.info("score it: python -m mrrate_r2v.cli.evaluate --task reconstruction "
             "--gt %s --pred %s --out <results>", args.cohort, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
