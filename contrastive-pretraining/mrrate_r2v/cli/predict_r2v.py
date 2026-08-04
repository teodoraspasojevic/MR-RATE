#!/usr/bin/env python3
"""Stage 2: report-to-volume inference. Computes no metrics.

Reads a cohort, generates one volume per case from that case's report text, and writes a
prediction set that `python -m mrrate_r2v.cli.evaluate --task report2volume` scores. The model
is a frozen NVIDIA base UNet plus a trained report adapter (`cli.train_r2v`); it is assembled
by `cli.generate_r2v.build_sampler`, which is also what the free-form single-report script uses.

    python -m mrrate_r2v.cli.predict_r2v \\
        --cohort <workspace>/cohorts/test_v1 \\
        --checkpoint <workspace>/runs/r2v_adapter_v1/adapter_last.pt \\
        --base-checkpoint <workspace>/models/diff_unet_3d_rflow-mr-brain_v0.pt \\
        --vae-checkpoint <workspace>/models/autoencoder_v1.pt \\
        --text-checkpoint <workspace>/pretrained/RadBERT-RoBERTa-4m \\
        --out <workspace>/predictions/r2v_v1

To score a checkpoint that already saved `.nii.gz` files elsewhere, you do not need this
script -- use `python -m mrrate_r2v.cli.import_predictions` instead.

Two things a real implementation must honor:

- Output on the cohort's own grid (`case.shape` / `case.spacing_mm`). Emitting a different
  grid is allowed but the evaluator will exclude those cases rather than resize them.
- Derive the per-case seed from `stable_seed(args.seed, case.case_id)` so a rerun reproduces
  the same volumes.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np

from ..cohort import Cohort, sha256_file
from ..predictions import PredictionItem, PredictionSet, write_prediction_set
from ..volumes import VolumeWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict_r2v")


def stable_seed(base_seed: int, case_id: str) -> int:
    h = hashlib.sha256(f"{base_seed}:{case_id}".encode()).hexdigest()
    return int(h[:8], 16)


def load_r2v_model(args):
    """Load the report-conditioned generator described by `args`.

    Delegates to `cli.generate_r2v.build_sampler`, so this script and the free-form single-report
    script cannot diverge on how a model is assembled. `--checkpoint` is the *adapter*: the frozen
    base UNet, the VAE and the text encoder are separate artefacts.
    """
    from types import SimpleNamespace

    from .generate_r2v import build_sampler

    sampler, _embedder, _payload = build_sampler(SimpleNamespace(
        base_checkpoint=args.base_checkpoint, vae_checkpoint=args.vae_checkpoint,
        adapter=args.checkpoint, network_config=args.network_config,
        text_encoder=None, text_checkpoint=args.text_checkpoint, max_report_tokens=None,
        device=args.device, latent_only=False,
        report_guidance_scale=args.report_guidance_scale,
        modality_guidance_scale=args.modality_guidance_scale,
        num_inference_steps=args.num_inference_steps, seed=args.seed,
        batched_guidance=True, allow_base_mismatch=args.allow_base_mismatch,
    ))
    return sampler


def generate_one(model, report_text: str, case, seed: int, modality: str = "T1w") -> np.ndarray:
    """One report in, one volume out on the case's own grid."""
    return model.generate(report_text, tuple(case.shape), tuple(case.spacing_mm), seed, modality=modality)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="adapter checkpoint (cli.train_r2v)")
    p.add_argument("--base-checkpoint", type=Path, required=True, help="frozen NVIDIA diffusion UNet")
    p.add_argument("--vae-checkpoint", type=Path, required=True, help="frozen NVIDIA autoencoder")
    p.add_argument("--text-checkpoint", type=Path, default=None,
                   help="text encoder directory; default = whatever the adapter recorded")
    p.add_argument("--network-config", type=Path, default=None)
    p.add_argument("--model-name", default="report2volume", help="recorded in the prediction set's provenance")
    p.add_argument("--modality", default="T1w", choices=["T1w", "T2w", "FLAIR", "SWI", "unknown"])
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--report-guidance-scale", type=float, default=4.0)
    p.add_argument("--modality-guidance-scale", type=float, default=10.0)
    p.add_argument("--allow-base-mismatch", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} exists and is non-empty; pass --overwrite to replace it")

    cohort = Cohort(args.cohort)
    log.info("cohort %s", cohort.summary())
    model = load_r2v_model(args)

    cases = cohort.cases[:args.limit] if args.limit else cohort.cases
    items, failures = [], []
    t0 = time.time()
    with VolumeWriter(args.out) as writer:
        for n, case in enumerate(cases, start=1):
            seed = stable_seed(args.seed, case.case_id)
            try:
                volume = generate_one(model, cohort.load_report(case.case_id), case, seed,
                                      modality=args.modality)
            except Exception as e:  # noqa: BLE001
                failures.append({"case_id": case.case_id, "error": f"{type(e).__name__}: {e}"})
                log.warning("case %s failed: %s", case.case_id, e)
                continue
            writer.add(case.bucket, case.case_id, volume)
            items.append(PredictionItem(
                prediction_id=case.case_id, case_id=case.case_id, sequence=case.sequence,
                plane=case.acquisition_plane, shape=list(volume.shape),
                spacing_mm=list(case.spacing_mm), seed=seed,
            ))
            if n % 50 == 0 or n == len(cases):
                log.info("[%d/%d] %.1fs elapsed", n, len(cases), time.time() - t0)

    elapsed = time.time() - t0
    pset = PredictionSet(
        task="report2volume", cohort_id=cohort.cohort_id, cohort_cases_sha256=cohort.cases_sha256,
        model={"name": args.model_name, "checkpoint": str(args.checkpoint),
               "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint.is_file() else None},
        items=items, failures=failures, created_by="mrrate_r2v.cli.predict_r2v",
        runtime={"device": args.device, "elapsed_sec": round(elapsed, 1), "seed": args.seed},
    )
    out = write_prediction_set(args.out, pset)
    log.info("%d volumes, %d failures, %.1fs -> %s", len(items), len(failures), elapsed, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
