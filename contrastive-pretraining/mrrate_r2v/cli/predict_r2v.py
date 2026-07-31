#!/usr/bin/env python3
"""Stage 2: report-to-volume inference. Computes no metrics.

No report-conditioned checkpoint exists in this project yet, so this script has nothing to
load. It is written now because everything *around* the model is already fixed: it reads a
cohort, generates one volume per case from that case's report text, and writes a prediction
set that `python -m mrrate_r2v.cli.evaluate --task report2volume` scores. When a checkpoint
appears, implement `load_r2v_model` and `generate_one` below and the rest of the pipeline works
unchanged.

    python -m mrrate_r2v.cli.predict_r2v \\
        --cohort <workspace>/cohorts/test_v1 \\
        --checkpoint <your_checkpoint> \\
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


def load_r2v_model(checkpoint: Path, device: str):
    """Load a report-conditioned generator. Not implemented -- no such checkpoint exists yet.

    A model returned here must expose `generate(report_text, shape, spacing_mm, seed) ->
    np.ndarray[X, Y, Z]`. Wire in the real loader when a checkpoint exists; do not substitute
    a different model, because the resulting numbers would be attributed to a report-to-volume
    system that was never run.
    """
    raise SystemExit(
        "predict_r2v: no report-to-volume checkpoint is implemented yet.\n"
        "Implement load_r2v_model() and generate_one() in mrrate_r2v/cli/predict_r2v.py, or -- if "
        "your checkpoint already wrote .nii.gz files -- use\n"
        "    python -m mrrate_r2v.cli.import_predictions --cohort <cohort> "
        "--predictions-csv <csv> --out <pred_dir>"
    )


def generate_one(model, report_text: str, case, seed: int) -> np.ndarray:
    """One report in, one volume out on the case's own grid."""
    return model.generate(report_text, tuple(case.shape), tuple(case.spacing_mm), seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-name", default="report2volume", help="recorded in the prediction set's provenance")
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
    model = load_r2v_model(args.checkpoint, args.device)  # raises until implemented

    cases = cohort.cases[:args.limit] if args.limit else cohort.cases
    items, failures = [], []
    t0 = time.time()
    with VolumeWriter(args.out) as writer:
        for n, case in enumerate(cases, start=1):
            seed = stable_seed(args.seed, case.case_id)
            try:
                volume = generate_one(model, cohort.load_report(case.case_id), case, seed)
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
