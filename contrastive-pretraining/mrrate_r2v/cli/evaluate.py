#!/usr/bin/env python3
"""Stage 3: the evaluator. One command scores every task.

    python -m mrrate_r2v.cli.evaluate --task <task> --gt <cohort_dir> --pred <pred_dir> --out <results_dir>

`--task` decides which metrics run; nothing else does:

    reconstruction   voxelwise fidelity + detail preservation + distribution
    report2volume    the above + report alignment
    generation       distribution only -- no real patient corresponds to a generated volume

`--gt` is always a cohort directory from `cli.preprocess`. For paired tasks its volumes are the
per-case ground truth; for `generation` they are the real reference population. Either way the
FOV and the number of samples come from the cohort, so two runs against the same cohort are
comparable by construction -- and a prediction set built against a *different* cohort is
rejected outright rather than scored.

Distribution metrics (FID and friends) need a GPU-ish workload and a MedicalNet checkpoint, so
they are opt-in via `--distribution-metrics` for paired tasks. They are always on for
`generation`, which has no other metrics.

Read `summary.json` first; `run_manifest.json` is the record of what actually ran.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from ..cohort import Cohort
from ..eval.runner import EvaluationInputs, run_evaluation
from ..eval.tasks import TASK_NAMES, get_task
from ..eval.wandb_logging import WANDB_MODES, WandbRun
from ..predictions import PredictionReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(TASK_NAMES))
    p.add_argument("--gt", type=Path, required=True, help="ground-truth cohort directory")
    p.add_argument("--pred", type=Path, required=True, help="prediction directory")
    p.add_argument("--out", type=Path, required=True, help="results directory to create")
    p.add_argument("--overwrite", action="store_true")

    dm = p.add_argument_group("distribution metrics")
    dm.add_argument("--distribution-metrics", action="store_true",
                    help="compute FID / Inception Score / precision-recall-density-coverage "
                         "(always on for --task generation)")
    dm.add_argument("--medicalnet-checkpoint", type=Path, default=None,
                    help="MedicalNet ResNet-10 weights for the 3D FID; without it only the 2.5D "
                         "Inception variant is computed")
    dm.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    dm.add_argument("--fid-bootstrap", type=int, default=30)
    dm.add_argument("--min-subgroup-n", type=int, default=10,
                    help="skip per-sequence distribution metrics below this many cases -- they are "
                         "not stable on small subgroups")
    dm.add_argument("--diversity-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-metric-groups", nargs="*", default=[],
                   choices=["fidelity", "perceptual", "distribution", "report_alignment"],
                   help="drop metric groups the task would otherwise compute. `perceptual` "
                        "(edge preservation, Laplacian variance, HF energy, per-plane SSIM) is "
                        "~40%% of per-case time. Can only remove, never add -- generation still "
                        "cannot get a voxelwise metric. Recorded in summary.json as "
                        "metric_groups_skipped.")
    p.add_argument("--workers", type=int, default=1,
                   help="worker processes for per-case scoring. Evaluation is CPU-compute-bound "
                        "(measured: I/O is 0.5%% of the work), so this is the one knob that "
                        "actually speeds it up -- scaling is near-linear in cores. Results are "
                        "identical regardless of value. In Slurm, pass $SLURM_CPUS_PER_TASK.")

    wb = p.add_argument_group("optional W&B")
    wb.add_argument("--wandb-mode", default="disabled", choices=list(WANDB_MODES))
    wb.add_argument("--wandb-entity", default=None)
    wb.add_argument("--wandb-project", default=None)
    wb.add_argument("--wandb-group", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} exists and is non-empty; pass --overwrite to replace it")

    task = get_task(args.task)
    cohort = Cohort(args.gt)
    predictions = PredictionReader(args.pred)

    if predictions.task != task.name:
        log.warning("prediction set was produced for task=%r but you asked to score it as %r. "
                    "Proceeding -- the metric set follows --task -- but check this is intended.",
                    predictions.task, task.name)

    distribution = args.distribution_metrics or not task.paired
    if not task.paired and not args.distribution_metrics:
        log.info("--task %s has no paired metrics, so distribution metrics are enabled "
                 "automatically", task.name)

    if args.device == "cuda" and distribution:
        import torch
        if not torch.cuda.is_available():
            log.warning("--device cuda but no GPU visible; falling back to cpu for feature extraction")
            args.device = "cpu"

    inputs = EvaluationInputs(
        cohort=cohort, predictions=predictions, task=task, output_dir=args.out,
        distribution_metrics=distribution, medicalnet_checkpoint=args.medicalnet_checkpoint,
        device=args.device, fid_bootstrap=args.fid_bootstrap,
        min_subgroup_n=args.min_subgroup_n, diversity_k=args.diversity_k, seed=args.seed,
        workers=args.workers, skip_metric_groups=tuple(args.skip_metric_groups),
        extra_run_metadata={"slurm_job_id": os.environ.get("SLURM_JOB_ID")},
    )
    summary = run_evaluation(inputs)

    if args.wandb_mode != "disabled":
        run = WandbRun(
            mode=args.wandb_mode, entity=args.wandb_entity, project=args.wandb_project,
            run_name=f"{task.name}-{cohort.cohort_id}-{os.environ.get('SLURM_JOB_ID', 'local')}",
            group=args.wandb_group or f"mr-rate-{task.name}",
            tags=[task.name, cohort.spec["split"]],
            config={"task": task.name, "cohort_id": cohort.cohort_id, "model": predictions.model},
        )
        run.log({"summary": summary.get("paired_metrics", {}).get("overall", {})})
        (args.out / "wandb_run.json").write_text(json.dumps(run.finish(), indent=2))

    print(f"\n{'=' * 72}")
    print(f"task={task.name}  cohort_id={cohort.cohort_id}")
    print(f"scored {summary['n_scored']} / {summary['n_cohort_cases']} cases "
          f"({summary['n_excluded']} excluded)")
    overall = summary.get("paired_metrics", {}).get("overall", {})
    for name in ("psnr_fg", "ssim3d_whole", "mae_fg"):
        if name in overall and overall[name].get("mean") is not None:
            print(f"  {name:24s} {overall[name]['mean']:.4f} +/- {overall[name]['std']:.4f}")
    print(f"full results -> {args.out}/summary.json")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
