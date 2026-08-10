"""The one evaluation pipeline. Every task, every model, goes through `run_evaluation`.

    ground-truth cohort  +  prediction set  +  task  ->  results directory

It reads `.npy` volumes and nothing else: no manifest, no archive, no Dataset, no model. That
is deliberate -- the evaluator cannot accidentally preprocess differently than the cohort did,
because it does not preprocess at all.

Order of operations, all of it non-negotiable:

1. Refuse to run unless the prediction set's `cohort_id` matches the ground-truth cohort's
   (`PredictionReader.assert_matches_cohort`). Same cases, same FOV, same N, or no numbers.
2. Refuse to run on an incomplete cohort or prediction set -- a missing volume file must never
   be silently treated as a smaller sample.
3. For paired tasks, pair each prediction to its case by `case_id`, then check geometry with
   `compare_geometry` before any voxelwise metric. A case that fails is excluded with a reason,
   never resized to fit.
4. Compute exactly the metric groups `tasks.py` declares for this task.
5. Write one canonical result layout (see `RESULT_FILES`).

Both `evaluate` entry points and the tests call `run_evaluation`; there is no second path.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..volumes import VolumeReader
from . import aggregate as AGG
from . import geometry_contract as G
from . import paired as M
from . import summary_csv as SUM
from . import tasks as T

log = logging.getLogger("mrrate_r2v.eval")

# One VolumeReader per (pid, artifact root). An NpzFile keeps its zip open, so building a new
# reader per case would re-read a multi-GB central directory every time.
#
# **The pid in the key is load-bearing, not cosmetic.** ProcessPoolExecutor forks, so a child
# inherits whatever handles the parent had already opened, and concurrent reads through a shared
# inherited file descriptor corrupt each other -- observed as `BadZipFile: Overlapped entries` and
# `zlib.error: invalid distance too far back`. Keying on the pid makes a forked child build its own
# handle instead of reusing the parent's.
_READERS: dict = {}


def _reader(root: str) -> VolumeReader:
    key = (os.getpid(), root)
    r = _READERS.get(key)
    if r is None:
        r = VolumeReader(root)
        _READERS[key] = r
    return r

EVALUATION_VERSION = "mr_rate_evaluation_v2"

RESULT_FILES = {
    "summary.json": "headline numbers per sequence and overall -- read this first",
    "per_case_metrics.csv": "one row per scored case",
    "distribution_metrics.json": "population-level metrics (FID, diversity), when computed",
    "excluded_cases.json": "every case that was NOT scored, with the reason",
    "run_manifest.json": "exactly what was run: cohort_id, task, model provenance, versions",
    "anatomy_metrics.json": "anatomical plausibility of the produced population vs the real one",
    "metrics_per_bucket.csv": "THE summary: one row per (modality, plane) with its geometry",
    "metrics_summary.csv": "aggregates: per modality, overall_macro, overall_weighted",
    "figures/": "example orthogonal-slice montages (and optional .nii.gz) for visual inspection",
}


@dataclass
class EvaluationInputs:
    """What an evaluation needs. Assembled by `cli/evaluate.py`; also the seam tests use."""

    cohort: object                        # cohort.Cohort
    predictions: object                   # predictions.PredictionReader
    task: T.TaskSpec
    output_dir: Path
    distribution_metrics: bool = False
    medicalnet_checkpoint: Path | None = None
    device: str = "cpu"
    fid_bootstrap: int = 30
    min_subgroup_n: int = 10
    diversity_k: int = 5
    seed: int = 42
    workers: int = 1                      # parallel worker processes for per-case scoring
    skip_metric_groups: tuple = ()         # groups to drop from what the task declares
    save_figures: int = 3                  # example montages per sequence (0 = none)
    report_image_model: object = None     # must expose .score(text, volume) -> float
    report_classifier: Path | None = None  # blinded pathology classifier (cli.train_report_classifier)
    report_labels_csv: Path | None = None  # default: the repo's mrrate_merged_labels.csv
    save_nifti_cases: int = 0              # also export gt/pred/absdiff .nii.gz for the first N
    extra_run_metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- paired metrics


def compute_paired_metrics(gt: np.ndarray, pred: np.ndarray, groups) -> dict:
    """Every paired metric for the requested groups, on one (ground truth, prediction) pair.

    `_fg` variants are restricted to a foreground mask derived from the *ground truth* only --
    never from the prediction, which would let a degenerate prediction choose its own easier
    evaluation region.
    """
    row = {}
    # Computed once and shared: both groups need it, and it is ~100 ms at 256^3.
    fg = M.foreground_mask_from_intensity(gt) if {"fidelity", "perceptual"} & set(groups) else None
    if "fidelity" in groups:
        row.update({
            "mae_whole": M.mae(gt, pred), "mse_whole": M.mse(gt, pred),
            "psnr_whole": M.psnr(gt, pred), "ncc_whole": M.ncc(gt, pred),
            "ssim3d_whole": M.ssim_3d(gt, pred),
            "mae_fg": M.mae(gt, pred, fg), "mse_fg": M.mse(gt, pred, fg),
            "psnr_fg": M.psnr(gt, pred, mask=fg), "ncc_fg": M.ncc(gt, pred, fg),
            "relative_intensity_error_fg": M.relative_intensity_error(gt, pred, fg),
            "foreground_voxel_fraction": float(fg.mean()),
        })
    if "perceptual" in groups:
        row.update({
            "edge_preservation_fg": M.edge_preservation_ratio(gt, pred, fg),
            "laplacian_variance_ratio_fg": M.laplacian_variance_ratio(gt, pred, fg),
            "hf_energy_ratio": M.high_frequency_energy_ratio(gt, pred),
        })
        for name, axis in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
            s = M.ssim_2d_mean(gt, pred, axis=axis)
            row[f"ssim2d_{name}_mean"] = s["mean"]
            row[f"ssim2d_{name}_n_slices_used"] = s["n_slices_used"]
    return row


def report_image_similarity(report_text, volume: np.ndarray, model=None) -> dict:
    """Pluggable hook for report-image agreement.

    `model` must expose `.score(report_text, volume) -> float`. No such model exists in this
    project yet, so this returns `available=False` with a concrete reason. It never substitutes
    a different model and calls the result a report-alignment score.
    """
    if model is None:
        return {"available": False, "score": None,
                "reason": "no validated MRI image-text model exists in this project yet"}
    if not report_text:
        return {"available": False, "score": None, "reason": "no report text for this case"}
    return {"available": True, "score": float(model.score(report_text, volume)), "reason": None}


# --------------------------------------------------------------------------- the pipeline


@dataclass(frozen=True)
class _ScoreJob:
    """One case's work, as plain picklable data so it can run in a worker process.

    Paths rather than open readers: a worker loads the two `.npy` files itself, which keeps the
    job small to pickle and means no shared state between processes.
    """

    cohort_root: str
    pred_root: str
    case: object          # cohort.CohortCase (frozen dataclass)
    item: object          # predictions.PredictionItem (dataclass)
    groups: tuple
    needs_report: bool
    report_text: str = ""   # carried in the job: a worker must not re-read reports.json per case


def _score_one(job: _ScoreJob, report_image_model=None):
    """Score one case. Returns ("row", metric_row) or ("excluded", reason_dict).

    Module-level and side-effect-free so a ProcessPoolExecutor can call it. Results depend only
    on the job, so running N of these in parallel gives byte-identical output to running them
    serially -- parallelism here is purely a wall-clock concern.
    """
    case, item = job.case, job.item
    # Bucket comes from the case (ground truth) and the item (prediction) independently. If a
    # prediction were filed under the wrong bucket this read fails loudly rather than silently
    # scoring the wrong pair.
    gt = _reader(job.cohort_root).read(case.bucket, case.case_id)
    pred = _reader(job.pred_root).read(item.bucket, item.prediction_id)

    ok, comparison = _check_geometry(case, item, pred.shape)
    if not ok:
        return "excluded", {
            "prediction_id": item.prediction_id, "category": "geometry_incompatible",
            "reason": "; ".join(comparison.reasons) or comparison.decision.value,
            "geometry_comparison": comparison.as_dict(),
        }

    row = {"case_id": case.case_id, "sequence": case.sequence,
           "acquisition_plane": case.acquisition_plane, "bucket": case.bucket,
           "prediction_id": item.prediction_id, "shape": list(case.shape)}
    row.update(compute_paired_metrics(gt, pred, job.groups))
    if job.needs_report:
        # In a worker report_image_model is always None -- a real model may not pickle, so
        # supplying one forces serial execution (see _score_all), where it IS passed through.
        sim = report_image_similarity(job.report_text, pred, report_image_model)
        row["report_image_similarity_available"] = sim["available"]
        row["report_image_similarity_score"] = sim["score"]
        row["report_image_similarity_unavailable_reason"] = sim["reason"]
    return "row", row


def _score_all(jobs, inputs: EvaluationInputs):
    """Run every case's scoring, in parallel when asked. Order is preserved either way.

    Falls back to serial when `workers <= 1`, when there is nothing to gain, or when a real
    `report_image_model` is supplied -- that object lives in the parent process and is not
    assumed picklable.
    """
    workers = max(1, int(inputs.workers or 1))
    if inputs.report_image_model is not None and workers > 1:
        log.info("report_image_model supplied -- scoring serially (the model is not sent to workers)")
        workers = 1
    if workers == 1 or len(jobs) < 2:
        return [_score_one(j, inputs.report_image_model) for j in jobs]

    workers = min(workers, len(jobs))
    log.info("scoring %d cases across %d worker processes", len(jobs), workers)
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # map() preserves input order, so results are deterministic regardless of worker timing.
        return list(pool.map(_score_one, jobs, chunksize=1))


def _pair_items(cohort, predictions, task):
    """[(case, item)] for paired tasks, plus exclusions for unmatched predictions.

    Matching is by `case_id` only -- both sides were written against the same cohort, so there
    is nothing to infer. An item whose `case_id` is absent from the cohort is an error in the
    prediction set, not something to guess about.
    """
    paired, excluded = [], []
    if not task.paired:
        return paired, excluded
    by_case = {c.case_id: c for c in cohort.cases}
    seen = set()
    for item in predictions.items:
        if item.case_id is None:
            excluded.append({"prediction_id": item.prediction_id, "category": "unpaired_item",
                             "reason": f"task={task.name} is paired but this item has no case_id"})
            continue
        if item.case_id in seen:
            excluded.append({"prediction_id": item.prediction_id, "category": "duplicate_case",
                             "reason": f"more than one prediction for case {item.case_id}"})
            continue
        case = by_case.get(item.case_id)
        if case is None:
            excluded.append({"prediction_id": item.prediction_id, "category": "no_matching_case",
                             "reason": f"case_id {item.case_id} is not in this cohort"})
            continue
        seen.add(item.case_id)
        paired.append((case, item))
    for case in cohort.cases:
        if case.case_id not in seen:
            excluded.append({"prediction_id": None, "category": "missing_prediction",
                             "reason": f"cohort case {case.case_id} ({case.sequence}) has no prediction"})
    return paired, excluded


def _check_geometry(case, item, pred_shape):
    """`compare_geometry` between a cohort case and a prediction. Returns (ok, comparison)."""
    gt_geom = G.GeometryRecord.from_cohort_case(case, preprocessing_version=EVALUATION_VERSION)
    pred_geom = G.GeometryRecord(
        shape=tuple(int(x) for x in pred_shape), axis_order=G.DATASET_AXIS_ORDER,
        anatomical_axis_meaning=G.DATASET_ANATOMICAL_AXIS_MEANING,
        spacing_mm=tuple(float(x) for x in item.spacing_mm), orientation=G.DATASET_ORIENTATION,
        affine=None, modality=item.sequence, acquisition_plane=case.acquisition_plane,
        crop_pad=None, valid_bounds=None, preprocessing_version=EVALUATION_VERSION,
        source="prediction", study_key=case.study_key, series_key=case.series_key,
    )
    comparison = G.compare_geometry(gt_geom, pred_geom)
    return comparison.decision == G.GeometryDecision.STRICT_MATCH, comparison


def run_evaluation(inputs: EvaluationInputs) -> dict:
    """Run the whole pipeline and write `inputs.output_dir`. Returns the summary dict."""
    cohort, predictions, task = inputs.cohort, inputs.predictions, inputs.task
    out = Path(inputs.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- gate 1: same cohort, or nothing
    predictions.assert_matches_cohort(cohort)

    # ---- gate 2: both sides complete
    missing_gt = cohort.verify_complete()
    if missing_gt:
        raise SystemExit(
            f"{len(missing_gt)} cohort volumes are missing from {cohort.root} (e.g. "
            f"{missing_gt[:3]}). Re-run preprocess; a partial cohort must not be scored as if "
            f"it were the full one."
        )
    missing_pred = predictions.verify_complete()
    if missing_pred:
        raise SystemExit(
            f"{len(missing_pred)} prediction volumes are missing from {predictions.root} (e.g. "
            f"{missing_pred[:3]}). Re-run the predict step."
        )

    groups = task.groups_to_run(distribution_enabled=inputs.distribution_metrics,
                                skip=inputs.skip_metric_groups)
    if not groups:
        if not task.paired and not inputs.distribution_metrics:
            raise SystemExit(
                f"--task {task.name} has only distribution metrics ({task.unpaired_reason}), so "
                f"disabling them leaves nothing to compute. Enable distribution metrics -- "
                f"cli/evaluate.py does this automatically for unpaired tasks."
            )
        raise SystemExit(
            f"--task {task.name} declares {list(task.metric_groups)} but skip="
            f"{list(inputs.skip_metric_groups)} removes all of them. Drop fewer groups."
        )
    log.info("task=%s paired=%s metric groups=%s", task.name, task.paired, list(groups))
    if inputs.skip_metric_groups:
        log.warning("metric groups SKIPPED by request: %s -- recorded in summary.json",
                    list(inputs.skip_metric_groups))
    log.info("cohort_id=%s  %d cases  %d predictions", cohort.cohort_id, len(cohort.cases),
             len(predictions.items))

    t0 = time.time()
    metric_rows, excluded = [], []
    paired_items, pair_exclusions = _pair_items(cohort, predictions, task)
    excluded.extend(pair_exclusions)

    # ---- paired metrics
    if task.paired:
        needs_report = "report_alignment" in groups
        jobs = [_ScoreJob(cohort_root=str(cohort.root), pred_root=str(predictions.root),
                          case=case, item=item, groups=tuple(groups),
                          needs_report=needs_report,
                          report_text=cohort.load_report(case.case_id) if needs_report else "")
                for case, item in paired_items]
        for outcome, payload in _score_all(jobs, inputs):
            (metric_rows if outcome == "row" else excluded).append(payload)

    # ---- distribution metrics
    distribution_result, case_features = None, []
    if "distribution" in groups:
        distribution_result, case_features = _run_distribution_metrics(inputs, paired_items)

    # ---- blinded classifier consistency (reuses the MedicalNet features extracted just above)
    #
    # Always produces a result object, never None: every task writes report_consistency.json, and a
    # task that does not declare the group says so in it. "Not applicable here" and "the file is
    # missing" must not look the same, and the result layout is identical across tasks by
    # invariant (test_eval_tasks_and_runner.py::test_result_layout_is_identical_across_tasks).
    if "report_consistency" not in groups:
        report_consistency_result = {
            "available": False,
            "reason": (f"--task {task.name} does not declare report_consistency"
                       if "report_consistency" not in task.metric_groups
                       else "report_consistency was skipped by --skip-metric-groups"),
        }
    elif not case_features:
        report_consistency_result = {
            "available": False,
            "reason": "no MedicalNet features available -- report_consistency reuses the "
                      "distribution pass's features, so it needs --distribution-metrics and "
                      "--medicalnet-checkpoint",
        }
        log.warning("report_consistency unavailable: %s", report_consistency_result["reason"])
    else:
        report_consistency_result = _run_report_consistency_metrics(
            inputs, paired_items, case_features)

    # ---- anatomical plausibility (population-level, so it works for unpaired tasks too)
    anatomy_result = None
    if "anatomy" in groups:
        anatomy_result = _run_anatomy_metrics(inputs, paired_items)

    elapsed = time.time() - t0

    # ---- example figures (and optional NIfTI exports) for visual inspection
    figures_written = _save_examples(inputs, out, metric_rows, paired_items)

    # ---- write results
    paired_names = T.paired_metric_names(groups)
    per_sequence = AGG.aggregate_metric_rows(metric_rows, lambda r: r["sequence"], paired_names) if metric_rows else {}
    per_bucket = AGG.aggregate_metric_rows(metric_rows, lambda r: r["bucket"], paired_names) if metric_rows else {}

    summary = {
        "task": task.name,
        "task_summary": task.summary,
        "paired": task.paired,
        "unpaired_reason": task.unpaired_reason,
        "metric_groups_computed": list(groups),
        "metric_groups_skipped": list(inputs.skip_metric_groups),
        "cohort_id": cohort.cohort_id,
        "n_cohort_cases": len(cohort.cases),
        "n_predictions": len(predictions.items),
        "n_scored": len(metric_rows),
        "n_excluded": len(excluded),
        "paired_metrics": per_sequence,
        "paired_metrics_per_bucket": per_bucket,
        "bucket_geometry": {b: cohort.bucket_geometry(b) for b in cohort.buckets},
        "distribution_metrics": distribution_result,
        "report_consistency": report_consistency_result,
        "anatomy": anatomy_result,
        "elapsed_sec": round(elapsed, 1),
        "figures": figures_written,
    }

    _write_csv(out / "per_case_metrics.csv", metric_rows)
    # The clean, human-readable summary: per bucket, then per modality, then two overall rows.
    csv_written = SUM.write_summary_csv(out, cohort, metric_rows, paired_names,
                                       distribution_result, anatomy_result)
    summary["csv_files"] = csv_written
    (out / "excluded_cases.json").write_text(json.dumps(excluded, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if distribution_result is not None:
        (out / "distribution_metrics.json").write_text(json.dumps(distribution_result, indent=2))
    # Its own file, like the other population-level groups: it carries a per-label table and a
    # per-case column, both too wide for summary.json's inline view. The per-case CSV is written
    # with a fixed header even when empty -- it is the input to the challenge's case-level
    # permutation test, and a consumer should find an empty table rather than a missing path.
    (out / "report_consistency.json").write_text(
        json.dumps(report_consistency_result, indent=2, default=str))
    _write_csv(out / "report_consistency_per_case.csv",
               report_consistency_result.get("per_case") or [],
               fieldnames=("case_id", "bucket", "consistency", "consistency_real"))
    # The readable deliverable: one row per clinical label, generated score next to the same
    # classifier's real-volume ceiling and the image-blind floor. This is the table to quote.
    _write_csv(
        out / "report_consistency_per_label.csv",
        [{"label": name, **{k: v for k, v in entry.items() if k != "interpretation"}}
         for name, entry in (report_consistency_result.get("per_label") or {}).items()],
        fieldnames=("label", "auroc", "real_reference_auroc", "prevalence_baseline_auroc",
                    "retention", "average_precision", "prevalence", "n", "n_positive",
                    "mean_predicted_probability", "usable", "low_support"),
    )
    if anatomy_result is not None:
        (out / "anatomy_metrics.json").write_text(json.dumps(anatomy_result, indent=2))

    run_manifest = {
        "evaluation_version": EVALUATION_VERSION,
        "geometry_contract_version": G.GEOMETRY_CONTRACT_VERSION,
        "task": task.name,
        "metric_groups_computed": list(groups),
        "metric_groups_skipped": list(inputs.skip_metric_groups),
        "cohort": {"root": str(cohort.root), **cohort.summary()},
        "predictions": {"root": str(predictions.root), "task": predictions.task,
                        "model": predictions.model, "n_items": len(predictions.items)},
        "distribution_metrics_enabled": inputs.distribution_metrics,
        "seed": inputs.seed, "device": inputs.device,
        "elapsed_sec": round(elapsed, 1),
        **inputs.extra_run_metadata,
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True))

    log.info("done: %d scored, %d excluded, %.1fs -> %s", len(metric_rows), len(excluded), elapsed, out)
    return summary


def _save_examples(inputs: EvaluationInputs, out: Path, metric_rows, paired_items) -> list:
    """Write example figures (and optional NIfTI triplets) into `<out>/figures`.

    Never fatal: a plotting failure must not throw away a completed evaluation, so problems are
    logged and the metrics still get written. Returns the relative paths written.
    """
    if inputs.save_figures <= 0 and inputs.save_nifti_cases <= 0:
        return []
    try:
        from . import figures as F
    except ImportError as e:  # pragma: no cover - PIL missing
        log.warning("cannot import figure support (%s) -- skipping example figures", e)
        return []

    cohort, predictions, task = inputs.cohort, inputs.predictions, inputs.task
    fig_dir = out / F.FIGURES_DIR
    written = []

    try:
        if task.paired:
            by_id = {item.prediction_id: case for case, item in paired_items}
            # Rank by the primary fidelity metric when it was computed; otherwise fall back to
            # cohort order so figures still appear for a fidelity-skipped run.
            metric = "psnr_fg" if any("psnr_fg" in r for r in metric_rows) else None
            per_bucket: dict = {}
            for row in metric_rows:
                per_bucket.setdefault(row["bucket"], []).append(row)

            for bucket, rows in sorted(per_bucket.items()):
                chosen = (F.select_ranked(rows, metric, inputs.save_figures) if metric
                          else sorted(rows, key=lambda r: r["case_id"])[:inputs.save_figures])
                for rank, row in enumerate(chosen):
                    case = by_id.get(row["prediction_id"])
                    if case is None:
                        continue
                    gt = cohort.load_volume(case.case_id)
                    pred = predictions.load_volume(row["prediction_id"])
                    caption = (f"{metric}={row[metric]:.3f}" if metric and metric in row else "")
                    name = f"{bucket}_rank{rank}_{case.case_id}.png"
                    F.save_paired_figure(gt, pred, fig_dir / name, case_id=case.case_id,
                                         sequence=case.sequence, plane=case.acquisition_plane,
                                         caption=caption)
                    written.append(f"{F.FIGURES_DIR}/{name}")
                    if rank < inputs.save_nifti_cases:
                        stem = f"{bucket}_rank{rank}_{case.case_id}"
                        for tag, arr in (("gt", gt), ("pred", pred), ("absdiff", np.abs(gt - pred))):
                            F.save_example_nifti(arr, case.spacing_mm,
                                                 fig_dir / f"{stem}_{tag}.nii.gz")
                            written.append(f"{F.FIGURES_DIR}/{stem}_{tag}.nii.gz")
        else:
            # Unconditional generation: no counterpart, so show a real volume from the same bucket
            # alongside for scale -- same modality AND plane, or the comparison misleads.
            for bucket in cohort.buckets:
                gen_items = [i for i in predictions.items if i.bucket == bucket]
                real = cohort.cases_for_bucket(bucket)
                for rank, item in enumerate(gen_items[:inputs.save_figures]):
                    reference = (cohort.load_volume(real[rank % len(real)].case_id) if real else None)
                    name = f"{bucket}_gen{rank}_{item.prediction_id}.png"
                    F.save_unpaired_figure(predictions.load_volume(item.prediction_id), reference,
                                           fig_dir / name, prediction_id=item.prediction_id,
                                           sequence=bucket)
                    written.append(f"{F.FIGURES_DIR}/{name}")
    except Exception as e:  # noqa: BLE001 - figures are a convenience, metrics are the result
        log.warning("example figure generation failed (%s: %s) -- metrics are unaffected",
                    type(e).__name__, e)

    if written:
        log.info("wrote %d example file(s) -> %s", len(written), fig_dir)
    return written


def _run_anatomy_metrics(inputs: EvaluationInputs, paired_items):
    """Anatomical plausibility, real population vs produced population.

    Unpaired by construction (a KS test between two distributions), so it is meaningful for
    unconditional generation as well as for paired tasks. Parallelised with the same worker count
    as the paired scoring -- the measures are pure functions of one volume.
    """
    from . import anatomy as A

    cohort, predictions, task = inputs.cohort, inputs.predictions, inputs.task
    real_refs = [(str(cohort.root), c.bucket, c.case_id) for c in cohort.cases]
    if task.paired:
        prod_refs = [(str(predictions.root), i.bucket, i.prediction_id) for _c, i in paired_items]
    else:
        prod_refs = [(str(predictions.root), i.bucket, i.prediction_id) for i in predictions.items]

    workers = max(1, int(inputs.workers or 1))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            real = list(pool.map(_measure_one, real_refs, chunksize=1))
            prod = list(pool.map(_measure_one, prod_refs, chunksize=1))
    else:
        real = [_measure_one(r) for r in real_refs]
        prod = [_measure_one(r) for r in prod_refs]

    out = {"overall": A.compare_populations(real, prod)}
    # Per bucket and per sequence, using the same grouping every other metric family uses.
    prod_keys = ([(i.bucket, i.sequence) for _c, i in paired_items] if task.paired
                 else [(i.bucket, i.sequence) for i in predictions.items])
    for level, real_key, prod_key in ((0, lambda c: c.bucket, lambda k: k[0]),
                                      (1, lambda c: c.sequence, lambda k: k[1])):
        g_real, g_prod = {}, {}
        for c, m in zip(cohort.cases, real):
            g_real.setdefault(real_key(c), []).append(m)
        for k, m in zip(prod_keys, prod):
            g_prod.setdefault(prod_key(k), []).append(m)
        for name in sorted(set(g_real) & set(g_prod)):
            out[name] = A.compare_populations(g_real[name], g_prod[name])
    return out


def _measure_one(ref) -> dict:
    """Anatomical measures for one volume. `ref` is (root, bucket, volume_id); module-level and
    tuple-argument so a ProcessPoolExecutor can map over it."""
    from . import anatomy as A

    root, bucket, volume_id = ref
    return A.measure(_reader(root).read(bucket, volume_id))


def _run_distribution_metrics(inputs: EvaluationInputs, paired_items):
    """Population-level metrics. Imported lazily: this is the only part of the evaluator that
    needs torch and a MedicalNet checkpoint, so a fidelity-only run stays lightweight.

    For paired tasks the real/produced populations are the matched pairs. For generation there
    is no pairing, so real and produced volumes are lined up **within a bucket**, truncated to
    `min(n_real, n_gen)` -- Frechet distance needs two populations of comparable size, not a
    correspondence. Bucket, not sequence: only within a bucket do all volumes share a geometry,
    and pairing across planes would compare an axial real volume with a sagittal generated one.
    """
    from . import distribution as DM

    cohort, predictions, task = inputs.cohort, inputs.predictions, inputs.task
    medicalnet = DM.MedicalNetFeatureExtractor(inputs.medicalnet_checkpoint, inputs.device) \
        if inputs.medicalnet_checkpoint else None
    inception = DM.InceptionFeatureExtractor(inputs.device)

    def features_for(case_id_or_none, sequence, bucket, real_arr, gen_arr):
        cf = DM.CaseFeatures(case_id=case_id_or_none or "gen", sequence=sequence, bucket=bucket)
        if real_arr is not None:
            if medicalnet is not None:
                cf.medicalnet_real = medicalnet.extract(real_arr)
            cf.inception_2p5d_real = DM.extract_2p5d_inception_features(real_arr, inception)
            cf.mid_slice_real = DM.mid_slice(real_arr, axis=2)
            _, p = inception.extract_batch(cf.mid_slice_real[None])
            cf.inception_mid_probs_real = p[0]
        if gen_arr is not None:
            if medicalnet is not None:
                cf.medicalnet_gen = medicalnet.extract(gen_arr)
            cf.inception_2p5d_gen = DM.extract_2p5d_inception_features(gen_arr, inception)
            cf.mid_slice_gen = DM.mid_slice(gen_arr, axis=2)
            _, p = inception.extract_batch(cf.mid_slice_gen[None])
            cf.inception_mid_probs_gen = p[0]
        return cf

    all_features = []
    if task.paired:
        for case, item in paired_items:
            all_features.append(features_for(
                case.case_id, case.sequence, case.bucket,
                cohort.load_volume(case.case_id), predictions.load_volume(item.prediction_id),
            ))
    else:
        gen_by_bucket: dict = {}
        for item in predictions.items:
            gen_by_bucket.setdefault(item.bucket, []).append(item)
        for bucket in cohort.buckets:
            real_cases = cohort.cases_for_bucket(bucket)
            gen_items = gen_by_bucket.get(bucket, [])
            n = min(len(real_cases), len(gen_items))
            if n < max(len(real_cases), len(gen_items)):
                log.info("%s: %d real vs %d generated -- using min(%d) for population metrics",
                         bucket, len(real_cases), len(gen_items), n)
            for i in range(n):
                all_features.append(features_for(
                    f"{bucket}_{i}", real_cases[i].sequence, bucket,
                    cohort.load_volume(real_cases[i].case_id),
                    predictions.load_volume(gen_items[i].prediction_id),
                ))

    if not all_features:
        log.warning("no feature pairs available -- skipping distribution metrics")
        return None, []
    result = DM.compute_distribution_metrics(
        all_features, cohort.sequences, min_subgroup_n=inputs.min_subgroup_n,
        n_bootstrap=inputs.fid_bootstrap, seed=inputs.seed, k_diversity=inputs.diversity_k,
        buckets=cohort.buckets,
    )
    # The features are returned as well as consumed: `report_consistency` needs the very same
    # MedicalNet vectors, and extracting them twice would double the GPU cost of an evaluation for
    # nothing. They are per-case and keyed by case_id, so the two metrics cannot line up differently.
    return result, all_features


def _run_report_consistency_metrics(inputs: EvaluationInputs, paired_items, case_features):
    """The blinded-classifier metric: does a classifier trained only on real volumes read, off each
    generated volume, the findings its conditioning report described?

    Runs the same classifier over BOTH populations -- real and generated -- so the generated score
    always arrives with the ceiling it should be read against. A classifier that cannot separate a
    label on real volumes says nothing about generated ones, and `evaluate_consistency` marks those
    labels rather than quietly averaging them in.

    Returns a dict that is always written, even when it could not run: an unavailable metric with a
    stated reason is information, a missing key is not.
    """
    from .report_classifier import (
        auroc,
        evaluate_consistency,
        load_classifier_or_none,
        per_case_consistency,
        prevalence_baseline_auroc,
    )
    from .report_labels import ReportLabels

    classifier, reason = load_classifier_or_none(inputs.report_classifier, inputs.device)
    if classifier is None:
        log.warning("report_consistency unavailable: %s", reason)
        return {"available": False, "reason": reason}

    cohort = inputs.cohort
    try:
        labels = ReportLabels(inputs.report_labels_csv)
    except SystemExit as e:
        return {"available": False, "reason": str(e)}
    if tuple(labels.labels) != tuple(classifier.labels):
        return {"available": False,
                "reason": f"label set mismatch: classifier was fitted on {list(classifier.labels)}, "
                          f"{labels.path} provides {list(labels.labels)}"}

    joined = labels.for_cohort(cohort)
    # Features are keyed by case_id and were built in paired_items order; rebuild the row order
    # from the cases that have BOTH a label and a feature pair.
    by_case = {cf.case_id: cf for cf in case_features}
    rows = [(case, by_case[case.case_id]) for case, _item in paired_items
            if case.case_id in by_case and case.case_id in joined
            and by_case[case.case_id].medicalnet_real is not None
            and by_case[case.case_id].medicalnet_gen is not None]
    if not rows:
        return {"available": False,
                "reason": "no case had both a report label and a MedicalNet feature pair "
                          "(is --medicalnet-checkpoint set? distribution metrics enabled?)"}

    truth = np.array([joined[case.case_id] for case, _ in rows], dtype=np.int64)
    probabilities_real = classifier.predict_proba(np.stack([cf.medicalnet_real for _, cf in rows]))
    probabilities_gen = classifier.predict_proba(np.stack([cf.medicalnet_gen for _, cf in rows]))

    real_reference = {
        name: {"auroc": auroc(truth[:, i], probabilities_real[:, i])}
        for i, name in enumerate(classifier.labels)
    }
    result = evaluate_consistency(
        probabilities_gen, truth, classifier.labels,
        real_reference=real_reference,
        prevalence_baseline=prevalence_baseline_auroc(classifier.labels),
    )
    # The same computation on the real volumes, in full -- this is the ceiling row of the table,
    # not a footnote.
    result["real_reference"] = evaluate_consistency(
        probabilities_real, truth, classifier.labels,
        real_reference=real_reference,
        prevalence_baseline=prevalence_baseline_auroc(classifier.labels),
    )
    per_case = per_case_consistency(probabilities_gen, truth, classifier.labels,
                                    usable_labels=result["labels_usable"])
    per_case_real = per_case_consistency(probabilities_real, truth, classifier.labels,
                                         usable_labels=result["labels_usable"])
    result.update({
        "available": True,
        "n_scored": len(rows),
        "n_cases_without_labels": len(paired_items) - len(rows),
        "label_coverage": labels.cohort_coverage(cohort),
        "classifier": {
            "path": str(inputs.report_classifier),
            "labels": list(classifier.labels),
            "provenance": vars(classifier.provenance),
        },
        # Per case, for the challenge's case-level permutation test. case_id only: no identifiers.
        "per_case": [
            {"case_id": case.case_id, "bucket": case.bucket,
             "consistency": None if np.isnan(v) else float(v),
             "consistency_real": None if np.isnan(r) else float(r)}
            for (case, _), v, r in zip(rows, per_case, per_case_real)
        ],
    })
    finite = per_case[~np.isnan(per_case)]
    result["mean_per_case_consistency"] = float(finite.mean()) if finite.size else None
    finite_real = per_case_real[~np.isnan(per_case_real)]
    result["mean_per_case_consistency_real"] = float(finite_real.mean()) if finite_real.size else None
    log.info("report_consistency: %d cases, macro AUROC %s (real reference %s) over %d usable labels",
             len(rows), result["macro_auroc_usable_labels"],
             result["real_reference"]["macro_auroc_usable_labels"], result["n_labels_usable"])
    return result


def _write_csv(path: Path, rows, fieldnames=None) -> None:
    """`fieldnames` fixes the header so a table with no rows is still a readable, well-formed CSV
    rather than a zero-byte file. Without it the columns are the union of the rows', as before."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows and not fieldnames:
            return
        w = csv.DictWriter(f, fieldnames=list(fieldnames or sorted({k for r in rows for k in r})))
        w.writeheader()
        w.writerows(rows)
