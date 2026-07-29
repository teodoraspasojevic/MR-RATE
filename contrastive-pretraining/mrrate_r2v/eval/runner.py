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
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import aggregate as AGG
from . import geometry_contract as G
from . import paired as M
from . import tasks as T

log = logging.getLogger("mrrate_r2v.eval")

EVALUATION_VERSION = "mr_rate_evaluation_v2"

RESULT_FILES = {
    "summary.json": "headline numbers per sequence and overall -- read this first",
    "per_case_metrics.csv": "one row per scored case",
    "distribution_metrics.json": "population-level metrics (FID, diversity), when computed",
    "excluded_cases.json": "every case that was NOT scored, with the reason",
    "run_manifest.json": "exactly what was run: cohort_id, task, model provenance, versions",
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
    report_image_model: object = None     # must expose .score(text, volume) -> float
    save_nifti_cases: int = 0
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


def _score_one(job: _ScoreJob, report_image_model=None):
    """Score one case. Returns ("row", metric_row) or ("excluded", reason_dict).

    Module-level and side-effect-free so a ProcessPoolExecutor can call it. Results depend only
    on the job, so running N of these in parallel gives byte-identical output to running them
    serially -- parallelism here is purely a wall-clock concern.
    """
    case, item = job.case, job.item
    gt = np.load(f"{job.cohort_root}/volumes/{case.case_id}.npy").astype(np.float32, copy=False)
    pred = np.load(f"{job.pred_root}/volumes/{item.prediction_id}.npy").astype(np.float32, copy=False)

    ok, comparison = _check_geometry(case, item, pred.shape)
    if not ok:
        return "excluded", {
            "prediction_id": item.prediction_id, "category": "geometry_incompatible",
            "reason": "; ".join(comparison.reasons) or comparison.decision.value,
            "geometry_comparison": comparison.as_dict(),
        }

    row = {"case_id": case.case_id, "sequence": case.sequence,
           "acquisition_plane": case.acquisition_plane,
           "prediction_id": item.prediction_id, "shape": list(case.shape)}
    row.update(compute_paired_metrics(gt, pred, job.groups))
    if job.needs_report:
        report_path = Path(job.cohort_root) / "reports" / f"{case.case_id}.txt"
        text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
        # In a worker this is always None -- a real model may not pickle, so supplying one forces
        # serial execution (see _score_all), where it IS passed through.
        sim = report_image_similarity(text, pred, report_image_model)
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
        jobs = [_ScoreJob(cohort_root=str(cohort.root), pred_root=str(predictions.root),
                          case=case, item=item, groups=tuple(groups),
                          needs_report="report_alignment" in groups)
                for case, item in paired_items]
        for outcome, payload in _score_all(jobs, inputs):
            (metric_rows if outcome == "row" else excluded).append(payload)

    # ---- distribution metrics
    distribution_result = None
    if "distribution" in groups:
        distribution_result = _run_distribution_metrics(inputs, paired_items)

    elapsed = time.time() - t0

    # ---- write results
    paired_names = T.paired_metric_names(groups)
    per_sequence = AGG.aggregate_metric_rows(metric_rows, lambda r: r["sequence"], paired_names) if metric_rows else {}

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
        "distribution_metrics": distribution_result,
        "elapsed_sec": round(elapsed, 1),
    }

    _write_csv(out / "per_case_metrics.csv", metric_rows)
    (out / "excluded_cases.json").write_text(json.dumps(excluded, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if distribution_result is not None:
        (out / "distribution_metrics.json").write_text(json.dumps(distribution_result, indent=2))

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


def _run_distribution_metrics(inputs: EvaluationInputs, paired_items):
    """Population-level metrics. Imported lazily: this is the only part of the evaluator that
    needs torch and a MedicalNet checkpoint, so a fidelity-only run stays lightweight.

    For paired tasks the real/produced populations are the matched pairs. For generation there
    is no pairing, so the real population is the whole cohort and the produced population is
    every prediction item, truncated to `min(n_real, n_gen)` per sequence -- Frechet distance
    needs two populations of comparable size, not a correspondence.
    """
    from . import distribution as DM

    cohort, predictions, task = inputs.cohort, inputs.predictions, inputs.task
    medicalnet = DM.MedicalNetFeatureExtractor(inputs.medicalnet_checkpoint, inputs.device) \
        if inputs.medicalnet_checkpoint else None
    inception = DM.InceptionFeatureExtractor(inputs.device)

    def features_for(case_id_or_none, sequence, real_arr, gen_arr):
        cf = DM.CaseFeatures(case_id=case_id_or_none or "gen", sequence=sequence)
        if real_arr is not None:
            if medicalnet is not None:
                cf.medicalnet_real = medicalnet.extract(real_arr)
            cf.inception_2p5d_real = DM.extract_2p5d_inception_features(real_arr, inception)
            _, p = inception.extract_batch(DM.mid_slice(real_arr, axis=2)[None])
            cf.inception_mid_probs_real = p[0]
        if gen_arr is not None:
            if medicalnet is not None:
                cf.medicalnet_gen = medicalnet.extract(gen_arr)
            cf.inception_2p5d_gen = DM.extract_2p5d_inception_features(gen_arr, inception)
            _, p = inception.extract_batch(DM.mid_slice(gen_arr, axis=2)[None])
            cf.inception_mid_probs_gen = p[0]
        return cf

    all_features = []
    if task.paired:
        for case, item in paired_items:
            all_features.append(features_for(
                case.case_id, case.sequence,
                cohort.load_volume(case.case_id), predictions.load_volume(item.prediction_id),
            ))
    else:
        for seq in cohort.sequences:
            real_cases = cohort.cases_for_sequence(seq)
            gen_items = [i for i in predictions.items if i.sequence == seq]
            n = min(len(real_cases), len(gen_items))
            if n < len(real_cases) or n < len(gen_items):
                log.info("sequence=%s: %d real vs %d generated -- using min(%d) for "
                         "population metrics", seq, len(real_cases), len(gen_items), n)
            for i in range(n):
                all_features.append(features_for(
                    f"{seq}_{i}", seq,
                    cohort.load_volume(real_cases[i].case_id),
                    predictions.load_volume(gen_items[i].prediction_id),
                ))

    if not all_features:
        log.warning("no feature pairs available -- skipping distribution metrics")
        return None
    return DM.compute_distribution_metrics(
        all_features, cohort.sequences, min_subgroup_n=inputs.min_subgroup_n,
        n_bootstrap=inputs.fid_bootstrap, seed=inputs.seed, k_diversity=inputs.diversity_k,
    )


def _write_csv(path: Path, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        fieldnames = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
