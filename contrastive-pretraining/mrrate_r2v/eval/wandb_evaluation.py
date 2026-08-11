"""What an evaluation sends to Weights & Biases: one metrics table, and a few example panels.

`wandb_logging.WandbRun` is deliberately dataset-agnostic (it was ported unchanged from the older
implementation). This module is the R2V-specific assembly on top of it, so that separation survives.

Two deliverables, matching what a reader actually wants from a finished evaluation:

1. **`metrics/all` -- one table with every metric that ran**, per bucket and then the aggregates,
   ending in provenance rows including **how many samples the model was trained on**. A W&B run
   whose only artefact is a scatter of scalars cannot be read next to another run; a single table
   can be sorted, filtered and diffed in the UI.

2. **A handful of ground-truth vs generated panels**, rendered by the *same*
   `figures.validation_panel_html` the training loop uses -- same matched slice indices, same
   ground-truth-derived intensity window, same physical aspect ratio, same report text alongside.
   Using a second renderer here would mean a panel from an evaluation and a panel from training
   could not be compared, which is the one thing panels are for.

**Panels are a sample, never the whole cohort.** 2,000 interactive panels is ~1 GB of base64 and
an unusable workspace. `select_panel_cases` takes the worst, the median and the best case by a
paired metric, spread across buckets -- the same rank-spread rationale `_save_examples` uses for the
on-disk figures, because the worst case is where a failure mode is visible and the best case is
where you check it is not a fluke.

**Report text is patient text.** Panels embed it, so they are logged only under
`--wandb-log-reports`, mirroring `cli.train_r2v`'s gate. Without that flag the metrics table still
goes up; only the panels are withheld.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("mrrate_r2v.eval.wandb_evaluation")

#: Columns of `metrics/all`. Fixed and shared by every row kind, so bucket rows, aggregate rows and
#: provenance rows land in one sortable table instead of three.
TABLE_COLUMNS = ("scope", "kind", "metric", "value", "std", "n", "note")

#: Default number of example panels. Small on purpose -- see the module docstring.
DEFAULT_N_PANELS = 6


def _finite(value):
    """A JSON/W&B-safe float, or None. numpy scalars and NaN both break a `wandb.Table` cell."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if np.isfinite(as_float) else None


def _rows_from_aggregate(scope: str, kind: str, metrics: dict):
    """`{metric: {mean, std, n}}` -> table rows. `aggregate.py`'s shape, used for paired metrics."""
    rows = []
    for name, stats in sorted((metrics or {}).items()):
        if not isinstance(stats, dict):
            rows.append((scope, kind, name, _finite(stats), None, None, ""))
            continue
        rows.append((scope, kind, name, _finite(stats.get("mean")), _finite(stats.get("std")),
                     stats.get("n"), ""))
    return rows


def _rows_from_distribution(distribution: dict):
    """Flatten `distribution_metrics.json` into table rows.

    Only the numbers a reader quotes: the two Frechet distances, the diversity quartet, the
    intra-set mode-collapse pair. Bootstrap CIs and per-plane breakdowns stay in the JSON -- a
    table that carries everything is as unreadable as one that carries nothing.
    """
    rows = []
    for scope, entry in sorted((distribution or {}).items()):
        if not isinstance(entry, dict):
            continue
        for key, label in (("medicalnet_fid_3d", "medicalnet_fid_3d"),
                           ("inception_2p5d_fid", "inception_2p5d_fid")):
            block = entry.get(key)
            if isinstance(block, dict):
                rows.append((scope, "distribution", label, _finite(block.get("fid")), None,
                             block.get("n_gen"), block.get("skipped") or ""))
        diversity = entry.get("diversity_precision_recall_density_coverage")
        if isinstance(diversity, dict):
            for name in ("precision", "recall", "density", "coverage"):
                rows.append((scope, "diversity", name, _finite(diversity.get(name)), None, None, ""))
        for name, label in (("intra_set_ms_ssim_real", "intra_set_ssim_real"),
                            ("intra_set_ms_ssim_produced", "intra_set_ssim_produced")):
            block = entry.get(name)
            value = block.get("mean") if isinstance(block, dict) else block
            if value is not None:
                rows.append((scope, "mode_collapse", label, _finite(value), None, None,
                             "produced above real = less variety than the data"))
    return rows


def _rows_from_anatomy(anatomy: dict):
    """Anatomical plausibility: the KS statistic per measure, which is the comparison that matters
    (a mean on its own says nothing about whether two populations differ)."""
    rows = []
    for scope, entry in sorted((anatomy or {}).items()):
        if not isinstance(entry, dict):
            continue
        for measure, block in sorted(entry.items()):
            if not isinstance(block, dict):
                continue
            ks = block.get("ks_statistic")
            if ks is None:
                continue
            rows.append((scope, "anatomy", measure, _finite(ks), None, block.get("n"),
                         "KS statistic vs the real population; larger = more different"))
    return rows


def _rows_from_report_consistency(consistency: dict):
    """The blinded-classifier table: per label, then the two aggregates and their real-volume
    ceiling. Always emits at least one row -- an unavailable group must be visible in the table as
    unavailable, not absent from it."""
    if not consistency:
        return [("overall", "report_consistency", "available", None, None, None, "not computed")]
    if not consistency.get("available"):
        return [("overall", "report_consistency", "available", 0.0, None, None,
                 str(consistency.get("reason", ""))[:180])]

    rows = []
    for label, entry in sorted((consistency.get("per_label") or {}).items()):
        note = []
        if not entry.get("usable"):
            note.append("classifier cannot read this label on REAL volumes")
        if entry.get("low_support"):
            note.append(f"only {entry.get('n_positive')} positives")
        rows.append((label, "report_consistency", "auroc", _finite(entry.get("auroc")), None,
                     entry.get("n"), "; ".join(note)))
        rows.append((label, "report_consistency", "real_reference_auroc",
                     _finite(entry.get("real_reference_auroc")), None, entry.get("n"),
                     "the ceiling: same classifier on the real volumes"))
    reference = consistency.get("real_reference") or {}
    for metric, value, note in (
        ("macro_auroc_usable_labels", consistency.get("macro_auroc_usable_labels"),
         "HEADLINE. floor 0.5 (image-blind); ceiling below"),
        ("macro_auroc_usable_labels_REAL", reference.get("macro_auroc_usable_labels"),
         "the ceiling this is read against"),
        ("macro_retention_usable_labels", consistency.get("macro_retention_usable_labels"),
         "fraction of the classifier's real-data margin that survives"),
        ("mean_per_case_consistency", consistency.get("mean_per_case_consistency"),
         "per-case score; the input to a case-level permutation test"),
        ("n_labels_usable", consistency.get("n_labels_usable"), ""),
    ):
        rows.append(("overall", "report_consistency", metric, _finite(value), None,
                     consistency.get("n_scored"), note))
    return rows


def _rows_from_provenance(summary: dict, cohort, predictions) -> list:
    """The rows that make the table self-describing, including the training-sample count.

    This is why `cli.predict_r2v` records `model.training`: the evaluator never opens a checkpoint,
    so without that recording "how much was this trained" would be unanswerable from the results
    directory alone.
    """
    model = getattr(predictions, "model", None) or {}
    training = model.get("training") or {}
    rows = [
        ("provenance", "run", "cohort_id", None, None, None, str(cohort.cohort_id)),
        ("provenance", "run", "task", None, None, summary.get("n_scored"), str(summary.get("task"))),
        ("provenance", "run", "n_cases_scored", summary.get("n_scored"), None, None,
         f"{summary.get('n_excluded')} excluded"),
        ("provenance", "run", "model", None, None, None, str(model.get("name", "?"))),
    ]
    samples = training.get("samples_seen")
    per_epoch = training.get("samples_per_epoch")
    if samples is not None:
        rows.append((
            "provenance", "training", "train_samples_seen", float(samples), None, None,
            f"volumes the optimizer consumed = {training.get('optimizer_step')} optimizer steps x "
            f"effective batch {training.get('effective_batch_size')}"
            + (f"; ~{per_epoch:,} per epoch over {training.get('epochs_completed')} epochs"
               if per_epoch else "")
            + (f" (world_size from {training.get('world_size_source')})"
               if training.get("world_size_source") else ""),
        ))
    else:
        rows.append((
            "provenance", "training", "train_samples_seen", None, None, None,
            "unavailable: the prediction set records no effective batch size (the training run "
            "wrote no train_summary.json). Pass --train-world-size to cli.predict_r2v.",
        ))
    for key, note in (("optimizer_step", ""), ("epochs_completed", ""),
                      ("effective_batch_size", ""), ("learning_rate", ""),
                      ("trainable_parameters", ""), ("skipped_steps", "non-finite-gradient steps discarded")):
        if training.get(key) is not None:
            rows.append(("provenance", "training", key, _finite(training[key]), None, None, note))
    for key in ("conditioning", "report_format"):
        if training.get(key) is not None:
            rows.append(("provenance", "training", key, None, None, None, str(training[key])))
    return rows


#: The main table: one row per headline metric, one column per scope. Logged as `metrics/headline`.
HEADLINE_TABLE_COLUMNS = ("metric", "direction", "overall_macro", "overall_weighted",
                          "overall_pooled", "batched_fixed_n", "batched_std", "batch_size",
                          "n_batches", "n_scored", "note")


def _bucket_mean(distribution, buckets, getter, weights=None):
    """Mean of a per-bucket value, optionally population-weighted. None when nothing is available."""
    import numpy as _np

    pairs = []
    for b in buckets:
        v = getter((distribution or {}).get(b) or {})
        if isinstance(v, (int, float)) and _np.isfinite(v):
            pairs.append((b, float(v)))
    if not pairs:
        return None
    if weights is None:
        return float(_np.mean([v for _b, v in pairs]))
    num = sum(float(weights.get(b, 0.0)) * v for b, v in pairs)
    den = sum(float(weights.get(b, 0.0)) for b, _v in pairs)
    return num / den if den else None


def headline_table(summary: dict, cohort) -> tuple:
    """`(columns, rows)` for `metrics/headline` -- THE table to read.

    One row per headline metric, one column per way of aggregating it:

        overall_macro     unweighted mean of the per-bucket values
        overall_weighted  population-weighted mean of the per-bucket values
        overall_pooled    recomputed once over every case (buckets mixed)
        batched_fixed_n   recomputed on consecutive batches of a FIXED size, buckets ignored;
                          mean across batches, with the batch-to-batch std beside it

    **For FID and FVD, read `overall_pooled` and `batched_fixed_n`; the two averaged columns are
    diagnostics.** A Frechet distance carries a sample-size bias, not just noise, so averaging
    per-bucket values leaves the bias untouched -- every bucket is inflated in the same direction.
    Pooling cuts it by raising N; the fixed-N column keeps the bias *constant* so two runs of
    different scale stay comparable and the std gives an error bar for ranking.

    The blinded-classifier consistency is a single AUROC over all scored cases, so only the pooled
    column is defined for it -- a per-bucket AUROC is undefined wherever a label has one class in
    that bucket, and averaging over the buckets where it happens to exist would silently change the
    population. That is stated in the row's note rather than filled with a number.
    """
    buckets = list(getattr(cohort, "buckets", []) or [])
    weights = dict(getattr(cohort, "population_bucket_counts", {}) or {})
    distribution = summary.get("distribution_metrics") or {}
    pooled = distribution.get("overall") or {}
    n_scored = summary.get("n_scored") or summary.get("n_cohort_cases")

    def frechet_row(label, key, batched_key, note):
        getter = (lambda e: (e.get(key) or {}).get("combined_unweighted_mean")) if key != "medicalnet_fid_3d" \
            else (lambda e: (e.get(key) or {}).get("fid"))
        batched = pooled.get(batched_key) or {}
        return (
            label, "lower is better",
            _finite(_bucket_mean(distribution, buckets, getter)),
            _finite(_bucket_mean(distribution, buckets, getter, weights)),
            _finite(getter(pooled)),
            _finite(batched.get("value")),
            _finite(batched.get("std")),
            batched.get("batch_size"),
            batched.get("n_batches"),
            pooled.get("n") or n_scored,
            note if batched.get("available", True) else f"{note}. batched: {batched.get('reason')}",
        )

    rows = [
        frechet_row("FVD (r3d18 / medicalnet)", "fvd", "fvd_batched",
                    "MRI-volume adaptation of FVD -- Kinetics-400 r3d_18, not I3D. Per-plane, then "
                    "the unweighted mean, matching the challenge's FID_2p5D_Avg shape"),
        frechet_row("FID 2.5D (Inception)", "inception_2p5d_fid", "inception_2p5d_fid_batched",
                    "the challenge's FID_2p5D_Avg shape exactly: per-plane Frechet, unweighted mean"),
    ]

    medicalnet_getter = lambda e: (e.get("medicalnet_fid_3d") or {}).get("fid")  # noqa: E731
    rows.append((
        "FID 3D (MedicalNet)", "lower is better",
        _finite(_bucket_mean(distribution, buckets, medicalnet_getter)),
        _finite(_bucket_mean(distribution, buckets, medicalnet_getter, weights)),
        _finite(medicalnet_getter(pooled)), None, None, None, None,
        pooled.get("n") or n_scored,
        "UNVALIDATED at scale: the preprocessing fix (z-score + foreground crop) landed after the "
        "last full run. Check FID(real T1w vs real T2w) > FID(real vs its own reconstruction) "
        "before trusting it",
    ))

    consistency = summary.get("report_consistency") or {}
    if consistency.get("available"):
        real_ref = (consistency.get("real_reference") or {}).get("macro_auroc_usable_labels")
        rows.append((
            "Blinded classifier consistency (macro AUROC)", "higher is better",
            None, None, _finite(consistency.get("macro_auroc_usable_labels")), None, None, None,
            None, consistency.get("n_scored"),
            f"one AUROC over all scored cases, so only the pooled column is defined. Read against "
            f"the same classifier's REAL-volume ceiling = {_finite(real_ref)}; the image-blind "
            f"floor is 0.5. {consistency.get('n_labels_usable')} usable labels",
        ))
    else:
        rows.append((
            "Blinded classifier consistency (macro AUROC)", "higher is better",
            None, None, None, None, None, None, None, None,
            f"UNAVAILABLE: {consistency.get('reason', 'not computed')}",
        ))
    return list(HEADLINE_TABLE_COLUMNS), rows


def metrics_table(summary: dict, cohort, predictions) -> tuple:
    """`(columns, rows)` for `metrics/all` -- every metric family that ran, in one table.

    Order is deliberate: per-bucket rows first (the primary unit of this evaluation), then the
    per-modality and overall aggregates, then the population-level groups, then provenance. So the
    table reads top-to-bottom from most specific to most general.
    """
    rows = []
    for bucket, metrics in sorted((summary.get("paired_metrics_per_bucket") or {}).items()):
        if bucket == "overall":
            continue
        rows.extend(_rows_from_aggregate(bucket, "paired", metrics))
    for scope, metrics in sorted((summary.get("paired_metrics") or {}).items()):
        rows.extend(_rows_from_aggregate(scope, "paired", metrics))
    rows.extend(_rows_from_distribution(summary.get("distribution_metrics")))
    rows.extend(_rows_from_anatomy(summary.get("anatomy")))
    rows.extend(_rows_from_report_consistency(summary.get("report_consistency")))
    rows.extend(_rows_from_provenance(summary, cohort, predictions))
    return list(TABLE_COLUMNS), rows


def headline_scalars(summary: dict) -> dict:
    """The few numbers worth having as run-level values so two runs can be sorted by them in the
    W&B run table. Everything else lives in `metrics/all`."""
    overall = (summary.get("paired_metrics") or {}).get("overall") or {}
    out = {}
    for name in ("psnr_fg", "ssim3d_whole", "mae_fg", "ncc_fg"):
        value = _finite((overall.get(name) or {}).get("mean"))
        if value is not None:
            out[f"eval/{name}"] = value
    distribution = (summary.get("distribution_metrics") or {}).get("overall") or {}
    for key, label in (("medicalnet_fid_3d", "medicalnet_fid_3d"),
                       ("inception_2p5d_fid", "inception_2p5d_fid")):
        block = distribution.get(key)
        if isinstance(block, dict):
            value = _finite(block.get("fid"))
            if value is not None:
                out[f"eval/{label}"] = value
    consistency = summary.get("report_consistency") or {}
    if consistency.get("available"):
        for key in ("macro_auroc_usable_labels", "macro_retention_usable_labels",
                    "mean_per_case_consistency"):
            value = _finite(consistency.get(key))
            if value is not None:
                out[f"eval/report_consistency/{key}"] = value
        ceiling = _finite((consistency.get("real_reference") or {})
                          .get("macro_auroc_usable_labels"))
        if ceiling is not None:
            out["eval/report_consistency/real_reference_macro_auroc"] = ceiling
    return out


class _PanelCase:
    """The attribute surface `figures.validation_panel_html` expects, backed by a cohort case.

    A shim rather than a refactor of the panel renderer: the renderer is shared with training and
    its inputs are already right, so the cheap and non-breaking move is to present a cohort case in
    the shape it already accepts. `case_id` is a hash and `study_key`/`series_key` are not carried,
    so a panel still contains no identifier.
    """

    def __init__(self, case, target, report_text: str, report_sections):
        self.index = 0
        self.case_id = case.case_id
        self.report_text = report_text or ""
        self.report_sections = report_sections or {}
        self.modality = case.sequence
        self.plane = case.acquisition_plane
        self.shape_xyz = tuple(case.shape)
        self.spacing_xyz = tuple(case.spacing_mm)
        self.study_hash = ""
        self.target = target


def select_panel_cases(metric_rows, n_panels: int, rank_metric: str = "psnr_fg") -> list:
    """Which cases get a panel: worst, median and best by `rank_metric`, spread over buckets.

    Returns `[(case_id, label)]`, where the label says *why* that case was picked -- a panel with no
    provenance invites reading a cherry-picked best case as typical.

    Falls back to cohort order when the metric is absent (an unpaired task has no per-case metric to
    rank by), and never returns more than `n_panels`.
    """
    if n_panels <= 0:
        return []
    scored = [r for r in metric_rows if _finite(r.get(rank_metric)) is not None]
    if not scored:
        return [(r["case_id"], "arbitrary (no rankable metric)") for r in metric_rows[:n_panels]]

    by_bucket: dict = {}
    for row in scored:
        by_bucket.setdefault(row.get("bucket", "?"), []).append(row)
    for rows in by_bucket.values():
        rows.sort(key=lambda r: float(r[rank_metric]))

    # Round-robin over buckets taking worst, then best, then median, so a small n_panels still
    # spans several anatomies instead of three panels from one bucket.
    picks, seen = [], set()
    for position in ("worst", "best", "median"):
        for bucket in sorted(by_bucket):
            rows = by_bucket[bucket]
            if not rows:
                continue
            row = {"worst": rows[0], "best": rows[-1], "median": rows[len(rows) // 2]}[position]
            if row["case_id"] in seen:
                continue
            seen.add(row["case_id"])
            picks.append((row["case_id"],
                          f"{position} {rank_metric} in {bucket} ({float(row[rank_metric]):.3f})"))
            if len(picks) >= n_panels:
                return picks
    return picks


def log_evaluation(run, summary: dict, cohort, predictions, *, metric_rows=(),
                   n_panels: int = DEFAULT_N_PANELS, log_reports: bool = False,
                   rank_metric: str = "psnr_fg") -> dict:
    """Send the table, the headline scalars and a few panels. Returns what was logged, for the
    results directory.

    Never raises: `WandbRun` swallows its own failures, and a panel that cannot be rendered is
    logged as a warning. An evaluation that has already computed its metrics must not be lost to a
    logging problem.
    """
    # The main table first: three headline metric families x four aggregation scopes. `metrics/all`
    # stays as the exhaustive per-bucket view underneath it.
    head_columns, head_rows = headline_table(summary, cohort)
    run.log_table("metrics/headline", head_columns, head_rows)
    columns, rows = metrics_table(summary, cohort, predictions)
    run.log_table("metrics/all", columns, rows)
    scalars = headline_scalars(summary)
    # The training-sample count is a run-level property, so it belongs in the sortable run table
    # too, not only in the metrics table.
    training = ((getattr(predictions, "model", None) or {}).get("training") or {})
    if training.get("samples_seen"):
        scalars["eval/train_samples_seen"] = float(training["samples_seen"])
    run.set_summary(scalars)
    run.log(scalars)

    logged = {"table_rows": len(rows), "headline_rows": len(head_rows),
              "scalars": sorted(scalars), "panels": [],
              "panels_withheld_reason": None}

    if n_panels <= 0:
        logged["panels_withheld_reason"] = "n_panels=0"
        return logged
    if not log_reports:
        # Same gate as cli.train_r2v: the panel embeds report text.
        logged["panels_withheld_reason"] = (
            "--wandb-log-reports not set; panels embed patient report text")
        log.info("W&B panels withheld: %s", logged["panels_withheld_reason"])
        return logged

    from . import figures as F

    picks = select_panel_cases(list(metric_rows), n_panels, rank_metric)
    # case_id -> prediction_id. For a report2volume set the two are equal, but that is a property of
    # how `cli.predict_r2v` names things rather than a contract -- `cli.import_predictions` assigns
    # its own prediction ids -- so the mapping is looked up rather than assumed.
    prediction_for_case = {item.case_id: item.prediction_id
                           for item in getattr(predictions, "items", ()) or ()
                           if getattr(item, "case_id", None)}
    for case_id, why in picks:
        case = cohort.case_by_id(case_id)
        if case is None:
            continue
        try:
            panel = F.validation_panel_html(
                _PanelCase(case, cohort.load_volume(case_id), cohort.load_report(case_id),
                           cohort.load_report_sections(case_id)),
                generated=predictions.load_volume(prediction_for_case.get(case_id, case_id)),
                step=int(((predictions.model or {}).get("training") or {}).get("optimizer_step") or 0),
                epoch=int(((predictions.model or {}).get("training") or {}).get("epochs_completed") or 0),
                validation_index=0, full=True,
            )
        except Exception as e:  # noqa: BLE001 -- a panel is never worth an evaluation
            log.warning("could not render panel for %s (%s)", case_id, e)
            continue
        run.log_html(f"examples/{case.bucket}/{case_id}", panel)
        logged["panels"].append({"case_id": case_id, "bucket": case.bucket, "selected_as": why})
    log.info("W&B: %d table rows, %d scalars, %d panels", len(rows), len(scalars),
             len(logged["panels"]))
    return logged


__all__ = [
    "DEFAULT_N_PANELS",
    "HEADLINE_TABLE_COLUMNS",
    "TABLE_COLUMNS",
    "headline_table",
    "headline_scalars",
    "log_evaluation",
    "metrics_table",
    "select_panel_cases",
]
