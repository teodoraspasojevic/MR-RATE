"""One evaluation path for the whole benchmark, the way `eval/runner.py` is the one path for
volume evaluation. The CLI and the tests both call `run_benchmark`; there is no second scorer.

Input is a directory of embedding caches (`embed.py`), never a model -- so scoring cannot
accidentally encode differently than the cached run did, and re-scoring is a CPU job.

Output is a directory:

    metrics_matrix.csv    one row per (encoder, report_format), one column per metric
    per_label_auroc.csv   the pathology probe broken out by label
    summary.json          the same numbers, machine-readable, plus every run's provenance
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

from . import tasks
from .embed import read_cache

#: Column order in metrics_matrix.csv. Cost columns come last but are not an afterthought:
#: the selection rule weighs them explicitly (see docs/TEXT_ENCODERS.md).
METRIC_COLUMNS = (
    "pathology_probe_auroc",
    "bucket_probe_auroc",
    "negation_delta",
    "negation_delta_median",
    "negation_dominance",
    "negation_auroc",
    "nn_jaccard_delta",
    "sim_spearman",
    "embed_dim",
    "truncated_pct",
    "reports_per_second",
    "n_pathology_labels",
)


def _label_matrix(records, names):
    return np.array([[bool(r.labels.get(n, False)) for n in names] for r in records], dtype=bool)


def _bucket_matrix(records, names):
    return np.array([[n in r.buckets for n in names] for r in records], dtype=bool)


def _align(records, study_uids):
    """Reorder `records` to the cache's own study order. The cache stores its uids precisely so
    this never has to be assumed."""
    index = {r.study_uid: r for r in records}
    missing = [u for u in study_uids if u not in index]
    if missing:
        raise KeyError(f"{len(missing)} cached study ids are absent from the corpus "
                       f"(first: {missing[0][:4]}...). The cache and the corpus disagree.")
    return [index[u] for u in study_uids]


def _features(mean, maximum, use_max=True):
    """concat(mean, max) by default -- see `embed.py` for why a mean-only probe flatters
    mean-pooling encoders."""
    return np.concatenate([mean, maximum], axis=1) if use_max else mean


def run_benchmark(cache_dir, train_records, test_records, encoders, report_formats,
                  out_dir, label_names=None, bucket_names=None, negation_cache=None,
                  use_max_pooling=True, seed=0, min_prevalence=tasks.MIN_PREVALENCE,
                  train_split="train", test_split="test"):
    """Score every (encoder, report_format) whose cache exists. Missing caches are reported and
    skipped, never silently dropped from the averages."""
    os.makedirs(out_dir, exist_ok=True)
    label_names = label_names or sorted({k for r in train_records for k in r.labels})
    bucket_names = bucket_names or sorted({b for r in train_records for b in r.buckets})

    rows, per_label_rows, provenance, skipped = [], [], {}, []
    for encoder in encoders:
        for report_format in report_formats:
            key = f"{encoder}__{report_format}"
            try:
                mean_tr, max_tr, uid_tr, stats_tr = read_cache(cache_dir, encoder, report_format, train_split)
                mean_te, max_te, uid_te, stats_te = read_cache(cache_dir, encoder, report_format, test_split)
            except FileNotFoundError:
                skipped.append(key)
                continue
            started = time.time()

            train_aligned = _align(train_records, uid_tr)
            test_aligned = _align(test_records, uid_te)
            y_tr_lab = _label_matrix(train_aligned, label_names)
            y_te_lab = _label_matrix(test_aligned, label_names)
            y_tr_buc = _bucket_matrix(train_aligned, bucket_names)
            y_te_buc = _bucket_matrix(test_aligned, bucket_names)

            x_train = _features(mean_tr, max_tr, use_max_pooling)
            x_test = _features(mean_te, max_te, use_max_pooling)

            path_macro, path_per_label, path_skipped = tasks.multilabel_probe(
                x_train, y_tr_lab, x_test, y_te_lab, label_names, seed=seed,
                min_prevalence=min_prevalence)
            buc_macro, buc_per_label, _ = tasks.multilabel_probe(
                x_train, y_tr_buc, x_test, y_te_buc, bucket_names, seed=seed,
                min_prevalence=min_prevalence)
            nn_score, nn_random, nn_delta = tasks.nearest_neighbour_label_agreement(
                mean_te, y_te_lab, seed=seed)
            spearman = tasks.similarity_label_spearman(mean_te, y_te_lab, seed=seed)

            negation = {}
            if negation_cache is not None:
                negation = negation_cache.get(key, {})

            truncation = (stats_te.get("truncation") or {})
            row = {
                "encoder": encoder, "report_format": report_format,
                "pathology_probe_auroc": path_macro,
                "bucket_probe_auroc": buc_macro,
                "negation_delta": negation.get("negation_delta", float("nan")),
                "negation_delta_median": negation.get("negation_delta_median", float("nan")),
                "negation_dominance": negation.get("negation_dominance", float("nan")),
                "negation_auroc": negation.get("negation_auroc", float("nan")),
                "nn_jaccard_delta": nn_delta,
                "sim_spearman": spearman,
                "embed_dim": int(mean_te.shape[1]),
                "truncated_pct": 100.0 * float(truncation.get("fraction_truncated", 0.0)),
                "reports_per_second": float(stats_te.get("reports_per_second", float("nan"))),
                "n_pathology_labels": len(path_per_label),
            }
            rows.append(row)
            for name, value in sorted(path_per_label.items()):
                per_label_rows.append({"encoder": encoder, "report_format": report_format,
                                       "label": name, "auroc": value})
            provenance[key] = {
                "encoder_identity": stats_te.get("encoder", {}),
                "truncation_test": truncation,
                "truncation_train": (stats_tr.get("truncation") or {}),
                "n_train": int(mean_tr.shape[0]), "n_test": int(mean_te.shape[0]),
                "labels_skipped_low_prevalence": path_skipped,
                "nn_jaccard_raw": nn_score, "nn_jaccard_random": nn_random,
                "negation": negation,
                "scoring_seconds": time.time() - started,
            }
            print(f"  [{key}] pathology={path_macro:.4f} bucket={buc_macro:.4f} "
                  f"negΔ={row['negation_delta']:.4f} nnΔ={nn_delta:+.4f} rho={spearman:.4f} "
                  f"trunc={row['truncated_pct']:.2f}%", flush=True)

    _write_csv(os.path.join(out_dir, "metrics_matrix.csv"),
               rows, ["encoder", "report_format", *METRIC_COLUMNS])
    _write_csv(os.path.join(out_dir, "per_label_auroc.csv"),
               per_label_rows, ["encoder", "report_format", "label", "auroc"])
    summary = {
        "rows": rows, "provenance": provenance, "skipped_missing_cache": skipped,
        "label_names": label_names, "bucket_names": bucket_names,
        "n_train_records": len(train_records), "n_test_records": len(test_records),
        "train_split": train_split, "test_split": test_split,
        "use_max_pooling": use_max_pooling, "seed": seed,
        "weak_label_warning": (
            "pathology_probe_auroc, nn_jaccard_delta and sim_spearman use labels.json, which was "
            "LLM-derived from the same report text. Valid for ranking encoders against each other; "
            "not an absolute measure of clinical accuracy. bucket_probe_auroc (DICOM-derived) and "
            "negation_auroc (rule-constructed) do not share this caveat."
        ),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1, default=float)
    if skipped:
        print(f"[bench] {len(skipped)} (encoder, format) pairs skipped: no cache "
              f"(first: {skipped[0]})")
    return summary


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = ["METRIC_COLUMNS", "run_benchmark"]
