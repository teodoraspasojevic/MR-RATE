"""The clean metrics summary, as CSV.

`summary.json` is the machine-readable record; these two files are what you actually read and
paste into a paper.

    metrics_per_bucket.csv   one row per (modality, plane) with the geometry it was scored at,
                             the sample counts, and every metric
    metrics_summary.csv      the aggregate rows: per modality, then two overall rows

Two overall rows, deliberately, because they answer different questions and disagree here:

    overall_macro       unweighted mean across buckets -- every anatomy counts equally
    overall_weighted    weighted by the ELIGIBLE POPULATION counts recorded in cohort.json

The cohort is sampled to equal size per bucket (so per-bucket FID is stable), which means cohort
counts are a sampling artefact and must not be used as weights. `population_bucket_counts` holds
the real frequencies -- e.g. T1w AXIAL is ~4000 eligible cases against T2w SAGITTAL's ~1300 -- so
`overall_weighted` is what the test split would actually look like.

Every row also carries `nvidia_train_n`, NVIDIA's own published count of training images for that
bucket. Three T2w buckets were trained on 195/125/551 images and NVIDIA explicitly says output
quality is not guaranteed there; they are kept in the aggregates (nothing is silently dropped) but
the column is there so a weak number can be read in context.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..volumes import split_bucket

# NVIDIA's published per-bucket training-image counts, from NV-Generate-CTMR/docs/inference.md.
# Present so a reader can tell a model failure from a coverage gap.
NVIDIA_TRAIN_N = {
    "T1w__AXIAL": 47810, "T1w__SAGITTAL": 69268, "T1w__CORONAL": 38756,
    "T2w__AXIAL": 195, "T2w__SAGITTAL": 551, "T2w__CORONAL": 125,
    "FLAIR__AXIAL": 27990, "FLAIR__SAGITTAL": 58421, "FLAIR__CORONAL": 27698,
    "SWI__AXIAL": 47859, "SWI__SAGITTAL": 2, "SWI__CORONAL": 4,
    "MRA__AXIAL": 37, "MRA__SAGITTAL": 98, "MRA__CORONAL": 11,
}
LOW_TRAIN_N = 1000       # NVIDIA's own "quality not guaranteed" threshold, rounded


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _mean(rows, key):
    vals = [r[key] for r in rows
            if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else None


def bucket_rows(cohort, metric_rows, metric_names, distribution, anatomy) -> list:
    """One dict per bucket: geometry, counts, paired-metric means, and the population metrics."""
    by_bucket: dict = {}
    for r in metric_rows:
        by_bucket.setdefault(r.get("bucket", ""), []).append(r)

    out = []
    for bucket in cohort.buckets:
        geom = cohort.bucket_geometry(bucket)
        rows = by_bucket.get(bucket, [])
        modality, plane = split_bucket(bucket)
        row = {
            "bucket": bucket,
            "modality": modality,
            "plane": plane,
            "shape_xyz": "x".join(str(v) for v in geom.get("shape_xyz", [])),
            "spacing_mm_xyz": "x".join(f"{v:.4f}" for v in geom.get("spacing_mm_xyz", [])),
            "fov_mm_xyz": "x".join(f"{v:g}" for v in geom.get("fov_mm_xyz", [])),
            "n_cohort": geom.get("n", 0),
            "n_scored": len(rows),
            "n_population": cohort.population_bucket_counts.get(bucket, 0),
            "nvidia_train_n": NVIDIA_TRAIN_N.get(bucket, ""),
            "nvidia_low_train_n": NVIDIA_TRAIN_N.get(bucket, 0) < LOW_TRAIN_N,
        }
        for name in metric_names:
            row[name] = _mean(rows, name)

        d = (distribution or {}).get(bucket, {})
        row["medicalnet_fid"] = (d.get("medicalnet_fid_3d") or {}).get("fid")
        row["inception_2p5d_fid"] = (d.get("inception_2p5d_fid") or {}).get("combined_unweighted_mean")
        row["intra_set_ssim_real"] = (d.get("intra_set_ms_ssim_real") or {}).get("mean")
        row["intra_set_ssim_produced"] = (d.get("intra_set_ms_ssim_produced") or {}).get("mean")

        a = (anatomy or {}).get(bucket, {})
        for name in ("lr_symmetry_ncc", "intracranial_fraction", "tissue_contrast_separation",
                     "background_purity"):
            row[f"anat_{name}_real"] = (a.get(name) or {}).get("real_mean")
            row[f"anat_{name}_produced"] = (a.get(name) or {}).get("produced_mean")
        out.append(row)
    return out


def aggregate_rows(cohort, rows, metric_columns) -> list:
    """Per-modality means, then `overall_macro` and `overall_weighted`.

    `overall_weighted` uses the eligible-population counts, not the cohort counts -- see the module
    docstring. If a bucket has no population count recorded it falls back to its cohort count so a
    weight is never silently zero.
    """
    pop = cohort.population_bucket_counts
    out = []

    for modality in sorted({r["modality"] for r in rows if r["modality"]}):
        subset = [r for r in rows if r["modality"] == modality]
        entry = {"scope": f"modality:{modality}", "n_buckets": len(subset),
                 "n_scored": sum(r["n_scored"] for r in subset),
                 "n_population": sum(pop.get(r["bucket"], r["n_cohort"]) for r in subset)}
        for c in metric_columns:
            entry[c] = _mean(subset, c)
        out.append(entry)

    macro = {"scope": "overall_macro", "n_buckets": len(rows),
             "n_scored": sum(r["n_scored"] for r in rows),
             "n_population": sum(pop.get(r["bucket"], r["n_cohort"]) for r in rows)}
    for c in metric_columns:
        macro[c] = _mean(rows, c)
    out.append(macro)

    weights = {r["bucket"]: float(pop.get(r["bucket"], r["n_cohort"])) for r in rows}
    weighted = {"scope": "overall_weighted", "n_buckets": len(rows),
                "n_scored": macro["n_scored"], "n_population": macro["n_population"]}
    for c in metric_columns:
        num = den = 0.0
        for r in rows:
            v = r.get(c)
            if isinstance(v, (int, float)) and np.isfinite(v):
                w = weights[r["bucket"]]
                num += w * v
                den += w
        weighted[c] = num / den if den else None
    out.append(weighted)
    return out


def write_summary_csv(out_dir, cohort, metric_rows, metric_names, distribution, anatomy) -> list:
    """Write both CSVs. Returns the filenames written."""
    out_dir = Path(out_dir)
    rows = bucket_rows(cohort, metric_rows, metric_names, distribution, anatomy)
    if not rows:
        return []

    per_bucket = out_dir / "metrics_per_bucket.csv"
    fields = list(rows[0])
    with open(per_bucket, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})

    metric_columns = [c for c in fields if c not in (
        "bucket", "modality", "plane", "shape_xyz", "spacing_mm_xyz", "fov_mm_xyz",
        "n_cohort", "n_scored", "n_population", "nvidia_train_n", "nvidia_low_train_n")]
    aggs = aggregate_rows(cohort, rows, metric_columns)
    summary = out_dir / "metrics_summary.csv"
    agg_fields = ["scope", "n_buckets", "n_scored", "n_population"] + metric_columns
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for r in aggs:
            w.writerow({k: _fmt(r.get(k)) for k in agg_fields})

    return [per_bucket.name, summary.name]
