"""The clean metrics summary, as CSV.

`summary.json` is the machine-readable record; these two files are what you actually read and
paste into a paper.

    metrics_per_bucket.csv   one row per (modality, plane) with the geometry it was scored at,
                             the sample counts, and every metric
    metrics_summary.csv      the aggregate rows: per modality, then two overall rows

Three overall rows, deliberately, because they answer different questions and disagree here:

    overall_macro       unweighted mean across buckets -- every anatomy counts equally
    overall_weighted    weighted by the ELIGIBLE POPULATION counts
    overall_pooled      recomputed once over ALL cases at once, not averaged from the buckets.
                        **For FID and FVD this is the row to quote** -- see `aggregate_rows`.

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
        row["fvd"] = (d.get("fvd") or {}).get("combined_unweighted_mean")
        row["intra_set_ssim_real"] = (d.get("intra_set_ms_ssim_real") or {}).get("mean")
        row["intra_set_ssim_produced"] = (d.get("intra_set_ms_ssim_produced") or {}).get("mean")

        a = (anatomy or {}).get(bucket, {})
        for name in ("lr_symmetry_ncc", "intracranial_fraction", "tissue_contrast_separation",
                     "background_purity"):
            row[f"anat_{name}_real"] = (a.get(name) or {}).get("real_mean")
            row[f"anat_{name}_produced"] = (a.get(name) or {}).get("produced_mean")
        out.append(row)
    return out


def aggregate_rows(cohort, rows, metric_columns, distribution=None) -> list:
    """Per-modality means, then `overall_macro`, `overall_weighted` and `overall_pooled`.

    **Three overall rows, and for a Frechet distance only one of them is the right one.**

    - `overall_macro` / `overall_weighted` average the per-bucket values. That is correct for a
      per-case metric (PSNR, SSIM, an anatomy measure): each bucket's value is an unbiased mean and
      averaging them is just a re-weighting.
    - `overall_pooled` is the metric recomputed **once over every case at once**, taken from
      `distribution["overall"]` rather than derived from the bucket rows.

    For FID and FVD, `overall_pooled` is the number to quote and the averaged rows are diagnostics.
    A Frechet distance carries a sample-size **bias** (roughly proportional to feature_dim / N), not
    just noise, and averaging biased estimates does not remove the bias -- every bucket's estimate is
    inflated in the same direction, so their mean is inflated by the same amount. Pooling instead
    multiplies N by the bucket count and cuts the bias roughly in proportion. Measured on 512-d
    features with a known-zero ground truth: N=200 sits around 2,400 while N=2,000 sits around 240.

    It is also what the field does. The VLM3D challenge's own `ranking_config` exposes exactly
    `FID_2p5D_{XY,XZ,YZ}` and `FID_2p5D_Avg` -- per *plane*, averaged -- with **no** per-anatomy or
    per-sequence stratification, i.e. one pooled number per plane over the whole test set. CTFlow
    (ICCV 2025 VLM3D) likewise reports a single FID/FVD over all 3,000 CT-RATE validation volumes.
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

    # Pooled: every case in one population. Only the population-level metrics have one -- a paired
    # metric's pooled value IS its weighted mean, so those cells stay blank rather than repeating.
    overall = (distribution or {}).get("overall") or {}
    if overall:
        pooled = {"scope": "overall_pooled", "n_buckets": len(rows),
                  "n_scored": overall.get("n", macro["n_scored"]),
                  "n_population": macro["n_population"]}
        for c in metric_columns:
            pooled[c] = None
        pooled["medicalnet_fid"] = (overall.get("medicalnet_fid_3d") or {}).get("fid")
        pooled["inception_2p5d_fid"] = (overall.get("inception_2p5d_fid") or {}).get(
            "combined_unweighted_mean")
        pooled["fvd"] = (overall.get("fvd") or {}).get("combined_unweighted_mean")
        out.append(pooled)

        # Fixed-N, buckets ignored. Comparable across runs of different scale; see
        # `distribution.compute_batched_frechet`.
        batched_fid = overall.get("inception_2p5d_fid_batched") or {}
        batched_fvd = overall.get("fvd_batched") or {}
        if batched_fid.get("available") or batched_fvd.get("available"):
            batched = {"scope": f"overall_batched_n{batched_fid.get('batch_size') or batched_fvd.get('batch_size')}",
                       "n_buckets": len(rows),
                       "n_scored": (batched_fid.get("n_batches") or batched_fvd.get("n_batches", 0))
                       * (batched_fid.get("batch_size") or batched_fvd.get("batch_size") or 0),
                       "n_population": macro["n_population"]}
            for c in metric_columns:
                batched[c] = None
            batched["inception_2p5d_fid"] = batched_fid.get("value")
            batched["fvd"] = batched_fvd.get("value")
            out.append(batched)
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
    aggs = aggregate_rows(cohort, rows, metric_columns, distribution)
    summary = out_dir / "metrics_summary.csv"
    agg_fields = ["scope", "n_buckets", "n_scored", "n_population"] + metric_columns
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for r in aggs:
            w.writerow({k: _fmt(r.get(k)) for k in agg_fields})

    return [per_bucket.name, summary.name]
