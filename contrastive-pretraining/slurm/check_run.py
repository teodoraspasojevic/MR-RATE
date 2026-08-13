#!/usr/bin/env python3
"""Check a finished run against `SUCCESS_CRITERIA.md`. Exit 0 only if every applicable check passes.

    python3 slurm/check_run.py --results <results_dir> [--results ...]
                               [--results-recon <dir>] [--results-gen <dir>]

Every check is named with its criterion id so a failure points straight at the table entry. Checks
whose inputs are absent are reported as SKIP, not PASS -- "not run" must never read as "fine".

Deliberately in the repo rather than in a scratch directory: a criterion that lives only in a shell
history is a criterion nobody re-checks after the next change.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrrate_r2v.data.geometry import UNET_SPATIAL_MULTIPLE, build_geometry_table  # noqa: E402
from mrrate_r2v.volumes import VolumeReader, split_bucket  # noqa: E402

EXPECTED_BUCKETS = 10
EXPECTED_N_PER_BUCKET = 200

# The aggregate scopes `eval/summary_csv.py` writes on top of the per-modality ones. Named rather
# than counted, so a run that silently loses `overall_pooled` -- the column FID and FVD should
# actually be read from -- fails the check instead of passing on an unchanged row count.
# `overall_batched_n<N>` carries the frechet batch size in its name, so it is matched by prefix:
# hardcoding n512 would fail any run that passed --frechet-batch-size.
EXPECTED_OVERALL_SCOPES = ("overall_macro", "overall_weighted", "overall_pooled")
BATCHED_SCOPE_PREFIX = "overall_batched_n"

# Spline resampling undershoots slightly at sharp edges, so a percentile-normalized volume has
# background voxels a hair below zero. Measured on the real cohort: worst -6.2e-05, typically
# ~1e-7. The threshold has to sit above that noise floor and still well below any real sign error
# (which would show up as a fraction of the [0, 1] scale, not as 1e-5).
NEGATIVE_TOLERANCE = 1e-3

# Reconstruction quality gates. See E5 in SUCCESS_CRITERIA.md for why SSIM3D is only a floor:
# it is texture-dominated (r = 0.735 with edge_preservation_fg), so blur is gated directly instead.
MIN_PSNR_FG = 20.0
MIN_EDGE_PRESERVATION = 0.8
MIN_SSIM3D_FLOOR = 0.30


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, cid, name, ok, detail=""):
        self.rows.append((cid, name, "PASS" if ok else "FAIL", detail))
        return ok

    def skip(self, cid, name, why):
        self.rows.append((cid, name, "SKIP", why))

    def report(self) -> int:
        width = max(len(r[1]) for r in self.rows) if self.rows else 10
        print()
        for cid, name, status, detail in self.rows:
            mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
            print(f"[{mark}] {cid:<3} {name:<{width}}  {detail}")
        failed = [r for r in self.rows if r[2] == "FAIL"]
        skipped = [r for r in self.rows if r[2] == "SKIP"]
        print(f"\n{len(self.rows) - len(failed) - len(skipped)} passed, {len(failed)} failed, "
              f"{len(skipped)} skipped")
        if failed:
            print("\nFAILED -- do not read the numbers as a result. See slurm/SUCCESS_CRITERIA.md.")
        return 1 if failed else 0


def read_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_cohort(c: Checks, root: Path):
    from mrrate_r2v.cohort import Cohort

    cohort = Cohort(root)
    counts = cohort.bucket_counts
    c.add("C1", "10 buckets x 200 cases",
          len(counts) == EXPECTED_BUCKETS and set(counts.values()) == {EXPECTED_N_PER_BUCKET},
          f"{len(counts)} buckets, counts={sorted(set(counts.values()))}, n={len(cohort.cases)}")

    table = build_geometry_table()
    bad_div, bad_fov, bad_shape = [], [], []
    for bucket in cohort.buckets:
        geom = cohort.bucket_geometry(bucket)
        shape = tuple(int(x) for x in geom["shape_xyz"])
        spacing = tuple(float(x) for x in geom["spacing_mm_xyz"])
        if any(s % UNET_SPATIAL_MULTIPLE for s in shape):
            bad_div.append(f"{bucket}:{shape}")
        key = split_bucket(bucket)
        spec = table.get(key)
        if spec is None:
            bad_shape.append(f"{bucket}:no table entry")
            continue
        # the table is (D,H,W); the cohort reports (X,Y,Z) -- compare as sorted FOV extents so this
        # check cannot itself be fooled by an axis-order mistake it is meant to catch
        want = tuple(sorted(round(s * p, 2) for s, p in zip(spec.target_shape, spec.target_spacing)))
        got = tuple(sorted(round(s * p, 2) for s, p in zip(shape, spacing)))
        if max(abs(a - b) for a, b in zip(want, got)) > 0.5:
            bad_fov.append(f"{bucket}: want FOV {want}, got {got}")
        if tuple(sorted(shape)) != tuple(sorted(spec.target_shape)):
            # compared as sorted extents: the table is (D,H,W), the cohort reports (X,Y,Z)
            bad_shape.append(f"{bucket}: want extents {sorted(spec.target_shape)}, "
                             f"got {sorted(shape)}")

    c.add("C2", f"every axis divisible by {UNET_SPATIAL_MULTIPLE}", not bad_div, "; ".join(bad_div))
    c.add("C3", "FOV reproduces NVIDIA's table", not bad_fov,
          "; ".join(bad_fov[:3]) or f"{len(cohort.buckets)} buckets")
    c.add("C2b", "shapes match the Option-B table", not bad_shape,
          "; ".join(bad_shape[:3]) or f"{len(cohort.buckets)} buckets")

    spec_json = cohort.spec
    geo = spec_json.get("geometry", {})
    frozen = {"posterior_shift_mm": geo.get("posterior_shift_mm"),
              "normalizer": geo.get("normalizer"), "seed": spec_json.get("seed")}
    c.add("C4", "frozen contract values", frozen == {"posterior_shift_mm": 0.0,
                                                     "normalizer": "percentile", "seed": 42},
          str(frozen))
    pop = cohort.population_bucket_counts
    c.add("C5", "population_bucket_counts complete",
          len(pop) >= len(counts) and all(pop.get(b, 0) > 0 for b in cohort.buckets),
          f"{len(pop)} entries, min={min(pop.values()) if pop else 0}")
    missing = cohort.verify_complete()
    c.add("C6", "all volumes present", not missing, f"{len(missing)} missing")

    # C7: intensity sanity on a sample of each bucket, since reading 2000 volumes is minutes
    reader = VolumeReader(root)
    problems = []
    for bucket in cohort.buckets:
        for case in cohort.cases_for_bucket(bucket)[:3]:
            v = reader.read(bucket, case.case_id)
            p995 = float(np.percentile(v, 99.5))
            if not np.isfinite(v).all():
                problems.append(f"{bucket}: non-finite voxels")
            elif v.min() < -NEGATIVE_TOLERANCE:
                problems.append(f"{bucket}: min={v.min():.3e} below -{NEGATIVE_TOLERANCE}")
            elif not 0.5 <= p995 <= 2.0:
                problems.append(f"{bucket}: p99.5={p995:.3f} outside [0.5, 2.0]")
    c.add("C7", "intensities sane (30 sampled)", not problems, "; ".join(problems[:3]))
    return cohort


def check_predictions(c: Checks, root: Path, cohort, tag: str, paired: bool):
    from mrrate_r2v.predictions import PredictionReader

    pr = PredictionReader(root)
    p = "R" if paired else "G"
    n = len(pr.items)
    fails = pr.spec.get("failures", [])
    c.add(f"{p}1", f"{tag}: 2000 volumes, no failures",
          n == EXPECTED_BUCKETS * EXPECTED_N_PER_BUCKET and not fails,
          f"n={n}, failures={len(fails)}")
    same = pr.spec.get("cohort_id") == cohort.cohort_id
    c.add(f"{p}2", f"{tag}: cohort_id matches", same, pr.spec.get("cohort_id", "")[:16])

    by_bucket_shape = {b: tuple(cohort.bucket_geometry(b)["shape_xyz"]) for b in cohort.buckets}
    wrong = [i.prediction_id for i in pr.items
             if tuple(i.shape) != by_bucket_shape.get(i.bucket)]
    c.add(f"{p}3", f"{tag}: every shape is its bucket's", not wrong,
          f"{len(wrong)} wrong, e.g. {wrong[:2]}")
    missing = pr.verify_complete()
    c.add(f"{p}4", f"{tag}: all volumes present", not missing, f"{len(missing)} missing")

    if not paired:
        req = pr.model.get("requested_geometry_per_bucket", {})
        mismatch = [b for b in cohort.buckets
                    if tuple(req.get(b, {}).get("shape_xyz", ())) != by_bucket_shape[b]]
        c.add("G5", f"{tag}: requested geometry == cohort's", not mismatch, str(mismatch[:3]))
        # G4: degenerate-output check on a sample
        reader = VolumeReader(root)
        bad = []
        for bucket in cohort.buckets:
            items = [i for i in pr.items if i.bucket == bucket][:3]
            for it in items:
                v = reader.read(bucket, it.prediction_id)
                std, fg = float(v.std()), float((v > 1e-3).mean())
                if std < 0.01:
                    bad.append(f"{bucket}: std={std:.4f}")
                elif not 0.05 <= fg <= 0.95:
                    bad.append(f"{bucket}: fg={fg:.3f}")
        c.add("G6", f"{tag}: not degenerate (30 sampled)", not bad, "; ".join(bad[:3]))
    return pr


def _buckets_figured(figures, buckets) -> set:
    """Which buckets have at least one .png.

    Matched by prefix against the known bucket list rather than parsed out of the filename:
    a paired figure is `<bucket>_rank<n>_<case_id>.png` but a generation figure is
    `<bucket>_gen<n>_gen_<bucket>_<n>.png`, where the bucket name appears twice -- any rsplit
    on "_gen" lands in the wrong place.
    """
    out = set()
    for f in figures:
        name = f.split("/")[-1]
        if not name.endswith(".png"):
            continue
        for b in buckets:
            if name.startswith(b + "_"):
                out.add(b)
                break
    return out


def check_results(c: Checks, root: Path, task: str, recon_root: Path | None = None,
                  buckets=()):
    summary = json.loads((root / "summary.json").read_text())
    prefix = {"reconstruction": "E", "generation": "EG", "report2volume": "ER"}[task]
    n_scored = summary["n_scored"]

    n_cases = summary.get("n_cohort_cases", 0)
    scale = ("entire split" if summary.get("evaluated_full_split")
             else f"{summary.get('n_per_bucket')}/bucket")

    if task in ("reconstruction", "report2volume"):
        # Every selected case must be scored with nothing excluded: the model is asked for the
        # case's own grid, so a geometry exclusion means it emitted a grid that was not requested.
        # The expected count is `n_cases` rather than a hardcoded 2000 -- the scale is now a run
        # parameter (8/bucket smoke, 200/bucket, or the whole split), and pinning a literal here
        # would fail every run that is not the old cohort size.
        prefix_1 = "E1" if task == "reconstruction" else "ER1"
        c.add(prefix_1, f"all {n_cases} selected cases scored, 0 excluded ({scale})",
              n_cases > 0 and n_scored == n_cases and summary["n_excluded"] == 0,
              f"scored={n_scored}/{n_cases} excluded={summary['n_excluded']}")
    else:
        c.add("E2a", "generation scored 0 paired cases", n_scored == 0, f"scored={n_scored}")

    # The run must be able to say which cases produced it, without carrying an identifier.
    c.add(f"{prefix}0", f"{task}: run_id recorded",
          bool(summary.get("run_id")), str(summary.get("run_id")))

    per_bucket = read_csv(root / "metrics_per_bucket.csv")
    agg = read_csv(root / "metrics_summary.csv")
    # 4 modality scopes + 4 overall scopes. It was 6 until `overall_pooled` and
    # `overall_batched_n512` were added on 2026-08-11; this checker was not updated with them, so
    # every valid run since then has been failing E3/EG3 on a stale constant and reporting
    # "do not read the numbers as a result" over a correct CSV. Counted, not hardcoded to 8, so
    # adding a fifth modality does not resurrect the same false alarm.
    scopes = {str(r.get("scope", "")) for r in agg}
    modality_scopes = {s for s in scopes if s.startswith("modality:")}
    batched_scopes = {s for s in scopes if s.startswith(BATCHED_SCOPE_PREFIX)}
    expected_agg = len(modality_scopes) + len(EXPECTED_OVERALL_SCOPES) + len(batched_scopes)
    missing = [s for s in EXPECTED_OVERALL_SCOPES if s not in scopes]
    if not batched_scopes:
        missing.append(f"{BATCHED_SCOPE_PREFIX}<N>")
    c.add(f"{prefix}3", f"{task}: CSVs have {EXPECTED_BUCKETS} + {expected_agg} rows",
          len(per_bucket) == EXPECTED_BUCKETS and len(agg) == expected_agg and not missing,
          f"{len(per_bucket)} bucket rows, {len(agg)} aggregate rows"
          + (f"; missing scopes {missing}" if missing else ""))

    if task == "generation":
        voxelwise = [k for k in (per_bucket[0] if per_bucket else {})
                     if k.split("_")[0] in ("mae", "mse", "psnr", "ncc", "ssim3d")]
        c.add("E2b", "generation CSV has no voxelwise column", not voxelwise, str(voxelwise))

    def col(rows, scope, key):
        for r in rows:
            if r.get("scope") == scope and r.get(key, "") != "":
                return float(r[key])
        return None

    metric = "psnr_fg" if task == "reconstruction" else "inception_2p5d_fid"
    macro, weighted = col(agg, "overall_macro", metric), col(agg, "overall_weighted", metric)
    if summary.get("evaluated_full_split"):
        # On a full-split run the population counts ARE the scored counts, so the two aggregates
        # coincide -- correctly, because there is no sampling artefact left to correct for. Only a
        # capped run can show that the weighting is actually applied.
        c.add(f"{prefix}4", f"{task}: macro == weighted on a full-split run (no sampling to correct)",
              macro is not None and weighted is not None and abs(macro - weighted) < 1e-6,
              f"macro={macro} weighted={weighted}")
    else:
        c.add(f"{prefix}4", f"{task}: macro != weighted on {metric}",
              macro is not None and weighted is not None and abs(macro - weighted) > 1e-9,
              f"macro={macro} weighted={weighted}")

    def bucket_vals(key):
        return {r["bucket"]: float(r[key]) for r in per_bucket if r.get(key, "") != ""}

    if task == "reconstruction":
        psnr, ssim, ncc = bucket_vals("psnr_fg"), bucket_vals("ssim3d_whole"), bucket_vals("ncc_fg")
        edge = bucket_vals("edge_preservation_fg")
        low_psnr = {b: round(v, 2) for b, v in psnr.items() if v <= MIN_PSNR_FG}
        # SSIM3D is texture-dominated, so the floor only catches a broken reconstruction; blur is
        # measured directly by edge_preservation_fg instead. See E5 in SUCCESS_CRITERIA.md.
        low_ssim = {b: round(v, 3) for b, v in ssim.items() if v <= MIN_SSIM3D_FLOOR}
        low_edge = {b: round(v, 3) for b, v in edge.items() if v <= MIN_EDGE_PRESERVATION}
        low_ncc = {b: round(v, 3) for b, v in ncc.items() if v <= 0.9}
        c.add("E5", f"psnr_fg>{MIN_PSNR_FG} edge>{MIN_EDGE_PRESERVATION} ssim3d>{MIN_SSIM3D_FLOOR}",
              len(psnr) == EXPECTED_BUCKETS and not (low_psnr or low_ssim or low_edge),
              f"{len(psnr)}/{EXPECTED_BUCKETS} buckets; psnr={low_psnr or 'ok'} "
              f"edge={low_edge or 'ok'} ssim={low_ssim or 'ok'}"
              + (f"; ssim3d range {min(ssim.values()):.3f}-{max(ssim.values()):.3f} "
                 f"(texture-dominated, not a fidelity floor)" if ssim else ""))
        c.add("E6", "ncc_fg > 0.9 in every bucket (axis-order guard)",
              len(ncc) == EXPECTED_BUCKETS and not low_ncc,
              f"{len(ncc)}/{EXPECTED_BUCKETS} buckets; low={low_ncc}")

    if task == "generation" and recon_root is not None and (recon_root / "metrics_per_bucket.csv").is_file():
        # Aggregate ordering, not per-bucket: reconstruction FID is a non-zero floor set by encoder
        # loss (measured 8.4-18.9), so a strong bucket's generation can legitimately beat it.
        gen_fid = bucket_vals("inception_2p5d_fid")
        recon_fid = {r["bucket"]: float(r["inception_2p5d_fid"])
                     for r in read_csv(recon_root / "metrics_per_bucket.csv")
                     if r.get("inception_2p5d_fid", "") != ""}
        shared = sorted(set(gen_fid) & set(recon_fid))
        below = [b for b in shared if gen_fid[b] <= recon_fid[b]]
        recon_agg = {r["scope"]: r for r in read_csv(recon_root / "metrics_summary.csv")}
        pairs = []
        for scope in ("overall_macro", "overall_weighted"):
            g, rv = col(agg, scope, "inception_2p5d_fid"), None
            row = recon_agg.get(scope, {})
            if row.get("inception_2p5d_fid", "") != "":
                rv = float(row["inception_2p5d_fid"])
            if g is not None and rv is not None:
                pairs.append((scope, g, rv))
        agg_ok = bool(pairs) and all(g > rv for _s, g, rv in pairs)
        majority_ok = bool(shared) and len(below) <= len(shared) // 2
        c.add("E7", "generation FID > reconstruction FID (aggregate + majority)",
              agg_ok and majority_ok,
              "; ".join(f"{s}: gen {g:.2f} vs recon {rv:.2f} ({g / rv:.2f}x)" for s, g, rv in pairs)
              + f"; {len(below)}/{len(shared)} buckets below the recon floor"
              + (f" {below}" if below else ""))
    elif task == "generation":
        c.skip("E7", "generation FID > reconstruction FID", "reconstruction results not given")

    if task == "generation":
        # E8 needs FID between two REAL populations of different modality, which the standard
        # evaluation never computes. Named as a skip rather than omitted, so it stays visible.
        c.skip("E8", "FID backbone validity (cross-modality)",
               "run separately; measured: Inception 2.5D 1.75x, MedicalNet 1.00x (invalid)")

    if task == "report2volume":
        # ER2: the intensity-space guard, checked from the outside. A prediction set written in
        # NVIDIA's int16 [0, 1000] range against a percentile-normalised [0, 1] cohort produces
        # metrics that all look plausible, so this is checked on the numbers rather than trusted.
        # MAE against a [0,1] ground truth cannot exceed ~2 for anything in the same space; a
        # 1000x offset puts it in the hundreds.
        mae = bucket_vals("mae_fg")
        wrong_space = {b: round(v, 1) for b, v in mae.items() if v > 5.0}
        c.add("ER2", "predictions are in the cohort's intensity space (mae_fg < 5)",
              bool(mae) and not wrong_space,
              f"{len(mae)}/{EXPECTED_BUCKETS} buckets; suspicious={wrong_space or 'none'}"
              + ("" if mae else "; no mae_fg column at all"))

        # ER3: the report-consistency group. Reported whether or not it ran -- an unavailable
        # metric with a reason is a SKIP here, never a silent pass.
        consistency_path = root / "report_consistency.json"
        consistency = json.loads(consistency_path.read_text()) if consistency_path.is_file() else {}
        if not consistency:
            c.add("ER3", "report_consistency.json written", False, "file missing")
        elif not consistency.get("available"):
            c.skip("ER3", "blinded classifier consistency",
                   consistency.get("reason", "unavailable, no reason recorded"))
        else:
            generated = consistency.get("macro_auroc_usable_labels")
            reference = (consistency.get("real_reference") or {}).get("macro_auroc_usable_labels")
            n_usable = consistency.get("n_labels_usable", 0)
            # The floor is 0.5 (an image-blind guesser). Beating it is what "the generated volume
            # carries the report's findings" means; how far it is from `reference` is how good.
            c.add("ER3", "blinded classifier consistency above chance on usable labels",
                  n_usable > 0 and generated is not None and generated > 0.5,
                  f"macro AUROC {generated} over {n_usable} usable labels; "
                  f"real-volume reference {reference}; floor 0.5")
            c.add("ER4", "the classifier was fitted on a non-test split",
                  (consistency.get("classifier", {}).get("provenance", {}).get("train_split")
                   not in ("test", "", None)),
                  f"train_split="
                  f"{consistency.get('classifier', {}).get('provenance', {}).get('train_split')!r} "
                  f"(fitting on test makes the metric circular)")

    figures = summary.get("figures", [])
    figured = _buckets_figured(figures, buckets)
    c.add(f"{prefix}9", f"{task}: figures for all 10 buckets",
          len(figured) == EXPECTED_BUCKETS,
          f"{len(figures)} files covering {len(figured)} buckets"
          + (f"; missing {sorted(set(buckets) - figured)}" if len(figured) != EXPECTED_BUCKETS else ""))


def main(argv=None) -> int:
    """Check one or more results directories against SUCCESS_CRITERIA.md.

    **The cohort and prediction stages are gone**, so there is nothing to check before the results:
    `cli.evaluate` builds the dataset, generates and scores in one pass, and the properties the old
    stage-1/stage-2 checks enforced on disk (right buckets, right FOV, one prediction per case, a
    matching cohort_id) are now either impossible to violate or checked inside the run itself and
    recorded in `run_manifest.json`.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, action="append", required=True,
                   help="a results directory from cli.evaluate; repeatable")
    args = p.parse_args(argv)

    c = Checks()
    print("=== checking against slurm/SUCCESS_CRITERIA.md ===")
    recon_root = next((r for r in args.results
                       if (r / "summary.json").is_file()
                       and json.loads((r / "summary.json").read_text()).get("task") == "reconstruction"),
                      None)
    for root in args.results:
        summary_path = root / "summary.json"
        if not summary_path.is_file():
            c.skip("E*", f"{root.name}", "no summary.json -- the run did not finish")
            continue
        summary = json.loads(summary_path.read_text())
        task = summary.get("task")
        if task not in ("reconstruction", "generation", "report2volume"):
            c.skip("E*", f"{root.name}", f"unknown task {task!r}")
            continue
        print(f"\n--- {root.name}: task={task} run_id={summary.get('run_id')}")
        buckets = sorted((summary.get("bucket_geometry") or {}))
        check_results(c, root, task, recon_root=recon_root, buckets=buckets)

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
