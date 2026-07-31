#!/usr/bin/env python3
"""Convert an external checkpoint's saved NIfTI files into a prediction set.

Use this when a model outside this pipeline already wrote `.nii.gz` volumes and you want them
scored against a cohort. It does the identifier matching *once*, records the result, and
produces a directory `cli/evaluate.py` accepts like any other -- so the evaluator never has to
guess which file belongs to which case.

    python -m mrrate_r2v.cli.import_predictions \\
        --cohort <workspace>/cohorts/test_v1 \\
        --predictions-csv /path/to/predictions.csv \\
        --out <workspace>/predictions/external_v1

CSV schema (`study_key`/`series_key` must match the manifest's `study_uid`/`series_id` exactly
-- never a filename or row position):

    study_key,prediction_path[,series_key,modality,acquisition_plane]

`series_key` may be omitted only for a study with exactly one case in the cohort. Anything
ambiguous, duplicated, or unmatched is rejected with a reason into `import_report.json` rather
than guessed at.

Geometry: each file's real affine is read and checked against the cohort's grid. A file on a
different grid is imported anyway (with its own shape recorded) and it is the evaluator that
excludes it -- so the exclusion appears in one place, with a reason, next to everything else.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from ..cohort import Cohort, case_id_for
from ..eval.pairing import load_predictions_csv, pair_predictions_to_targets, target_rows_from_cohort
from ..predictions import PredictionItem, PredictionSet, write_prediction_set
from ..volumes import VolumeWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("import_predictions")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", default="report2volume", choices=["reconstruction", "report2volume"],
                   help="recorded in the prediction set; --task on the evaluator is what selects metrics")
    p.add_argument("--model-name", default="external_checkpoint")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} exists and is non-empty; pass --overwrite to replace it")

    cohort = Cohort(args.cohort)
    log.info("cohort %s", cohort.summary())

    records = load_predictions_csv(args.predictions_csv)
    result = pair_predictions_to_targets(records, target_rows_from_cohort(cohort),
                                         expected_split=cohort.spec["split"])
    log.info("pairing: %s", result.summary())

    items, failures = [], []
    writer = VolumeWriter(args.out)
    for paired in result.paired:
        case_id = case_id_for(paired.target.study_key, paired.target.series_key)
        case = cohort.case_by_id(case_id)
        if case is None:  # pairing matched a target the cohort does not carry -- should not happen
            failures.append({"prediction_path": paired.prediction.prediction_path,
                             "category": "not_in_cohort",
                             "reason": "paired target is not one of this cohort's cases"})
            continue
        try:
            img = nib.load(paired.prediction.prediction_path)
            volume = np.asarray(img.dataobj, dtype=np.float32)
            spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
            orientation = "".join(nib.aff2axcodes(img.affine))
        except Exception as e:  # noqa: BLE001
            failures.append({"prediction_path": paired.prediction.prediction_path,
                             "category": "unreadable", "reason": f"{type(e).__name__}: {e}"})
            continue
        writer.add(case.bucket, case_id, volume)
        items.append(PredictionItem(
            prediction_id=case_id, case_id=case_id, sequence=case.sequence,
            plane=case.acquisition_plane, shape=list(volume.shape), spacing_mm=list(spacing),
            extra={"source_file": paired.prediction.prediction_path, "source_orientation": orientation},
        ))

    writer.close()

    for rejected in result.rejected:
        failures.append({"prediction_path": rejected.prediction.prediction_path,
                         "category": rejected.category, "reason": rejected.reason})

    pset = PredictionSet(
        task=args.task, cohort_id=cohort.cohort_id, cohort_cases_sha256=cohort.cases_sha256,
        model={"name": args.model_name, "source": "imported_nifti",
               "predictions_csv": str(args.predictions_csv)},
        items=items, failures=failures, created_by="mrrate_r2v.cli.import_predictions",
    )
    out = write_prediction_set(args.out, pset)
    (args.out / "import_report.json").write_text(json.dumps(
        {"pairing": result.summary(), "failures": failures}, indent=2))
    log.info("imported %d volumes, %d rejected -> %s", len(items), len(failures), out)
    if failures:
        log.warning("see %s for why each rejection happened", args.out / "import_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
