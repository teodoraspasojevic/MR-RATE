"""Report-derived clinical labels for a cohort's cases.

The label side of the blinded-classifier metric. `mrrate_merged_labels.csv` is keyed by
`study_uid`; a cohort's cases are keyed by `case_id`. This module is the join, and it is the
*only* place the two meet -- so identifiers stay in the cohort's `index.csv` and everything the
evaluator carries onward is `case_id`.

The labels themselves are the 14 merged clinical groups from
`scripts/eval_labels/splits_merged_majority/`: the 37 raw LLM pathology classes collapsed by a
neuroradiologist into pathophysiology (`PP_*`) and imaging-phenotype (`BP_*`) groups, after a
3-model majority vote. They are derived from the *report*, which is exactly what this metric
needs: the conditioning text and the label come from the same source, so "does the generated
volume carry the finding the report describes" is a well-posed question.

**Three `PP_`/`BP_` pairs are the same column.** `PP_Neurodegenerative` == `BP_Atrophies`,
`PP_Neoplastic` == `BP_Contrast_enhancing_intracranial`, `PP_Infectious` == `BP_Infectious_lesions`
-- identical counts on every split, because those groups collapse to the same underlying classes.
They are kept (dropping one of a pair would silently change what a macro average means) and flagged
in `duplicate_label_groups` so an aggregate can be read for what it is.

Stdlib only -- no torch, no numpy, no pandas. `data/{storage,manifest,reports,geometry}.py` keep
that property for a reason and this file is read by the same CPU-only paths.
"""
from __future__ import annotations

import csv
from pathlib import Path

#: Shipped with the repo, so a run never depends on a path outside it.
DEFAULT_LABELS_CSV = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval_labels" / "splits_merged_majority"
    / "mrrate_merged_labels.csv"
)

#: Columns that are byte-identical to another column (see the module docstring). Reported, not
#: removed.
DUPLICATE_LABEL_GROUPS = (
    ("PP_Neurodegenerative", "BP_Atrophies"),
    ("PP_Neoplastic", "BP_Contrast_enhancing_intracranial"),
    ("PP_Infectious", "BP_Infectious_lesions"),
)


class ReportLabels:
    """`study_uid -> {label: 0|1}`, plus the join onto a cohort."""

    def __init__(self, csv_path=None) -> None:
        self.path = Path(csv_path or DEFAULT_LABELS_CSV)
        if not self.path.is_file():
            raise SystemExit(
                f"pathology label file not found: {self.path}. It ships in the repo at "
                f"scripts/eval_labels/splits_merged_majority/mrrate_merged_labels.csv."
            )
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or reader.fieldnames[0] != "study_uid":
                raise SystemExit(f"{self.path}: expected a leading 'study_uid' column")
            self.labels = tuple(c for c in reader.fieldnames if c != "study_uid")
            self._by_study = {
                row["study_uid"]: tuple(int(row[c]) for c in self.labels) for row in reader
            }

    def __len__(self) -> int:
        return len(self._by_study)

    def for_study(self, study_uid: str):
        return self._by_study.get(study_uid)

    def for_cohort(self, cohort) -> dict:
        """`{case_id: (0|1, ...)}` for every case whose study has labels.

        Cases with no labelled study are simply absent, and `cohort_coverage` says how many. A
        missing label is never imputed as negative: 'the report was not classified' and 'the report
        says no' are different statements, and averaging them together would quietly deflate every
        prevalence.
        """
        out = {}
        for case in cohort.cases:
            row = self._by_study.get(case.study_key)
            if row is not None:
                out[case.case_id] = row
        return out

    def cohort_coverage(self, cohort) -> dict:
        matched = self.for_cohort(cohort)
        return {
            "n_cases": len(cohort.cases),
            "n_labelled": len(matched),
            "n_unlabelled": len(cohort.cases) - len(matched),
            "labels_csv": str(self.path),
            "n_labels": len(self.labels),
            "duplicate_label_groups": [list(p) for p in DUPLICATE_LABEL_GROUPS],
        }

    def prevalence(self, case_labels: dict) -> dict:
        """`{label: positive fraction}` over an already-joined `{case_id: row}` mapping."""
        n = len(case_labels)
        if not n:
            return {name: None for name in self.labels}
        return {
            name: sum(row[i] for row in case_labels.values()) / n
            for i, name in enumerate(self.labels)
        }


__all__ = ["DEFAULT_LABELS_CSV", "DUPLICATE_LABEL_GROUPS", "ReportLabels"]
