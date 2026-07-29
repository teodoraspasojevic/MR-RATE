"""Matching an *external* checkpoint's saved files to ground-truth cases, by identifier only.

Used by `cli/import_predictions.py`, and only there. Predictions produced by this package's
own predict scripts are already keyed by `case_id` and need no matching -- this module exists
for the case where someone hands you a directory of `.nii.gz` files and a CSV.

Pairing is decided ONLY by `study_key`/`series_key` (the manifest's `study_uid`/`series_id`) --
never by filename order, directory listing order, or batch position. A study can legitimately
have several eligible series, so a prediction must name which one; it may omit `series_key`
only when the study has exactly one. This module never guesses, and every prediction ends up
in `paired` or `rejected` with a reason -- nothing is silently dropped.

CSV schema: `study_key,prediction_path[,series_key,modality,acquisition_plane]`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PredictionRecord:
    """One prediction to be scored. `series_key=None` is only valid when the target study has
    exactly one eligible series (checked at pairing time, not assumed here).
    """

    study_key: str
    prediction_path: str
    series_key: str | None = None
    modality: str | None = None
    acquisition_plane: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TargetRow:
    """The subset of a manifest row (`manifest.ManifestRow`) pairing needs. Build via
    `target_rows_from_manifest_rows` -- never hand-construct against a different identifier
    source than the manifest already produced.
    """

    study_key: str
    series_key: str
    split: str
    modality: str | None
    acquisition_plane: str | None


def target_rows_from_manifest_rows(manifest_rows) -> dict:
    """`manifest_rows`: an iterable of `manifest.ManifestRow` (or any object with the same
    `study_uid`/`series_id`/`split`/`modality`/`plane` attributes). Returns
    `{(study_key, series_key): TargetRow}` -- the canonical target index every pairing call needs.
    """
    out = {}
    for r in manifest_rows:
        key = (r.study_uid, r.series_id)
        out[key] = TargetRow(study_key=r.study_uid, series_key=r.series_id, split=r.split, modality=r.modality, acquisition_plane=r.plane)
    return out


def target_rows_from_cohort(cohort) -> dict:
    """The same index built from a frozen `cohort.Cohort` -- so an external checkpoint's
    predictions are matched against exactly the cases that cohort contains, not against the
    whole manifest. `split` is taken from the cohort's own spec.
    """
    split = cohort.spec["split"]
    return {
        (c.study_key, c.series_key): TargetRow(
            study_key=c.study_key, series_key=c.series_key, split=split,
            modality=c.sequence, acquisition_plane=c.acquisition_plane,
        )
        for c in cohort.cases
    }


PREDICTION_CSV_REQUIRED_COLUMNS = {"study_key", "prediction_path"}
PREDICTION_CSV_OPTIONAL_COLUMNS = {"series_key", "modality", "acquisition_plane"}


def load_predictions_csv(path) -> list:
    """Read a prediction CSV into `PredictionRecord`s. Relative `prediction_path` values
    resolve against the CSV's own directory, so a manifest travels with its files.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows:
        missing = PREDICTION_CSV_REQUIRED_COLUMNS - set(rows[0])
        if missing:
            raise SystemExit(
                f"{path}: missing required column(s) {sorted(missing)} -- schema is "
                f"study_key,prediction_path[,series_key,modality,acquisition_plane]"
            )
    out = []
    for r in rows:
        base = Path(r["prediction_path"])
        out.append(PredictionRecord(
            study_key=r["study_key"],
            prediction_path=str(base if base.is_absolute() else (path.parent / base)),
            series_key=(r.get("series_key") or None), modality=(r.get("modality") or None),
            acquisition_plane=(r.get("acquisition_plane") or None),
        ))
    return out


@dataclass(frozen=True)
class RejectedPrediction:
    prediction: PredictionRecord
    category: str
    reason: str


@dataclass(frozen=True)
class PairedItem:
    prediction: PredictionRecord
    target: TargetRow


@dataclass(frozen=True)
class PairingResult:
    paired: tuple
    rejected: tuple

    def summary(self) -> dict:
        by_category = {}
        for r in self.rejected:
            by_category[r.category] = by_category.get(r.category, 0) + 1
        return {"n_paired": len(self.paired), "n_rejected": len(self.rejected), "rejected_by_category": by_category}


REJECTION_CATEGORIES = (
    "missing_identifier", "duplicate_prediction", "no_matching_target", "ambiguous_target_series_unspecified",
    "split_mismatch", "modality_mismatch", "plane_mismatch",
)


def pair_predictions_to_targets(
    predictions, target_rows_by_key: dict, *,
    expected_split: str | None = None, require_modality_match: bool = True, require_plane_match: bool = True,
) -> PairingResult:
    """`predictions`: iterable of `PredictionRecord`. `target_rows_by_key`: output of
    `target_rows_from_manifest_rows`. Every prediction ends up in exactly one of `paired`/
    `rejected` -- nothing is silently dropped.
    """
    study_to_series_keys: dict = {}
    for study_key, series_key in target_rows_by_key:
        study_to_series_keys.setdefault(study_key, set()).add(series_key)

    seen_keys: dict = {}
    duplicate_keys = set()
    for p in predictions:
        if not p.study_key:
            continue
        resolved_series = p.series_key
        if resolved_series is None:
            candidates = study_to_series_keys.get(p.study_key, set())
            if len(candidates) == 1:
                resolved_series = next(iter(candidates))
        key = (p.study_key, resolved_series)
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys[key] = p

    paired, rejected = [], []
    for p in predictions:
        if not p.study_key:
            rejected.append(RejectedPrediction(p, "missing_identifier", "prediction has no study_key"))
            continue

        resolved_series = p.series_key
        if resolved_series is None:
            candidates = study_to_series_keys.get(p.study_key, set())
            if len(candidates) == 0:
                rejected.append(RejectedPrediction(p, "no_matching_target", f"no target series found for study_key={p.study_key!r}"))
                continue
            if len(candidates) > 1:
                rejected.append(RejectedPrediction(p, "ambiguous_target_series_unspecified", f"study_key={p.study_key!r} has {len(candidates)} eligible target series ({sorted(candidates)}) but prediction did not specify series_key"))
                continue
            resolved_series = next(iter(candidates))

        key = (p.study_key, resolved_series)
        if key in duplicate_keys:
            rejected.append(RejectedPrediction(p, "duplicate_prediction", f"more than one prediction resolved to the same (study_key, series_key)={key}"))
            continue

        target = target_rows_by_key.get(key)
        if target is None:
            rejected.append(RejectedPrediction(p, "no_matching_target", f"no target row for (study_key, series_key)={key}"))
            continue
        if expected_split is not None and target.split != expected_split:
            rejected.append(RejectedPrediction(p, "split_mismatch", f"target split={target.split!r} != expected_split={expected_split!r}"))
            continue
        if require_modality_match and p.modality is not None and target.modality is not None and p.modality != target.modality:
            rejected.append(RejectedPrediction(p, "modality_mismatch", f"prediction modality={p.modality!r} != target modality={target.modality!r}"))
            continue
        if require_plane_match and p.acquisition_plane is not None and target.acquisition_plane is not None and p.acquisition_plane != target.acquisition_plane:
            rejected.append(RejectedPrediction(p, "plane_mismatch", f"prediction acquisition_plane={p.acquisition_plane!r} != target acquisition_plane={target.acquisition_plane!r}"))
            continue

        paired.append(PairedItem(prediction=p, target=target))

    return PairingResult(paired=tuple(paired), rejected=tuple(rejected))
