"""A prediction set -- the output of any model, and the mirror of a `Cohort`.

Every predict script writes one of these. `cli/evaluate.py` reads it and refuses to score it
unless its `cohort_id` matches the ground-truth cohort it is being compared against. That
single check is what stops a stale prediction directory from being silently scored against a
cohort it was never produced for.

On disk:

    <pred_dir>/
      predictions.json      task, cohort_id, model provenance (checkpoint sha256), item list
      volumes/<modality>__<plane>.npz   one compressed archive per bucket, members keyed by
                            prediction_id (same layout as a cohort -- see volumes.py)

`prediction_id` equals the ground-truth `case_id` for paired tasks (reconstruction,
report2volume) and is `gen_<sequence>_<NNN>` for unconditional generation, which has no
ground-truth counterpart. `case_id=None` on an item is how "there is no specific real volume
this should match" is stated explicitly rather than inferred.

Stdlib + numpy only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .volumes import VolumeReader, VolumeWriter, bucket_name

# 2.0: volumes bundled per (modality, plane) bucket, and PredictionItem carries `plane` so its
# bucket is derivable without a cohort lookup (unconditional generation has no case_id).
PREDICTIONS_SCHEMA_VERSION = "2.0"

PREDICTIONS_JSON = "predictions.json"


@dataclass
class PredictionItem:
    """One produced volume. `shape`/`spacing_mm` are (X, Y, Z)."""

    prediction_id: str
    sequence: str
    shape: list
    spacing_mm: list
    plane: str = "unknown"          # with `sequence`, determines which archive holds this volume
    case_id: str | None = None      # None for unconditional generation
    seed: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def bucket(self) -> str:
        return bucket_name(self.sequence, self.plane)


@dataclass
class PredictionSet:
    """The contract written to `predictions.json`."""

    task: str
    cohort_id: str
    cohort_cases_sha256: str
    model: dict                     # name, checkpoint path + sha256, configs -- provenance
    items: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    predictions_schema_version: str = PREDICTIONS_SCHEMA_VERSION
    volume_dtype: str = "float32"
    volume_axis_order: str = "XYZ"
    created_by: str = ""
    runtime: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["items"] = [asdict(i) if not isinstance(i, dict) else i for i in self.items]
        d["n_items"] = len(self.items)
        d["n_failures"] = len(self.failures)
        return d


def write_prediction_set(root, pset: PredictionSet) -> Path:
    """Write `predictions.json`. Volumes are written separately by a `volumes.VolumeWriter`."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = root / PREDICTIONS_JSON
    out.write_text(json.dumps(pset.to_dict(), indent=2, sort_keys=True))
    return out


class PredictionReader:
    """Read-only view of a prediction directory."""

    def __init__(self, root):
        self.root = Path(root)
        pred_json = self.root / PREDICTIONS_JSON
        if not pred_json.is_file():
            raise SystemExit(
                f"{self.root} is not a prediction directory (no {PREDICTIONS_JSON}). Produce one "
                f"with a `mrrate_r2v.cli.predict_*` script, or convert saved NIfTI files with "
                f"`python -m mrrate_r2v.cli.import_predictions`."
            )
        self.spec = json.loads(pred_json.read_text())
        version = self.spec.get("predictions_schema_version")
        if version != PREDICTIONS_SCHEMA_VERSION:
            raise SystemExit(
                f"{pred_json}: predictions_schema_version={version!r}, this code expects "
                f"{PREDICTIONS_SCHEMA_VERSION!r}."
            )
        self.items = [PredictionItem(**{k: v for k, v in i.items() if k in PredictionItem.__dataclass_fields__})
                      for i in self.spec.get("items", [])]
        self._volumes = VolumeReader(self.root)

    @property
    def task(self) -> str:
        return self.spec["task"]

    @property
    def cohort_id(self) -> str:
        return self.spec["cohort_id"]

    @property
    def model(self) -> dict:
        return dict(self.spec.get("model", {}))

    def item_by_id(self, prediction_id: str) -> PredictionItem | None:
        return next((i for i in self.items if i.prediction_id == prediction_id), None)

    def load_volume(self, prediction_id: str) -> np.ndarray:
        item = self.item_by_id(prediction_id)
        if item is None:
            raise KeyError(f"prediction_id {prediction_id!r} is not in this prediction set")
        return self._volumes.read(item.bucket, prediction_id)

    def verify_complete(self) -> list:
        """Prediction ids missing from their bucket archive."""
        return [i.prediction_id for i in self.items
                if not self._volumes.has(i.bucket, i.prediction_id)]

    def assert_matches_cohort(self, cohort) -> None:
        """Hard-fail unless this prediction set was produced against exactly `cohort`.

        This is the guarantee that makes runs comparable: a prediction directory built from a
        different case list, FOV, normalizer, or seed has a different `cohort_id` and is
        rejected here instead of producing numbers that look valid.
        """
        if self.cohort_id != cohort.cohort_id:
            raise SystemExit(
                f"cohort mismatch -- refusing to evaluate.\n"
                f"  predictions were produced against cohort_id={self.cohort_id}\n"
                f"  --gt cohort is                    cohort_id={cohort.cohort_id}\n"
                f"These differ in at least one of: case list, split, sequences, seed, geometry, "
                f"normalizer, or report sections. Re-run the predict step against this cohort, or "
                f"point --gt at the cohort the predictions were made for."
            )
        if self.spec.get("cohort_cases_sha256") != cohort.cases_sha256:
            raise SystemExit(
                f"cohort_id matched but the case-list hash did not -- one of the two directories "
                f"has been modified after it was written. Rebuild rather than score this."
            )
