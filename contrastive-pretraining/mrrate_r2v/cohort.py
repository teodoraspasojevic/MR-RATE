"""The frozen ground-truth cohort -- the artifact that makes experiments comparable.

`cli/preprocess.py` writes one of these once. Every prediction run and every evaluation
reads it. Nothing downstream re-decides which cases, what FOV, or how many samples, so two
runs pointed at the same cohort directory *cannot* disagree about any of it.

On disk:

    <cohort_dir>/
      cohort.json          the contract: geometry, filters, seed, case count, cohort_id.
                           Carries no patient identifiers, so it is safe to copy into
                           results directories and share.
      index.csv            case_id -> study_key/series_key/modality/plane. Stays in the
                           workspace; this is the only file with real identifiers.
      volumes/<modality>__<plane>.npz  one compressed archive per bucket; members are
                           float32 [X, Y, Z] keyed by case_id (see volumes.py)
      reports.json         {case_id: conditioning text}

`cohort_id` is a hash over the whole contract, including the ordered case list. Quote it in
a paper; two directories with the same `cohort_id` hold the same cases at the same geometry
with the same preprocessing. A prediction set records the `cohort_id` it was produced
against and `cli/evaluate.py` refuses to score it against any other.

Stdlib + numpy only -- no torch, so an evaluation never needs the data stack loaded.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .volumes import VolumeReader, VolumeWriter, bucket_name

# 2.0: volumes bundled one compressed archive per (modality, plane) bucket instead of one .npy
# per case, reports in a single JSON instead of one .txt per case, per-bucket geometry from
# NVIDIA's published FOV table, and posterior_shift defaulting to 0. A 1.0 cohort cannot be read
# by this code and is rejected with a clear message rather than half-loaded.
COHORT_SCHEMA_VERSION = "2.0"

COHORT_JSON = "cohort.json"
INDEX_CSV = "index.csv"
REPORTS_JSON = "reports.json"

_INDEX_FIELDS = ("case_id", "study_key", "series_key", "sequence", "acquisition_plane",
                 "bucket", "shape", "spacing_mm")


def case_id_for(study_key: str, series_key: str) -> str:
    """A stable, non-identifying id for one (study, series) pair.

    Used as the filename for that case's volume and report so nothing on disk or in a log
    line carries a raw identifier. Deterministic, so the same pair always maps to the same
    id across runs and machines.
    """
    return hashlib.sha256(f"{study_key}|{series_key}".encode("utf-8")).hexdigest()[:16]


def canonical_hash(obj) -> str:
    """sha256 of a JSON-canonical form -- key-order and whitespace independent."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class CohortCase:
    """One ground-truth case. `shape`/`spacing_mm` are (X, Y, Z), matching the stored .npy."""

    case_id: str
    study_key: str
    series_key: str
    sequence: str               # the modality: T1w / T2w / FLAIR / SWI
    acquisition_plane: str
    shape: tuple
    spacing_mm: tuple

    @property
    def bucket(self) -> str:
        """Which volume archive holds this case. Derived, never stored twice."""
        return bucket_name(self.sequence, self.acquisition_plane)


# --------------------------------------------------------------------------- selection


def select_cohort(dataset, sequences, n_per_sequence, seed) -> dict:
    """{sequence: [dataset_index, ...]} -- deterministic given
    (dataset.samples, sequences, n_per_sequence, seed).

    Two details that make it reproducible rather than merely random:

    - Candidates are sorted by (study_uid, series_id) *before* sampling, never left in
      manifest-row or dict-iteration order, which vary between callers and Python versions.
    - A fresh `RandomState` per sequence, so asking for `T1w` alone selects exactly the same
      T1w cases as asking for `T1w T2w` would. Sequence order can never shift a draw.
    """
    selected = {}
    for seq in sequences:
        idxs = sorted(
            (i for i, r in enumerate(dataset.samples) if r.modality == seq),
            key=lambda i: (dataset.samples[i].study_uid, dataset.samples[i].series_id),
        )
        if n_per_sequence is not None and len(idxs) > n_per_sequence:
            rng = np.random.RandomState(seed)
            chosen = sorted(rng.choice(len(idxs), size=n_per_sequence, replace=False).tolist())
            idxs = [idxs[p] for p in chosen]
        selected[seq] = idxs
    return selected


def select_cohort_buckets(dataset, sequences, n_per_bucket, seed) -> dict:
    """{(modality, plane): [dataset_index, ...]} -- the per-bucket counterpart of `select_cohort`.

    Sampling per (modality, plane) rather than per modality gives every bucket equal statistical
    power, which is what per-bucket FID needs: in the real test split the plane mix is ~81% axial,
    so a per-sequence sample leaves coronal buckets with a handful of cases and an unusable FID.

    The cost is that the cohort no longer mirrors the real plane distribution, so a
    "frequency-weighted overall" number must be weighted by the *eligible population* frequencies
    (recorded separately in `cohort.json` as `population_bucket_counts`), not by cohort counts.

    Same determinism rules as `select_cohort`: candidates sorted by (study_uid, series_id) before
    sampling, and a fresh RandomState per bucket so which buckets you ask for never shifts another
    bucket's draw.
    """
    selected = {}
    for i, r in enumerate(dataset.samples):
        if r.modality in sequences:
            selected.setdefault((r.modality, r.plane), []).append(i)
    out = {}
    for key in sorted(selected):
        idxs = sorted(selected[key],
                      key=lambda i: (dataset.samples[i].study_uid, dataset.samples[i].series_id))
        if n_per_bucket is not None and len(idxs) > n_per_bucket:
            rng = np.random.RandomState(seed)
            chosen = sorted(rng.choice(len(idxs), size=n_per_bucket, replace=False).tolist())
            idxs = [idxs[p] for p in chosen]
        out[key] = idxs
    return out


def population_bucket_counts(manifest_rows, sequences) -> dict:
    """{bucket: n} over every *eligible* (study, sequence) pair, before any cohort sampling.

    These are the weights a frequency-weighted aggregate must use, since the cohort itself is
    deliberately balanced across buckets. One row per (study, modality, plane), preferring the
    center-modality series -- the same grouping as `series_selection="one_per_study_per_bucket"`,
    so the weights and the cohort agree about what counts as one observation.
    """
    per = {}
    for r in manifest_rows:
        if r.modality not in sequences:
            continue
        key = (r.study_uid, r.modality, r.plane)
        if key not in per or (r.is_center_modality and not per[key].is_center_modality):
            per[key] = r
    counts = {}
    for r in per.values():
        b = bucket_name(r.modality, r.plane)
        counts[b] = counts.get(b, 0) + 1
    return dict(sorted(counts.items()))


def cases_fingerprint(cases) -> str:
    """A hash of the ordered case list -- identifies *which* cases without exposing them."""
    return canonical_hash([[c.study_key, c.series_key] for c in cases])


# --------------------------------------------------------------------------- writing


@dataclass
class CohortSpec:
    """The contract written to `cohort.json`. Everything here affects the numbers a run
    produces; nothing here is a patient identifier."""

    split: str
    sequences: list
    series_selection: str
    n_per_bucket: int | None
    seed: int
    geometry: dict              # R2VDatasetConfig.geometry_fingerprint()
    manifest_csv: str
    manifest_sha256: str
    report_source: str
    counts: dict = field(default_factory=dict)
    bucket_counts: dict = field(default_factory=dict)
    # eligible-population counts per bucket, BEFORE cohort sampling. The cohort is balanced across
    # buckets, so a frequency-weighted aggregate must weight by these, not by bucket_counts.
    population_bucket_counts: dict = field(default_factory=dict)
    n_cases: int = 0
    cases_sha256: str = ""
    cohort_schema_version: str = COHORT_SCHEMA_VERSION
    volume_dtype: str = "float32"
    volume_axis_order: str = "XYZ"
    created_by: str = "mrrate_r2v.cli.preprocess"

    def cohort_id(self) -> str:
        """Hash of this whole contract. Changing any field -- geometry, seed, case list,
        normalizer -- changes the id, so a stale prediction set can never be scored as if it
        matched."""
        return canonical_hash(asdict(self))[:16]

    def to_dict(self) -> dict:
        return {"cohort_id": self.cohort_id(), **asdict(self)}


def write_cohort(root, spec: CohortSpec, cases, reports: dict) -> Path:
    """Write `cohort.json`, `index.csv` and `reports.json` once the case list is final.

    Volumes are written separately, as they are produced, by a `volumes.VolumeWriter`. Counts are
    recorded per bucket as well as per sequence, because the per-bucket count is what the
    evaluation weights by and what generation has to match.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    spec.n_cases = len(cases)
    spec.counts = {seq: sum(1 for c in cases if c.sequence == seq) for seq in spec.sequences}
    spec.bucket_counts = {}
    for c in cases:
        spec.bucket_counts[c.bucket] = spec.bucket_counts.get(c.bucket, 0) + 1
    spec.cases_sha256 = cases_fingerprint(cases)

    with open(root / INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_INDEX_FIELDS))
        w.writeheader()
        for c in cases:
            w.writerow({
                "case_id": c.case_id, "study_key": c.study_key, "series_key": c.series_key,
                "sequence": c.sequence, "acquisition_plane": c.acquisition_plane,
                "bucket": c.bucket,
                "shape": json.dumps(list(c.shape)), "spacing_mm": json.dumps(list(c.spacing_mm)),
            })
    # One file rather than one .txt per case: same inode pressure argument as the volume archives.
    (root / REPORTS_JSON).write_text(json.dumps(reports, indent=0, sort_keys=True))
    (root / COHORT_JSON).write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
    return root / COHORT_JSON


# --------------------------------------------------------------------------- reading


class Cohort:
    """Read-only view of a cohort directory. This is what `cli/evaluate.py` and every
    predict script consume -- none of them touch a manifest, an archive, or the Dataset.
    """

    def __init__(self, root):
        self.root = Path(root)
        cohort_json = self.root / COHORT_JSON
        if not cohort_json.is_file():
            raise SystemExit(
                f"{self.root} is not a cohort directory (no {COHORT_JSON}). Build one with "
                f"`python -m mrrate_r2v.cli.preprocess`."
            )
        self.spec = json.loads(cohort_json.read_text())
        version = self.spec.get("cohort_schema_version")
        if version != COHORT_SCHEMA_VERSION:
            raise SystemExit(
                f"{cohort_json}: cohort_schema_version={version!r}, this code expects "
                f"{COHORT_SCHEMA_VERSION!r}. Rebuild the cohort rather than scoring against a "
                f"schema this version does not understand."
            )
        self.cases = self._read_index()
        self._volumes = VolumeReader(self.root)
        self._reports = None

    def _read_index(self):
        index_csv = self.root / INDEX_CSV
        if not index_csv.is_file():
            raise SystemExit(f"{index_csv} missing -- cohort directory is incomplete.")
        with open(index_csv, newline="", encoding="utf-8") as f:
            return [
                CohortCase(
                    case_id=r["case_id"], study_key=r["study_key"], series_key=r["series_key"],
                    sequence=r["sequence"], acquisition_plane=r["acquisition_plane"],
                    shape=tuple(json.loads(r["shape"])), spacing_mm=tuple(json.loads(r["spacing_mm"])),
                )
                for r in csv.DictReader(f)
            ]

    @property
    def cohort_id(self) -> str:
        return self.spec["cohort_id"]

    @property
    def cases_sha256(self) -> str:
        return self.spec["cases_sha256"]

    @property
    def sequences(self) -> list:
        return list(self.spec["sequences"])

    @property
    def geometry(self) -> dict:
        return dict(self.spec["geometry"])

    @property
    def population_bucket_counts(self) -> dict:
        """Eligible-population counts per bucket -- the weights for a frequency-weighted aggregate."""
        return dict(self.spec.get("population_bucket_counts", {}))

    @property
    def bucket_counts(self) -> dict:
        """{bucket: n} -- the unit the evaluation groups and weights by."""
        return dict(self.spec.get("bucket_counts", {}))

    @property
    def buckets(self) -> list:
        return sorted(self.bucket_counts)

    def case_by_id(self, case_id: str) -> CohortCase | None:
        return next((c for c in self.cases if c.case_id == case_id), None)

    def cases_for_sequence(self, sequence: str) -> list:
        return [c for c in self.cases if c.sequence == sequence]

    def cases_for_bucket(self, bucket: str) -> list:
        return [c for c in self.cases if c.bucket == bucket]

    def bucket_geometry(self, bucket: str) -> dict:
        """The (shape, spacing) every case in this bucket shares, read off the first case rather
        than recomputed -- so what is reported is what is actually on disk."""
        cases = self.cases_for_bucket(bucket)
        if not cases:
            return {}
        c = cases[0]
        return {"shape_xyz": list(c.shape), "spacing_mm_xyz": list(c.spacing_mm),
                "fov_mm_xyz": [round(s * p, 2) for s, p in zip(c.shape, c.spacing_mm)],
                "n": len(cases)}

    def load_volume(self, case_id: str) -> np.ndarray:
        case = self.case_by_id(case_id)
        if case is None:
            raise KeyError(f"case_id {case_id!r} is not in this cohort")
        return self._volumes.read(case.bucket, case_id)

    def load_report(self, case_id: str) -> str:
        if self._reports is None:
            path = self.root / REPORTS_JSON
            self._reports = json.loads(path.read_text()) if path.is_file() else {}
        return self._reports.get(case_id, "")

    def verify_complete(self) -> list:
        """Case ids missing from their bucket archive. An incomplete cohort must never be scored
        as if it were whole -- `cli/evaluate.py` fails on a non-empty result."""
        return [c.case_id for c in self.cases if not self._volumes.has(c.bucket, c.case_id)]

    def summary(self) -> dict:
        return {
            "cohort_id": self.cohort_id, "split": self.spec["split"],
            "n_cases": len(self.cases), "counts": self.spec["counts"],
            "bucket_counts": self.bucket_counts, "geometry": self.geometry,
        }
