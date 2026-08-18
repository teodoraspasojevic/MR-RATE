"""The manifest: one CSV row per eligible (study, series) pair. Stdlib only at import time.

The manifest is built once and reused forever. It records *where* each series is and what
it is (modality, plane, split) -- never anything about geometry policy, report source, or
sampling. So switching `series_selection`, `geometry_mode`, or report store costs nothing;
only a change of underlying storage requires a rebuild.

One builder, for the one storage layout this pipeline actually reads from:
`build_manifest_rows_from_shards_parquet`, over SHARDS_PATH's WebDataset-style
`shard-*.tar` + `series.parquet`. It works purely from that root's existing index, so a
full build takes seconds and touches no image bytes. Use `verify_archive_locators_sample`
afterwards to spot-check that the filename convention still holds before trusting a build.

(Two other builders -- for an already-extracted directory tree, and for DATA_PATH's
`batchNN.tar` of per-study zips -- existed here and were removed 2026-08-18: nothing in
this repo built a manifest from either layout. Re-add one only if a real storage location
needs it; `git log` has the removed implementations if so.)

**This module deliberately imports no torch.** pyarrow is imported lazily inside the one
builder that needs it, so a pyarrow-only interpreter can build a manifest without ever
importing torch.
"""
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

from .storage import ArchiveReader, Locator, iter_csv_dict_rows

# NV-Generate-MR-Brain supports T1w/T2w/FLAIR/SWI only, no MRA. MRA is also the rarest
# MR-RATE modality by a wide margin (~0.02% of series), so excluding it by default costs
# negligible coverage while matching the downstream generative model's actual support.
DEFAULT_EXCLUDED_MODALITIES = frozenset({"MRA"})

MANIFEST_FIELDS = (
    "study_uid", "series_id", "image_path", "split",
    "modality", "plane", "is_center_modality",
    "native_shape", "native_spacing_mm",
    "cache_index",
    "backend", "archive_path", "member_chain",
)

BACKENDS = ("file", "archive")


# --------------------------------------------------------------------------- eligibility


@dataclass
class SeriesMeta:
    """The fields read from MR-RATE's per-series metadata CSV.

    Deliberately excludes its array_shape/array_spacing_mm/array_fov_mm columns: those are
    computed by a plain `nib.load()` with NO RAS reorientation, i.e. raw on-disk axis order,
    despite once being named "ras_array_shape". Using them as native geometry would silently
    swap axes for any series not already stored in RAS. Native geometry is always
    independently derived via `read_native_geometry`, which does reorient.
    """

    series_id: str
    modality: Optional[str]
    plane: Optional[str]
    is_derived: bool
    is_localizer: bool
    is_center_modality: bool


def _parse_bool(value):
    return str(value).strip().lower() in ("1", "true", "t", "yes")


class MetadataStore:
    """MR-RATE's per-series metadata CSV (or `.tar.gz` of per-batch CSVs), keyed by
    (study_uid, series_id) -- series_id is unique within a study, not globally.

    Optional. Without it, modality/plane conditioning and the exclusion policy are
    unavailable and every discovered series is treated as eligible.
    """

    def __init__(self, metadata_csv):
        self.by_study = {}
        for row in iter_csv_dict_rows(metadata_csv):
            study_uid = row.get("study_uid")
            series_id = row.get("series_id")
            if not study_uid or not series_id:
                continue
            self.by_study.setdefault(study_uid, {})[series_id] = SeriesMeta(
                series_id=series_id,
                modality=row.get("classified_modality") or None,
                plane=row.get("acquisition_plane") or None,
                is_derived=_parse_bool(row.get("is_derived", "")),
                is_localizer=_parse_bool(row.get("is_localizer", "")),
                is_center_modality=_parse_bool(row.get("is_center_modality", "")),
            )

    def get(self, study_uid, series_id):
        return self.by_study.get(study_uid, {}).get(series_id)


def series_id_from_path(path, study_uid):
    """Recover the official series_id from a released filename.

    Released images are named `{study_uid}_{series_id}.nii.gz`, where series_id is
    `{modality}-{role}-{plane}`. Upload pseudonymizes only the study_uid substring, so the
    `_{series_id}` suffix survives to the release unchanged.

    Falls back to the bare filename stem when the `{study_uid}_` prefix is absent (e.g. test
    fixtures with a different naming scheme) -- treat that fallback as a best-effort key,
    not an authoritative series_id.
    """
    stem = os.path.basename(path)
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    prefix = f"{study_uid}_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def is_eligible(meta, excluded_modalities=DEFAULT_EXCLUDED_MODALITIES,
                exclude_derived=True, exclude_localizer=True):
    """Whether to keep a series, given its metadata (or None if unavailable).

    Derived/localizer/scout/DWI series are already excluded upstream of the public release,
    so those two checks are defensive re-checks (both are 100% False dataset-wide). The only
    exclusion that actually changes anything relative to the release is MRA.

    With `meta=None` there is no data-driven basis to exclude anything, so filtering
    degrades to "off" rather than guessing from a filename.
    """
    if meta is None:
        return True
    if exclude_derived and meta.is_derived:
        return False
    if exclude_localizer and meta.is_localizer:
        return False
    if meta.modality is not None and meta.modality in excluded_modalities:
        return False
    return True


# --------------------------------------------------------------------------- the row


@dataclass
class ManifestRow:
    """One eligible (study, series) pair.

    `backend="file"`: `image_path` is a real filesystem path in an extracted tree;
    archive_path/member_chain unused.

    `backend="archive"`: the series lives inside an un-extracted archive. `image_path` is
    "", `archive_path` is the outer on-disk tar, and `member_chain` is the sequence of
    member names to descend to reach the NIfTI bytes (see `storage.Locator`).
    """

    study_uid: str
    series_id: str
    image_path: str            # absolute path to the source NIfTI; "" when backend="archive"
    split: Optional[str]
    modality: Optional[str]
    plane: Optional[str]
    is_center_modality: bool
    native_shape: Optional[tuple]        # (D, H, W) voxels, RAS, pre-resample; None until lazily resolved for archives
    native_spacing_mm: Optional[tuple]   # (D, H, W) mm, RAS, pre-resample; None until lazily resolved for archives
    cache_index: int           # position within this study's sorted image_paths; matches
                               # preprocess_volumes.py's .npz stacking order 1:1. Meaningless
                               # for backend="archive" (that cache needs an extracted tree).
    backend: str = "file"
    archive_path: Optional[str] = None
    member_chain: Optional[tuple] = None

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown ManifestRow backend '{self.backend}'. Choose from: {BACKENDS}")
        if self.backend == "archive" and (not self.archive_path or not self.member_chain):
            raise ValueError("backend='archive' requires archive_path and member_chain.")

    def locator(self):
        """The `storage.Locator` this row resolves to -- the single translation point from
        persisted manifest fields to runtime storage access."""
        if self.backend == "archive":
            return Locator(kind="archive", archive_path=self.archive_path,
                           member_chain=tuple(self.member_chain))
        return Locator(kind="file", path=self.image_path)

    def to_csv_row(self):
        return {
            "study_uid": self.study_uid,
            "series_id": self.series_id,
            "image_path": self.image_path,
            "split": self.split or "",
            "modality": self.modality or "",
            "plane": self.plane or "",
            "is_center_modality": int(self.is_center_modality),
            "native_shape": json.dumps(list(self.native_shape)) if self.native_shape else "",
            "native_spacing_mm": json.dumps(list(self.native_spacing_mm)) if self.native_spacing_mm else "",
            "cache_index": self.cache_index,
            "backend": self.backend,
            "archive_path": self.archive_path or "",
            "member_chain": json.dumps(list(self.member_chain)) if self.member_chain else "",
        }

    @classmethod
    def from_csv_row(cls, row):
        return cls(
            study_uid=row["study_uid"],
            series_id=row["series_id"],
            image_path=row["image_path"],
            split=row.get("split") or None,
            modality=row.get("modality") or None,
            plane=row.get("plane") or None,
            is_center_modality=bool(int(row.get("is_center_modality") or 0)),
            native_shape=tuple(json.loads(row["native_shape"])) if row.get("native_shape") else None,
            native_spacing_mm=tuple(json.loads(row["native_spacing_mm"])) if row.get("native_spacing_mm") else None,
            cache_index=int(row["cache_index"]),
            # Explicit defaults: manifests written before the archive backend existed have no
            # such columns and must still load as ordinary backend="file" rows.
            backend=row.get("backend") or "file",
            archive_path=row.get("archive_path") or None,
            member_chain=tuple(json.loads(row["member_chain"])) if row.get("member_chain") else None,
        )


def write_manifest_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def read_manifest_csv(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [ManifestRow.from_csv_row(row) for row in reader]


# --------------------------------------------------------------------------- builders


# The released splits.csv and this package use "val". SHARDS_PATH's series.parquet `split`
# column agrees ("val"), but its *directory* is named "validation/" -- a real naming mismatch
# inside that repackaging, mapped here rather than silently gotten wrong.
_SHARDS_SPLIT_TO_DIR = {"train": "train", "val": "validation", "test": "test"}

# series.parquet's `plane` column uses abbreviated lowercase codes, NOT the
# AXIAL/SAGITTAL/CORONAL vocabulary used by the official metadata CSV and by
# `geometry.NV_BRAIN_FOV_MM`'s keys. Without this normalization every shards row silently
# falls back to the 256^3 fallback bucket regardless of its real modality/plane -- not a
# crash, so easy to miss. An unrecognized code passes through unchanged (falls back safely)
# rather than being assumed exhaustive; no oblique code was observed locally.
_SHARDS_PLANE_TO_CANONICAL = {"axi": "AXIAL", "sag": "SAGITTAL", "cor": "CORONAL"}


def build_manifest_rows_from_shards_parquet(
    shards_root, splits=("train", "val", "test"),
    excluded_modalities=DEFAULT_EXCLUDED_MODALITIES,
    exclude_derived=True, exclude_localizer=True,
):
    """Build rows for SHARDS_PATH's WebDataset-style layout from its own `series.parquet`,
    without opening a shard tar. That index already carries `shard_name` +
    `tar_member_path`, so this is a thin adapter, not a new index.

    Requires pyarrow, imported lazily so nothing else in the package depends on it. Because
    this module has no torch import either, an interpreter with only pyarrow can call this
    function directly -- which is why there is no separate standalone copy of it.

    `native_shape`/`native_spacing_mm` are intentionally NOT taken from series.parquet's own
    shape/spacing columns: that repackaging's build code is not available to verify their
    axis convention, and the analogous columns in the official metadata CSV were found to be
    in raw pre-reorientation order (see `SeriesMeta`). Left None, resolved lazily.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "build_manifest_rows_from_shards_parquet requires pyarrow. Run it under an "
            "interpreter that has pyarrow -- this module imports no torch, so a pyarrow-only "
            "interpreter is enough."
        ) from e

    table = pq.read_table(
        os.path.join(shards_root, "series.parquet"),
        columns=["study_uid", "series_id", "split", "shard_name", "modality", "plane",
                 "is_derived", "is_localizer", "is_center_modality", "image_present",
                 "tar_member_path"],
    )
    cols = {name: table.column(name) for name in table.column_names}

    rows = []
    for i in range(table.num_rows):
        if not cols["image_present"][i].as_py():
            continue
        split = cols["split"][i].as_py()
        if split not in splits:
            continue
        shard_name = cols["shard_name"][i].as_py()
        tar_member_path = cols["tar_member_path"][i].as_py()
        if not shard_name or not tar_member_path:
            continue
        raw_plane = cols["plane"][i].as_py()
        meta = SeriesMeta(
            series_id=cols["series_id"][i].as_py(),
            modality=cols["modality"][i].as_py(),
            plane=_SHARDS_PLANE_TO_CANONICAL.get(raw_plane, raw_plane) if raw_plane else None,
            is_derived=bool(cols["is_derived"][i].as_py()),
            is_localizer=bool(cols["is_localizer"][i].as_py()),
            is_center_modality=bool(cols["is_center_modality"][i].as_py()),
        )
        if not is_eligible(meta, excluded_modalities, exclude_derived, exclude_localizer):
            continue
        split_dir = _SHARDS_SPLIT_TO_DIR.get(split, split)
        # shard_name is the bare stem ("shard-000000"), not the on-disk filename.
        archive_path = os.path.join(shards_root, split_dir, shard_name + ".tar")
        rows.append(ManifestRow(
            study_uid=cols["study_uid"][i].as_py(), series_id=meta.series_id,
            image_path="", split=split, modality=meta.modality, plane=meta.plane,
            is_center_modality=meta.is_center_modality,
            native_shape=None, native_spacing_mm=None, cache_index=0,
            backend="archive", archive_path=archive_path,
            member_chain=(tar_member_path,),
        ))
    return rows


def build_shard_report_index(shards_root, splits=("train", "val", "test")):
    """The (study_uid, archive_path) index `reports.ShardReportStore` needs, built from
    `studies.parquet`'s own `has_report` column. Same lazy-pyarrow rule as above.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("build_shard_report_index requires pyarrow.") from e

    table = pq.read_table(
        os.path.join(shards_root, "studies.parquet"),
        columns=["study_uid", "split", "shard_name", "has_report"],
    )
    cols = {name: table.column(name) for name in table.column_names}

    rows = []
    for i in range(table.num_rows):
        if not cols["has_report"][i].as_py():
            continue
        split = cols["split"][i].as_py()
        if split not in splits:
            continue
        shard_name = cols["shard_name"][i].as_py()
        if not shard_name:
            continue
        split_dir = _SHARDS_SPLIT_TO_DIR.get(split, split)
        rows.append({
            "study_uid": cols["study_uid"][i].as_py(),
            "archive_path": os.path.join(shards_root, split_dir, shard_name + ".tar"),
        })
    return rows


def write_report_index_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["study_uid", "archive_path"])
        writer.writeheader()
        writer.writerows(rows)


def verify_archive_locators_sample(rows, n=20, seed=0):
    """Resolve `n` random archive-backed rows for real and confirm the bytes look like a
    gzip-compressed NIfTI. A bounded sanity check on the filename convention -- it does not
    decode voxels. Returns (n_ok, failures) where each failure is
    (index, redacted_locator, error) and never contains a study/series identifier.
    """
    archive_rows = [r for r in rows if r.backend == "archive"]
    rng = random.Random(seed)
    sample = rng.sample(archive_rows, min(n, len(archive_rows)))
    reader = ArchiveReader()
    n_ok = 0
    failures = []
    for idx, row in enumerate(sample):
        locator = row.locator()
        try:
            data = reader.read_bytes(locator)
            if len(data) < 2 or data[0] != 0x1F or data[1] != 0x8B:
                raise ValueError("resolved bytes do not look gzip-compressed")
            n_ok += 1
        except Exception as e:  # noqa: BLE001 -- report and continue sampling
            failures.append((idx, locator.redacted(), f"{type(e).__name__}: {e}"))
    return n_ok, failures
