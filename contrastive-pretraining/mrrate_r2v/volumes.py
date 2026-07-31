"""On-disk volume storage: one compressed archive per (modality, plane) bucket.

Used by both `cohort.py` and `predictions.py`, so ground truth and predictions are always stored
and read the same way.

Why bundled rather than one file per case: `/hnvme` has a **file-count** quota (61k soft), and one
file per volume put three artifact directories at ~2,000 files each. Bundling by bucket gives ~10
files per directory instead. Bucket is also the unit the evaluation already groups by, so a
per-bucket evaluation touches exactly one archive.

Why `.npz` rather than a tar or a custom format: an `.npz` is a zip of `.npy` members, and zip
members are **individually readable** -- `np.load(archive)[case_id]` decompresses only that one
member. So random access stays a one-liner and nothing has to be unpacked first.

    writer:  with VolumeWriter(root) as w: w.add(bucket, case_id, array)
    reader:  VolumeReader(root).read(bucket, case_id)

Compression measured on real 256^3 volumes: **2.91x** lossless (67.1 MB -> 23.1 MB), because ~50%
of a padded volume is exact zeros. Costs ~0.7 s/volume to write, ~0.14 s to read.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

VOLUMES_DIR = "volumes"


def bucket_name(sequence: str, plane: str) -> str:
    """Filesystem-safe bucket key. The single place this string is formed, so a writer and a
    reader can never disagree about it."""
    return f"{sequence}__{plane}"


def split_bucket(bucket: str) -> tuple:
    """Inverse of `bucket_name`: "T1w__AXIAL" -> ("T1w", "AXIAL").

    Generation needs this: it iterates a cohort's buckets and has to recover the modality (for the
    class-conditioning code) and the plane (for logging) from the archive key.
    """
    sequence, _, plane = bucket.partition("__")
    return sequence, plane


def bucket_path(root, bucket: str) -> Path:
    return Path(root) / VOLUMES_DIR / f"{bucket}.npz"


def list_buckets(root) -> list:
    return sorted(p.stem for p in (Path(root) / VOLUMES_DIR).glob("*.npz"))


class VolumeWriter:
    """Appends volumes into per-bucket archives, holding one open handle per bucket.

    Memory stays at one volume: each array is serialized to a buffer and written as a zip member,
    rather than accumulating a bucket in RAM (200 volumes would be ~8 GB).

    Always use as a context manager -- the zip central directory is only written on close, so an
    un-closed archive is unreadable.
    """

    def __init__(self, root):
        self.root = Path(root)
        (self.root / VOLUMES_DIR).mkdir(parents=True, exist_ok=True)
        self._handles: dict = {}
        self._counts: dict = {}

    def _handle(self, bucket: str) -> zipfile.ZipFile:
        zf = self._handles.get(bucket)
        if zf is None:
            zf = zipfile.ZipFile(bucket_path(self.root, bucket), "w",
                                 compression=zipfile.ZIP_DEFLATED, allowZip64=True)
            self._handles[bucket] = zf
            self._counts[bucket] = 0
        return zf

    def add(self, bucket: str, case_id: str, array: np.ndarray) -> None:
        """Store one volume as float32. Dtype is forced here so a caller cannot silently change
        the on-disk precision."""
        buf = io.BytesIO()
        npy_format.write_array(buf, np.ascontiguousarray(array, dtype=np.float32),
                               allow_pickle=False)
        self._handle(bucket).writestr(f"{case_id}.npy", buf.getvalue())
        self._counts[bucket] += 1

    def counts(self) -> dict:
        return dict(self._counts)

    def close(self) -> None:
        for zf in self._handles.values():
            zf.close()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class VolumeReader:
    """Reads volumes back, caching one open archive per bucket.

    Cheap to construct, so each worker process in a parallel evaluation can hold its own -- an
    `NpzFile` keeps its zip open, and sharing one across processes would not be safe.
    """

    def __init__(self, root):
        self.root = Path(root)
        self._open: dict = {}

    def _archive(self, bucket: str):
        z = self._open.get(bucket)
        if z is None:
            path = bucket_path(self.root, bucket)
            if not path.is_file():
                raise FileNotFoundError(f"no volume archive for bucket {bucket!r}: {path}")
            z = np.load(path)
            self._open[bucket] = z
        return z

    def read(self, bucket: str, case_id: str) -> np.ndarray:
        return self._archive(bucket)[case_id].astype(np.float32, copy=False)

    def has(self, bucket: str, case_id: str) -> bool:
        try:
            return case_id in self._archive(bucket).files
        except FileNotFoundError:
            return False

    def case_ids(self, bucket: str) -> list:
        return sorted(self._archive(bucket).files)

    def close(self) -> None:
        for z in self._open.values():
            z.close()
        self._open.clear()
