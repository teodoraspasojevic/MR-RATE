"""Read MR-RATE series straight out of un-extracted archives. Stdlib only, no torch.

Owns two things:

- `Locator` + `ArchiveReader` -- read one NIfTI's bytes out of a tar (optionally
  containing per-study zips) by true random access, never a sequential scan and never
  extracting the archive.
- `iter_csv_dict_rows` -- read a CSV, or every CSV inside a `.tar.gz`, without extracting.

Invariants: archives are opened read-only and never modified.

Why random access is safe here (measured on the real local copies): every outer archive
is a plain uncompressed POSIX tar, so `getmembers()` on a 592 GB tar takes ~1s and
reading an arbitrary member 27-50ms regardless of file size. Per-study inner zips are
STORED (uncompressed), so `zipfile` reads their central directory off the tar member's
own seekable handle in ~2ms with no intermediate copy. Full evidence:
docs/design/archive/07_archive_backed_mrrate_storage.md.
"""
import csv
import hashlib
import io
import json
import os
import tarfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


def iter_csv_dict_rows(path):
    """Yield dict rows from a plain CSV, or from every `.csv` member of a `.tar.gz`.

    MR-RATE ships `metadata.tar.gz` / `reports.tar.gz` as small tarballs of one CSV per
    batch. Archive members are read in memory, member by member -- nothing is ever
    extracted to disk, matching this module's no-extraction rule.
    """
    path_str = str(path)
    if path_str.endswith(".tar.gz") or path_str.endswith(".tgz"):
        with tarfile.open(path_str, mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                text = tf.extractfile(member).read().decode("utf-8", errors="replace")
                yield from csv.DictReader(io.StringIO(text))
    else:
        with open(path_str, "r", newline="") as f:
            yield from csv.DictReader(f)


# ---------------------------------------------------------------------------
# Locator: the smallest robust "where is this series" abstraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Locator:
    """Where to find one NIfTI's bytes.

    kind="file": ordinary extracted path, `path` set, everything else None --
      this is exactly today's `ManifestRow.image_path` behavior, unchanged.
    kind="archive": `archive_path` is the outer, on-disk archive (a plain
      tar file in every local case seen so far); `member_chain` is the
      sequence of container-member names to descend through to reach the
      target NIfTI bytes. len(member_chain)==1 for a NIfTI stored directly
      as a tar member (SHARDS_PATH); len(member_chain)==2 for a NIfTI nested
      inside a per-study ZIP that is itself a tar member (DATA_PATH) --
      chosen as one general "chain of containers" representation rather
      than a separate class per nesting depth, since the resolution logic
      (open archive_path, descend through each chain segment) is identical
      either way and generalizes to any future nesting depth without a new
      Locator subtype.
    """

    kind: str  # "file" | "archive"
    path: Optional[str] = None
    archive_path: Optional[str] = None
    member_chain: Optional[tuple] = None

    def __post_init__(self):
        if self.kind not in ("file", "archive"):
            raise ValueError(f"Unknown locator kind '{self.kind}'.")
        if self.kind == "file" and not self.path:
            raise ValueError("kind='file' requires path.")
        if self.kind == "archive" and (not self.archive_path or not self.member_chain):
            raise ValueError("kind='archive' requires archive_path and member_chain.")

    def cache_key(self):
        """Stable, collision-resistant, content-free key -- never a raw
        study/series identifier, safe to use as a cache filename."""
        if self.kind == "file":
            basis = json.dumps({"kind": "file", "path": self.path}, sort_keys=True)
        else:
            basis = json.dumps(
                {"kind": "archive", "archive_path": self.archive_path,
                 "member_chain": list(self.member_chain)},
                sort_keys=True,
            )
        return hashlib.sha256(basis.encode()).hexdigest()

    def redacted(self):
        """Safe-to-log summary: shapes and lengths only, never a literal
        path/study/series component (satisfies "log only in redacted
        form" -- study/series identifiers appear as path components in both
        `path` and `member_chain`)."""
        def shape(s):
            return f"<len={len(s)}>" if s else None
        if self.kind == "file":
            return {"kind": "file", "path": shape(self.path)}
        return {
            "kind": "archive",
            "archive_path": shape(self.archive_path),
            "member_chain_depth": len(self.member_chain),
            "member_chain_shapes": [shape(m) for m in self.member_chain],
        }


# ---------------------------------------------------------------------------
# ArchiveReader: process-local, fork/spawn-safe random access to bytes
# ---------------------------------------------------------------------------

class _ProcessLocalHandleCache:
    """Caches open TarFile handles for the *outer* archive only, keyed by
    (pid, archive_path). Never stored as Dataset instance state (which would
    make the Dataset unpicklable and, worse, risk sharing a live file
    descriptor's read position across a fork -- see the module docstring's
    multi-worker discussion). Each DataLoader worker process gets its own
    entry, opened lazily on that process's first access; a stale entry left
    behind by a different pid (e.g. after `fork()`) is never reused.
    Bounded to `max_open` handles (LRU-closed) so a job touching many
    different archives doesn't accumulate unbounded open file descriptors.
    """

    def __init__(self, max_open=8):
        self.max_open = max_open
        self._entries = OrderedDict()  # (pid, archive_path) -> TarFile

    def get(self, archive_path):
        pid = os.getpid()
        key = (pid, archive_path)
        tf = self._entries.get(key)
        if tf is not None:
            self._entries.move_to_end(key)
            return tf
        tf = tarfile.open(archive_path, mode="r:")
        self._entries[key] = tf
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_open:
            _, old_tf = self._entries.popitem(last=False)
            try:
                old_tf.close()
            except Exception:  # noqa: BLE001 -- best-effort close
                pass
        return tf

    def close_all(self):
        for tf in self._entries.values():
            try:
                tf.close()
            except Exception:  # noqa: BLE001
                pass
        self._entries.clear()


_HANDLE_CACHE = _ProcessLocalHandleCache()


class ArchiveReadError(Exception):
    """Raised when a locator cannot be resolved to bytes -- e.g. a member
    name that no longer exists (stale index / archive changed underneath
    us), a corrupt member, or a malicious/invalid path in member_chain."""


def _reject_path_traversal(member_name):
    # Defensive: cache keys are always sha256 hashes (see Locator.cache_key),
    # so a traversal attempt can only reach this code via a hand-built or
    # corrupted manifest -- still rejected explicitly rather than trusted.
    if member_name.startswith("/") or ".." in member_name.split("/"):
        raise ArchiveReadError(f"Refusing suspicious member path (redacted length={len(member_name)}).")


class ArchiveReader:
    """Resolves a `Locator(kind="archive")` to raw NIfTI bytes (still
    gzip-compressed, exactly as stored -- decompression happens later, in
    data.py's bytes-based loaders, not here).

    Each `read_bytes` call is self-contained and sequential (open outer
    member -> optionally open nested zip -> read target member -> return) --
    deliberately not holding any nested-zip handle open across calls, so
    there is no interleaved-read hazard on the outer tar's shared underlying
    file descriptor (see module docstring). The *outer* TarFile handle is
    cached and reused across calls (safe: each call's use of it is a
    complete, non-interleaved sequence), amortizing its one-time member-index
    build cost (sub-second even for the largest local archive, 592.8 GB).
    """

    def __init__(self, handle_cache=None):
        self._handles = handle_cache or _HANDLE_CACHE

    def __getstate__(self):
        # Never pickle the handle cache itself (it may hold open, unpicklable
        # TarFile objects by the time a DataLoader with a spawn-based worker
        # start method pickles the Dataset) -- an unpickled ArchiveReader
        # simply falls back to the process-global _HANDLE_CACHE, which is
        # exactly what a fresh worker process wants anyway (see the module
        # docstring's multi-worker discussion: handles are always meant to be
        # opened lazily, per-process, not inherited).
        return {}

    def __setstate__(self, state):
        self._handles = _HANDLE_CACHE

    def read_bytes(self, locator: Locator) -> bytes:
        if locator.kind != "archive":
            raise ValueError("ArchiveReader only resolves kind='archive' locators.")
        for m in locator.member_chain:
            _reject_path_traversal(m)

        try:
            tf = self._handles.get(locator.archive_path)
        except (FileNotFoundError, tarfile.ReadError) as e:
            raise ArchiveReadError(
                f"Cannot open outer archive (path length={len(locator.archive_path)}): "
                f"{type(e).__name__}: source archive may be missing or changed."
            ) from e

        outer_name = locator.member_chain[0]
        try:
            outer_info = tf.getmember(outer_name)
        except KeyError as e:
            raise ArchiveReadError(
                f"Outer member not found (name length={len(outer_name)}) -- "
                f"stale index or the source archive changed since indexing."
            ) from e

        if len(locator.member_chain) == 1:
            fobj = tf.extractfile(outer_info)
            if fobj is None:
                raise ArchiveReadError("Outer member is not a regular file.")
            return fobj.read()

        if len(locator.member_chain) != 2:
            raise ArchiveReadError(
                f"Unsupported member_chain depth {len(locator.member_chain)} "
                f"(only direct-tar-member and tar-of-zip nesting are implemented)."
            )

        outer_fobj = tf.extractfile(outer_info)
        if outer_fobj is None:
            raise ArchiveReadError("Outer member is not a regular file.")
        try:
            zf = zipfile.ZipFile(outer_fobj)
        except zipfile.BadZipFile as e:
            raise ArchiveReadError("Nested member is not a valid zip (corrupt?).") from e

        inner_name = locator.member_chain[1]
        try:
            return zf.read(inner_name)
        except KeyError as e:
            raise ArchiveReadError(
                f"Inner member not found (name length={len(inner_name)}) -- "
                f"stale index or the source archive changed since indexing."
            ) from e

