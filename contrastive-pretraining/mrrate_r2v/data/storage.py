"""Read MR-RATE series straight out of un-extracted archives. Stdlib only, no torch.

Owns three things:

- `Locator` + `ArchiveReader` -- read one NIfTI's bytes out of a tar (optionally
  containing per-study zips) by true random access, never a sequential scan and never
  extracting the archive.
- `NodeLocalCache` + `resolve_node_local_root` -- optional bounded LRU disk cache on
  node-local scratch ($TMPDIR), for when a real file path beats in-memory streaming.
  Fails loudly if no valid node-local root exists rather than filling a shared workspace.
- `iter_csv_dict_rows` -- read a CSV, or every CSV inside a `.tar.gz`, without extracting.

Invariants: archives are opened read-only and never modified; only the one requested
series is ever materialized; nothing is written outside the caller's budgeted cache root.

Why random access is safe here (measured on the real local copies): every outer archive
is a plain uncompressed POSIX tar, so `getmembers()` on a 592 GB tar takes ~1s and
reading an arbitrary member 27-50ms regardless of file size. Per-study inner zips are
STORED (uncompressed), so `zipfile` reads their central directory off the tar member's
own seekable handle in ~2ms with no intermediate copy. Full evidence:
docs/design/archive/07_archive_backed_mrrate_storage.md.
"""
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import tarfile
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional


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


# ---------------------------------------------------------------------------
# FAU node-local temp root resolution -- see design doc for the verification
# ---------------------------------------------------------------------------

# Verified (docs/nhr_official_docs/{data_filesystems,clusters_helma,
# data_staging}.md, re-read directly this session) for Helma: $TMPDIR is a
# node-local NVMe SSD (15 TB/node), created automatically and deleted
# automatically at job end, mounted only inside SLURM jobs -- confirmed
# absent on the login node in this very session (`echo $TMPDIR` -> unset,
# `hostname` -> a Helma login node, no $SLURM_JOB_ID set). Other NHR@FAU
# clusters (Alex, TinyFat, TinyGPU, Woody) also expose $TMPDIR as node-local
# SSD; Fritz's $TMPDIR is node-local *RAM disk* (data_filesystems.md) -- a
# real caveat if this pipeline ever runs there (see design doc). No FAU doc
# states an inode limit for $TMPDIR -- UNKNOWN, treated conservatively via
# the configurable byte/file budget below, not assumed unlimited.
DEFAULT_NODE_LOCAL_ENV_VARS = ("TMPDIR", "TMP")

# Best-effort heuristic only (see resolve_node_local_root's docstring) --
# filesystem types that plausibly indicate node-local, not shared/network,
# storage on a Linux HPC node.
_LOCAL_LOOKING_FSTYPES = {"tmpfs", "ext4", "xfs", "overlay", "btrfs"}
_NETWORK_LOOKING_FSTYPES = {"nfs", "nfs4", "lustre", "cifs", "smb"}


class NodeLocalRootError(Exception):
    """Raised when no valid node-local temp location can be resolved.
    Deliberately never caught internally to fall back to a persistent
    workspace -- callers that want a fallback must catch this explicitly and
    choose one themselves, so a large persistent workspace is never used as
    a silent default."""


def _fstype_of(path):
    """Best-effort /proc/mounts lookup of the filesystem type backing `path`.
    Returns None if undeterminable (non-Linux, /proc unavailable, etc.) --
    callers must treat None as UNKNOWN, not as evidence of anything."""
    try:
        path = os.path.realpath(path)
        best_match = None
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt_point, fstype = parts[1], parts[2]
                if path.startswith(mnt_point) and (
                    best_match is None or len(mnt_point) > len(best_match[0])
                ):
                    best_match = (mnt_point, fstype)
        return best_match[1] if best_match else None
    except OSError:
        return None


def resolve_node_local_root(env_vars=DEFAULT_NODE_LOCAL_ENV_VARS, require_exists=True):
    """Return a verified-as-well-as-possible node-local temp directory, or
    raise NodeLocalRootError. Never returns a guessed/default persistent
    path (e.g. $WORK, $HOME, an hnvme workspace) -- if none of `env_vars` is
    set to an existing, writable directory, this fails loudly.

    Returns (path, diagnostics) where diagnostics records what was checked,
    for logging/debugging without ever needing to log a full path
    (diagnostics reports only which env var matched and the best-effort
    filesystem-type heuristic).
    """
    for var in env_vars:
        val = os.environ.get(var)
        if not val:
            continue
        if require_exists and not os.path.isdir(val):
            continue
        fstype = _fstype_of(val)
        diagnostics = {
            "env_var": var,
            "fstype": fstype,
            "looks_node_local": fstype in _LOCAL_LOOKING_FSTYPES if fstype else None,
            "looks_network": fstype in _NETWORK_LOOKING_FSTYPES if fstype else None,
        }
        return val, diagnostics
    raise NodeLocalRootError(
        f"No node-local temp directory found among {env_vars} (checked "
        f"os.environ, then os.path.isdir). This usually means you are not "
        f"inside a SLURM job (on Helma, $TMPDIR only exists inside a job -- "
        f"see docs/nhr_official_docs/data_filesystems.md) or you're on a "
        f"cluster/queue where node-local scratch isn't provisioned. Refusing "
        f"to fall back to a persistent workspace automatically -- pass an "
        f"explicit cache_root, or disable the node-local cache and use "
        f"direct streaming instead."
    )


# ---------------------------------------------------------------------------
# NodeLocalCache: bounded, LRU-evicted, process-safe materialization cache
# ---------------------------------------------------------------------------

@dataclass
class CacheBudget:
    """All limits are hard caps this cache will not exceed, checked before
    admitting a new entry (evicting LRU entries first if needed). There is
    no unbounded mode -- max_bytes/max_files must both be finite."""

    max_bytes: int
    max_files: int
    max_studies: Optional[int] = None  # reserved for a future per-study granularity variant


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_materialized: int = 0

    def as_dict(self):
        return {"hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "bytes_materialized": self.bytes_materialized}


class NodeLocalCache:
    """A bounded disk cache under `root`, holding at most one materialized
    file per requested series (granularity: individual, gzip-compressed
    NIfTI -- chosen over whole-study or whole-ZIP granularity because a
    series-level Dataset sample only ever needs one series at a time, and
    several `series_selection` modes deliberately touch only a subset of a
    study's series per epoch; materializing whole studies would cache data
    that specific run never uses).

    Populated entries are plain `.nii.gz` files, readable by the *unmodified*
    `data.load_and_resample_nii`/`preprocess_nii` path -- once a series is
    cached, the rest of the pipeline (RAS reorient, resample, normalize,
    crop/pad) is byte-for-byte the same code as the extracted-directory
    backend; nothing preprocessing-related is duplicated here.
    """

    ENTRY_SUFFIX = ".nii.gz"
    META_SUFFIX = ".meta.json"

    def __init__(self, root: str, budget: CacheBudget):
        if not os.path.isdir(root):
            raise NodeLocalRootError(f"Cache root does not exist or is not a directory (length={len(root)}).")
        self.root = root
        self.budget = budget
        self.stats = CacheStats()
        os.makedirs(os.path.join(self.root, ".tmp"), exist_ok=True)
        os.makedirs(os.path.join(self.root, ".locks"), exist_ok=True)
        self._cleanup_stale_partials()

    # -- paths --------------------------------------------------------
    def _entry_path(self, key, hint=""):
        safe_hint = re.sub(r"[^A-Za-z0-9_.-]", "", hint)[:24]
        name = f"{safe_hint}_{key}" if safe_hint else key
        return os.path.join(self.root, name + self.ENTRY_SUFFIX)

    def _meta_path(self, entry_path):
        return entry_path[: -len(self.ENTRY_SUFFIX)] + self.META_SUFFIX

    def _lock_path(self, key):
        return os.path.join(self.root, ".locks", key + ".lock")

    # -- invalidation ---------------------------------------------------
    @staticmethod
    def _source_fingerprint(archive_path):
        try:
            st = os.stat(archive_path)
            return {"size": st.st_size, "mtime": st.st_mtime}
        except OSError:
            return None

    def _is_valid(self, entry_path, expected_fingerprint):
        if not os.path.exists(entry_path):
            return False
        meta_path = self._meta_path(entry_path)
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if expected_fingerprint is None:
            return True  # source archive unavailable to check (e.g. removed) -- trust the cache
        return meta.get("source_fingerprint") == expected_fingerprint

    # -- eviction --------------------------------------------------------
    def _entries(self):
        for name in os.listdir(self.root):
            if name.endswith(self.ENTRY_SUFFIX):
                path = os.path.join(self.root, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                yield path, st

    def _current_usage(self):
        total_bytes, total_files = 0, 0
        for _, st in self._entries():
            total_bytes += st.st_size
            total_files += 1
        return total_bytes, total_files

    def _evict_until_fits(self, incoming_bytes):
        total_bytes, total_files = self._current_usage()
        candidates = sorted(self._entries(), key=lambda pe: pe[1].st_mtime)  # oldest first (LRU)
        i = 0
        while (total_bytes + incoming_bytes > self.budget.max_bytes
               or total_files + 1 > self.budget.max_files) and i < len(candidates):
            path, st = candidates[i]
            i += 1
            try:
                os.remove(path)
                meta = self._meta_path(path)
                if os.path.exists(meta):
                    os.remove(meta)
            except OSError:
                continue
            total_bytes -= st.st_size
            total_files -= 1
            self.stats.evictions += 1
        if total_bytes + incoming_bytes > self.budget.max_bytes or total_files + 1 > self.budget.max_files:
            raise NodeLocalRootError(
                f"Cache budget too small to admit one more entry of "
                f"{incoming_bytes} bytes even after evicting every other "
                f"entry (max_bytes={self.budget.max_bytes}, "
                f"max_files={self.budget.max_files})."
            )

    def _cleanup_stale_partials(self, older_than_s=3600):
        tmp_dir = os.path.join(self.root, ".tmp")
        now = time.time()
        for name in os.listdir(tmp_dir):
            path = os.path.join(tmp_dir, name)
            try:
                if now - os.stat(path).st_mtime > older_than_s:
                    os.remove(path)
            except OSError:
                continue

    # -- public API -------------------------------------------------------
    def get_or_materialize(self, locator: Locator, fetch_bytes: Callable[[], bytes], hint="") -> str:
        """Return a real filesystem path to `locator`'s materialized bytes,
        fetching (via `fetch_bytes`) and caching them only on a miss.
        Process-safe: concurrent callers (e.g. two DataLoader workers, or
        two ranks on the same node) requesting the SAME key block on a
        per-key flock rather than racing to both materialize it.
        """
        key = locator.cache_key()
        entry_path = self._entry_path(key, hint=hint)
        fingerprint = self._source_fingerprint(locator.archive_path) if locator.kind == "archive" else None

        lock_path = self._lock_path(key)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if self._is_valid(entry_path, fingerprint):
                os.utime(entry_path, None)  # touch mtime: LRU recency signal
                self.stats.hits += 1
                return entry_path

            self.stats.misses += 1
            data = fetch_bytes()
            self._evict_until_fits(len(data))

            tmp_path = os.path.join(self.root, ".tmp", f"{key}.{os.getpid()}.{time.time_ns()}.partial")
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, entry_path)  # atomic: no partial file is ever visible under the final name

            with open(self._meta_path(entry_path), "w") as f:
                json.dump({"source_fingerprint": fingerprint, "bytes": len(data),
                           "created_at": time.time()}, f)

            self.stats.bytes_materialized += len(data)
            return entry_path
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
