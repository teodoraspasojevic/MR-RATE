# 07 — Archive-Backed Storage for the Report-to-Volume Dataset

Scope: adapt the report-to-volume Dataset/DataLoader implementation
(`docs/design/06_report_to_volume_dataset_implementation.md`) to read
directly from the two un-extracted local MR-RATE copies, without extracting
either one, without converting either into one file per series, and within
a strictly bounded, verified node-local cache when a real materialized file
is wanted. Code: `contrastive-pretraining/scripts/r2v_storage.py` (new),
extensions to `data_r2v.py`, `build_r2v_manifest.py`, two small additive
functions in `data.py`, and `tests/test_r2v_storage.py` (new) +
`tests/test_data_r2v.py` (extended).

Confidence key: **VERIFIED** (directly re-checked against the real, local
archives or docs in this session) / **INFERRED** / **ASSUMED** / **UNKNOWN**.

---

## How a Sample Is Read

```
manifest row (ManifestRow, backend="file" | "archive")
  → row.locator()                                    Locator(kind, path | archive_path+member_chain)
  → ArchiveReader.read_bytes(locator)                  [backend="archive" only]
      open/reuse outer archive (process-local handle cache, per (pid, archive_path))
      -> tf.getmember(outer_name)                      O(1)-ish dict lookup after one-time index
      -> if member_chain has 2 segments: zipfile.ZipFile(tf.extractfile(outer)) -> zf.read(inner_name)
         (reads ONLY the target member -- zip central directory read directly off
          the tar member's own seekable file object, no whole-zip materialization)
      -> returns raw, still gzip-compressed .nii.gz bytes
  → EITHER:
      "stream" (default): data.load_and_resample_nii_from_bytes(raw_bytes, ...)
        -- gzip.decompress in memory, nib.Nifti1Image.from_bytes(...), same
           RAS-canonicalize + resample as the path-based loader. NOTHING is
           ever written to disk in this mode.
      OR "node_local_cache": NodeLocalCache.get_or_materialize(locator, fetch)
        -- writes the raw (still gzip-compressed) bytes to one file under the
           configured cache root, then calls the UNMODIFIED
           data.preprocess_nii(cached_path, ...) exactly as the
           extracted-directory backend does.
  → data.preprocess_nii[_from_bytes]: RAS reorient -> resample -> normalize -> crop/pad
  → torch.Tensor[1, D, H, W]
  → collate_fn_r2v -> torch.Tensor[B, 1, D, H, W]
```

For `backend="file"` rows (the original, unmodified extracted-directory
path), this is unchanged: `preprocess_nii(row.image_path, ...)`, no
archive/locator/cache code is ever consulted.

**What is decompressed, and where:**
- The *outer* container (tar) is never decompressed -- every outer archive
  found locally is a plain, uncompressed POSIX tar (verified, see below).
- The *nested* container (a per-study ZIP, DATA_PATH only) is also
  uncompressed at the zip level (`compress_type=0`/STORED, verified) --
  `.nii.gz` members are stored as-is, since gzip-compressing an already-
  gzipped file would waste CPU for no size benefit.
- The **target `.nii.gz`'s own gzip stream** is decompressed exactly once
  per access: in memory only, in "stream" mode (nothing touches disk); or
  once when writing to the node-local cache in "node_local_cache" mode
  (still stored *compressed* on the cache disk -- see "Node-local cache
  design" -- so `nibabel`'s own, ordinary gzip handling decompresses it
  again on load, identically to the extracted-directory path).

**Where the cache writes, maximum size, when deleted:** under the
caller-configured `cache_root` (default: auto-resolved `$TMPDIR`, see
"Verified FAU cache location"), bounded to `cache_max_bytes`/`cache_max_files`
(defaults 200 GB / 20,000 files, both overridable), LRU-evicted as needed,
and never explicitly deleted by this code at job end -- it relies on
`$TMPDIR`'s own automatic, scheduler-driven deletion (verified for Helma;
see below) as the backstop, since a job-level "clean up at exit" hook would
run *after* training already benefited from the cache within that job.

**Do cache hits avoid decompression?** Yes for the *outer/nested container*
layers always (nothing there was ever compressed); for the *target NIfTI's*
own gzip stream, a "stream"-mode cache hit still means "no disk I/O
happened," but does NOT avoid re-running gzip decompression on that access
(there is nothing cached across calls in stream mode by design -- see
"Selected storage abstraction" for why streaming was chosen as the default
regardless). A "node_local_cache"-mode hit *does* avoid the archive-member
lookup and any archive I/O, but `nib.load()` on the cached `.nii.gz` file
still gzip-decompresses it exactly as it would for any `.nii.gz` file,
extracted-directory or not -- caching here saves archive-access and
network-filesystem-read cost, not gzip-decompression cost.

**Files/inodes created per cached sample:** exactly 2 (`<key>.nii.gz` +
`<key>.meta.json`) in `node_local_cache` mode; 0 in `stream` mode (the
default).

---

## Do I Need to Reshard?

| Local root | Change needed | Why |
|---|---|---|
| **DATA_PATH** (`/hnvme/workspace/<acct>-MR-RATE`, 28 un-extracted `batchNN.tar`, each a tar of per-study `.zip`) | **No change to the data. A lightweight index/manifest, built purely from existing metadata (no archive opened), plus runtime streaming.** | Confirmed by direct testing this session: indexing a 592.8 GB tar takes <1s, random member access 27-50ms (isolated) to ~0.3-5s (real, cold/warm cache, see "Performance measurements"); the outer tar member and nested zip's naming convention is fully verified and locators can be constructed from `metadata.tar.gz`+`splits.csv` alone. |
| **SHARDS_PATH** (`/hnvme/workspace/<acct>-MR-Rate-raw`, WebDataset-style `shard-*.tar`) | **No change to the data. A thin adapter reading its own pre-built `series.parquet` index into this pipeline's manifest schema, plus runtime streaming.** | This root already ships its own lightweight index (`series.parquet`'s `shard_name`+`tar_member_path` columns) -- building a *new* index from scratch would duplicate work this repackaging's own pipeline already did. Verified directly against real shard tars this session (15/15 real locators resolved after fixing one naming assumption -- see "Test and smoke-test results"). |
| *(hypothetical: neither metadata source available)* | Would require opening every archive at index-build time (still no extraction, just slower indexing) | Not the case for either currently-present local root. |

**Permanent resharding is not recommended for either root** -- every
non-rewriting approach evaluated (metadata-only indexing + runtime random
access, with an optional bounded node-local cache) is demonstrably
sufficient, verified against the real archives, not merely argued for in
the abstract.

---

## Executive conclusion

Neither local MR-RATE copy needs to be extracted, resharded, or converted
into a directory of individual files. Both are plain, **uncompressed** POSIX
tars (directly confirmed via `file` and via live `tarfile` indexing/random-
access timing against the real, full-size archives, not samples) --
uncompressed tar supports genuine random access, and both roots already
carry (or can trivially derive) a lightweight, pre-existing index mapping
each eligible series to its exact archive location. The Dataset now reads
either root directly, by default via pure in-memory streaming (no disk
write at all), with an optional, strictly bounded, LRU-evicted node-local
cache (verified FAU location: `$TMPDIR`) as a configurable alternative for
workloads that want a real file on disk. The existing extracted-directory
path (`docs/design/06_....md`) is completely unmodified and still the
default `backend`.

---

## Local dataset representations inspected

All facts below were re-checked directly against the real, current local
files this session (not assumed from the prior audit), using only
`tarfile`/`zipfile` metadata operations (`getmembers()`, `extractfile()`,
`ZipFile.infolist()`) and `pyarrow.parquet.read_schema` -- no full-archive
read, no `extractall`, nothing written back to either root. All example
paths/member names shown below are synthetic/sanitized per the task's
logging constraints; only structural shape (depth, extension, counts) is
real.

### DATA_PATH (`/hnvme/workspace/<acct>-MR-RATE`)

- **VERIFIED**: 28 top-level files, `batch00.tar` … `batch27.tar`, sizes
  7.4 GB (`batch26`) to 781 GB (`batch12`); `file batch00.tar` → *"POSIX tar
  archive (GNU)"* -- **plain, uncompressed tar**, not `.tar.gz`.
- **VERIFIED (live, `batch26.tar`, 7.4 GB)**: 312 members = 1 top-level
  directory entry + 311 `<dir>/<ID>.zip` files -- i.e. every outer member is
  a **per-study ZIP**, one directory level deep.
- **VERIFIED (live, `batch00.tar`, 592.8 GB, the largest local archive)**:
  `getmembers()` (full index of 4,325 members) completed in **0.96s**;
  reading the archive's **last** member (55.9 MB) took **27ms**, its
  **middle** member (119.8 MB) took **50ms** -- not a function of file
  position, confirming genuine seek-based random access, not a sequential
  scan.
- **VERIFIED (live, one per-study zip from `batch26.tar`, read into memory,
  4.7ms)**: 6 inner members, all `compress_type=0` (STORED) --
  `<study>/img/<series>.nii.gz` (2 members) and `<study>/seg/<mask>.nii.gz`
  (4 members -- brain + defacing masks, one pair per series; not consumed
  by this Dataset, matching `discover_subjects`'s existing behavior of only
  ever looking in `img/`).
- **VERIFIED (live)**: opening a nested ZIP **directly on the tar member's
  own seekable file object** (`zipfile.ZipFile(tf.extractfile(outer_member))`,
  no intermediate full-zip read into memory) took **2ms** to read the
  central directory of a 53 MB nested zip, then **0.9ms** to read one
  specific 5.6 KB inner member -- confirming the nested zip's own central
  directory supports true random access to one target member without
  materializing the whole nested zip.
- **VERIFIED (filename convention, via a dedicated research pass over
  `data-preprocessing/src/mr_rate_preprocessing/mri_preprocessing/
  {dcm2nii,modality_filtering,brain_segmentation_and_defacing,
  zip_and_upload}.py` and `prepare_metadata.py`)**: released filename =
  `{study_uid}_{series_id}.nii.gz`; `series_id` = `{modality_abbr}-
  {role_abbr}-{plane_abbr}[-N]`. `zip_and_upload.py`'s `_zip_study`
  pseudonymizes only the `study_uid` substring, so this suffix survives
  unchanged to the release. This is the basis for
  `_data_path_member_chain()`'s zero-archive-access locator construction.
- **VERIFIED**: `metadata.tar.gz` (125 MB) and `reports.tar.gz` (28 MB) are
  themselves small `.tar.gz` archives of ~28 per-batch CSVs each (not single
  CSV files) -- `MetadataStore`/`StructuredReportStore` now read these
  directly (`_iter_csv_dict_rows`), so even this small metadata layer is
  never extracted to disk either.
- **Small archive/report/metadata reads, not full extraction**: this
  session's hard constraints treat "do not extract any local MR-RATE copy"
  as applying to the *image* archives; reading the small `metadata.tar.gz`/
  `reports.tar.gz` fully in-memory (a few dozen small CSVs, single-digit
  seconds) was judged in-scope and implemented directly rather than asking
  the user to extract them separately, since doing so keeps the "nothing
  written to disk" property complete end-to-end.

### SHARDS_PATH (`/hnvme/workspace/<acct>-MR-Rate-raw`)

- **VERIFIED**: `train/`, `validation/`, `test/` directories, each holding
  `shard-NNNNNN.tar` + `.result.json` + `.tar.sha256` triples; one sampled
  shard (`shard-000000.tar`, 2.05 GB) is `file` → *"POSIX tar archive"* --
  again plain, uncompressed.
- **VERIFIED (live, that shard)**: `getmembers()` (303 members) in 0.17s.
  Structure: `<study>/series/<series>.nii.gz` (270 members) and
  `<study>/masks/<mask>.nii.gz` (not consumed here), **plus 3 per-study JSON
  sidecars named literally `report.json`, `labels.json`, `study.json`** --
  generic, non-identifying filenames.
- **VERIFIED (live, one study's members, contiguity check by comparing only
  parent-directory equality, never printing real names)**: every member
  belonging to one study is **contiguous** within the shard tar (WebDataset-
  style grouping), 11 distinct study-groups in the sampled shard, 15-45
  members/study (mean 27.6).
- **VERIFIED (live, `report.json`'s keys, values redacted to length-only)**:
  `{"report", "clinical_information", "technique", "findings",
  "impression"}` -- **the exact structured report schema this pipeline
  already prefers** (`StructuredReportStore`), available *inside every
  shard*, no separate reports CSV needed for this root. Not wired up as a
  new `ReportStore` implementation in this task (out of scope -- this task
  is the storage/locator layer, not a new report source), but flagged as a
  natural, low-effort future extension in "Known limitations."
- **VERIFIED (live, `study.json`'s keys)**: `study_uid, patient_uid,
  batch_id, split, shard_name, source_status, source_archive, source_member,
  source_member_size, n_series_written, has_report, has_labels,
  pipeline_version, processing_timestamp` -- study-level only, no per-series
  modality/plane (that lives in `series.parquet`, below).
- **VERIFIED (`pyarrow.parquet.read_schema`, system `python3`, this
  cluster)**: `series.parquet` (190 MB, 636,218 rows per the prior audit)
  has columns `study_uid, series_id, split, shard_name, modality, plane,
  repeat, is_derived, is_localizer, is_center_modality, metadata_matched,
  body_region, source_archive, source_member, source_zip_member,
  image_present, shape, spacing, orientation, dtype, header_error,
  source_read_error, tar_member_path, compressed_size_bytes,
  checksum_sha256, brain_mask_present, brain_mask_tar_member_path, ...` --
  **this repackaging already carries a per-series archive locator
  (`shard_name` + `tar_member_path`) and QC flags (`image_present`,
  `header_error`, `source_read_error`)** -- a pre-existing lightweight
  index this task adapts into, rather than rebuilding.
- **VERIFIED, a real naming discrepancy found and fixed this session**:
  `series.parquet`'s `split` column values are the *canonical* `train`/
  `val`/`test` strings (independently re-counted: 578,016/23,529/34,673),
  but the on-disk **directory** is named `validation/`, not `val/`, AND
  `shard_name`'s stored value is the **bare shard stem** (`"shard-000000"`),
  **without** the `.tar` extension the real filename has. Both mismatches
  were caught by an actual real-archive test run (see "Test and smoke-test
  results") and fixed in `build_manifest_rows_from_shards_parquet` --
  neither would have been caught by unit tests alone, since a synthetic
  test fixture will only be wrong if the person writing it makes the same
  wrong assumption, which is exactly what happened on the first pass here.
- **`shape`/`spacing` columns -- deliberately NOT trusted**: this
  repackaging's own build code is not present locally to verify their axis
  convention, and DATA_PATH's own, code-verified metadata CSV was
  independently found to store its analogous `array_shape`/
  `array_spacing_mm` columns in **raw, pre-reorientation axis order**
  (`modality_filtering.py:311-319`'s `load_image_properties`, a plain
  `nib.load()` with no `as_closest_canonical` call) -- see
  `docs/design/06_....md`'s equivalent finding. Rather than assume
  `series.parquet`'s `shape`/`spacing` are RAS-canonical without evidence,
  this design leaves `native_shape`/`native_spacing_mm` `None` for every
  archive-backed row and resolves them lazily, from the same bytes already
  read for training (see `data_r2v.py`'s `SeriesMeta` docstring and
  `build_manifest_rows_from_shards_parquet`'s docstring for the full
  reasoning).

---

## Compatibility matrix

| Dataset root | Storage representation | Current (extracted-dir) loader compatible? | Direct streaming possible? | Temporary extraction needed? | Recommended strategy |
|---|---|---:|---:|---:|---|
| DATA_PATH | 28 uncompressed `batchNN.tar`, each member a per-study uncompressed-zip, `<study>/img/*.nii.gz` inside | No (confirmed: `discover_subjects` returns 0 subjects against the un-extracted root, both this session and the prior audit) | **Yes** -- verified random access at every level | No | Metadata-only manifest (`build_manifest_rows_from_data_path_zips`) + `archive_access_mode="stream"` (default); `node_local_cache` mode available if disk-backed caching is preferred |
| SHARDS_PATH | `train/validation/test/shard-*.tar`, uncompressed, NIfTI directly as tar members, `study.json`/`report.json`/`labels.json` sidecars, `series.parquet`/`studies.parquet` pre-built index | No (same reason) | **Yes** -- verified random access, no nested container at all | No | Adapt `series.parquet` (`build_manifest_rows_from_shards_parquet`) + `archive_access_mode="stream"` (default); `node_local_cache` available |
| Extracted directory (either root, post-download/merge, or any future layout matching `discover_subjects`) | Real files on a real filesystem | **Yes, unchanged** | N/A (already direct) | N/A | Unchanged: `backend="file"`, `build_manifest_rows` |

---

## Why full extraction is unacceptable

Beyond the task's explicit hard constraints, the concrete numbers back this
up: DATA_PATH alone is ~7.5 TB (28 batches, `du`-equivalent from the
directory listing) of per-study zips holding ~705K series; the FAU
`hnvme` workspace inode budget was already independently measured (prior
session, `docs/design/fau_hpc_execution_profile.md` §11) at **59,939/81,920 (≈73%)
of its soft inode limit consumed by pre-existing workspaces**, before this
project writes anything. Extracting even one root to "one file per series"
would need on the order of 2-3 files per series (image + 2 masks) × ~636K-
705K series -- well over a million new inodes, guaranteed to blow through
that already-nearly-exhausted budget (hard constraint #3), independent of
the extra multi-TB duplicate storage it would also consume on a beta,
no-SLA filesystem (`data_workspaces.md`).

---

## Current loader assumptions (traced)

Traced the exact path `manifest builder → image discovery → manifest
image_path → Dataset.__getitem__ → nibabel load → preprocessing`
(`docs/design/06_....md`'s own citations, re-verified against the current
`data_r2v.py`/`data.py`):

| Assumption | Where | Abstracted? |
|---|---|---|
| Discovery walks a real directory tree | `discover_subjects` (`data.py`), called from `build_manifest_rows` | **Kept, unchanged** -- still the only discovery mechanism for `backend="file"`; archive-backed manifests use their own metadata-only builders instead of `discover_subjects` entirely, so this assumption is simply *not exercised* for archive roots, not worked around |
| One `image_path: str`, a real filesystem path | `ManifestRow.image_path`, `MRReportToVolumeDataset.__getitem__` | **Abstracted**: `ManifestRow.locator()` returns a `Locator` (kind="file" wraps the unchanged `image_path`; kind="archive" wraps `archive_path`+`member_chain`) |
| Direct `nib.load(path)` | `data.load_and_resample_nii`, `data.read_native_geometry` | **Extended via extraction, not duplicated**: both were split into a path-independent core (`_resample_canonical`, `_geometry_of_canonical`) shared by new bytes-based siblings (`load_and_resample_nii_from_bytes`, `read_native_geometry_from_bytes`) -- the path-based functions' external behavior is byte-for-byte unchanged (full test suite re-run confirms) |
| Directory iteration / `os.path.exists` checks | `discover_subjects`, `list_nii_files` | **Kept, unchanged** for `backend="file"`; archive-backed rows never call these -- existence is instead determined by attempting the archive read itself (raising a clear `ArchiveReadError` on failure, see "Archive security considerations") |
| One image per file | Implicit throughout | **Still holds** -- an archive member IS one file (or one nested-zip member); no change needed |
| `.npz` cache paths, stable numeric `cache_index` | `preprocess_volumes.py`'s per-study `.npz`, `ManifestRow.cache_index` | **Kept, unchanged, and explicitly scoped out for archives**: `use_preprocessed=True` still requires `backend="file"`-style extracted data (that whole mechanism is built around `discover_subjects`); `cache_index` is simply unused/meaningless for `backend="archive"` rows (documented in `ManifestRow`'s docstring), not repurposed or overloaded |

**No preprocessing logic was duplicated.** `load_and_resample_nii_from_bytes`
and `preprocess_nii_from_bytes` are the bytes-based *siblings* of the
existing path-based functions, sharing their core via extraction
(`_resample_canonical`/`_geometry_of_canonical`), exactly mirroring how
`docs/design/06_....md` already established `read_native_geometry`'s
relationship to `load_and_resample_nii`. RAS reorientation, resampling,
normalization, and crop/pad are each defined exactly once regardless of
storage backend.

---

## Access strategies evaluated

| Strategy | Verdict |
|---|---|
| Direct tar-member streaming | **Chosen** (default, `archive_access_mode="stream"`) -- verified fast (tens of ms isolated, ~0.3-5s real cold/warm; see Performance), zero disk footprint, zero new inodes, trivially satisfies every hard constraint |
| Direct WebDataset iteration (an `IterableDataset`/`webdataset`-library-style sequential reader) | **Rejected** -- would abandon this pipeline's existing map-style `Dataset` contract (random-access `__getitem__`, needed for `series_selection` modes, modality-balanced sampling, and arbitrary train/val/test filtering by index) for no benefit: SHARDS_PATH's own grouping is by directory-prefix, not extension-sharing basename, and plain `tarfile` random access is already fast without adopting a new library dependency |
| Indexed random access to tar members | **Chosen, implicitly** -- this is exactly what `tarfile.getmembers()` + the process-local handle cache provides; no separate index format needed beyond each root's own existing metadata (see "Manifest/index design") |
| Reading NIfTI bytes into a seekable in-memory object | **Chosen** (this is "stream" mode) |
| Spooling only one selected NIfTI to node-local storage | **Chosen** as the `node_local_cache` alternative -- granularity discussion below |
| Spooling one per-study ZIP to node-local storage, then reading selected members | **Rejected** -- would materialize series a given run's `series_selection` mode doesn't even use (e.g. `one_per_study_deterministic` touches 1 of N series/study); wastes cache budget vs. per-series granularity for no access-pattern benefit |
| Extracting one selected study into a bounded node-local cache | **Rejected**, same reasoning as above (study granularity over-materializes relative to a series-level Dataset's actual access pattern) |
| Converting archives to another permanent shard format | **Rejected** -- this *is* "resharding," explicitly discouraged unless every non-rewriting approach is unsuitable (none were) |
| Building only a lightweight archive-member index without rewriting image data | **Chosen**, but as an *adaptation* of each root's own pre-existing metadata (DATA_PATH's `metadata.tar.gz`+`splits.csv`; SHARDS_PATH's `series.parquet`) rather than a new index built by scanning archives -- see "Manifest/index design" |

**Full-study-ZIP in-memory materialization was explicitly avoided** for
DATA_PATH's nested case: `ArchiveReader` opens `zipfile.ZipFile` directly on
the *tar member's own seekable file object* and reads only the target inner
member (verified 2ms central-directory read + <1ms targeted member read on
a 53 MB nested zip) -- never materializing the whole nested zip in memory,
which matters for the documented outlier studies with up to 83 series (a
zip that size could be very large). No configurable size-check-before-full-
read was needed for this path specifically because it never does a
whole-zip read in the first place; the size-check principle is still
respected where it *is* relevant -- the node-local cache path writes exactly
the one target member's bytes (bounded by the cache's own byte budget), not
an unbounded read.

---

## Selected storage abstraction

**`Locator`** (`r2v_storage.py`): the smallest representation that covers
every case found -- `kind="file"` (unchanged extracted-path contract) or
`kind="archive"` with `archive_path` + a `member_chain` tuple (length 1 for
a NIfTI stored directly as a tar member; length 2 for a NIfTI nested inside
a per-study ZIP that is itself a tar member). One general "chain of
containers" shape was chosen over a separate class per nesting depth,
since the resolution logic (open the archive, descend through each chain
segment) is identical regardless of depth and generalizes without a new
type if a third format ever nests one level deeper.

**`ArchiveReader`**: resolves a `kind="archive"` `Locator` to raw bytes.
Caches the *outer* `TarFile` handle per `(pid, archive_path)` (never the
nested zip handle -- each `read_bytes` call is self-contained and
sequential, avoiding any interleaved-read hazard on the outer tar's shared
file descriptor). The pid-keyed cache is what makes this safe across
`fork()`-based DataLoader workers (see "Multi-worker and distributed
behavior"). `Locator`/`ArchiveReader` never touch report text or image
voxel data -- they resolve *where bytes are*, not what they mean.

**`ManifestRow`** gained three new fields (`backend`, `archive_path`,
`member_chain`), all with defaults, so every manifest CSV
`docs/design/06_....md` already produces still loads unchanged
(`from_csv_row` treats a missing `backend` column as `"file"`). A
`locator()` method is the one place that translates the persisted fields
into the runtime `Locator` -- the seam a future challenge adapter (per
`docs/design/06_....md`'s "Challenge-adapter boundary") could also target,
by constructing `ManifestRow`s directly rather than the on-disk CSV.

---

## Manifest/index design

**No new index format was invented for either root** -- both builders are
thin adapters into the *existing* `ManifestRow`/CSV persistence layer:

- `build_manifest_rows_from_data_path_zips(data_root, metadata_source,
  splits_csv, ...)`: iterates `splits_csv` (must carry `batch_id`, which the
  real released `splits.csv` does) and `metadata_source`'s per-study series
  list, constructing each locator via the documented, verified filename
  convention (`_data_path_member_chain`) -- **opens zero archives**.
  `verify_archive_locators_sample(rows, n=20)` is a separate, explicit,
  bounded correctness check (never run implicitly / never run at full
  scale without being asked) that resolves a small random sample against
  the real archives and reports pass/fail counts + redacted failure
  reasons, never raw identifiers.
- `build_manifest_rows_from_shards_parquet(shards_root, ...)`: reads
  `series.parquet`'s existing `shard_name`+`tar_member_path`+QC columns
  directly (via `pyarrow`, imported lazily -- see "Configuration
  reference"), filters by `image_present` and the same eligibility policy
  (`is_eligible`) as every other manifest source, and maps its `split`
  column's canonical value to the real (differently-named) directory.
  **Opens zero shard tars.**

Both are deterministic (pure functions of the source metadata's own
content) and require no incremental checkpointing for the same reason
`docs/design/06_....md`'s original `build_manifest_rows` doesn't: re-running
against unchanged metadata reproduces the same rows. `native_shape`/
`native_spacing_mm` are left `None` in both (would require opening every
eligible series' bytes at index time, which the task explicitly asks to
avoid) and are resolved lazily, once, from the same bytes `__getitem__`
already reads for training -- never during index/manifest construction.

Neither report text nor image voxel data is ever stored in the manifest
CSV, matching the persistent-index requirements.

---

## Node-local cache design

**Granularity: individual, gzip-compressed NIfTI** (not per-study ZIP, not
whole-shard, not VAE latent -- a generative model's latent doesn't exist
yet, per `docs/design/06_....md`'s explicit non-goal of touching model
conditioning). Chosen because this Dataset is series-level: a study's
non-requested siblings (under `one_per_study_*` selection, or simply not
yet visited under `"all"`) should not occupy cache budget, and the *raw*
compressed bytes (not the decompressed/preprocessed tensor) are cached so
that switching `geometry_mode`/normalizer between runs doesn't invalidate
the cache -- only the *source* changing does.

**Budget**: `CacheBudget(max_bytes, max_files)`, both hard caps, no
unbounded mode. Default 200 GB / 20,000 files (`R2VDatasetConfig`'s
`cache_max_bytes`/`cache_max_files`) -- see "Cache sizing recommendation."

**Eviction**: LRU by file mtime (touched via `os.utime` on every hit),
computed by directory scan -- deliberately not a separate SQLite/database
index for this (small, single-node-scoped) cache, to avoid one more moving
part with its own lock-contention profile; the archive/manifest layer
above it (`series.parquet`, `metadata.tar.gz`+`splits.csv`) is where a
"real" index already exists and is reused, not reinvented.

**Process/lock safety**: a per-key `fcntl.flock` (POSIX advisory lock,
correct across threads *and* processes on one node's local filesystem) is
held for the duration of a miss's fetch+write, so two DataLoader workers
requesting the *same* series block on each other rather than racing to
both materialize it -- verified directly (`test_concurrent_requests_for_
same_key_fetch_once`: two threads request the same key concurrently with a
0.2s artificial fetch delay; the fetch function is confirmed called exactly
once).

**Atomicity**: writes go to `<root>/.tmp/<key>.<pid>.<time_ns>.partial`,
then `os.replace()` (atomic rename) into the final `<hint>_<key>.nii.gz`
name -- no partial file is ever visible under a name any reader would treat
as valid; stale `.partial` files (crash/kill mid-write) older than 1h are
swept on `NodeLocalCache.__init__`.

**Invalidation**: a `.meta.json` sidecar per entry records the source
archive's `(size, mtime)` at population time; a later request re-checks
this against the archive's *current* `os.stat` and treats a mismatch as a
cache miss -- verified directly (`test_source_archive_invalidation`:
changing the source file's bytes after a cache hit forces exactly one
re-fetch).

**Security**: cache filenames are always a sha256 hex digest of the
locator's content (plus a short, non-identifying `hint` like
`"T1w_AXIAL_"`), never a raw study/series identifier; `member_chain`
segments are checked for `..`/absolute-path components before ever being
handed to `tarfile`/`zipfile` (`_reject_path_traversal`), even though the
only realistic way to hit this is a corrupted or maliciously hand-edited
manifest, not the normal build path.

**Counters**: `NodeLocalCache.stats` (`hits`, `misses`, `evictions`,
`bytes_materialized`) -- integers only, no keys, safe to log verbatim.

---

## Verified FAU cache location and lifecycle

Re-read directly this session (`docs/nhr_official_docs/{data_filesystems,
clusters_helma,data_staging,data_workspaces}.md`), plus a live check on the
actual login node this session is running on:

| Question | Answer | Confidence |
|---|---|---|
| Exact variable | `$TMPDIR` (both `data_filesystems.md:14,45-54` and `clusters_helma.md:27-29` name it explicitly; `resolve_node_local_root` checks `$TMPDIR` then `$TMP` by default, both configurable) | VERIFIED |
| Node-local? | Yes -- "Node-local job-specific directory" (`data_filesystems.md:14`); on Helma specifically, a dedicated 15 TB NVMe SSD *per node* (`clusters_helma.md:11-12,29,122`) | VERIFIED |
| SSD-backed? | Yes on Helma (and Alex/TinyFat/TinyGPU/Woody); **Fritz's `$TMPDIR` is node-local RAM disk instead**, reducing available application RAM there -- a real caveat if this pipeline ever runs on Fritz (`data_filesystems.md:51`) | VERIFIED |
| Login vs. compute vs. job-only? | **Job-scoped only** -- "automatically created and removed" within SLURM jobs (`data_filesystems.md:47`); **directly confirmed live this session**: on a Helma login node (`hostname -f` → a Helma frontend), with no `$SLURM_JOB_ID` set, `echo $TMPDIR`/`echo $TMP` are both **unset** | VERIFIED (doc + live check, not doc-only) |
| Capacity | 15 TB/node (Helma) | VERIFIED |
| Inode limit | **Not documented anywhere in the local FAU docs** | **UNKNOWN** -- `R2VDatasetConfig.cache_max_files` defaults conservatively (20,000) rather than assuming a large or unlimited inode budget |
| Lifecycle/auto cleanup | Created at job start, deleted at job end, automatically, no user action needed (`data_filesystems.md:47`, `clusters_helma.md:29`) | VERIFIED |
| Multiple tasks/jobs sharing it on one node | **Within one job**: yes, trivially (`$TMPDIR` is one path for that job's whole process tree, including every DataLoader worker -- this is exactly how the node-local cache is shared across workers). **Across two independent SLURM jobs landing on the same node**: **UNKNOWN/ambiguous** -- `data_staging.md`'s own "Sharing Data Across Concurrent Jobs" section describes a *manual*, lock-file-based pattern under `/tmp/$USER-$JOB_CLASS` (not `$TMPDIR` itself) for exactly this case, and explicitly warns "Slurm does not guarantee concurrent job placement on the same node" -- this design does not rely on or attempt cross-job cache sharing at all (each job's cache is scoped to `$TMPDIR`, whatever that resolves to for that job) | Doc: ambiguous; this design: doesn't depend on the answer |
| Does SLURM allocate it automatically? | Appears to, per the docs (no `--tmp=`-style request flag is shown in any FAU example script) | INFERRED (no FAU doc explicitly says "no request needed," but none shows one being made either) |
| Must a job request local temp capacity? | Not shown as necessary in any FAU example | INFERRED, not fully confirmed |
| Multi-node behavior | Each node has its **own, independent** `$TMPDIR` -- no doc suggests any cross-node pooling/sharing mechanism, consistent with "node-local" | VERIFIED by description, not by a live multi-node test (this session ran no jobs, per hard constraint #11) |
| Recommended cleanup | Rely on `$TMPDIR`'s own automatic deletion as the backstop; this code additionally does its own bounded eviction *during* the job (so a long job doesn't creep toward using the whole 15 TB) and stale-partial sweeping at `NodeLocalCache` construction -- but performs no explicit "delete everything at process exit" step, since the scheduler's own cleanup already guarantees that | This design's choice, consistent with the doc-verified lifecycle |

**Fail-safe behavior, verified by test**: `resolve_node_local_root()` raises
`NodeLocalRootError` (never silently substitutes `$WORK`/`$HOME`/an `hnvme`
workspace) if neither `$TMPDIR` nor `$TMP` is set to an existing directory
-- directly tested (`test_raises_when_no_env_var_set`,
`test_never_returns_a_default_persistent_path`) and directly exercised via
the Dataset (`test_node_local_cache_mode_fails_loudly_without_valid_root`).

---

## Cache sizing recommendation

```
required cache working set
  ≈ workers_per_node × prefetch_factor × max_concurrently_materialized_samples
    × estimated_compressed_bytes_per_series
```

Using this session's own measured real-archive sizes (per-series compressed
`.nii.gz`, DATA_PATH sample: 3.1-4.3 MB img members in the smallest sampled
zip, up to ~120 MB middle-sized outer zip members overall; prior audit's
p95/p99/p100 compressed-size percentiles: 34.6/63.1/167.6 MB) --
**budget for ~100-150 MB/series compressed, not the decompressed/float32
size**, since the cache stores compressed bytes:

| Scenario | workers/node | prefetch_factor | concurrently materialized | Suggested `cache_max_bytes` |
|---|---:|---:|---:|---|
| Single-GPU dev (`preempt`, Profile 1 in `docs/design/fau_hpc_execution_profile.md`) | 4-8 | 2 | ~16-32 | 5-10 GB |
| Single-node 4-GPU pilot (Profile 2) | 16-24/GPU × 4 GPUs | 2 | ~256-768 | 50-100 GB |
| Default (`R2VDatasetConfig`) | -- | -- | -- | **200 GB / 20,000 files** (deliberately conservative relative to Helma's 15 TB/node, leaving room for checkpoints-in-progress, other application temp files, and the undocumented inode ceiling) |

Do **not** size the cache as a fixed percentage of the node's total 15 TB
`$TMPDIR` -- reserve explicit headroom for checkpoints being written mid-job
and any other application temp usage sharing the same disk, per this
session's FAU-doc review (no FAU doc states a formal reservation
convention, so this is this design's own conservative choice, not a
verified requirement).

Also relevant, not folded into the formula above: decompressed float32
volume size dominates *transient* memory (not cache disk) in "stream" mode
-- for the default per-(modality,plane) geometry buckets
(`docs/design/06_....md`), e.g. T1w axial's `176×240×240` target is 40.6 MB
as float32, comparable to or larger than its compressed source in some
cases; this is ordinary Python/NumPy working memory, already implicitly
bounded by `num_workers × prefetch_factor`, not something this cache design
needs to separately account for.

---

## Multi-worker and distributed behavior

- **Each DataLoader worker is a separate process.** No archive/zip handle
  is ever stored as `Dataset` instance state that gets pickled with
  meaningful content: `ArchiveReader.__getstate__` returns `{}` (verified,
  `test_archive_reader_is_picklable_even_with_open_handles` populates the
  handle cache, pickles the reader, unpickles it, and confirms it still
  works) -- every real handle lives in a **module-level, per-`(pid,
  archive_path)`-keyed cache**, so a `fork()`ed or `spawn()`ed worker always
  either inherits an empty cache (fork, before any access) or gets a fresh
  one (spawn, or fork after the pid check misses), never a shared,
  concurrently-mutated file descriptor.
- **Workers on one node reading the same study**: safe under both storage
  modes -- "stream" mode has no shared mutable state at all (each call is
  independent); "node_local_cache" mode serializes concurrent same-key
  requests via the per-key `flock` (verified, see "Node-local cache
  design").
- **Persistent workers / worker restart**: a restarted worker gets a new
  pid, so it's indistinguishable from a fresh worker to the handle cache --
  no stale-handle risk.
- **Prefetching**: unaffected -- `__getitem__` is a plain, synchronous,
  self-contained call regardless of backend; `DataLoader`'s own prefetch
  machinery works exactly as it does for the extracted-directory backend.
- **Distributed ranks on the same node**: each rank is again a separate
  process; the same per-process handle-cache and per-key cache-file locking
  applies identically across ranks as across plain DataLoader workers --
  no additional rank-awareness was needed or added.
- **Different nodes**: `$TMPDIR` is node-local and NOT shared across nodes
  (see "Verified FAU cache location") -- a series cached on node A's
  `$TMPDIR` is invisible to node B. This design does not attempt any
  cross-node cache coordination (out of scope, and would contradict
  "node-local"); each node independently streams/caches whatever it needs.
  **What must change for multi-node training**: nothing in this code --
  each rank/node already resolves its own `$TMPDIR` independently and reads
  archives directly off the shared `hnvme` filesystem (which *is* reachable
  from every node, unlike `$TMPDIR`); the only operational change is
  ensuring each node's job script still sets a per-node cache budget
  appropriate to that node (the defaults already are, being per-node
  numbers, not aggregate).

---

## Archive security considerations

- **Read-only, always**: every archive is opened with `tarfile.open(...,
  mode="r:")` / `zipfile.ZipFile(fileobj)` in read mode; nothing in
  `r2v_storage.py` ever calls a write/append/delete method on a source
  archive. No test or code path renames, moves, or overwrites a source file.
- **Path traversal**: `_reject_path_traversal` rejects any `member_chain`
  segment starting with `/` or containing a literal `..` path component,
  raising `ArchiveReadError` before any archive is even touched -- tested
  directly (`test_path_traversal_rejected`). In practice this can only be
  reached via a hand-edited or corrupted manifest, since every locator this
  code itself constructs comes from verified metadata, never from
  unsanitized user input.
- **Corrupt/truncated members**: `zipfile.BadZipFile` and `KeyError` (member
  not found) are both caught and re-raised as a clear `ArchiveReadError`
  with a redacted (length-only) description -- never an unhandled traceback
  exposing a raw path, and never silently returning wrong bytes. Directly
  observed on **real** DATA_PATH archives this session (2/20 and 8/15 in
  two different bounded random samples -- see "Test and smoke-test
  results"): the error path correctly identifies genuinely corrupt nested
  zips rather than crashing or hanging.
- **Cache filenames never contain raw identifiers**: sha256-keyed, plus an
  optional short, sanitized (`[A-Za-z0-9_.-]` only) hint -- verified
  (`test_entry_filename_has_no_raw_identifier`).
- **Nothing here executes archive content**: no `eval`, no deserialization
  of untrusted pickle/exec-capable formats -- the only content interpreted
  is NIfTI header/array bytes (via `nibabel`, the same trust boundary the
  extracted-directory backend already has) and, for indexing, plain CSV/
  Parquet rows.

---

## Code changes

- **New**: `contrastive-pretraining/scripts/r2v_storage.py` -- `Locator`,
  `ArchiveReader`, `_ProcessLocalHandleCache`, `ArchiveReadError`,
  `resolve_node_local_root`, `NodeLocalRootError`, `CacheBudget`,
  `CacheStats`, `NodeLocalCache`.
- **New**: `contrastive-pretraining/tests/test_r2v_storage.py` (33 tests).
- **Extended**: `contrastive-pretraining/scripts/data.py` -- added
  `_canonicalize`, `_resample_canonical`, `_geometry_of_canonical`,
  `_looks_gzipped` (internal, shared cores), `load_and_resample_nii_from_bytes`,
  `read_native_geometry_from_bytes`, `preprocess_nii_from_bytes` (public,
  bytes-based siblings). `load_and_resample_nii`/`read_native_geometry`'s
  *external* behavior is unchanged (full pre-existing test suite re-run
  confirms this, see below); their bodies now delegate to the shared cores.
- **Extended**: `contrastive-pretraining/scripts/data_r2v.py` -- `ManifestRow`
  gained `backend`/`archive_path`/`member_chain` (all defaulted, so old
  manifest CSVs load unchanged) and a `locator()` method; new
  `_iter_csv_dict_rows` (feeds both `StructuredReportStore` and
  `MetadataStore`, now `.tar.gz`-capable); new
  `build_manifest_rows_from_data_path_zips`,
  `build_manifest_rows_from_shards_parquet`,
  `verify_archive_locators_sample`; `R2VDatasetConfig` gained
  `archive_access_mode`/`cache_root`/`cache_env_vars`/`cache_max_bytes`/
  `cache_max_files`; `MRReportToVolumeDataset.__init__`/`__getitem__`
  dispatch on `row.backend`, with lazy native-geometry resolution for
  archive rows.
- **Extended**: `contrastive-pretraining/scripts/build_r2v_manifest.py` --
  `--source {extracted_dir,data_path_archive,shards_parquet}` (default
  `extracted_dir`, fully backward compatible with the original flag set),
  plus `--verify_sample`.
- **Extended**: `contrastive-pretraining/tests/test_data_r2v.py` -- new
  classes covering archive/extracted equivalence, lazy geometry, node-local
  cache mode, fail-safe cache-root behavior, batch collation, geometry
  bucketing, and both new manifest builders.

No changes were made to `preprocess_volumes.py`, `MRReportDataset`,
`MRReportDatasetInfer`, `mr_rate_trainer.py`, or any contrastive-training
code path.

---

## Configuration reference

New `R2VDatasetConfig` fields (all optional, all defaulted):

| Field | Default | Meaning |
|---|---|---|
| `archive_access_mode` | `"stream"` | `"stream"` (no disk write) or `"node_local_cache"` |
| `cache_root` | `None` | `None` → auto-resolve via `resolve_node_local_root(cache_env_vars)` on first genuine need; an explicit path is used as-is (must exist) |
| `cache_env_vars` | `("TMPDIR", "TMP")` | Checked in order; never includes a persistent-storage variable by default |
| `cache_max_bytes` | `200 * 1024**3` (200 GB) | Hard cap, see "Cache sizing recommendation" |
| `cache_max_files` | `20_000` | Hard cap; conservative given `$TMPDIR`'s undocumented inode limit |

New `build_r2v_manifest.py` flags: `--source`, `--data_root`,
`--batch_tar_pattern`, `--verify_sample`, `--shards_root` (see the script's
own module docstring for full usage examples).

**Dependency note (pyarrow)**: `build_manifest_rows_from_shards_parquet`
imports `pyarrow.parquet` lazily, inside the function, specifically so that
importing `data_r2v`/constructing `MRReportToVolumeDataset` never requires
it. `pyarrow` is confirmed **not installed** in the `pytorch2.5.1` conda
environment this repo's tests run under, and **is** confirmed installed via
this cluster's system `python3`. Per the task's "do not install a new
dependency without approval," this was **not installed** anywhere this
session -- run `build_manifest_rows_from_shards_parquet` (or
`build_r2v_manifest.py --source shards_parquet`) under an interpreter that
already has `pyarrow` (the system `python3`, confirmed) instead.

---

## Tests and smoke-test results

**Synthetic tests, all passing:**
- `tests/test_r2v_storage.py`: 33 tests -- `Locator` (validation, cache-key
  determinism/uniqueness, redaction), `ArchiveReader` (direct-tar-member and
  nested-zip reads with byte-identical round-trip verification against
  `data.load_and_resample_nii_from_bytes`, missing outer/inner member,
  missing archive, path traversal, corrupt nested zip, unsupported chain
  depth, handle-cache reuse and LRU bounding, picklability with a populated
  handle cache), `resolve_node_local_root` (raises with no env var set,
  resolves an existing dir, skips a nonexistent dir and tries the next var,
  never falls back to `$HOME`/`$WORK`), `NodeLocalCache` (root-must-exist,
  miss-then-hit, no-raw-identifier filenames, LRU eviction, budget-too-
  small error, no visible partial files, stale-partial cleanup on init,
  source-archive invalidation, concurrent-same-key single-fetch via
  threads).
- `tests/test_data_r2v.py` additions: byte-identical equivalence between
  `backend="file"` and `backend="archive"` (`stream` mode) for the same
  underlying NIfTI; `node_local_cache` mode producing the identical tensor
  as `stream` mode plus a real cached file on disk; lazy native-geometry
  resolution for archive rows (confirmed `None` at manifest-build time,
  correct non-zero values after `__getitem__`); fail-loud behavior with no
  valid cache root; batch collation across archive-backed samples;
  `GeometryBucketBatchSampler` treating archive rows identically to file
  rows; `build_manifest_rows_from_data_path_zips` building rows with
  `tarfile.open` monkey-patched to raise (proving zero archives are opened
  during the build), `verify_archive_locators_sample` resolving those rows
  against a real synthetic archive, and a full end-to-end
  `MRReportToVolumeDataset` built from such rows; `build_manifest_rows_from_
  shards_parquet` (via `pytest.importorskip("pyarrow")` -- **skipped in this
  session's test environment**, since pyarrow isn't installed there by
  design; the equivalent logic was separately verified against the real
  `series.parquet`, see below) and its `ImportError` message when pyarrow
  is genuinely absent.
- **Full pre-existing suite re-run**: **236 passed, 1 skipped** (the
  pyarrow-gated test above), 0 failures -- confirms the extracted-directory
  backend, `MRReportDataset`/`MRReportDatasetInfer`, and every test added in
  `docs/design/06_....md`'s task are all still exactly as before.

**Real-archive smoke tests (bounded, sanitized -- no identifiers or report
text printed, only counts/timings/redacted structure):**
- **DATA_PATH, full-scale metadata-only build**: `build_manifest_rows_from_
  data_path_zips` against the *real* `metadata.tar.gz`+`splits.csv`
  produced **705,090 rows in 20.4s**, with a modality distribution (T1w
  36.7%, T2w 27.6%, FLAIR 25.3%, SWI 10.3%) matching the prior audit's
  independently-measured dataset-wide percentages (36/28/25/11%) almost
  exactly -- strong cross-validation that the metadata-only locator
  construction is correct at full scale, without opening a single archive.
- **DATA_PATH, `verify_archive_locators_sample(rows, n=20)` against the
  real archives**: **18/20 resolved successfully**; the 2 failures were
  both `ArchiveReadError: Nested member is not a valid zip (corrupt?)` --
  i.e. the *outer* member (this design's filename-convention assumption)
  was found correctly, and the failure is consistent with the prior audit's
  independently-documented ~0.43-7.9% real corruption rate in this dataset,
  not a bug in locator construction. Wall time: 98.2s (≈4.9s/sample,
  matching the prior audit's own measured per-series rate almost exactly).
- **DATA_PATH, warm-handle-cache follow-up**: 15 more samples drawn from a
  *single already-touched* batch tar (45,396 rows) resolved in 4.34s total
  (≈289ms/sample, ~17x faster than the cross-batch cold-cache rate above) --
  8/15 failures in this particular batch, a real, honestly-reported finding
  (see "Known limitations"), not glossed over.
- **SHARDS_PATH, `series.parquet` schema + locator construction, real
  data, system `python3`**: first pass (15 real locators) returned
  **0/15** -- caught a real bug (`shard_name` lacks the `.tar` extension in
  the parquet, contrary to this design's first assumption). Fixed in
  `build_manifest_rows_from_shards_parquet`; **re-verified 15/15** after
  the fix. This bug and fix are reported explicitly per the task's "report
  what could not be verified" instruction -- it was caught specifically
  *because* a real-archive check was run, not because the synthetic test
  (which had encoded the same wrong assumption) caught it.

## Performance measurements

| Operation | Measured |
|---|---|
| Index (`getmembers()`) a 7.4 GB outer tar (311 members) | 0.53s |
| Index a 592.8 GB outer tar (4,325 members) | 0.96s |
| Random-read the last member of that 592.8 GB tar (55.9 MB) | 27ms |
| Random-read a middle member (119.8 MB) | 50ms |
| Open a nested zip directly on a tar member's file object + read its central directory (53 MB nested zip) | 2ms |
| Read one targeted 5.6 KB inner member from that nested zip | 0.9ms |
| Full real-sample resolve, 20 samples spread across many batches (cold handle cache) | 4.9s/sample |
| Full real-sample resolve, 15 samples within one already-open batch (warm handle cache) | 289ms/sample |
| DATA_PATH full metadata-only manifest build (705,090 rows) | 20.4s |
| SHARDS_PATH real-locator verification (15 samples, post-fix) | a few seconds total (not separately timed) |

Bytes materialized / files-inodes created: 0 in every "stream"-mode
measurement above (by construction); `node_local_cache` mode creates
exactly 2 files (`.nii.gz` + `.meta.json`) per distinct cached series,
verified directly in `test_node_local_cache_mode_matches_stream_mode`.
Peak temporary usage was not separately profiled at scale (no job was
submitted, per hard constraint #11) -- the unit tests verify the *budget
enforcement mechanism* (eviction, hard-cap errors), not a real large-N
measurement.

---

## Known limitations

1. **`$TMPDIR`'s inode limit is undocumented** -- `cache_max_files=20,000`
   is a conservative guess, not a verified ceiling.
2. **DATA_PATH's real per-batch corruption rate varies enough that a small
   random sample can show anywhere from 0% to >50% failures** depending on
   which batch(es) it happens to touch (directly observed this session,
   two different samples: 2/20 and 8/15) -- consistent with, but not a
   precise re-measurement of, the prior audit's per-batch gap findings.
   Production use should filter using `image_present`/QC columns where
   available (SHARDS_PATH's `series.parquet` has these; DATA_PATH's own
   metadata does not directly, so a `data_path_archive`-sourced manifest
   currently has no equivalent pre-filter -- a real, not-yet-closed gap).
3. **No cross-node cache coordination** -- by design (`$TMPDIR` is
   node-local), but worth restating: a multi-node job will independently
   warm each node's cache from cold.
4. **SHARDS_PATH's embedded `report.json`/`labels.json` per-shard sidecars
   are not wired into a new `ReportStore`** -- observed and documented, but
   out of this task's storage-layer scope; a natural, low-effort future
   extension (would let `StructuredReportStore`-equivalent report text come
   directly from the shard, with zero separate reports-CSV dependency for
   that root).
5. **`series.parquet`'s own `shape`/`spacing` columns are not used**,
   deliberately, for the reason given above (unverified axis convention) --
   means archive-backed rows always pay a lazy, per-series geometry
   resolution cost at first access rather than getting it for free from an
   already-computed column, even though such a column exists.
6. **No SQLite/Parquet-backed index was built for the *cache* itself**
   (only for eviction bookkeeping, via plain directory scan) -- acceptable
   at the cache's own bounded scale (tens of thousands of files at most),
   but would not scale to a much larger cache budget without revisiting.
7. **Peak temporary usage under real, sustained multi-worker load was not
   measured** (no job was submitted this session) -- the budget-enforcement
   *mechanism* is unit-tested; a real-scale measurement is future work.
8. **Fritz's `$TMPDIR` is RAM disk, not SSD** -- this design's cache
   defaults (200 GB) would be inappropriate there without reconsideration;
   not a concern for the Helma-centric workflow this and the prior audit
   both anchor on, but worth flagging if this pipeline is ever run
   elsewhere.

---

## Recommended training usage

Configuration and SLURM snippets only -- **nothing below was submitted or
executed as a job this session** (hard constraint #11).

**1. Choosing/verifying the cache root inside a job:**

```bash
#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module load python cuda/12.6.2
conda activate mrrate-r2v

# $TMPDIR is set automatically by SLURM on Helma once inside the job --
# do not hardcode a path; let resolve_node_local_root() find it. This one
# line is enough to sanity-check it before a long training run:
python3 -c "
from r2v_storage import resolve_node_local_root
path, diag = resolve_node_local_root()
print('cache root:', path)
print('diagnostics (no path/content, just filesystem-type heuristic):', diag)
"
```

**2. Configuring cache size / workers / cleanup:**

```python
from data_r2v import R2VDatasetConfig

config = R2VDatasetConfig(
    split="train",
    archive_access_mode="node_local_cache",   # or "stream" for zero disk usage
    cache_root=None,                          # auto-resolve $TMPDIR/$TMP; never guesses a persistent path
    cache_max_bytes=100 * 1024**3,             # 100 GB -- see "Cache sizing recommendation" for your worker count
    cache_max_files=20_000,
)
```

`num_workers` DataLoader workers, and (if applicable) multiple ranks on the
same node, all resolve the *same* `$TMPDIR` for that job automatically
(it's one env var for the whole job's process tree) -- no extra
coordination code is needed; the per-key `flock` inside `NodeLocalCache`
already handles concurrent access safely (verified, see "Tests").
Cleanup: rely on `$TMPDIR`'s own automatic deletion at job end (verified
behavior on Helma); this code's own eviction keeps the cache within budget
*during* the job, it does not need to (and does not) delete anything extra
at exit.

**3. Small smoke test (dry-run, no archive opened):**

```bash
python scripts/build_r2v_manifest.py --source data_path_archive --dry_run \
    --data_root /hnvme/workspace/<acct>-MR-RATE \
    --metadata_csv /hnvme/workspace/<acct>-MR-RATE/metadata.tar.gz \
    --splits_csv /hnvme/workspace/<acct>-MR-RATE/splits.csv
```

**4. Small smoke test with a bounded real-archive check (run on a login
node briefly, or inside a short interactive `salloc` -- this itself is not
a training job and completes in seconds to low minutes per this session's
own measurements):**

```bash
python scripts/build_r2v_manifest.py --source data_path_archive \
    --data_root /hnvme/workspace/<acct>-MR-RATE \
    --metadata_csv /hnvme/workspace/<acct>-MR-RATE/metadata.tar.gz \
    --splits_csv /hnvme/workspace/<acct>-MR-RATE/splits.csv \
    --out_csv /hnvme/workspace/<acct>-r2v-manifests/data_path.csv \
    --verify_sample 20
```

**5. Why archive access should not run from a login node for anything
beyond a small, bounded check:** login nodes are shared, not accounted
against your job allocation, and not intended for sustained I/O or compute
-- the dry-run/`--verify_sample 20`-style checks above are the right scale
for a login node (seconds, per this session's own measurements); a real
training run (sustained per-epoch archive reads across many workers) must
run inside a job, both for policy reasons and because `$TMPDIR` (needed for
`node_local_cache` mode) does not exist outside one.

**6. Multi-node training -- what must change:** nothing in the Dataset/
storage code (see "Multi-worker and distributed behavior"); operationally,
size `cache_max_bytes`/`cache_max_files` per the *per-node* formula in
"Cache sizing recommendation" (these are already per-node numbers, not
aggregate-across-nodes), and expect each node to independently warm its own
cache from cold at job start.

---

## Source-code index

- `contrastive-pretraining/scripts/r2v_storage.py` -- `Locator` (dataclass +
  `cache_key`/`redacted`), `_ProcessLocalHandleCache`, `_HANDLE_CACHE`,
  `ArchiveReadError`, `_reject_path_traversal`, `ArchiveReader`
  (`__getstate__`/`__setstate__` for safe pickling, `read_bytes`),
  `resolve_node_local_root`, `NodeLocalRootError`, `CacheBudget`,
  `CacheStats`, `NodeLocalCache` (`get_or_materialize`, eviction/invalidation
  internals).
- `contrastive-pretraining/scripts/data.py` -- `_canonicalize`,
  `_resample_canonical`, `_geometry_of_canonical`, `_looks_gzipped`,
  `load_and_resample_nii`/`load_and_resample_nii_from_bytes`,
  `read_native_geometry`/`read_native_geometry_from_bytes`,
  `preprocess_nii`/`preprocess_nii_from_bytes`.
- `contrastive-pretraining/scripts/data_r2v.py` -- `ManifestRow` (backend/
  archive_path/member_chain/`locator()`), `_iter_csv_dict_rows`,
  `_data_path_member_chain`, `build_manifest_rows_from_data_path_zips`,
  `_SHARDS_SPLIT_TO_DIR`, `build_manifest_rows_from_shards_parquet`,
  `verify_archive_locators_sample`, `R2VDatasetConfig` (archive/cache
  fields), `MRReportToVolumeDataset.__init__`/`_node_local_cache_for`/
  `_read_archive_bytes`/`__getitem__`.
- `contrastive-pretraining/scripts/build_r2v_manifest.py` -- `--source`
  dispatch, `_dry_run_data_path_archive`.
- `contrastive-pretraining/tests/test_r2v_storage.py`,
  `contrastive-pretraining/tests/test_data_r2v.py` (archive-related classes
  appended at the end of the latter).
- `data-preprocessing/src/mr_rate_preprocessing/mri_preprocessing/
  modality_filtering.py:311-319` (`load_image_properties` -- the
  pre-reorientation axis-order finding this design relies on for both
  DATA_PATH's metadata CSV and, by the same caution, `series.parquet`'s
  `shape`/`spacing` columns).
- `docs/nhr_official_docs/data_filesystems.md`,
  `docs/nhr_official_docs/clusters_helma.md`,
  `docs/nhr_official_docs/data_staging.md`,
  `docs/nhr_official_docs/data_workspaces.md` -- FAU temp-storage
  verification.
- `docs/design/06_report_to_volume_dataset_implementation.md` -- the
  Dataset/DataLoader design this task adapts, entirely reused and
  unmodified in its extracted-directory form.
- `docs/design/mr_rate_local_audit.md`, `logs/mr_rate_audit_metrics.json`,
  `docs/design/fau_hpc_execution_profile.md`, `docs/design/audit_progress.md` -- prior
  audits this task cross-validates against (modality distribution match,
  per-series timing match, inode-budget context, the "known incident" that
  motivated this session's own extra caution around archive-member
  inspection).
