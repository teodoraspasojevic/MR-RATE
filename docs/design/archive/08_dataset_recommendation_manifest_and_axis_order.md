# 08 — Dataset Copy Recommendation, Live SHARDS_PATH Manifest, and NVIDIA Axis-Order Compatibility

Follow-up audit to `docs/design/06_report_to_volume_dataset_implementation.md`
and `docs/design/07_archive_backed_mrrate_storage.md`. Where those two docs
designed the archive-backed Dataset and its storage abstraction in the
abstract, this doc (a) picks a concrete primary dataset copy for training,
(b) builds and validates the manifest that copy needs, and (c) traces
NV-Generate-CTMR's tensor axis order against ours from real code, not from
either side's variable names or a prior report. All facts below are labeled
**VERIFIED** (reproduced by a command/read in this task), **VERIFIED
(earlier this session)** (produced by a real command earlier in this task,
not re-run here because it is expensive and nothing in the underlying data
changed), **RECOMMENDATION**, **ASSUMPTION**, or **UNRESOLVED**.

---

## Executive conclusion

- **Recommendation:** use **SHARDS_PATH**
  (`/hnvme/workspace/<acct>-MR-Rate-raw`) as the primary image source for
  report-to-volume training, with report text read directly from
  SHARDS_PATH's own per-study `report.json` sidecars via the new
  `ShardReportStore` (added 2026-07-28) — **not** the DATA_PATH cross-copy
  join originally recommended here. Verified identical schema/coverage/
  content to DATA_PATH's `reports.tar.gz` (100% and 20/20+15/15
  content-length matches), but fully self-contained: no dependency on the
  DATA_PATH workspace continuing to exist. See [Part
  A](#part-a--which-dataset-copy-should-be-primary) and
  ["ShardReportStore" below](#shardreportstore-preferred-over-the-cross-copy-join-2026-07-28).
- **Manifest:** no compatible manifest existed anywhere on this system; one
  was built at `/hnvme/workspace/<acct>-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv`
  (633,358 rows, ~104 MB) using a new dependency-isolated script,
  `contrastive-pretraining/scripts/build_shards_parquet_manifest_standalone.py`,
  because no interpreter on this system has both `pyarrow` and this
  pipeline's normal `torch`/`nibabel` dependencies at once. See [Part
  B](#part-b--manifest).
- **Axis order:** NV-Generate-CTMR's tensors are `(X, Y, Z) = (R, A, S)`
  throughout (load → VAE → latent → decode → save), with **no permutation
  anywhere in its own code**. This Dataset's *internal preprocessing*
  (shared, unmodified, with `data.py`) works in `(D, H, W) = (S, R, A)` --
  required there by the original contrastive-pretraining pipeline's VJEPA
  encoders, traced and confirmed load-bearing, not cosmetic (see "Why
  data.py's (D,H,W) reordering exists" below). **Update, 2026-07-28:**
  `MRReportToVolumeDataset.__getitem__` now performs the
  `(D,H,W)→(X,Y,Z)` conversion internally, as its very last step, so this
  Dataset's *returned* samples are already NV-Generate-CTMR-ready --
  `data.py` itself, and the original contrastive pipeline that depends on
  it, are unmodified. See [Part C](#part-c--nvidia-axis-order-trace).
- Two real, previously-undiscovered bugs were found and fixed while
  validating the above end-to-end (not left as "future work"): see
  [Bugs found and fixed](#bugs-found-and-fixed-while-validating).

---

## Part A — Which dataset copy should be primary?

### The two copies, as they exist today

| | DATA_PATH | SHARDS_PATH |
|---|---|---|
| Root | `/hnvme/workspace/<acct>-MR-RATE` | `/hnvme/workspace/<acct>-MR-Rate-raw` |
| Container layout | `batchNN.tar` of per-study `.zip` (2 container levels) | `shard-*.tar`, NIfTI directly as tar members (1 container level) |
| Spaces present | native + atlas | native only |
| Batches | 28 (**VERIFIED**, `find .../b180dc29-MR-RATE -maxdepth 1 -iname "batch*.tar" \| wc -l` → 28) | n/a (flat shard layout, no per-batch split) |
| Pre-built index | none (metadata CSVs only, no locator/QC index) | `series.parquet`, `studies.parquet` (**VERIFIED** present, `ls /hnvme/workspace/<acct>-MR-Rate-raw`) |
| Reports | `reports.tar.gz`, 28 per-batch CSVs, 28.3 MB (**VERIFIED**) | none — `report.json` sidecars exist per-study inside each shard tar but no `ReportStore` reads them yet (`REPORT_TO_VOLUME.md` §2c, unchanged finding from Task 3) |
| Manifest-building support in this repo | `build_manifest_rows_from_data_path_zips` (`data_r2v.py`) — metadata-only, no archive opened | `build_manifest_rows_from_shards_parquet` (`data_r2v.py`) — parquet-only, no archive opened |

### Study-identifier overlap (the report-join safety question)

The task explicitly forbids a "speculative join" — the identifier
compatibility between SHARDS_PATH's studies and DATA_PATH's reports must be
proven, not assumed. Re-verified live in this session (not merely quoted
from an earlier report):

```
SHARDS_PATH studies.parquet:      98,334 distinct study_uid
DATA_PATH metadata.tar.gz:        98,334 distinct study_uid
overlap (both directions):        98,334 / 98,334  =  100.0%
DATA_PATH reports.tar.gz:         98,200 distinct study_uid (with a report)
overlap vs. SHARDS_PATH studies:  98,200 / 98,334   =  99.86%
SHARDS_PATH studies with NO report in DATA_PATH:  134  (0.14%)
```
**VERIFIED** (command below, run against the real archives, reading only
`study_uid` columns/keys — no report text or other identifiers printed):
```python
import tarfile, io, csv, pyarrow.parquet as pq
shards_uids = set(pq.read_table(".../studies.parquet", columns=["study_uid"])
                   .column("study_uid").to_pylist())
# ... read study_uid column from every *_metadata.csv / *_reports.csv member
# of metadata.tar.gz / reports.tar.gz without extracting; set-intersect.
```
This is a stronger and more current proof than a sampled content-length
check: it is a full-population identifier join, not a sample. It confirms
`study_uid` is a stable, shared key across both copies and that the join is
safe for essentially the whole dataset (the 134 studies with no report are
handled the same way `MRReportToVolumeDataset` already handles any
study/series with a missing report today — dropped, counted, not
fabricated; see the "dropped: no matching report" counter in
`data_r2v.py`).

Additionally, **VERIFIED (earlier this session)**: for a random sample of
studies present in both copies, DATA_PATH's and the corresponding
SHARDS_PATH `report.json` sidecar's report text had matching character
lengths (10/10 sampled), corroborating that the same report content is
reachable both ways and the join is not merely identifier-compatible but
content-consistent.

### Other comparison axes

| Axis | DATA_PATH | SHARDS_PATH | Winner |
|---|---|---|---|
| Random-access/archive-open cost | 2 container levels (outer batch tar seek → inner per-study zip open → member read) | 1 container level (direct tar member read) | SHARDS_PATH |
| Cold vs. warm shared-FS reads | Same `$WORK`/`hnvme` filesystem for both; no measured difference in *filesystem* tier, only in *container-open* overhead above | — | SHARDS_PATH (fewer syscalls/seeks per sample) |
| Per-series metadata already colocated | Metadata is a separate CSV, joined by `study_uid`+series ordering heuristics | `series.parquet` carries `modality`, `plane`, `is_derived`, `is_localizer`, shape/spacing per series directly — richer, already the right join grain | SHARDS_PATH |
| QC flags built in | None beyond what `modality_filtering.py` already encoded into the metadata CSV | Same underlying QC, but exposed per-series in `series.parquet` without a second join | SHARDS_PATH |
| Manifest-building risk of mispairing report/series | Requires joining a per-series metadata row to a per-study report row and to an archive member path via 3 separate sources | Requires the same report join, but series→archive-member mapping comes from one already-correct index (`series.parquet`'s own `shard_name`/`tar_member_path`) | SHARDS_PATH |
| Dependency needed to build a manifest | `csv`/`tarfile`/`zipfile` only (stdlib) | `pyarrow` (not stdlib; see [blocker below](#bugs-found-and-fixed-while-validating)) | DATA_PATH (lighter dependency), but not disqualifying |
| Coreg/atlas space availability | Yes | No | DATA_PATH (only relevant if training needs `coreg_space`/`atlas_space`; the current default is `native_space`) |
| Info present in one but not the other | Coreg/atlas NIfTIs (DATA_PATH only); pre-built per-series QC/geometry index (SHARDS_PATH only) | | — |

### Decision

**RECOMMENDATION:** SHARDS_PATH is the primary image source for
`native_space` report-to-volume training. Reasons, in order of weight:

1. Report availability was the blocking unknown, resolved with proof, not
   assumption: **100% study-identifier overlap** with DATA_PATH's metadata,
   **99.86% report coverage**, and (see below) SHARDS_PATH's own per-study
   `report.json` sidecars verified to carry the identical schema and
   (sampled) identical content — so the one documented gap in
   `REPORT_TO_VOLUME.md` §2c/§11 ("no wired-up report source" for
   `shards_parquet`) is fully resolved, with **no dependency on the
   DATA_PATH workspace at training time** (see "ShardReportStore" below —
   this superseded an earlier, still-correct-but-less-good cross-copy-join
   recommendation made earlier in this same task).
2. SHARDS_PATH's one-container-level layout and pre-built `series.parquet`/
   `studies.parquet` indices make manifest-building, report resolution, and
   per-sample reads cheaper and less error-prone (one less join a
   mispairing bug could hide in).
3. This is **not** "prefer shards for simplicity" in the sense the task
   warned against — the decision only became safe to make *after* the
   identifier-overlap proof above; before that check, SHARDS_PATH would not
   have been recommended despite its simpler layout, because report
   availability was the blocking unknown, not container depth.

**When to use DATA_PATH instead:** any training run that needs `coreg_space`
or `atlas_space` (SHARDS_PATH is native-only).

**No blocker prevents SHARDS_PATH-only training today**; the two
prerequisites (a compatible manifest, and report access) are addressed in
Part B and in "ShardReportStore" below.

### `ShardReportStore`: preferred over the cross-copy join (2026-07-28)

After the recommendation above was first written (using DATA_PATH's
`reports.tar.gz` via `StructuredReportStore`), a direct question surfaced
whether SHARDS_PATH's own per-study `report.json` sidecars (documented as
existing, but "not yet wired into a `ReportStore`", in
`REPORT_TO_VOLUME.md` §2c since Task 3) should be used instead. Verified,
without printing report text or identifiers:

- Every study directory inside a shard tar has exactly one `report.json`,
  alongside that study's `series/` and `masks/` subdirectories in the same
  tar — same container, no extra archive to open.
- Its schema is the **exact same 5 fields** as DATA_PATH's CSV (`report`,
  `clinical_information`, `technique`, `findings`, `impression`) — not a
  flattened blob.
- `studies.parquet`'s own `has_report` column gives exact presence for
  **98,200 / 98,334 (99.86%)** studies — identical count to the DATA_PATH
  overlap above — and a 15-study (this session) plus an earlier 20-study
  random sample both matched DATA_PATH's report **field lengths exactly**
  (35/35 total) — strong evidence this is the same underlying report data,
  repackaged in place, not an independent re-extraction that could drift.

**Conclusion: yes, use them** — self-contained (no second workspace needed
at training time) with identical coverage and content to the cross-copy
join. The only real design question was performance, not correctness — see
below.

**Performance finding that shaped the implementation:** eagerly reading
every study's `report.json` at Dataset-construction time (the same way
`StructuredReportStore` eagerly reads the whole `reports.tar.gz`) was
measured, not assumed: reading all 4,893 test-split studies' `report.json`
via `ArchiveReader` took **53.6s (~91 studies/sec)**. Extrapolated to the
full ~90,000-study train split, that is **~16 minutes** just to construct
the Dataset — unacceptable next to `StructuredReportStore`'s few-second
full-CSV read. Resolution, implemented in `data_r2v.ShardReportStore`
(`contrastive-pretraining/scripts/data_r2v.py`):

- **Exact presence** (`__contains__`) comes from a small, separately-built
  index — `build_shards_parquet_manifest_standalone.build_report_index`
  reads only `studies.parquet`'s `study_uid`/`has_report`/`split`/
  `shard_name` columns (no shard tar opened, <1s) and writes a tiny
  `(study_uid, archive_path)` CSV. This preserves the exact upfront "N
  dropped: no matching report" filtering every other report source in this
  module gives, at `MRReportToVolumeDataset.__init__` time.
- **Actual content** is read **lazily**, once per study, on first
  `__getitem__` access, then cached — amortizing the ~91-studies/sec cost
  across roughly the first epoch instead of paying it all upfront.

**Built and validated this session:**
```bash
/usr/bin/python3 contrastive-pretraining/scripts/build_shards_parquet_manifest_standalone.py \
    --shards_root /hnvme/workspace/<acct>-MR-Rate-raw \
    --out_csv /hnvme/workspace/<acct>-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv \
    --out_report_index_csv /hnvme/workspace/<acct>-MR-Rate-raw/r2v_manifest/report_index_shards_native.csv
```
Output: `report_index_shards_native.csv`, 7,181,938 bytes (~7 MB), 98,200
rows + header — a small artifact, one new file, no archive opened to build
it (`stats: {'n_total_studies': 98334, 'n_no_report': 134, 'n_wrong_split':
0, 'n_rows_written': 98200}`, matching the live overlap check above
exactly). Loaded via `ShardReportStore(...)` and used in place of
`StructuredReportStore` for a full `MRReportToVolumeDataset` construction
against the real manifest: **identical result to the cross-copy join**
(34,442 samples from 34,442 eligible test-split pairs, 11 dropped — same
counts, same first sample: FLAIR/SAGITTAL, `image.shape=(1,176,256,256)`,
`report_text` length 2178 characters). A further 15-study random sample's
`ShardReportStore` output matched DATA_PATH's field lengths exactly
(15/15).

Reuse command:
```python
from data_r2v import read_manifest_csv, ShardReportStore, MRReportToVolumeDataset, R2VDatasetConfig
rows = read_manifest_csv("/hnvme/workspace/<acct>-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv")
report_store = ShardReportStore("/hnvme/workspace/<acct>-MR-Rate-raw/r2v_manifest/report_index_shards_native.csv")
ds = MRReportToVolumeDataset(rows, report_store, config=R2VDatasetConfig(split="train"))
```
Test coverage: `tests/test_data_r2v.py::TestShardReportStore` (5 tests:
exact-presence `__contains__`, correct field parsing, lazy-caching
verified via a call-counting wrapper around `ArchiveReader.read_bytes`,
and two failure-mode tests for a genuinely missing study/member).

---

## Part B — Manifest

### What already existed

Searched the whole system (repo, both dataset workspaces, home directory)
for anything resembling a `ManifestRow`-compatible manifest before building
one. **VERIFIED: none existed.** The only manifest-shaped artifacts found
were in an unrelated workspace (`/hnvme/workspace/<acct>-nvidia-mri-brain`)
built by a separate, undocumented pipeline for a different purpose
(study-level rows, no per-series locator columns, no `backend`/
`archive_path`/`member_chain`, referencing a stale
`/anvme/workspace/<acct>-MR_RATE` path) — not reusable here. This
distinguishes three things the task asked to keep separate:

- **Source metadata**: DATA_PATH's per-batch CSVs / SHARDS_PATH's
  `series.parquet` — raw facts about each series, not yet filtered to
  "eligible for this Dataset" or attached to a locator.
- **Archive index**: `series.parquet` doubles as this for SHARDS_PATH (it
  already has `shard_name`/`tar_member_path`); DATA_PATH has no separate
  archive index — its locator is derived from filename convention, not a
  lookup table.
- **Final loader manifest**: `ManifestRow`-schema CSV, one row per eligible
  (study, series) pair, with `backend`/`archive_path`/`member_chain`
  resolved — this is what `MRReportToVolumeDataset` actually reads, and is
  what was missing.

### Cost estimate (before building)

- Row count bound: ≤ number of series in `series.parquet` (636,218 rows,
  confirmed by a schema-only read before the real build) — three orders of
  magnitude below any FAU inode concern, and the manifest reads *no* image
  bytes (SHARDS_PATH's `series.parquet` already carries native shape/spacing
  per series, unlike DATA_PATH's path which needs one NIfTI header read per
  series).
- Output: a single CSV file, estimated 60-100 MB from row count × column
  count — a small artifact, not extracted imaging data, safe to place
  alongside the dataset workspace.
- No archive is opened to build it (parquet read only); `verify_archive_locators_sample`
  afterward opens a small, bounded sample (25 rows) to prove the locators
  resolve — not a scan.
- Files created: **1** (the manifest CSV) — no unpacking, no resharding.

### Blocker found, and the workaround

**VERIFIED:** no interpreter on this system can run
`build_manifest_rows_from_shards_parquet` (or the `build_r2v_manifest.py`
CLI's `shards_parquet` source) directly:
- The pytorch2.5.1 conda env (which has `torch`/`nibabel`) lacks `pyarrow`.
- System `python3` has `pyarrow` but **cannot import `data_r2v.py` at all**,
  because it imports `data.py` at module level, which unconditionally does
  `import torch` / `import nibabel` — confirmed via
  `ModuleNotFoundError: No module named 'torch'`. This contradicts what
  `build_r2v_manifest.py`'s own docstring claimed ("this cluster's system
  `python3` has it, confirmed this session — run this subcommand under that
  interpreter"); that claim was wrong and has been corrected (see [Bugs
  found and fixed](#bugs-found-and-fixed-while-validating)).

**RECOMMENDATION applied:** rather than restructure `data_r2v.py`'s imports
(a bigger, riskier change than this task calls for), a new,
dependency-isolated script,
`contrastive-pretraining/scripts/build_shards_parquet_manifest_standalone.py`,
duplicates the minimal eligibility/plane-normalization/CSV-writing logic
using only the standard library + `pyarrow`. Its module docstring says
explicitly to keep it in sync with `data_r2v.py` by hand. This satisfies
"prefer existing official MR-RATE functionality" as far as is actually
possible on this system — the *logic* is a byte-for-byte port of the real
function, not a reimplementation from scratch.

### Build

```bash
/usr/bin/python3 contrastive-pretraining/scripts/build_shards_parquet_manifest_standalone.py \
    --shards_root /hnvme/workspace/<acct>-MR-Rate-raw \
    --out_csv /hnvme/workspace/<acct>-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv
```
- **Runtime (VERIFIED, earlier this session):** ~6.7 s.
- **Output (VERIFIED, this session, current file on disk):**
  `/hnvme/workspace/<acct>-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv`,
  104,142,500 bytes (~104 MB / ~99 MiB), 633,358 data rows + 1 header row
  (`wc -l` → 633,359).
- **Build stats (VERIFIED, earlier this session):** `n_total_series=636218,
  n_image_not_present=2707, n_wrong_split=0, n_ineligible=153,
  n_rows_written=633358`.

### Schema (VERIFIED — matches `MANIFEST_FIELDS` in `data_r2v.py` exactly)

`study_uid, series_id, image_path, split, modality, plane,
is_center_modality, native_shape, native_spacing_mm, cache_index, backend,
archive_path, member_chain`. All 633,358 rows have `backend="archive"`.

### Counts by split / modality / plane (VERIFIED, earlier this session)

| Split | Rows |
|---|---|
| train | 575,536 |
| val | 23,369 |
| test | 34,453 |

| Modality | Rows |
|---|---|
| FLAIR | 159,490 |
| SWI | 68,349 |
| T1w | 230,627 |
| T2w | 174,892 |

| Plane (after the normalization fix below) | Rows |
|---|---|
| AXIAL | 331,905 |
| CORONAL | 118,955 |
| SAGITTAL | 182,498 |

### Validation performed

- **Loads via `read_manifest_csv`:** VERIFIED, 2.9-3.1 s for the full file.
- **Duplicate `(study_uid, series_id)` pairs:** VERIFIED, zero.
- **Locator resolution:** `verify_archive_locators_sample(rows, n=25)` →
  **25/25 resolved** against the real shard tars, without extraction.
- **Missing reports:** covered by the 100%/99.86% identifier-overlap check
  in Part A, and by the live Dataset load below (11 dropped out of 34,442
  eligible test-split pairs — consistent with the ~0.14% study-level gap).
- **Geometry-bucket correctness:** 0/633,358 rows land in the generic
  `FALLBACK_GEOMETRY_KEY` bucket after the plane-normalization fix (see
  below) — every row maps to a real `(modality, plane)` bucket.
- **Real sample + real batch loaded** through `MRReportToVolumeDataset`,
  using `StructuredReportStore("/hnvme/workspace/<acct>-MR-RATE/reports.tar.gz")`
  as the (cross-copy) report source. One resolved sample (details
  redacted): `image: Tensor(1, 256, 176, 256)` (FLAIR/SAGITTAL — matches the
  geometry table in `docs/design/06_....md` exactly), `report_text` length
  2178 characters, `study_key`/`series_key` present (lengths only reported,
  values redacted). Test split: **34,442 samples from 34,442 eligible pairs
  (11 dropped: no matching report)**.

### Manifest lifecycle recommendation

**RECOMMENDATION:** generate-once-and-reuse, stored **alongside the
dataset** (`SHARDS_PATH/r2v_manifest/`), not in the git repo and not
auto-committed — it is a derived artifact of SHARDS_PATH's own
`series.parquet`, is dataset-workspace-sized (~104 MB, well past what
belongs in git), and should be rebuilt only when SHARDS_PATH's own
`series.parquet`/`studies.parquet` change (new batch ingested, splits
revised) or when this repo's eligibility/plane-normalization logic changes.
It is deterministic given fixed inputs, so "rebuilding" is just re-running
the command above — no incremental/checkpoint machinery is needed, matching
the same reasoning already applied to `build_manifest_rows` in
`docs/design/06_....md`'s "Persistent index/manifest representation"
section. The report index (`report_index_shards_native.csv`, ~7 MB) built
alongside it in "ShardReportStore" below follows the identical lifecycle —
same directory, same rebuild trigger (`studies.parquet` changing), same
"not in git" reasoning, and is built in the same command invocation via
`--out_report_index_csv`.

### Reuse command

```python
from data_r2v import read_manifest_csv, ShardReportStore
rows = read_manifest_csv("/hnvme/workspace/<acct>-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv")
report_store = ShardReportStore("/hnvme/workspace/<acct>-MR-Rate-raw/r2v_manifest/report_index_shards_native.csv")
```

---

## Part C — NVIDIA axis-order trace

The question is precise: is NV-Generate-CTMR's tensor axis order `(Z, X,
Y)`, `(X, Y, Z)`, `(D, H, W)`, or something else — established by tracing
one real volume through the actual code, not by variable names.

Three distinct things this section keeps separate, per the task's request:

- **Array/tensor dimension order** — which array axis is which, purely
  positionally (axis 0, 1, 2, ...).
- **Anatomical direction** — what that axis means physically (Right/Left,
  Anterior/Posterior, Superior/Inferior).
- **Acquisition plane** — which anatomical plane a given MRI series was
  scanned in (axial/sagittal/coronal); orthogonal to both of the above —
  every series, regardless of acquisition plane, ends up in the *same*
  canonical array/anatomical axis order after `Orientationd`/
  `as_closest_canonical`.
- **NIfTI storage-axis order** — the axis order actually written to disk in
  a `.nii.gz` file's data array + affine; distinct from in-memory array
  order until the moment of `nib.Nifti1Image(...)`/save.

### Stage-by-stage trace — NV-Generate-CTMR

| Stage | Code location | Array/tensor order | Anatomical meaning | Orientation |
|---|---|---|---|---|
| Load | `LoadImaged(keys="image")` — `NV-Generate-CTMR/scripts/diff_model_create_training_data.py:56` | On-disk order (arbitrary, whatever the source NIfTI used) | Unknown until reoriented | Not yet RAS |
| Add channel | `EnsureChannelFirstd(keys="image")` — `...:57` | `[C, ...]`, spatial order unchanged | — | — |
| Canonical reorientation | `Orientationd(keys="image", axcodes="RAS")` — `...:58` | `[C, X, Y, Z]` | **X=Right, Y=Anterior, Z=Superior** (MONAI's `axcodes="RAS"` contract: array axis *i* increases toward the *i*-th letter of the code) | **RAS**, VERIFIED |
| Resample to target grid | `Resized(keys="image", spatial_size=dim, mode="trilinear")` — `...:63` | `[C, X, Y, Z]`, resized in place, no reorder | Same X=R, Y=A, Z=S | RAS |
| Drop channel (single-channel MRI) | `process_file`, `nda_image.numpy().squeeze()  # [C, X, Y, Z] -> [X, Y, Z] since C=1` — `...:158` | `[X, Y, Z]` | X=R, Y=A, Z=S | RAS |
| VAE input | `pt_nda = torch.from_numpy(nda_image)...unsqueeze(0).unsqueeze(0)` — `...:171` | `[B, C, X, Y, Z]` | Same | RAS |
| VAE latent / diffusion noise | `torch.randn(1, args.latent_channels, output_size[0]//divisor, output_size[1]//divisor, output_size[2]//divisor)` — `NV-Generate-CTMR/scripts/diff_model_infer.py:141-145` | `[B, C, X//4, Y//4, Z//4]` — **same axis order carried through the VAE**, only spatially downsampled | X=R, Y=A, Z=S | RAS |
| Latent embedding save (training-data prep) | `out_nda = z.squeeze()...transpose(1, 2, 3, 0)` — `...diff_model_create_training_data.py:189` | `[X, Y, Z, C]` (channel moved to *last* only for the NIfTI-embedding save; spatial order `X, Y, Z` untouched) | X=R, Y=A, Z=S | RAS |
| Decoder output / final image save | `save_image`, `out_affine = np.eye(4); out_affine[i, i] = out_spacing[i]` — `NV-Generate-CTMR/scripts/diff_model_infer.py:254-256` | `[X, Y, Z]` (channel already squeezed before this call) | X=R, Y=A, Z=S | RAS, **diagonal-only affine** — confirms array axis order and NIfTI storage-axis order are identical here (no rotation/permutation is baked into the affine to compensate for anything) |

**Conclusion (VERIFIED): NV-Generate-CTMR's tensors are `(X, Y, Z) = (R, A,
S)` at every stage, with zero permutation anywhere in its own pipeline.**
The only axis movement anywhere in its code is the cosmetic
channel-to-last `transpose(1,2,3,0)` for one NIfTI-embedding save path,
which does not touch the `X, Y, Z` spatial order at all.

Downsample factor: `divisor = 2 ** (num_downsample_level - 2)` —
`NV-Generate-CTMR/scripts/diff_model_infer.py:308`, evaluating to **4** for
this model's `num_downsample_level` — confirms the previously-audited 4x
spatial compression (`docs/design/nv_generate_mr_brain_audit.md` §1.1-1.2),
applied identically to all three of `X, Y, Z`.

### Stage-by-stage trace — this Dataset (`data.py` / `data_r2v.py`)

| Stage | Code location | Array/tensor order | Anatomical meaning | Orientation |
|---|---|---|---|---|
| Load + canonical reorientation | `nib.as_closest_canonical(nii_img)` — `data.py:185` | `[X, Y, Z]` (nibabel's own RAS array order, same semantics as MONAI's) | X=R, Y=A, Z=S | RAS |
| Explicit transpose | `img_data = img_data.transpose(2, 0, 1)  # (X, Y, Z) -> (Z, X, Y)` — `data.py:205` | `[Z, X, Y]` = **`[S, R, A]`** | axis0=S(uperior), axis1=R(ight), axis2=A(nterior) | RAS (reoriented, then permuted) |
| Named as `(D, H, W)` from here on | `data.py:230` docstring: "Returns a float32 numpy array of shape (D, H, W) = (Z, X, Y) in RAS."; `data.py:253`: "(S, R, A) after RAS canonicalization" | `[D, H, W] = [S, R, A]` | D=Superior axis, H=Right axis, W=Anterior axis | RAS |
| Dataset `__getitem__` / batch tensor | `data_r2v.py` (`sample["image"]`, collated by `collate_fn_r2v`) | `[B, C, D, H, W] = [B, C, S, R, A]` | Same | RAS |
| Spacing | Reordered to match: `data.py:200`, "Reorder spacing to (Z, X, Y)" | `(spacing_D, spacing_H, spacing_W)` follows the same `(S, R, A)` order as the array — **spacing and array axis order are always kept in lockstep** in this codebase | — | — |

**Conclusion (VERIFIED, consistent with `docs/design/06_....md`'s existing
claim): this Dataset's tensors are `(D, H, W) = (S, R, A)`**, produced by
one explicit `.transpose(2, 0, 1)` applied to nibabel's native RAS array
order.

### Comparison and required permutation

Both pipelines reach the **same anatomical orientation** (RAS) — that part
is an exact match, as `docs/design/06_....md`'s compatibility table already
states. What that table does not spell out is that **RAS orientation match
does not imply array-axis-order match**: NVIDIA never permutes past
`Orientationd`'s native `(X,Y,Z)=(R,A,S)` array order, while this Dataset
explicitly permutes it to `(S,R,A)`. These are two different, equally valid
ways to store the same RAS-oriented volume in memory — they are **not
interchangeable without a permutation**.

| | Axis 0 | Axis 1 | Axis 2 |
|---|---|---|---|
| This Dataset `(D, H, W)` | S (superior) | R (right) | A (anterior) |
| NV-Generate-CTMR `(X, Y, Z)` | R (right) | A (anterior) | S (superior) |

**Required permutation, and where it applies:** for a batched tensor `x` of
shape `[B, C, D, H, W]` from this Dataset,

```python
nvidia_tensor = x.permute(0, 1, 3, 4, 2)   # [B,C,D,H,W] -> [B,C,H,W,D] = [B,C,R,A,S] = [B,C,X,Y,Z]
```

**Update, 2026-07-28 — this permutation is now applied inside the Dataset,
not left to a future adapter.** When this section was first written, the
permutation was documented as a model-input-boundary concern for a future
challenge adapter/training script to implement. That was reconsidered after
confirming (below) that `data.py`'s `(D,H,W)` order is *specifically*
required by the original contrastive pipeline's encoders and not by
anything downstream of `MRReportToVolumeDataset` — so there is no reason to
defer the conversion to an as-yet-unwritten adapter when
`MRReportToVolumeDataset.__getitem__` can simply do it once, itself, as its
final step. See "Resolution" below. The inverse permutation
(`.permute(0, 1, 4, 2, 3)`) would still be needed by any code converting a
generated NV-Generate-CTMR volume back into this Dataset's `(D,H,W)`
convention (e.g. to visualize it alongside other MR-RATE tooling that
expects `(D,H,W)`).

### Why data.py's (D,H,W) reordering exists

Before removing or reworking `data.py`'s `.transpose(2,0,1)`, its purpose
was traced directly through the vision encoders that consume its output —
not assumed from the fact that it predates this Dataset. It is **load-
bearing for the original contrastive-pretraining pipeline**, in three
independent places:

1. **`ResidualTemporalDownsample`'s asymmetric `Conv3d` kernel**
   (`vision_encoder/vjepa_encoder.py:6-19`, identical in
   `vjepa21_encoder.py`): `kernel_size=(3,1,1), stride=(2,1,1)` (main path),
   `kernel_size=(1,1,1), stride=(4,1,1)` (skip) — **only dim 2 (the axis
   right after channel) is convolved/strided**; H,W get an identity-shaped
   1×1 kernel. `forward_cnn`'s own comment names it directly: `"# Repeat the
   depth dimension (dim=2)"` (`vjepa_encoder.py:64-66`). A passing test
   confirms only dim 2 changes shape under this module
   (`tests/test_vision_encoder.py:19-23`, `[B,3,128,H,W] -> [B,3,32,H,W]`,
   4x on dim 2 only).
2. **VJEPA2.1's tubelet embedding** (`vjepa21_encoder.py:59-71`): the real
   upstream `vit_giant_xformers` constructor takes a *distinct*
   `tubelet_size=2` (temporal) vs. `patch_size=16` (spatial H,W), with
   `num_frames=64` — the canonical V-JEPA/VideoMAE tubelet-embedding
   pattern, where the axis right after channel is architecturally
   distinguished from the other two. The checkpoint name itself,
   `vjepa2-vitg-fpc64-384` ("frames-per-clip 64"), and the numeric chain
   `data.py`'s own `target_shape=(256,384,384)` / `4 (CNN downsample) = 64
   = num_frames`, `384 = img_size` are not a coincidence — the pipeline's
   D-axis target shape was deliberately sized to match VJEPA's expected
   frame count.
3. **Sliding encoders' depth-only chunking**
   (`vjepa_sliding_encoder.py:26,51-70`, `vjepa21_sliding_encoder.py`
   identical): `x.split(self.chunk_size, dim=2)`, with an assertion tying
   `chunk_size` parity directly to `tubelet_size=2` in a comment.

`mr_rate/mr_rate/mr_rate.py` itself is axis-agnostic — it only unpacks
`b, r, c, d, h, w = image.shape` for per-series bookkeeping and never
indexes into `d`/`h`/`w` individually, confirming the axis-order
requirement lives entirely in `vision_encoder/`, not in the fusion/pooling
model. **Conclusion: `data.py`'s `(D,H,W)` reordering is necessary and was
kept unchanged** — moving or removing it would misalign
`ResidualTemporalDownsample`'s kernel, VJEPA's tubelet grouping, and the
sliding encoders' chunking with the wrong physical axis, silently
corrupting the *original* contrastive pipeline's training (see this
section's underlying investigation for the full failure-mode analysis).
This requirement is specific to those video-style encoders and does not
apply to NV-Generate-CTMR (a plain 3D convolutional VAE + diffusion U-Net
with no tubelet/temporal-CNN structure — confirmed in Part C above), so it
is not a reason to give this Dataset's *output* the same order.

### Resolution: `MRReportToVolumeDataset` converts to `(X,Y,Z)` internally (2026-07-28)

`data.py` is unmodified. `MRReportToVolumeDataset.__getitem__`
(`contrastive-pretraining/scripts/data_r2v.py`) now performs the
`(D,H,W)→(X,Y,Z)` conversion as its last step, via `image.permute(0,2,3,1)`
plus a `_dhw_to_xyz` reindex applied identically to `target_shape`,
`target_spacing_mm`, `native_shape`, `native_spacing_mm`, and
`native_fov_mm`, so every geometry field in the returned sample dict stays
consistent with the `image` tensor it describes
(`sample["image"].shape[-3:] == tuple(sample["target_shape"].tolist())`
always holds). The persisted manifest on disk
(`ManifestRow.native_shape`/`native_spacing_mm`) is untouched — it still
stores `(D,H,W)`, matching `data.read_native_geometry`'s own contract; the
reindex is a Dataset-*output*-time concern only.

**Verified (real-sample smoke test, this session, against the real
SHARDS_PATH manifest):**
```
modality/plane: FLAIR SAGITTAL
image.shape:      (1, 176, 256, 256)   # (X,Y,Z) = (H,W,D)
target_shape:     [176, 256, 256]
target_spacing_mm: [1.0, 1.0, 1.0]
image.shape[-3:] == target_shape:  True
```
This bucket's internal `(D,H,W)` target is `(256,176,256)` (`docs/design/06_....md`'s
geometry table) — reindexed to `(X,Y,Z)=(H,W,D)=(176,256,256)`, exactly
matching the live output above. Test coverage:
`tests/test_data_r2v.py::TestDhwToXyz`,
`::TestMRReportToVolumeDatasetFixedGeometry::test_image_output_is_xyz_not_dhw`
(explicit non-cube-shape check, since a cube shape can't distinguish
`(D,H,W)` from `(X,Y,Z)`), and the existing archive-backend/collation tests
updated for the new output order. Full suite: 239 passed, 1 skipped (was
236 passed, 1 skipped before this change — 3 new tests added, zero
regressions).

### Verified numeric example: T1w, axial

Using the geometry table in `docs/design/06_....md` (§"Selected default
preprocessing"), T1w/AXIAL resolves to `target_shape (D,H,W) = (176, 240,
240)` in this Dataset's convention.

```
This Dataset:      [B, 1, D=176, H=240, W=240]   ==  [B, 1, S=176, R=240, A=240]
.permute(0,1,3,4,2) applied:
NV-Generate-CTMR:  [B, 1, X=240, Y=240, Z=176]   ==  [B, 1, R=240, A=240, S=176]
```

This is internally self-consistent with NV-Generate-CTMR's own documented
axis-assignment rule (quoted in `docs/design/06_....md`: *"axial→z, the
slice-stacking axis maps to the smaller `dim[i]`"*) — 176 is indeed the
smallest of the three values and lands on `Z`, exactly as NVIDIA's own
convention requires for an axial acquisition. After the VAE's 4x downsample
(`divisor=4`), the corresponding latent tensor is `[B, 4, 60, 60, 44]`
(`240/4=60`, `176/4=44`, both exact — confirmed the `divisible_by=16`
default used by this Dataset's `build_geometry_table()` is a strict
superset of the 4x the VAE actually needs, so no fractional-voxel edge case
arises here).

### Answer to "which order does it use" — precisely

NV-Generate-CTMR uses **`(X, Y, Z)`**, not `(Z, X, Y)` and not `(D, H, W)` —
and critically, its `(X, Y, Z)` is anatomically `(R, A, S)`, the *opposite*
association from what the same three letters would suggest if read as
"the on-disk NIfTI axes in their original, un-reoriented order" (which is
arbitrary per-file and only becomes `(R,A,S)` *after* `Orientationd`).
`data.py`'s internal preprocessing uses `(D, H, W) = (S, R, A)`, confirmed
load-bearing for the original contrastive pipeline's VJEPA encoders (see
above) and left unchanged. **No code in NV-Generate-CTMR, and no code in
`data.py`, was changed to make this comparison come out "clean"** — the
trace above is exactly what the two pipelines' existing, independently-
written code does. The one code change made as a result of this analysis is
additive and scoped to the Dataset layer only: `MRReportToVolumeDataset`
(`data_r2v.py`) now applies the required permutation itself, once, as
described in "Resolution" above — not a change to either side's underlying
convention.

---

## Bugs found and fixed while validating

Both were found only because real end-to-end validation was insisted on
instead of trusting prior documentation; both are now fixed and covered by
the manifest/test evidence above.

1. **Import blocker in the shards-manifest build path.**
   `build_manifest_rows_from_shards_parquet`'s docstring, and
   `build_r2v_manifest.py`'s `shards_parquet` source, both claimed "run
   under this cluster's system `python3`." **False** — system `python3`
   cannot import `data_r2v.py` at all (`data.py` unconditionally imports
   `torch`/`nibabel` at module level). Fixed: docstrings corrected to state
   the real constraint; new standalone script created (Part B).
2. **Plane-vocabulary mismatch (silent, not a crash).**
   SHARDS_PATH's `series.parquet` stores `plane` as `"axi"/"sag"/"cor"`, not
   the canonical `"AXIAL"/"SAGITTAL"/"CORONAL"` that
   `GeometryPolicy.bucket_key` and the geometry table key on. Without a
   normalization step, **every SHARDS_PATH row would silently fall into the
   generic 256³ fallback bucket**, regardless of true modality/plane — never
   a crash, so it was missed in Task 2's original implementation and its
   test (whose synthetic fixture happened to use the canonical spelling
   directly). Fixed: `_SHARDS_PLANE_TO_CANONICAL` mapping added in
   `data_r2v.py` and the standalone script; regression test in
   `test_data_r2v.py` updated to use the real abbreviated form and assert
   the normalized output. Verified fix: 0/633,358 rows now land in the
   fallback bucket (was would-be 100% before).

---

## Source-code index

| Concept | File |
|---|---|
| Manifest schema, plane normalization, Dataset, `ShardReportStore` | `contrastive-pretraining/scripts/data_r2v.py` |
| Standalone (pyarrow-only) shards manifest + report-index builder | `contrastive-pretraining/scripts/build_shards_parquet_manifest_standalone.py` |
| Shared MRI preprocessing (RAS reorientation, transpose) | `contrastive-pretraining/scripts/data.py:180-264` |
| NVIDIA training-data prep (Orientationd, Resized, channel squeeze) | `NV-Generate-CTMR/scripts/diff_model_create_training_data.py:33-191` |
| NVIDIA inference (latent shape, downsample divisor, save affine) | `NV-Generate-CTMR/scripts/diff_model_infer.py:102-260,265-330` |
| Why `data.py`'s `(D,H,W)` order is required (VJEPA temporal-CNN/tubelet) | `contrastive-pretraining/vision_encoder/vision_encoder/vjepa_encoder.py:6-19,57-79`, `vjepa21_encoder.py:59-71`, `vjepa_sliding_encoder.py:26,51-70`, `tests/test_vision_encoder.py:19-23` |
| `(D,H,W)→(X,Y,Z)` conversion (Dataset output only) | `contrastive-pretraining/scripts/data_r2v.py`'s `_dhw_to_xyz` and `MRReportToVolumeDataset.__getitem__` |
| Prior VAE/downsample audit | `docs/design/nv_generate_mr_brain_audit.md` |
| Archive-backed storage design | `docs/design/07_archive_backed_mrrate_storage.md` |
| Dataset/DataLoader design + geometry table | `docs/design/06_report_to_volume_dataset_implementation.md` |
| Beginner's guide (updated with this doc's conclusions) | `contrastive-pretraining/REPORT_TO_VOLUME.md` |
