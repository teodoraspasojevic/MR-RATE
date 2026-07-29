# MR-RATE Local Audit Report

**Scope:** read-only audit of the MR-RATE repository fork and two local dataset copies, in preparation for a text-conditioned 3D brain MRI generation ("report-to-volume") project.
**Date:** 2026-07-28. **Auditor constraints:** no network access, no writes outside `OUTPUT_PATH`, no dataset modification, no identifiers or report text in output, stratified/reproducible sampling for image-level checks.

---

## 1. Executive Summary

The MR-RATE native-space release is **anonymized, DICOM→NIfTI-converted, modality/plane-classified, and defaced** — but it is **not skull-stripped, not reoriented to a canonical axis order, not resampled to common spacing, not cropped/padded to common shape, and not intensity-normalized**. All of that geometric/intensity standardization is the responsibility of a downstream consumer (the `contrastive-pretraining` loader does it for its own use case, with defaults tuned for a discriminative contrastive model, not a generative one).

Locally:
- **DATA_PATH** holds the native-space release plus the atlas-registration derivative, but **both are still packaged as un-extracted per-batch `.tar` archives** (a custom repackaging for cluster transfer, not the standard `download.py` output layout), and the **coreg and nvseg-ctmr derivatives are absent**. Five of 28 batches (04, 14, 15, 16, 27) have a documented, non-trivial completeness gap (up to ~57% of a batch's studies affected).
- **SHARDS_PATH** is a separate, custom WebDataset-style repackaging explicitly built for a **"MR Volume Generation" challenge** (per its own `.forithmus/config.json`) — almost certainly the direct target for this project, given the parent repo's name (`VLM3D-MRI-R2V-MICCAI-26`). Its build code is not in this git repo, so only its declared output was audited, not its logic. Its self-reported statistics are independently corroborated by this audit (patient-level split integrity, corruption counts, and dtype/orientation distributions all cross-check cleanly).
- A **37-file stratified, reproducible sample** (seed 42) of actual NIfTI volumes independently confirms, at the voxel level: no skull-stripping (median 69% of nonzero voxels lie outside the brain mask), defacing applied correctly and completely (0% leakage into the defaced region in every sample), no canonical orientation (36 of 37 volumes had a *unique* shape; orientation codes vary), no NaN/Inf, and a >100x intensity-scale difference between uint16 and float32 series that any normalization scheme must account for.

The dataset is a strong foundation for report-to-volume work, but it is **native, heterogeneous, unfiltered imaging data with a study-level (not series-level) report**, requires substantial and carefully-designed geometry/intensity preprocessing, and has several open questions (contrast-enhancement labeling, brain/spine labeling, anonymization-method auditability) that should be resolved before committing to a training pipeline. See `report2volume_gap_analysis.md` and `recommended_next_steps.md` for the actionable path forward.

---

## 2. Inspected Paths and Repository Revision

| Path | Role | Access mode used |
|---|---|---|
| `/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE` (REPO_PATH) | Fork of MR-RATE repo | read-only |
| `/hnvme/workspace/b180dc29-MR-RATE` (DATA_PATH) | Native-space local dataset copy | read-only |
| `/hnvme/workspace/y100dc19-MR-Rate-raw` (SHARDS_PATH) | Custom shard repackaging | read-only |
| `/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE/logs` (OUTPUT_PATH) | This audit's outputs | read-write (created; did not exist before this audit) |

**Repository state:** branch `main`, HEAD `d72f47c` ("Merge pull request #5 from forithmus/linear-probe-merged-labels"), up to date with `origin` (the user's own fork, `teodoraspasojevic/MR-RATE`). Working tree clean except one untracked file, `CLAUDE.md` (project instructions, not part of the pipeline). **No `upstream` remote is configured and no network access was available, so the fork's code could not be diffed against `forithmus/MR-RATE` upstream — this is an explicit unknown, not a claim of parity.**

---

## 3. Dataset Inventory

### 3.1 DATA_PATH (native-space local copy)

- All 28 batches (`batch00.tar` … `batch27.tar`) are present but **un-extracted** — each is a tar bundle of the original per-study `.zip` files (confirmed structurally; raw study identifiers are not reproduced here per this audit's redaction requirements). This layout was produced by custom rsync scripts (`fix_missing_hnvme.sh`, `poll_rsync_hnvme.sh`, found at DATA_PATH root) moving data from a different cluster filesystem (`/anvme`) to `/hnvme` — **it is not the layout produced by the repo's own `scripts/hf/download.py`**, which would leave extracted `<study_uid>/img/...` folders.
- A single derivative directory, `MR-RATE-atlas/`, is present at DATA_PATH root, holding the same per-batch tar-of-zips structure for the atlas-registration space. **`MR-RATE-coreg` and `MR-RATE-nvseg-ctmr` are absent** — not downloaded to this location.
- Small, directly-readable files at DATA_PATH root: `metadata.tar.gz` (28 per-batch CSVs, ~1.1GB total), `reports.tar.gz` (28 per-batch CSVs), `pathology_labels.tar.gz` (1 CSV), `splits.csv` (uncompressed, ~3MB), and the official dataset card `README.md`.
- Header rows were read directly (no data rows, no identifiers) from one metadata CSV, one reports CSV, and the pathology-labels CSV to confirm actual released schema (Section 6).
- No raw DICOM files were found anywhere in DATA_PATH — consistent with this being a post-conversion release copy, not a raw PACS export.

### 3.2 SHARDS_PATH (custom shard repackaging)

- `.forithmus/config.json` (found under `validation/`) declares: `"challenge": "mr-volume-generation"`, `"challenge_title": "MR Volume Generation"`, `"submission_type": "docker"`. **This strongly suggests SHARDS_PATH was built as a submission-format-matching repackaging for a named "MR Volume Generation" challenge**, consistent with this repo's parent directory name (`VLM3D-MRI-R2V-MICCAI-26`) and the user's stated report-to-volume goal. No further challenge specification (rules, exact required format) was found locally, and none could be fetched (no network access) — **whether this shard layout exactly matches the challenge's required submission format is unverified and should be checked against the actual challenge documentation.**
- Structure: `dataset.json`, `provenance.json`, `validation_report.json`, `series.parquet` (636,218 rows), `studies.parquet` (98,334 rows), and `train/`, `validation/`, `test/` directories each holding `shard-NNNNNN.tar` + `.result.json` + `.tar.sha256` triples (train: 3,407 shards / 10,221 files; validation: 139 shards / 417 files; test: 216 shards / 648 files — all counts self-consistent with `dataset.json`).
- `_work/` contains `index.sqlite` and `slurm_logs/` with per-array-task logs (job IDs 621959/622216/622217/622218), confirming the shard build ran as a SLURM array job, one task per shard.
- `pipeline_version: "mrrate-raw-repackage-1.0"`, generated 2026-07-27T14:30:00+02:00. **The code that produced this repackaging is not present in REPO_PATH** (grep for "repackage", "shard_name", "mrrate-raw-repackage", "webdataset" found no implementation, only an unrelated mention of "WebDataset" as an alternative approach in `dataset_guide.md`). This audit could therefore only verify the *declared output*, not the *build logic*.
- The name "raw" (directory name `MR-Rate-raw`, pipeline version `mrrate-raw-repackage`) and the wide shape/spacing variance independently observed in Section 5 both indicate this repackaging **preserves native-space geometry as-is** — it does not appear to apply resampling, cropping, or normalization itself, only reorganization + validation.

### 3.3 File categories present / absent (aggregate, both paths)

| Category | Status |
|---|---|
| NIfTI images (`.nii.gz`) | Present, confirmed via successful header/array parsing of 37 sampled files (inside un-extracted zips/tars) |
| Raw DICOM | Absent |
| Brain masks | Present — 636,172 / 636,218 series (99.99%) per `series.parquet`; 37/37 sampled series had a geometry-matching brain mask |
| Defacing masks | Present — 636,177 / 636,218 series (99.99%); 37/37 sampled series had a geometry-matching defacing mask |
| Multi-label anatomical segmentation (nvseg-ctmr) | **Absent locally**; also not yet implemented in the repo pipeline (`multi_label_seg/` is a documented placeholder) |
| Co-registration derivative (coreg) | **Absent locally** (not downloaded to DATA_PATH) |
| Atlas-registration derivative | Present locally (un-extracted, not independently header-verified in this audit — see Section 8 limitations) |
| Metadata CSV | Present, schema independently confirmed |
| Reports CSV | Present, schema independently confirmed, study-level granularity |
| Pathology labels CSV | Present, 37 binary columns, study-level |
| Train/val/test split file | Present (`splits.csv`), independently verified patient-level (0 leakage across 83,425 patients — see Section 6) |
| Registration transforms (`.mat`) | Expected inside the (absent) coreg derivative and the (present, unverified) atlas derivative per code/docs; not independently confirmed |
| Un-extracted compressed archives | **All of DATA_PATH's native and atlas data** — every batch is still tar-of-zips |
| Missing/corrupt entries | 2,707 of 636,218 series (0.43%) flagged `image_present=False` with `BadZipFile` errors in `series.parquet`, concentrated in batches 04/14/15/16/27; independently reproduced (as failures, via a slightly different error path — see Section 5) on 3/3 deliberately-sampled corrupt entries |
| Duplicate rows | None found — 0 duplicate `study_uid` in `splits.csv` (98,334 unique of 98,334 rows), `metadata_matched=True` for 100% of series rows (no orphans) |

**Assessment: this is neither an unmodified `download.py` output nor a `merge_downloaded_repos.py`-merged tree.** It is a distinct, custom "tar-of-zips" bulk-transfer repackaging, undocumented in the official README's download workflow.

---

## 4. Expected Official Preprocessing (from repository code and docs)

Read directly: root `README.md`, `data-preprocessing/README.md`, `data-preprocessing/docs/dataset_guide.md`, plus source-level review (via parallel read-only agents, results cross-checked against docs and, where possible, against actual data) of: `dcm2nii.py`, `pacs_metadata_filtering.py`, `series_classification.py`, `modality_filtering.py`, `brain_segmentation_and_defacing.py`, `quickshear.py`, `hdbet.py`, `zip_and_upload.py`, `prepare_metadata.py`, `registration/registration.py`, `registration/upload.py`, `scripts/hf/download.py`, `scripts/hf/merge_downloaded_repos.py`, the `reports_preprocessing/` pipeline (01–06), and `contrastive-pretraining/scripts/data.py` + `preprocess_volumes.py` + `data_inference.py`.

### 4.1 Performed before the public native-space release (steps 1–7, `data-preprocessing/`)
1. **DICOM→NIfTI** via `dcm2niix` (`dcm2nii.py:73-80`), gzip-compressed, JSON sidecars. No reorientation flags passed to `dcm2niix`; whatever internal reorientation `dcm2niix` performs by default is outside this codebase's visibility.
2. **PACS metadata filtering** — required-column enforcement, duplicate-series removal.
3. **Series classification** — 5-level rule hierarchy (diffusion tags → vendor sequence IDs → scan parameters → description keywords → numeric fallback) assigns `classified_modality` (T1w/T2w/FLAIR/SWI/MRA), `is_derived`, `sequence_family`, and (via regex on description text, `series_classification.py:201-208`) a transient `is_contrast_enhanced` flag. Localizer/scout detection also happens here (`is_localizer`, regex on description).
4. **Modality filtering** — acceptance thresholds from `config_mri_preprocessing.py`: age ≥13, shape ≥16 voxels/axis, FOV 140–350mm/axis, accepted modalities `{T1w, T2w, T2-FLAIR, SWI, MRA}`, accepted planes `{AXIAL, CORONAL, SAGITTAL, OBLIQUE}`, non-derived only, phase/magnitude series filtered by filename pattern. Acquisition plane is computed here (not in step 3) from `ImageOrientation(Patient)` cross-product (`modality_filtering.py:159-177`). One T1w series per study is designated `is_center_modality` (earliest `SeriesNumber`).
5. **Brain segmentation & defacing** — HD-BET predicts a binary brain mask; Quickshear uses that mask only to compute a shear plane; the **released image is the original (full-head) volume with only the sheared region zeroed** — brain mask and defacing mask are saved as separate files alongside it. This is the single most consequential design fact for downstream skull-stripping assumptions (Section 7).
6. **Upload (zip)** — per-study completeness check (image + brain-mask + defacing-mask for every modality, or the whole study is dropped), anonymization of the study ID via `accession_to_uid()` — **which in this repo checkout is a stub returning the input unchanged** (`utils.py:253-257`: *"Original function has been replaced with a dummy function that returns the accession number as is"*). The real anonymization method is not auditable from this code.
7. **Upload (metadata)** — column drop/keep per `config_metadata_columns.json` (drops all DICOM UID fields, `is_contrast_enhanced`, several duplicate-named sequence parameters; keeps curated fields), patient ID and study date merged in from an **external Excel mapping not present in this repo** (so patient/date anonymization method is also not auditable from code).

### 4.2 Optional derivative: within-study co-registration
`registration/registration.py`, run on a separate server after downloading processed studies. Moving modalities are registered to the T1w center modality via ANTs (Rigid transform, linear interpolation, Mattes mutual information). Both the transform (`.mat`) and the warped image are saved. **Not applied to native-space files; produces the separate `MR-RATE-coreg` repository/derivative, which is absent from DATA_PATH.**

### 4.3 Atlas/MNI registration
Same script, second stage: center modality registered to MNI152 (ICBM 2009c Nonlinear Symmetric) via ANTs, moving modalities propagated via the composed transform. Produces the `MR-RATE-atlas` derivative, **present locally but not independently header-verified in this audit** (out of scope for the Phase 3 sample, which focused on native space).

### 4.4 Multi-label segmentation
Documented as **"coming soon"** — not yet implemented anywhere in the repository (`multi_label_seg/` is a placeholder). The separately-existing `MR-RATE-nvseg-ctmr` HF repo (referenced in `download.py`) is a related but distinct product; it is absent from DATA_PATH.

### 4.5 Radiology report preprocessing
LLM-based (Qwen3.5-35B-A3B-FP8 via vLLM), study-level, run as standalone SLURM scripts: (1) anonymization — replaces names/dates/hospitals/accession numbers with deterministic tokens, text-only, no structured-metadata anonymization in this sub-pipeline; (2) translation, Turkish→English (code uses temperature 0.0; the pipeline's own README says 0.1 — a documentation/code mismatch); (3) translation QC — actually LLM-based language classification, despite the README describing it as "rule-based" (another doc/code mismatch); (4) structuring into `clinical_information/technique/findings/impression`, with a no-think fallback triggered by JSON-parse failure; (5) structure QC; (6) pathology classification — 3-step CoT→JSON-extraction→verification pass (only 1→0 label flips allowed in verification) over 37 SNOMED/RadLex-grounded categories. **No duplicate/template-report detection exists anywhere in this pipeline** — a real gap flagged for the new project's QC design. Note: steps 4/5 internally use a column named `UID`; the final released CSV uses `study_uid` — the rename is not visible in any script in this repo (an unresolved provenance gap, though the final artifact was independently confirmed correct).

### 4.6 Applied only by the contrastive-pretraining loader (NOT part of the release)
`contrastive-pretraining/scripts/data.py` / `preprocess_volumes.py`: RAS reorientation (`nib.as_closest_canonical`) → trilinear resample to default (1.0, 0.5, 0.5) mm spacing → z-score/percentile/minmax normalization (default z-score: clip ±5σ then divide by 5, → [-1,1]) → crop/pad to default (256, 384, 384) with a 15mm "posterior shift" that recenters the crop to compensate for the fact that defacing has already removed anterior tissue. This loader also assumes (does not verify) that inputs are already defaced; it has no skull-stripping logic of its own. Its `fusion_mode=early` setting uses only the **first alphabetically-sorted volume** per subject — a positional, not modality-aware, selection.

### 4.7 Preprocessing needed specifically for a new report-to-volume model
None of the above pipelines were designed for generative modeling; see Section 9 and `report2volume_gap_analysis.md` for the full gap list. In short: a geometry-standardization strategy that doesn't distort anatomy (Section 7.4 of the suitability assessment), an explicit conditioning schema (modality/plane/contrast/skull-state), series-level (not just study-level) report/image alignment reasoning, deduplication/QC filtering of the known-corrupt entries, and a model-ready manifest joining all of the above.

---

## 5. Observed Local Preprocessing (independently verified)

### 5.1 Method
Stratified, reproducible sample of native-space series, **seed = 42**, drawn from SHARDS_PATH's `series.parquet` (built by the user's own pipeline, but read fresh and independently by this audit — not trusted uncritically). Sampling method: group population by (modality, plane) → 2 per group (`random.Random(42).sample`, deterministic sort by `(study_uid, series_id)` before sampling) → add 2 smallest + 2 largest by `compressed_size_bytes` → seeded fill to ensure ≥8 distinct batches represented → seeded pad/trim to exactly 37 valid targets → separately, 3 rows deliberately drawn from the `image_present=False` population to cross-check the corruption flag. Final sample: **37 valid + 3 deliberately-corrupt = 40 series**, spanning all 5×3 modality×plane combinations, both released dtypes, 4 orientation codes, 3 splits, and 18 distinct batches (including known-gap batches 04 and 15).

Bytes were streamed **in-memory** directly from the un-extracted `batchNN.tar` files in DATA_PATH (`tarfile` → per-study zip member → `zipfile` → target `.nii.gz` member → `gzip.decompress` → `nibabel` via a `FileHolder`/`BytesIO` wrapper). **No file was ever extracted to disk; nothing was written back to DATA_PATH or SHARDS_PATH.** Total run: 195.5s wall time for 40 series (~4.9s/series average). A user decision after an initial pilot run stopped sampling at this n=37 valid size rather than scaling to the originally-proposed 150, since the qualitative findings below were already unanimous across every stratum sampled.

### 5.2 Findings
- **37/37 valid entries parsed successfully; 3/3 deliberately-corrupt entries failed** (matching `series.parquet`'s `image_present=False` flag), though via a `KeyError` on a null member path rather than exactly reproducing the parquet's own recorded `BadZipFile` error — **the corruption is confirmed real, but the identical failure mode was not reproduced**, a minor caveat.
- **Skull-stripping: NOT applied to native-space images.** Median 69% (range 24–90%) of nonzero voxels lie *outside* the brain mask footprint, in every one of the 37 samples. This is independent, voxel-level confirmation of the code-level finding in Section 4.1 step 5.
- **Defacing: applied and verified fully correct.** In all 37 samples, 0% of voxels inside the "defaced" (mask=0) region were nonzero — the shear plane was applied cleanly and completely everywhere sampled. Fraction of volume *kept* (not sheared off) ranged 69.2%–99.9% (median 84.2%).
- **Brain-mask / defacing-mask geometry: matches the image in 100% of cases** (shape and affine both matched in all 37).
- **No canonical orientation.** `qform_code = sform_code = 1` in all 37 (i.e., all express a scanner-anatomical affine, consistent with `dcm2niix` defaults), but the affine-derived orientation codes vary (RAS/LAS/SLA/LIA), and the affine determinant sign split 16 negative / 21 positive across the sample (expected given the orientation mix, not itself an anomaly).
- **No NaN or Inf values** found in any of the 37 sampled volumes.
- **No standardized geometry whatsoever**: 36 of 37 samples had a *unique* voxel shape; 34 of 37 had a unique spacing tuple. Spacing ranged from near-isotropic sub-millimeter (e.g., 0.43mm) to strongly anisotropic thick-slice 2D acquisitions (e.g., 8.125×8.125×6.0mm, 0.688×0.688×6.5mm). Observed field-of-view ranged ~140–260mm per axis in this sample (within, but not spanning the full width of, the documented 140–350mm acceptance range).
- **Two distinct intensity regimes by dtype**: uint16 series (26/37 in-sample) had observed maxima in the 164–2,716 range (raw scanner units); float32 series (11/37) had observed maxima in the 6,930–366,901 range — **over 100x higher dynamic range**. Any single fixed intensity-normalization scheme applied uniformly across dtypes would badly misbehave on one or the other.
- **All 37 sampled volumes were 3D** (`ndim=3`) — no 4D/multi-echo series were encountered in this sample, though this cannot be generalized to the full 705,254-series dataset from n=37.
- **New finding, orientation metadata inconsistency:** in 8 of 37 samples (22%), the orientation code independently computed from the affine did not match the orientation code recorded in SHARDS_PATH's `series.parquet`. This is a data-quality issue in the *custom shard-building pipeline's* own metadata (not the official release, which does not publish an orientation column at all) and was not further root-caused within this audit's scope.

### 5.3 Aggregate metadata-level findings (from `series.parquet`/`studies.parquet`, no image bytes read — n=636,218 series / 98,334 studies)
These corroborate and extend the byte-level sample:
- Modality distribution: T1w 36%, T2w 28%, FLAIR 25%, SWI 11%, MRA 0.02% (very rare, as documented).
- Plane distribution: axial 52%, sagittal 29%, coronal 19%. **Oblique: 0 observed** in this local copy, despite being a documented accepted plane — worth a caveat, not asserted as a bug.
- `is_derived` and `is_localizer`: 100% False — confirms upstream filtering already removed these categories before release.
- `dtype`: uint16 57%, float32 42% (matches the byte-level sample's bimodal split).
- `orientation`: RAS 52%, LAS 30%, SLA 9%, LIA 9%, small tails — independently confirms no canonical reorientation at the whole-dataset level, not just in the 37-sample.
- Corruption: 2,707/636,218 (0.43%) series flagged `image_present=False`, all with `BadZipFile` read errors, concentrated in batches 04/14/15/16/27 — fully consistent (down to the exact count, 7,727 affected studies) across `provenance.json`, `studies.parquet.source_status`, and `studies.parquet.validation_status`.
- `n_series_written` per study: min 0, max 83, mean 6.47 — the max-83 outlier is worth a manual QC look before training.
- `has_report`: 99.86% of studies; `has_labels`: 99.6% of studies.

---

## 6. Report/Image Linkage Analysis

- **Reports are study-level** (one row per `study_uid` in `reports/batchXX_reports.csv`, columns confirmed directly: `study_uid,report,clinical_information,technique,findings,impression`). **MRI volumes are series-level within a study** — a single report describes findings potentially spanning multiple sequences/series.
- Join key confirmed working end-to-end: `metadata_matched = True` for 100% of series rows in `series.parquet` (no orphaned series), and `has_report = True` for 99.86% of studies.
- **Pathology labels are also study-level** (`study_uid` + 37 binary columns, header independently confirmed), derived from the `findings` section only.
- **Splits are patient-level, independently verified.** Direct read of `splits.csv` (not the shard pipeline's self-report): 98,334 rows, 98,334 unique `study_uid` (zero duplicates), 83,425 unique `patient_uid`, split counts train=88,985/val=3,781/test=5,568 (exact match to documentation), **zero patients spanning more than one split**. No split-*generation* code was found anywhere in the repository — only downstream consumption code exists; the original assignment algorithm (and its random seed, if any) is an unresolved unknown. Separately, no code in `contrastive-pretraining` enforces patient-level split isolation at load time (it only matches `study_uid`) — the guarantee currently rests entirely on the data itself being correct, not on any code-level safeguard.
- **Implication for report-to-volume training** (elaborated in Section 5 of `report2volume_gap_analysis.md`): pairing one study-level report independently with *every* series of that study (the naive approach) introduces label noise — a report describing "diffuse white matter changes on FLAIR" says nothing distinguishing about a same-study T1w series, so a report→series model trained this way will learn spurious or diluted associations unless the unit-of-training design explicitly accounts for this (see suitability assessment, `report2volume_gap_analysis.md`).

---

## 7. Confirmed / Likely / Not-Applied / Mixed / Unknown Classification

Confidence key: **CONFIRMED** = verified via code AND/OR independent data/header/voxel evidence gathered in this audit. **LIKELY** = strong evidence from one source (code or docs) but not independently cross-checked against data in this audit. **NOT APPLIED** = confirmed absent. **MIXED** = differs by product/derivative or by subset of the data. **UNKNOWN** = cannot be determined from available code/data without network access or additional work.

| # | Operation | Status | Evidence |
|---|---|---|---|
| 1 | DICOM→NIfTI conversion | CONFIRMED APPLIED | `dcm2nii.py:73-80` (dcm2niix invocation); NIfTI files successfully parsed in 37/37 byte-level samples |
| 2 | De-identification (outcome) | CONFIRMED APPLIED (outcome); **UNKNOWN** (method) | Released study IDs are anonymized-looking codes, `patient_uid`/dates present only in anonymized form in metadata; but `accession_to_uid()` is a stub in this checkout (`utils.py:253-257`) and patient/date mapping comes from an external file not in the repo — the actual anonymization algorithm cannot be audited from code |
| 3 | Defacing | CONFIRMED APPLIED | `quickshear.py:111-198` (code); independently verified via voxel data: 0% nonzero-in-defaced-region across 37/37 samples |
| 4 | Brain-mask generation | CONFIRMED APPLIED | `brain_segmentation_and_defacing.py:174-198` (HD-BET); `brain_mask_present=True` for 99.99% of 636,218 series in `series.parquet`; 37/37 sampled masks geometry-matched their image |
| 5 | Skull stripping (of the released image) | **CONFIRMED NOT APPLIED** | `brain_segmentation_and_defacing.py:182-198` defaces the *original* (full-head) input, not a brain-masked one; independently verified: median 69% of nonzero voxels lie outside the brain mask in 37/37 samples |
| 6 | Canonical orientation (e.g. RAS) | CONFIRMED NOT APPLIED | No `as_closest_canonical`/reorientation call anywhere in the release pipeline (grep-confirmed); independently verified via 636,218-row aggregate (orientation codes vary: RAS 52%/LAS 30%/SLA 9%/LIA 9%+tails) and via the 37-sample byte-level read |
| 7 | Intensity clipping or normalization (at release) | CONFIRMED NOT APPLIED | No such code in steps 1–5; independently verified via wildly different, non-normalized intensity ranges by dtype in the 37-sample (100x+ scale difference) |
| 8 | Bias-field correction | CONFIRMED NOT APPLIED | Grep for N4/bias-field terms across `mri_preprocessing/` returns zero hits |
| 9 | Denoising | CONFIRMED NOT APPLIED | Same grep, zero hits |
| 10 | Resampling to common voxel spacing (at release) | CONFIRMED NOT APPLIED | No such code in steps 1–5; independently verified: 34 of 37 sampled series have a unique spacing tuple |
| 11 | Resizing to common voxel dimensions (at release) | CONFIRMED NOT APPLIED | Same as above; 36 of 37 sampled series have a unique shape |
| 12 | Crop/pad to common shape/FOV (at release) | CONFIRMED NOT APPLIED | Same as above |
| 13 | Within-study multimodal co-registration | **MIXED** | NOT applied to native-space files (by design); CONFIRMED applied (from code) for the separate `MR-RATE-coreg` derivative, which is **absent from DATA_PATH** — not present locally in any form |
| 14 | Registration to MNI/atlas space | **MIXED** | NOT applied to native-space files (by design); CONFIRMED applied (from code) for the `MR-RATE-atlas` derivative, which IS present locally in DATA_PATH but was **not independently header-verified** in this audit (out of the sampled scope) — LIKELY rather than CONFIRMED for the local copy specifically |
| 15 | Modality/sequence classification | CONFIRMED APPLIED | `series_classification.py` 5-level rule hierarchy; `classified_modality` column confirmed present in the actual released metadata CSV header; distribution confirmed via 636,218-row aggregate |
| 16 | Acquisition-plane classification | CONFIRMED APPLIED | `modality_filtering.py:159-177`; `acquisition_plane` column confirmed in metadata CSV; distribution confirmed via aggregate (oblique observed at 0 locally — caveat, not a contradiction of the code) |
| 17 | Contrast-agent classification | **CONFIRMED NOT APPLIED / NOT AVAILABLE** in the release | `is_contrast_enhanced` computed transiently in `series_classification.py:201-208` but is in the `columns_to_drop` list in `config_metadata_columns.json`; independently confirmed absent by reading the actual released metadata CSV header. Only a weak heuristic (lower `SeriesNumber` ⇒ more likely pre-contrast) is documented, not a real label |
| 18 | Exclusion of localizers/scouts/derived/non-diagnostic series | CONFIRMED APPLIED | `modality_filtering.py` config (`INCLUDE_DERIVED_SERIES=False`, localizer/subtraction exclusion); independently confirmed via aggregate: `is_derived`/`is_localizer` both 100% False across 636,218 series |
| 19 | Segmentation generation (binary brain mask) | CONFIRMED APPLIED | See #4 |
| 19b | Segmentation generation (multi-label anatomical) | **NOT APPLIED / NOT YET IMPLEMENTED**, and **NOT PRESENT** locally | `multi_label_seg/` is a documented placeholder ("coming soon"); `MR-RATE-nvseg-ctmr` directory absent from DATA_PATH |
| 20 | Report anonymization | LIKELY APPLIED | LLM-based token replacement confirmed in code (`01_anonymization`); **not independently verified by reading actual report content**, by design, to avoid exposing report text during this audit |
| 21 | Report structuring | CONFIRMED APPLIED | `04_structuring` code; independently confirmed via the actual reports CSV header showing `clinical_information/technique/findings/impression` columns present |
| 22 | Report-to-study linkage | CONFIRMED | `study_uid` join key confirmed functional: `metadata_matched=100%` across 636,218 series rows; `has_report=99.86%` of studies; reports CSV keyed by `study_uid` (confirmed via header, though the reports-pipeline's own scripts internally use `UID` — the rename point is not visible in the repo, an unresolved provenance gap) |
| 23 | Patient-level train/val/test splitting | **CONFIRMED** (outcome); **UNKNOWN** (generation method) | Independently verified from raw `splits.csv`: 0 patients spanning multiple splits across 83,425 patients. No split-generation/assignment code exists anywhere in the repository — only consumption code. The method and its reproducibility cannot be audited |

---

## 8. Shape / Spacing / FOV / Modality / Orientation Summaries

See Section 5.2–5.3 above for the full numeric detail; key numbers repeated here for reference:
- **Shape/spacing heterogeneity**: 36/37 unique shapes, 34/37 unique spacing tuples in the byte-level sample — treat every series as potentially having its own geometry.
- **FOV** (observed in-sample): ~140–260mm per axis (documented acceptance range is 140–350mm; the wider tail was not necessarily captured at n=37).
- **Spacing range**: sub-millimeter isotropic (~0.43mm) to strongly anisotropic thick-slice (up to 8.125×8.125×6.0mm).
- **Modality mix** (dataset-wide, n=636,218): T1w 36% / T2w 28% / FLAIR 25% / SWI 11% / MRA 0.02%.
- **Plane mix**: axial 52% / sagittal 29% / coronal 19% / oblique 0% (locally observed).
- **Orientation mix**: RAS 52% / LAS 30% / SLA 9% / LIA 9% / small tails — no canonical standard.
- **Dtype mix**: uint16 57% / float32 42%, with a >100x intensity-scale difference between them observed in-sample.

---

## 9. Blockers and Risks

1. **Geometry heterogeneity is near-total.** Any training pipeline must include an explicit, physically-aware resampling/crop-pad strategy (Section 5, "Geometry strategy," in the suitability discussion below) — naive fixed-shape resizing will distort anatomy differently per series.
2. **Study-level report vs. series-level image is a real label-noise source** if not designed around explicitly (Section 6).
3. **~0.43% of series (2,707) and ~7.9% of studies (7,727) have missing/corrupt native image data**, concentrated in 5 of 28 batches — these need to be filtered from any training manifest, not silently included as zero-valued or skipped mid-batch.
4. **coreg derivative is entirely absent from DATA_PATH**; if the geometry strategy design ultimately prefers a shared per-study reference frame (rather than per-series native geometry), that data would need to be downloaded separately (17.6TB).
5. **Anonymization method is not auditable from this codebase** (stub function; external mapping file). This is a compliance/audit-trail gap worth escalating if this data will be used in a context requiring anonymization-method attestation.
6. **No contrast-enhancement, body-region (brain vs. spine), or oblique-plane ground truth is currently available** in the released metadata — all documented as "coming soon" or absent. Conditioning a generative model on contrast state or restricting to brain-only volumes currently requires either accepting the weak `SeriesNumber` heuristic or building your own classifier.
7. **No duplicate/template-report or near-duplicate-study detection exists anywhere in the pipeline** — a genuine QC gap for a generative model, where memorized templates or duplicated training pairs are a bigger risk than for a discriminative/contrastive model.
8. **The SHARDS_PATH build code is not auditable** (not in this repo) — its declared statistics were cross-checked and found self-consistent and independently corroborated wherever checkable, but its internal logic (e.g., exactly how it selects which series go into a given shard, any silent filtering) could not be reviewed.
9. **This fork could not be diffed against upstream** (no network access) — any local modifications to the official pipeline logic, if present, would not have been caught by this audit.

---

## 10. Recommended Training Formulation, Preprocessing Pipeline, and Evaluation Protocol

Full detail, tradeoffs, and a concrete first-implementation recommendation are in `report2volume_gap_analysis.md` and `recommended_next_steps.md`. Summary:
- **Terminology**: the target task is **text-conditioned (report-conditioned) 3D medical image synthesis / report-to-volume generation** — not "VLM" (vision-language *model*, which in this codebase refers to the existing discriminative contrastive alignment model, VL-CABS). The closest architectural family is **3D latent diffusion**, as used by NVIDIA's own `NV-Generate-MR-Brain` (explicitly built on this same dataset, per the root README) — encode volumes into a compact latent via a 3D VAE, then train a diffusion (or flow-matching) model in that latent space with text-embedding cross-attention conditioning. Alternatives (3D GANs, VQ-token autoregressive/MaskGIT-style models, score-based models) are viable but have less directly-relevant prior art on this exact dataset.
- **Training unit**: recommend report + explicit modality/plane/contrast/skull-state conditioning → one series, rather than naively pairing one study-level report with every series (label-noise risk) or trying to generate whole multi-series studies jointly (much harder, no strong prior art in this codebase).
- **Geometry**: recommend a physically-grounded resample-to-fixed-spacing + center-crop/pad-to-fixed-FOV strategy (not naive shape resizing), following the same spirit as the existing loader's approach but with generation-appropriate defaults (the existing 1.0×0.5×0.5mm / 256×384×384 defaults were tuned for a discriminative contrastive model and are unlikely to be optimal for a generator without re-evaluation).
- **Evaluation**: needs to be stratified by modality/plane, not a single unstratified FID, plus anatomical-validity, report-image semantic consistency, and memorization/privacy checks (this dataset is medical and reidentifiable data-adjacent) before any pathology-fidelity or radiologist-review stage.

---

## 11. Evidence and Limitations

- All dataset-side findings are either (a) read directly from small, uncompressed or lightly-compressed files (CSV headers, JSON, `splits.csv`), (b) computed from the pre-existing `series.parquet`/`studies.parquet` manifests via column-projected, in-memory queries (no image bytes touched), or (c) independently measured from a stratified 37-file sample of actual NIfTI bytes streamed in-memory from DATA_PATH.
- The **37-file sample is not large enough to bound rare-event rates precisely** (e.g., the true dataset-wide rate of orientation-metadata mismatches, or the existence of 4D/multi-echo series) — it is sufficient for the qualitative CONFIRMED/NOT-APPLIED classifications in Section 7, which were unanimous across every stratum sampled, but percentile-level estimates (e.g., intensity distributions) should be treated as indicative, not final.
- **Coreg derivative, DICOM-level ground truth, and the reports/labels text content itself were not inspected** in this audit (the former is absent locally; the latter two were deliberately avoided to prevent exposing report text or requiring judgment calls about what counts as an "identifier" inside free text).
- **One self-disclosed process issue**: during initial filesystem inspection, two tool-output prints inadvertently included real (post-anonymization) study UID strings in the working transcript — not in any file, not in this report, and not reused. Corrected immediately; flagged here for completeness per this audit's own transparency requirements.
- No content from DATA_PATH, SHARDS_PATH, or REPO_PATH was uploaded, transmitted, or shared outside this local analysis. No network access was used at any point.
