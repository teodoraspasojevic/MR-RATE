# MR-RATE Audit — Progress Checkpoint

Last updated: 2026-07-28 — **AUDIT COMPLETE**. All 6 phases finished and all 5 required deliverables written to OUTPUT_PATH (listed at the bottom of this file). This section records the final state; the detailed phase-by-phase history below (written mid-audit, after the Phase 3 interruption) is preserved as-is for provenance.

Original header (kept for history): session resumed after a tool-call abort during the Phase 3 permission request; OUTPUT_PATH did not exist before this session — nothing from the interrupted run had been persisted to disk.

## How to use this file
This is the durable source of truth across context resets. Before resuming work, read this file in full. If it disagrees with anything reconstructed from conversation memory, this file wins for anything marked "confirmed" — those were derived from actual command output, not recollection.

## Repo state at last check
- Path: `/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE`
- Branch `main`, HEAD `d72f47c` ("Merge pull request #5 from forithmus/linear-probe-merged-labels"), up to date with `origin` (user's own fork `teodoraspasojevic/MR-RATE`, not upstream `forithmus/MR-RATE` — no `upstream` remote configured, and no network access, so code cannot be diffed against upstream).
- Working tree clean except one untracked file: `CLAUDE.md`. Unchanged across the interruption (re-verified 2026-07-28).
- **Do not modify.** Read-only per task constraints.

## Known incident (disclosed to user, contained)
During the first pass, two tool outputs printed real (post-anonymization) study UIDs into the conversation transcript:
1. `tar -tvf batch00.tar | head -40` listed ~39 literal `batch00/<study_uid>.zip` filenames.
2. A path-structure debug print leaked one more study UID because a redaction regex missed an underscore-adjacent occurrence (`\b` treats `_` as a word character).

Containment: nothing was written to OUTPUT_PATH or any file; the leak is confined to that conversation's transcript. Nothing further copied it. **Going forward: never dump raw tar/zip member listings; report counts/patterns only; any redaction script must be tested on a synthetic string before being trusted on real paths.**

## Phase status

### Phase 1 — Repository inspection: COMPLETE
Read directly: root `README.md`, `data-preprocessing/README.md`, `data-preprocessing/docs/dataset_guide.md`.
Delegated to 4 parallel read-only Explore agents (all returned, findings captured below with file:line citations already collected in this session's context — not re-verified on resume, but citations are specific enough to re-check quickly if needed):
- MRI preprocessing pipeline code (dcm2nii.py, series_classification.py, modality_filtering.py, brain_segmentation_and_defacing.py, quickshear.py, hdbet.py, registration.py)
- Upload/metadata/download/merge code (zip_and_upload.py, prepare_metadata.py, download.py, merge_downloaded_repos.py, registration/upload.py)
- Reports preprocessing pipeline (01_anonymization through 06_pathology_classification)
- contrastive-pretraining loader (data.py, preprocess_volumes.py, data_inference.py, split loading)

**Key confirmed facts (code-cited):**
- Native-space release is **defaced, NOT skull-stripped**. HD-BET brain mask is computed and saved separately (`seg/..._brain-mask.nii.gz`); the released image (`img/...nii.gz`) only has the anterior face region zeroed via Quickshear — brain tissue and skull are otherwise intact. Evidence: `data-preprocessing/src/mr_rate_preprocessing/mri_preprocessing/brain_segmentation_and_defacing.py:182-198`, `quickshear.py:111-130,190`.
- **No reorientation, resampling, intensity normalization, bias-field correction, or denoising** anywhere in the official release pipeline (steps 1-5). Confirmed absent by grep and later independently confirmed by header evidence (Phase 2/3 below).
- Acquisition plane derived from DICOM `ImageOrientation(Patient)` cross-product (`modality_filtering.py:159-177`). Contrast-enhancement (`is_contrast_enhanced`) is computed transiently during classification (`series_classification.py:201-208`) but is in the **dropped-columns list** in `config_metadata_columns.json` — confirmed absent from the actual released metadata CSV header (independently verified, see Phase 2).
- **Anonymization algorithm for study/patient IDs is NOT in this repo.** `accession_to_uid()` in `utils.py:253-257` is a stub: "Original function has been replaced with a dummy function that returns the accession number as is." Patient ID / study date anonymization comes from external Excel mappings not present in the codebase. This is a genuine unknown — we can see the *output* (10-char random-looking UIDs) but not the *method*.
- Co-registration and atlas registration are distinct steps; both save real warped images (not just `.mat` transforms) to disk (`registration.py:321-327,502-521`).
- **contrastive-pretraining loader** (not the released data) is where RAS reorientation (`nib.as_closest_canonical`), trilinear resampling to default target spacing (1.0, 0.5, 0.5) mm, normalization (zscore/percentile/minmax, default zscore), and crop/pad to default (256, 384, 384) with a 15mm "posterior shift" (compensates for defacing bias) all happen. `data.py:171-241,335-337`.
- Loader's `fusion_mode=early` uses only the **first alphabetically-sorted volume** per subject — a positional, not modality-aware, choice (`mr_rate/mr_rate/mr_rate.py:272-275`). Important for the "unit of training" design question in Phase 5.
- **No patient-level split enforcement code exists anywhere in the repo** — loader only matches on `study_uid`. The `patient_uid` column in `splits.csv` is carried through but unused by loader logic. (Independently verified the actual data has zero leakage anyway — see Phase 2.)
- Reports pipeline: LLM-only anonymization (text, deterministic tokens, no code-side NER/regex), Turkish→English via Qwen3.5-35B-A3B-FP8/vLLM (temperature actually 0.0 in code vs 0.1 documented in README — a doc/code mismatch), structuring extracts `clinical_information/technique/findings/impression` with a no-think fallback triggered by JSON parse failure, pathology classification is a 3-step CoT→JSON→verify pass over 37 SNOMED/RadLex-grounded categories with only 1→0 flips allowed in the verify step.
- **No duplicate/template-report detection or statistical language-detection exists** in the reports pipeline (language QC is LLM-based, not rule-based as the README claims — another doc/code mismatch). Real gap for the new project's QC design.
- 37→32 (shipped inference set, drops 5 low-agreement pathologies) and 37→14 (`splits_merged_majority`, clinical-group majority vote) label-merging both happen in `contrastive-pretraining/`, not in the reports pipeline itself.

### Phase 2 — Filesystem inventory: COMPLETE (bounded/cheap operations only, no full recursive scans needed — see rationale below)
**DATA_PATH** (`/hnvme/workspace/b180dc29-MR-RATE`):
- All 28 batches are still **un-extracted `batchNN.tar`** files, each a bundle of the original per-study `.zip` files (confirmed via bounded `tar -tvf ... | head`, output NOT to be re-run in raw form — see incident above). This was moved via custom rsync scripts (`/anvme` → `/hnvme`; see `fix_missing_hnvme.sh`, `poll_rsync_hnvme.sh` at DATA_PATH root), not via the repo's `download.py`.
- Only **native** and **atlas** derivatives present locally (`MR-RATE-atlas/batchNN.tar`, also un-extracted, same tar-of-zips pattern, sizes independently listed via `ls -la`). **coreg and nvseg-ctmr derivatives are absent from DATA_PATH.**
- `metadata.tar.gz` (28 per-batch CSVs), `reports.tar.gz` (28 per-batch CSVs), `pathology_labels.tar.gz` (1 CSV), `splits.csv` (uncompressed, 3MB) present at DATA_PATH root — all listed/peeked without extraction.
- DATA_PATH's `README.md` is the official public HF dataset card (matches dataset_guide.md numbers: 705,254 series / 98,334 studies / 83,425 patients / 28 batches / 4 repos, sizes 8.1/17.6/12.3TB/415GB).
- Confirmed metadata CSV header (batch00, read via in-memory tar stream, header row only): columns match the "kept" list from prepare_metadata.py's config, **`is_contrast_enhanced` confirmed absent** (dropped), no brain/spine body-part column yet (matches "coming soon" doc note).
- Confirmed reports CSV header: `study_uid,report,clinical_information,technique,findings,impression` — final artifact uses `study_uid` (the reports-pipeline code itself uses `UID` internally; the rename happens somewhere not visible in the repo — flagged as unknown).
- Confirmed pathology labels CSV header: `study_uid` + exactly 37 pathology name columns, matching the 37 names extracted from `06_pathology_classification` code.
- **Independent verification of splits.csv** (own script, not trusting any pipeline's self-report): 98,334 rows, 98,334 unique `study_uid` (0 duplicates), 83,425 unique `patient_uid`, split counts train=88,985/val=3,781/test=5,568 (exact match to docs), **0 patients spanning multiple splits** — patient-level split integrity CONFIRMED directly from data.
- No dedicated split-generation/assignment code found anywhere in the repo (grep for split-related function defs only found consumption code) — the algorithm that originally produced `splits.csv` is not in this codebase. Flagged as unknown.

**SHARDS_PATH** (`/hnvme/workspace/y100dc19-MR-Rate-raw`):
- `.forithmus/config.json` reveals this is built for a **"mr-volume-generation" / "MR Volume Generation" challenge** (`submission_type: docker`) — given the repo directory name `VLM3D-MRI-R2V-MICCAI-26`, this is almost certainly the target competition for the user's report-to-volume goal. **The shard-building code itself is NOT in this git repo** (grep found nothing) — only its declared output (`dataset.json`, `provenance.json`, `validation_report.json`, `series.parquet`, `studies.parquet`) could be audited.
- `pipeline_version: "mrrate-raw-repackage-1.0"`, generated 2026-07-27T14:30:00+02:00. Repackages native-space zips into WebDataset-style `.tar` shards: train 3,407 shards (10,221 files = shards×3: `.tar`+`.result.json`+`.tar.sha256`), val 139 shards (417 files), test 216 shards (648 files) — all counts independently cross-checked and consistent.
- `_work/` contains `index.sqlite` + `slurm_logs/` with per-array-task logs (job IDs 621959/622216/622217/622218) — confirms this was built via a SLURM array job, one task per shard.
- `provenance.json` documents a **known native-image completeness gap** in batches 04, 14, 15, 16, 27 (e.g. batch14: only 375/3,578 studies intact; batch27: 2,286/5,077) — explicitly attributed to "this local copy's construction," not an upstream MR-RATE issue.
- `series.parquet` (636,218 rows) and `studies.parquet` (98,334 rows) contain **pre-computed per-series header fields** (shape, spacing, orientation, dtype, checksums, presence flags) — read directly via pyarrow (column-projected, in-memory, no image I/O; pandas is NOT installed in the system python3, pyarrow.compute works fine and is what was used).

**Aggregate distributions already computed from series.parquet/studies.parquet (no re-run needed):**
- split: train 578,016 / test 34,673 / val 23,529 series (gap vs. official 705,254 total matches the known-gap batches)
- modality: T1w 231,800 (36%) / T2w 175,061 (28%) / FLAIR 160,490 (25%) / SWI 68,714 (11%) / MRA 153 (0.02%)
- plane: axi 333,282 (52%) / sag 183,267 (29%) / cor 119,669 (19%) / **oblique: 0 observed in this local copy** (docs say oblique is an accepted plane — worth a caveat, not necessarily a bug)
- body_region: 100% null (not yet populated, matches "coming soon")
- is_derived / is_localizer: 100% False (matches docs)
- is_center_modality: True for 89,809 of 98,334 studies (gap = studies whose center modality fell in a gap-affected batch)
- metadata_matched: 100% True (no orphaned rows)
- image_present: True 633,511 / False 2,707 (0.43%) — real corruption, concentrated in gap batches, `source_read_error` = `BadZipFile: Bad magic number for file header` (2,661 image-only, 41 image+both masks, 5 image+brain-mask)
- brain_mask_present: True 636,172 (99.99%) / defacing_mask_present: True 636,177 (99.99%)
- dtype: uint16 364,425 (57%) / float32 269,086 (42%) / null 2,707 (matches missing images)
- orientation: RAS 52% / LAS 30% / SLA 9% / LIA 9% / null 0.4% / small tails (LPS/SLP/LIP/LAI/ILA/ALS, <20 each) — **this is header-derived, independent confirmation that no canonical reorientation is applied dataset-wide**
- studies.source_status: ok 90,607 / empty_source_member 4,063 / missing_source_member 3,653 / corrupt_source_zip 11 (sums to exactly the 7,727 gap count in provenance.json and validation_report.json — self-consistent across all three sources)
- studies.validation_status: ok 90,564 / source_gap 7,727 / ok_zero_series 43 (edge case worth a QC note)
- has_report: True 98,200 (99.86%) / has_labels: True 97,896 (99.6%)
- n_series_written per study: min 0, max 83 (outlier worth a QC flag), mean 6.47
- compressed_size_bytes (image, present only, n=633,511): p0=0.02MB, p5=1.5MB, p25=4.0MB, median=6.9MB, p75=12.8MB, p95=34.6MB, p99=63.1MB, p100=167.6MB, mean=11.7MB, **sum ≈ 7.4 TB**

**Rationale for not running a full recursive `find`/`du` scan:** total byte counts are already known exactly from `ls -la` on the (small number of) large tar files, and per-study/per-series counts are already known exactly from the parquet manifests (built by a prior full pass over the data by the user's own pipeline). A redundant full filesystem walk would cost significant I/O for no new information.

### Phase 3 — Stratified NIfTI/metadata audit: PARTIAL
- **Done (aggregate/metadata-level, cheap, no image bytes read):** all of the above parquet-derived distributions, which cover most of Phase 3's aggregate requirements (shape/spacing/orientation/dtype distributions, mask presence rates, corruption rate) using data the custom pipeline already computed.
- **NOT done:** actual byte-level sample. Phase 3 also requires intensity min/max/percentiles, NaN/Inf presence, zero-voxel fraction, mask-geometry match, and independent cross-check of the parquet's own header claims — none of which exist in the parquet and all of which require opening compressed NIfTI files.
- **This is the exact point of interruption.** A specific plan was proposed and awaiting user approval when the permission-request tool call aborted (`AbortError: Tool permission stream closed before response received`) — the user never saw or answered it in a way that reached me.

**Proposed plan (not yet approved, not yet run):**
- N ≈ 150 series, **seed = 42**, stratified across modality/plane/dtype/orientation/split/is_center_modality/batch-spread/compressed-size extremes, drawn from `series.parquet` rows with `image_present=True` (plus ~3 deliberately-corrupt rows to cross-validate the error flag).
- Read via in-memory `tarfile`/`zipfile` streaming directly from the un-extracted `batchNN.tar` files in DATA_PATH — **no extraction to disk**, nothing written back to DATA_PATH/SHARDS_PATH, only aggregate results saved to OUTPUT_PATH.
- For ~40-50 of those, also pull the matching brain-mask/defacing-mask to check geometry alignment (shape/affine match) and defaced-vs-skull-stripped confirmation via mask statistics — no full image rendering.
- Estimated transient transfer: ~1-5 GB total (based on the actual compressed-size percentiles above), likely a few minutes, exact timing uncertain on shared Lustre.

### Phase 4 — Processing-state classification: NOT STARTED
Most evidence needed already exists in Phases 1-2 findings above; drafting the CONFIRMED/LIKELY/NOT APPLIED/MIXED/UNKNOWN table is mainly a synthesis task, though a few rows (intensity clipping applied at release? denoising?) benefit from the Phase 3 byte-level sample before being finalized as CONFIRMED rather than LIKELY.

### Phase 5 — Report-to-volume suitability assessment: NOT STARTED

### Phase 6 — Deliverables: NOT STARTED
OUTPUT_PATH did not exist before this session. None of the 5 required files have been created yet:
- `mr_rate_local_audit.md`
- `mr_rate_audit_metrics.json`
- `report2volume_gap_analysis.md`
- `proposed_model_manifest_schema.json`
- `recommended_next_steps.md`

## Exact next action (RESOLVED)
~~Ask the user (again) whether to proceed with the Phase 3 byte-level sample~~ — resolved: user chose "reduce scope first." A 40-file pilot (37 valid + 3 deliberately-corrupt) was run (seed 42, in-memory streaming, no disk writes, 195.5s wall / ~4.9s per file). All 37 valid files parsed successfully; all 3 corrupt files failed as expected (though via `KeyError` on a null member path rather than exactly reproducing the parquet's `BadZipFile`, a noted caveat). Findings were unanimous across every modality/plane/dtype/batch stratum sampled (skull-stripping not applied — median 69% of nonzero voxels outside brain mask; defacing applied and 100% correct; no canonical orientation; no NaN/Inf; >100x intensity-scale difference between uint16 and float32 series; 36/37 unique shapes, 34/37 unique spacings). Given this consistency, the user chose to **stop at n=37** rather than scale to the originally-proposed 150, and proceed to Phase 4/5/6.

Scripts used (in scratchpad, not OUTPUT_PATH, since they contain real study/series identifiers in their manifest — never copied into any OUTPUT_PATH file):
- `build_sample_manifest.py` (system `python3`, has pyarrow, no nibabel) — builds the stratified sample from `series.parquet`, writes `phase3_sample_manifest.json` to scratchpad.
- `run_pilot_sample.py` (must run under `/apps/python/3.12-conda/bin/python3` — the system python3 lacks nibabel/pandas; this env has nibabel 5.4.2 + numpy but lacks pyarrow, hence the two-interpreter split) — streams bytes in-memory from DATA_PATH's un-extracted `batchNN.tar` files, parses NIfTI via nibabel `FileHolder`/`BytesIO`, writes aggregate-only (no identifiers) `phase3_pilot_results.json` to scratchpad.

## Phase 4 — Processing-state classification: COMPLETE
Full CONFIRMED/LIKELY/NOT_APPLIED/MIXED/UNKNOWN table for 23 operations written into `mr_rate_local_audit.md` §7 and mirrored machine-readably in `mr_rate_audit_metrics.json.processing_state_classification`. Key results: skull-stripping CONFIRMED NOT APPLIED (independently verified at voxel level), defacing CONFIRMED APPLIED (100% correct in-sample), canonical orientation CONFIRMED NOT APPLIED, contrast-agent classification CONFIRMED NOT APPLIED/NOT AVAILABLE, anonymization method UNKNOWN (stub function in repo), split-generation method UNKNOWN (no code in repo, but outcome independently CONFIRMED clean).

## Phase 5 — Suitability assessment: COMPLETE
Terminology clarified (report-to-volume / text-conditioned 3D synthesis, closest architecture = 3D latent diffusion per NVIDIA's `NV-Generate-MR-Brain` built on this same dataset — explicitly not "VLM" in the discriminative sense already implemented in `contrastive-pretraining/`). Full discussion of unit-of-training, report/image alignment noise, sequence selection, geometry strategy, preprocessing, text preprocessing, leakage/splitting, manifest schema, and evaluation protocol is distributed across `mr_rate_local_audit.md` §9-10, `report2volume_gap_analysis.md`, and `recommended_next_steps.md`.

## Phase 6 — Deliverables: COMPLETE
All 5 required files written to OUTPUT_PATH (`/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE/logs/`):
1. `mr_rate_local_audit.md` — full narrative report (executive summary through evidence/limitations)
2. `mr_rate_audit_metrics.json` — machine-readable aggregates, validated as syntactically valid JSON, no identifiers/report text/full sensitive paths
3. `report2volume_gap_analysis.md` — requirement/current-state/evidence/action/priority/risk table
4. `proposed_model_manifest_schema.json` — schema + 3 fully synthetic example rows, validated as syntactically valid JSON
5. `recommended_next_steps.md` — pilot / production preprocessing / training / evaluation / unresolved-questions sections

All deliverables re-scanned with a regex for identifier-like tokens post-write — no leaks found (verified 2026-07-28).

## No further action pending. Audit complete.
