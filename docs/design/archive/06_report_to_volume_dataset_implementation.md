# 06 — Report-to-Volume Dataset/DataLoader Implementation

Scope: the Dataset/DataLoader layer for a provisional MICCAI VLM3D "MR Volume
Generation" report-to-volume training pipeline, built on the official MR-RATE
repository code. Does **not** cover model conditioning, full training,
weight downloads, full-dataset preprocessing, or challenge submission
packaging. Code: `contrastive-pretraining/scripts/data_r2v.py`,
`contrastive-pretraining/scripts/build_r2v_manifest.py`, two small additive
functions and one backward-compatible keyword in
`contrastive-pretraining/scripts/data.py`, and
`contrastive-pretraining/tests/test_data_r2v.py`.

Confidence key used throughout: **VERIFIED** (directly supported by code/data/
docs read in this session, cited `path:Lx-Ly`) / **INFERRED** (strongly
supported but not explicit) / **ASSUMED** (a provisional decision made
because the challenge spec is unpublished) / **UNKNOWN** (not determinable
locally).

Prior work this design builds on directly, without re-deriving: the full
local MR-RATE dataset audit (`docs/design/mr_rate_local_audit.md`,
`logs/mr_rate_audit_metrics.json`, `docs/design/report2volume_gap_analysis.md`,
`docs/design/recommended_next_steps.md`, `logs/proposed_model_manifest_schema.json`),
the dataset/dataloader code audit (`docs/design/mr_rate_dataset_and_dataloader_implementation.md`,
`logs/mr_rate_dataset_contract.json`), the NV-Generate-CTMR code audit
(`docs/design/nv_generate_mr_brain_audit.md`), the challenge-contract analysis
(`docs/design/challenge_contract.md`), and the FAU HPC execution profile
(`docs/design/fau_hpc_execution_profile.md`). Every claim reused from those reports
is re-cited here to its original code/data evidence, not just to the report.

---

## Concise Data Contract

### What is paired with what?

- **One sample = one (report-conditioning text, single target series) pair.**
  Not one study-with-all-its-volumes (`MRReportDataset`'s granularity), and
  not one report paired independently with every series without any
  conditioning signal.
- **Default: one report is paired with one series at a time**, drawn from a
  per-series manifest that lists every eligible series of every study. A
  study with N eligible series contributes N samples across the manifest —
  the *same* report text is attached to each, distinguished by that series'
  own `modality`/`acquisition_plane` fields.
- **Report sections used by default: `findings` + `impression`**, deterministic,
  whole text, no random sentence sampling.
- **Conditioning fields accompanying the text:** `modality`, `acquisition_plane`
  (both from the official metadata CSV), plus two constant/placeholder fields,
  `contrast_state="unknown"` and `skull_state="defaced_not_stripped"` (see
  "Target-series inclusion/exclusion policy" below for why).
- **Multiple eligible series in one study:** all are kept in the manifest;
  `MRReportToVolumeDataset(series_selection=...)` controls whether the
  *dataset* (not the manifest) exposes all of them, one per study
  deterministically, or one per study re-sampled stochastically each epoch —
  default is **`"all"`** (every eligible series is its own sample).
- **Can a study appear more than once per epoch?** Yes, under the default
  `series_selection="all"` — exactly once per eligible series it has. Under
  either `one_per_study_*` mode, exactly once per epoch, period.

### What preprocessing is applied?

| Aspect | Value |
|---|---|
| Source image space | `native_space` (configurable: `coreg_space`/`atlas_space`, same as `data.py`) |
| Orientation | RAS canonical (`nib.as_closest_canonical`), reused unchanged from `data.py` |
| Intensity normalization | `PercentileNormalizer(lower_percentile=0.0, upper_percentile=99.5, lower_limit=0.0, upper_limit=1.0, clip=False)` — matches NV-Generate-CTMR's own MRI intensity transform (see "Compatibility with NV-Generate-MR-Brain") |
| Resampling | Trilinear, `align_corners=False`, reused unchanged from `data.py`'s `resize_array` |
| Crop/pad | Center crop/pad with a 15mm posterior shift on the A-P axis (defacing compensation), reused unchanged from `data.py`'s `crop_or_pad` |
| Target spacing | **Default: 1.0×1.0×1.0mm isotropic**, per (modality, plane) bucket (see geometry table below); `geometry_mode="fixed"` reverts to the contrastive loader's 1.0×0.5×0.5mm single grid |
| Target shape | **Default: per-(modality, plane) bucket**, e.g. T1w/axial → (176, 240, 240) voxels; see full table below |
| Resulting physical FOV | Always ≥ the NV-Generate-MR-Brain median training FOV for that bucket (shapes are rounded *up*, never down) |
| Interpolation | Trilinear (unchanged) |
| Skull policy | Whole-brain, defaced-not-stripped (native-space default; matches NV-Generate-MR-Brain's whole-brain modality codes 9/10/11/16/20, not the skull-stripped 29–33 codes) |
| Anisotropic / partial-FOV scans | Handled the same way as any other series: resample-to-spacing then crop/pad; no special-cased rejection |

**No single fixed geometry is used by default** — see "Geometry strategies
considered" for why a configuration-driven per-(modality, plane) policy was
chosen instead, with the exact table given there.

### What does one sample contain?

`MRReportToVolumeDataset.__getitem__` returns a plain `dict`:

| Field | Python type | Tensor shape | Dtype | Meaning |
|---|---|---|---|---|
| `image` | `torch.Tensor` | `[1, X, Y, Z]` | `bfloat16` (configurable) | 1=channel, X=Right–Left, Y=Anterior–Posterior, Z=Superior–Inferior — NV-Generate-CTMR's own array axis order, converted once as the last step of `__getitem__` (added 2026-07-28; see "Compatibility with NV-Generate-MR-Brain" and `docs/design/08_....md` Part C) |
| `report_text` | `str` | — | — | Composed `findings`+`impression` (or configured sections), deterministic |
| `modality` | `str` | — | — | e.g. `"T1w"`, or `"unknown"` if no metadata was available |
| `acquisition_plane` | `str` | — | — | e.g. `"AXIAL"`, or `"unknown"` |
| `contrast_state` | `str` | — | — | Always `"unknown"` (not derivable from the release) |
| `skull_state` | `str` | — | — | Always `"defaced_not_stripped"` for `native_space` |
| `target_spacing_mm` | `torch.Tensor` | `[3]` | `float32` | (X,Y,Z) mm, this bucket's target — reindexed from the internal (D,H,W) geometry table the same way as `image` |
| `target_shape` | `torch.Tensor` | `[3]` | `int64` | (X,Y,Z) voxels, this bucket's target; always equals `image.shape[1:]` |
| `native_shape` | `torch.Tensor` | `[3]` | `int64` | (X,Y,Z) voxels, pre-resample, RAS-canonical |
| `native_spacing_mm` | `torch.Tensor` | `[3]` | `float32` | (X,Y,Z) mm, pre-resample, RAS-canonical |
| `native_fov_mm` | `torch.Tensor` | `[3]` | `float32` | native_shape × native_spacing_mm, (X,Y,Z) |
| `study_key` | `str` | — | — | Anonymized study UID (traceability only — never printed) |
| `series_key` | `str` | — | — | Series ID (traceability only — never printed) |

**Axis-order note:** everything in this table is (X,Y,Z)-ordered, NOT the
(D,H,W)=(S,R,A) order `MRReportDataset` (`data.py`) uses for the original
contrastive-pretraining pipeline. Internally, `MRReportToVolumeDataset`
still preprocesses in (D,H,W) (reusing `data.py`'s crop/resample/normalize
unchanged) and reindexes to (X,Y,Z) exactly once, at the very end of
`__getitem__` — see `data_r2v.py`'s module docstring and `_dhw_to_xyz` for
why (D,H,W) is required by the *original* pipeline's VJEPA encoders but not
by NV-Generate-CTMR, which this Dataset targets. The persisted manifest
(`ManifestRow.native_shape`/`native_spacing_mm`, on disk) is unaffected —
it stays (D,H,W); only this Dataset's returned sample dict is reindexed.

Required for training: `image`, `report_text`. Conditioning-only:
`modality`, `acquisition_plane`, `contrast_state`, `skull_state`.
Traceability-only: `study_key`, `series_key`. Debug/evaluation-only:
`target_*`, `native_*`.

### What does one batch contain?

`collate_fn_r2v(batch)` returns the same field names, batched:

| Field | Batched shape/type |
|---|---|
| `image` | `[B, 1, X, Y, Z]` |
| `target_spacing_mm`, `target_shape`, `native_shape`, `native_spacing_mm`, `native_fov_mm` | `[B, 3]` |
| `report_text`, `modality`, `acquisition_plane`, `contrast_state`, `skull_state`, `study_key`, `series_key` | `list[str]`, length B |

- **No text padding/tokenization at this layer** — `report_text` stays a raw
  `list[str]`; tokenization is deferred to the training loop/collator that
  owns the tokenizer, mirroring the existing pipeline's own separation
  (`mr_rate_trainer.py:430-435` tokenizes *after* its own collate step).
- **No image padding/masking** — every sample in a batch is exactly one
  volume (no per-study N-axis to pad, unlike `MRReportDataset`).
- **Batch size**: any `batch_size >= 1` is supported (unlike `data.py`'s
  `collate_fn`, which only ever reads `batch[0]`). With the default
  `geometry_mode="per_modality_plane"`, `batch_size > 1` additionally
  requires `GeometryBucketBatchSampler` (or `geometry_mode="fixed"`) so every
  item in a batch shares one target shape — otherwise `collate_fn_r2v` raises
  a clear `ValueError` naming the mismatched shapes, rather than a raw
  `torch.stack` traceback or (as in `data.py`) silently discarding samples.

---

## Executive conclusion

The existing MR-RATE dataset/dataloader stack (`MRReportDataset`,
`MRReportDatasetInfer`, their shared `collate_fn`/`collate_fn_infer`) is
architecturally unsuited to report-to-volume **generation** as-is: it returns
one *study* as a stack of *all* its volumes sharing one randomly-resampled
sentence list, and its collate functions only ever look at `batch[0]`
(VERIFIED, `data.py:667-672`/`data_inference.py:242-251`, confirmed by live
execution in `docs/design/mr_rate_dataset_and_dataloader_implementation.md` §9/§11).
Neither property is appropriate for a generative model that needs a
deterministic, whole-report-conditioned, single-volume target and a real
batch.

However, the *primitives underneath* those two classes — filesystem discovery,
NIfTI loading/RAS-reorientation/resampling, the three normalizer classes,
crop/pad, the `.npz` cache mechanism and its manifest validation, and the
splits-CSV loader — are report/task-agnostic pure functions
(`discover_subjects`'s own docstring says "Report/split agnostic",
`data.py:126`) and are reused **unchanged, by composition**, in
`data_r2v.py`. The new module (`MRReportToVolumeDataset`,
`collate_fn_r2v`, `build_manifest_rows`) is a new, series-level Dataset built
from those same primitives, not a fork of `MRReportDataset`. Two small,
backward-compatible additions were made to `data.py` itself (a header-only
geometry reader and a `splits.csv`-by-study-not-by-allow-set loader; see next
section) — everything else in `data.py` is untouched, and the full 106+2
pre-existing test suite passes unmodified (see "Tests and smoke-test
results").

The default pairing policy is **series-level**: one manifest row per eligible
(study, series) pair, one sample per row by default. The default geometry
policy is **configuration-driven per-(modality, plane) buckets**, sourced
from NV-Generate-MR-Brain's own documented training-FOV distribution, not a
single fixed grid — the evidence (near-total native-MR-RATE geometric
heterogeneity, and NV-Generate-MR-Brain's own per-bucket FOV table differing
by up to ~45% per axis across buckets) does not support one global default.

---

## Official MR-RATE components reused

All citations point at `contrastive-pretraining/scripts/data.py` unless noted.

| Component | Lines | Reuse mode |
|---|---|---|
| `discover_subjects` (+ `list_nii_files`, `SPACE_TO_IMG_SUBDIR`) | 124-176 | Reused unchanged, called from `build_manifest_rows` |
| `load_and_resample_nii` | 178-201 | Reused unchanged (indirectly, via `preprocess_nii`) |
| `resize_array` | 23-29 | Reused unchanged (indirectly) |
| `ZScoreNormalizer` / `PercentileNormalizer` / `MinMaxNormalizer` | 32-121 | Reused unchanged; `PercentileNormalizer` gained one new, backward-compatible `clip=` keyword (see below) |
| `crop_or_pad` | 250-289 | Reused unchanged (indirectly, via `preprocess_nii`) |
| `preprocess_nii` | 291-301 | Reused unchanged — the exact live/`__getitem__` path calls this, guaranteeing byte-identical behavior to `preprocess_volumes.py`'s cache-building path, same invariant `MRReportDataset` already relies on |
| `build_cache_manifest` / `CACHE_MANIFEST_NAME` / `CACHE_CONFIG_KEYS` / `validate_cache_manifest` | 303-372 | Reused unchanged for the `geometry_mode="fixed"` `.npz`-cache path |
| `preprocess_volumes.py` (whole script) | — | Reused unchanged as the cache-building tool for `geometry_mode="fixed"`; no changes needed |
| Splits-CSV loading | new `load_all_splits`/`load_split_uids` | New top-level functions, **promoted** from the identical private logic duplicated in `MRReportDataset._load_splits`/`MRReportDatasetInfer._load_splits` (`data.py:470-478` unchanged; the two private methods are untouched, the new top-level functions are additive) |

---

## Components extended or replaced and why

**Extended (backward-compatible additions to `data.py`), not modified:**

1. **`read_native_geometry(path)`** (`data.py:203-220`) — a new, small,
   header-only function returning `(D, H, W)` shape + spacing in the exact
   same axis convention as `load_and_resample_nii`, *without* calling
   `get_fdata()`. Needed because `load_and_resample_nii` resamples and
   discards the pre-resample geometry (`crop_or_pad`'s output carries no
   record of original shape/spacing/FOV — a gap the prior dataloader audit
   flagged explicitly, `docs/design/mr_rate_dataset_and_dataloader_implementation.md`
   §13). Verified self-consistent with `load_and_resample_nii` via a targeted
   test that constructs a NIfTI with a non-trivial axis permutation and
   confirms both functions agree (`test_data_r2v.py::TestReadNativeGeometryOrientation`).
2. **`load_all_splits(splits_csv)` / `load_split_uids(splits_csv, split)`**
   (`data.py:222-247`) — the latter is a straight promotion of the identical
   logic already duplicated in both dataset classes' private
   `_load_splits`; the former is new (keeps every row's split label instead
   of filtering to one split's allow-set), needed because the manifest
   records each row's split **once** and `MRReportToVolumeDataset` filters to
   whichever split it's constructed with, without re-reading the CSV per
   split.
3. **`PercentileNormalizer(..., clip=True)`** (`data.py:46-83`) — one new
   keyword, default `True` (= 100% unchanged behavior, verified by the
   pre-existing `test_output_range` test and a new explicit regression test,
   `test_data.py::test_clip_default_true_matches_prior_behavior`). `clip=False`
   matches NV-Generate-CTMR's own `ScaleIntensityRangePercentilesd(...,
   clip=False)` MRI transform (`NV-Generate-CTMR/scripts/transforms.py:42-71`,
   audited in `docs/design/nv_generate_mr_brain_audit.md` §5.2) — needed because the
   existing class always clamped, which the target model's own pipeline does
   not do for the upper tail.

**Replaced (new class/functions in `data_r2v.py`), and why reuse was not appropriate:**

1. **Per-study, multi-volume `__getitem__` granularity** (`MRReportDataset`,
   `data.py:373-724`) → replaced by series-level `MRReportToVolumeDataset`.
   A generative target needs exactly one volume per sample; stacking all of
   a study's volumes and sharing one report across them is a
   contrastive-objective-specific design (works there because the loss only
   ever needs *some* image-text pairs per step) that actively mismatches a
   generation target (see "Report/image relationship" below).
2. **Random per-`__getitem__` sentence subsampling**
   (`data.py:704-724`, `random.sample` reseeded from the process-global RNG
   on every call — confirmed non-reproducible across identical calls by the
   prior audit, §9) → replaced by `ReportRecord.compose()`, deterministic,
   whole-section text, no truncation. A generator should see the same,
   complete conditioning text every time it sees that series, not a
   different random 34-sentence subset each epoch.
3. **`collate_fn`/`collate_fn_infer`'s `batch[0]`-only behavior**
   (`data.py:726-731`, `data_inference.py:242-251`) → replaced by
   `collate_fn_r2v`, which stacks every item in the batch (confirmed live by
   the prior audit that the existing functions silently discard everything
   past the first item — `docs/design/mr_rate_dataset_and_dataloader_implementation.md`
   §11 Part D). Not reusable at all for a training loop that wants a real
   batch of independent (report, volume) pairs.
4. **`MRRATE`'s `real_volume_mask`/`fusion_mode` pooling machinery**
   (`mr_rate/mr_rate/mr_rate.py`) — entirely specific to combining multiple
   series into one contrastive embedding; there is no multi-series axis to
   pool in a series-level generative sample, so this has no analog here at all.

---

## Report/image relationship in MR-RATE

**VERIFIED** (`docs/design/mr_rate_local_audit.md` §6, independently re-confirmed by
this session's own header read, and by the exact CSV schema `study_uid,
report, clinical_information, technique, findings, impression`):
- Reports are **study-level**: one row per `study_uid`.
- Volumes are **series-level**: a study has anywhere from 1 to 83 series
  (mean 6.47, `logs/mr_rate_audit_metrics.json`), of varying modality/plane.
- A report describing, e.g., FLAIR findings says nothing distinguishing
  about a same-study T1w series — pairing one report with *every* series
  independently is a real label-noise risk (already flagged in
  `docs/design/report2volume_gap_analysis.md` row 1), not a decision this design
  can eliminate, only mitigate via explicit per-series conditioning fields
  (modality/plane) so the model has *some* signal distinguishing series
  beyond the shared text.
- `has_report=True` for 99.86% of studies (`logs/mr_rate_audit_metrics.json`);
  join key `study_uid` is 100% consistent with `series.parquet`'s
  `metadata_matched` column.

---

## Pairing alternatives considered

| Alternative | Verdict |
|---|---|
| One study-level report paired independently with each eligible series (naive) | **This is what the default policy actually does at the manifest level** — the mitigation is per-series modality/plane conditioning fields, not avoiding the pairing itself (no per-sequence report attribution exists anywhere in the pipeline to do better — a genuine gap, flagged as future work) |
| Report + explicit modality → one series | Subsumed by the chosen policy (modality *and* plane are both included) |
| Report + modality + plane → one series | **Chosen** (this is the manifest's actual per-row conditioning) |
| One report + variable collection of study series (study-level generation) | Rejected for v1: no prior art in this codebase for jointly generating a variable-length set of volumes from one report; `NV-Generate-MR-Brain` itself generates one volume per call (VERIFIED, `docs/design/nv_generate_mr_brain_audit.md` §2: "Batch size at inference is fixed to 1 volume per process") |
| Selecting one series dynamically per study | **Implemented as a configurable alternative**, not the default — see `series_selection` below |
| Separate modality-specific datasets/models | Rejected for v1 as unnecessary abstraction: the same `MRReportToVolumeDataset` class already supports per-modality/plane geometry via the manifest + `GeometryPolicy`; separate classes would duplicate the report/split/exclusion logic for no benefit |

---

## Selected default pairing policy

**Default (`series_selection="all"`):** one sample per eligible
(study, series) manifest row. A study with N eligible series contributes N
samples per epoch — deliberately **not** deduplicated to one-per-study by
default, because:
- it is the simplest, most transparent policy (no implicit resampling to
  reason about),
- it matches the manifest's natural granularity exactly (no information is
  discarded or needs to be reconstructed later), and
- the overrepresentation of larger studies is a real, known, explicitly
  documented tradeoff (not hidden), directly mitigable via
  `compute_modality_balance_weights`/`get_modality_balanced_sampler` (inverse
  modality-frequency weighting, pattern-reused from `MRReportDataset`'s own
  pathology rebalancing, `data.py:552-618`) if it proves harmful in practice.

**Configurable alternatives**, implemented on the *same* manifest (no
rebuild needed to switch):
- `"one_per_study_deterministic"` — one sample per study, always the
  `is_center_modality` series if flagged, else the `series_id`-sorted first.
  Removes overrepresentation entirely, at the cost of never training on a
  study's non-preferred series.
- `"one_per_study_random"` — one sample per study, re-chosen uniformly at
  random each epoch via `Dataset.set_epoch(epoch)`, seeded deterministically
  by `(config.seed, epoch, study_uid)` — **not** the ambient, non-reproducible
  reseeding `MRReportDataset` does today (`data.py:704-724`, reseeded from
  the process-global RNG on every `__getitem__` call). This satisfies
  "explicit random behavior during training" (deterministic given the seed
  and epoch) while still covering all of a study's series across enough
  epochs.

## How study and series weighting is controlled

Three independent knobs, composable:
1. `series_selection` (above) — controls how many samples a study contributes.
2. `compute_modality_balance_weights` / `get_modality_balanced_sampler` —
   inverse-frequency `WeightedRandomSampler` over any single categorical
   manifest field (default: `modality`), off by default (uniform sampling),
   directly mirroring `MRReportDataset`'s existing rare-pathology rebalancing
   mechanism (same idea, different label schema — pattern reuse, not code reuse).
3. `GeometryBucketBatchSampler` (for `batch_size > 1` under
   `geometry_mode="per_modality_plane"`) additionally determines *within-batch*
   composition (always single-bucket), but does not change the underlying
   per-epoch sample distribution — it only reorders which same-bucket
   samples land in the same batch.

---

## Selected report representation

- **Primary conditioning text: `findings` + `impression`**, in that order,
  each prefixed with a section label (`"Findings: ...\nImpression: ..."`),
  via `ReportRecord.compose(config.report_sections)`.
- **`technique` excluded from the default text** — ASSUMED, because
  `technique` explicitly names the acquired sequence
  (`docs/design/mr_rate_local_audit.md` §6's schema), which for a *per-series*
  generation target is redundant with (or could substitute for) the
  `modality`/`acquisition_plane` conditioning fields already supplied
  separately — including it in the free text risks the model learning to
  read the sequence name out of the report rather than the modality field,
  a shortcut-learning risk rather than a genuine generation signal.
  Configurable via `report_sections` if a future experiment wants it.
- **`clinical_information` excluded from the default text** — ASSUMED,
  contextual/indication text, off by default to keep conditioning text
  focused on diagnostic content; configurable.
- **Absent sections**: skipped in `compose()` (not padded with placeholder
  text); if every requested section is absent, `compose()` returns `""` and
  that sample's report_text is empty (not excluded — exclusion happens only
  when the *study* has no report row at all, i.e. isn't in the report
  store).
- **Three report sources supported**, behind one duck-typed interface
  (`__contains__`/`__getitem__` → `ReportRecord`):
  - `StructuredReportStore` (**default for `extracted_dir`/`data_path_archive`**)
    — the official structured CSV schema (`study_uid, report,
    clinical_information, technique, findings, impression`), section
    boundaries preserved exactly as released. Reads a plain CSV or a
    `.tar.gz` of per-batch CSVs (e.g. DATA_PATH's `reports.tar.gz`) eagerly,
    in full, at construction time.
  - `ShardReportStore` (**preferred for `shards_parquet`**, added
    2026-07-28) — reads each study's own `report.json` sidecar directly
    from its shard tar (same 5-field schema, verified content-identical to
    DATA_PATH's reports). Self-contained: no dependency on DATA_PATH.
    Exact presence comes from a small separately-built index
    (`build_shards_parquet_manifest_standalone.build_report_index`, from
    `studies.parquet`'s `has_report` column); actual content is read
    lazily per study on first access and cached, since eagerly reading
    every study's `report.json` at construction time was measured at ~91
    studies/sec (~16 minutes for the full train split) — see
    `docs/design/08_dataset_recommendation_manifest_and_axis_order.md`
    §"ShardReportStore" for the full evidence and design rationale.
  - `SentenceJSONLReportStore` (**legacy/fallback**) — the pre-extracted
    sentence-list JSONL format `MRReportDataset` already consumes
    (`data.py:480-494`); has no section boundaries, so every record's
    sentences are joined into a single string exposed as *both* `raw` and
    `findings`. This is the only report source **currently present, already
    extracted, on this local checkout**
    (`contrastive-pretraining/data/findings_sentences.jsonl`) — smoke-tested
    directly against it (see "Tests and smoke-test results").
  - **Randomly sampling report sentences is not reused** from
    `MRReportDataset` — deliberately: a generator should see the same,
    complete conditioning text deterministically, not a different random
    subset each epoch (this is exactly the "do not preserve contrastive
    behavior merely because it exists" instruction).
- **Whole selected text is returned before tokenization**: `report_text` is
  a raw Python `str`; **tokenization happens in neither the Dataset nor
  `collate_fn_r2v`**, deferred to the training loop/model adapter — this
  mirrors the existing pipeline's own convention exactly
  (`mr_rate_trainer.py:430-435` tokenizes *after* its own `collate_fn`, not
  inside the Dataset or collate function).
- **No report text is ever printed**: neither `data_r2v.py` nor its tests
  contain a `print`/log statement that includes a report-text field's value
  (only counts/lengths are ever printed, matching `MRReportDataset`'s own
  existing convention of printing counts, not content).

---

## Target-series inclusion/exclusion policy

`is_eligible(meta, ...)` (`data_r2v.py`):
- **Excludes `classified_modality == "MRA"` by default**
  (`DEFAULT_EXCLUDED_MODALITIES`) — ASSUMED/INFERRED-justified: MRA is
  ~0.02% of MR-RATE series (`docs/design/mr_rate_local_audit.md` §5.3) *and*
  `nvidia/NV-Generate-MR-Brain`'s public model card documents support for
  T1/T2/FLAIR/SWI only, not MRA (`docs/design/challenge_contract.md` §0.1's
  web-verified model-card cross-check) — excluding it costs negligible
  coverage while matching the actual downstream model.
- **Excludes `is_derived`/`is_localizer`** — defensive re-checks; both are
  100% False dataset-wide (`logs/mr_rate_audit_metrics.json`), because
  DWI/ADC/scout/localizer/derived series are already excluded upstream at
  the official `modality_filtering.py` acceptance stage (accepted
  modalities: `{T1w, T2w, T2-FLAIR→FLAIR, SWI, MRA}` only — no DWI/ADC
  anywhere in the release). The manifest builder's own eligibility check
  does not need to (and does not) special-case DWI/ADC separately.
- **Contrast state**: no exclusion possible — `is_contrast_enhanced` is
  computed transiently but dropped from the release
  (`config_metadata_columns.json:391`, confirmed absent from the actual
  metadata CSV header by the prior audit). Every sample's `contrast_state`
  field is the constant string `"unknown"`.
- **Skull state**: no exclusion needed — native-space MR-RATE is uniformly
  defaced-but-not-skull-stripped (independently confirmed at the voxel level,
  median 69% of nonzero voxels outside the brain mask,
  `docs/design/mr_rate_local_audit.md` §5.2). Every sample's `skull_state` field is
  the constant `"defaced_not_stripped"`.
- **Body region (brain vs. spine)**: **UNKNOWN/not filterable** — `body_region`
  is 100% null in the release (documented "coming soon"). CLAUDE.md
  describes MR-RATE as covering "brain and spine MRI"; this design (and the
  NV-Generate-MR-Brain target) is brain-focused, but nothing in the current
  metadata can separate the two. Flagged as an open, challenge-independent
  gap — not resolvable by this Dataset.
- **Near-duplicate series** (e.g. repeated same-modality/plane acquisitions,
  `modality_filtering.py`'s `-2`/`-3` counter suffix) — **retained**, not
  deduplicated, matching the prior audit's own finding that no
  duplicate/near-duplicate detection exists anywhere in the pipeline
  (`docs/design/mr_rate_local_audit.md` §9 item 7). Each survives as its own manifest
  row/sample.
- **Oblique-plane series**: **not excluded** — fall into the geometry
  policy's fallback bucket (see below), consistent with NV-Generate-MR-Brain
  having seen oblique scans in training (`NV-Generate-CTMR/docs/inference.md`,
  quoted in the geometry section).

---

## Geometry strategies considered

Compared explicitly, per the task's checklist:

| Strategy | Verdict |
|---|---|
| One fixed grid for everything | Rejected as the default (still offered as `geometry_mode="fixed"`) — MR-RATE's native geometry is near-totally heterogeneous (36/37 unique shapes, 34/37 unique spacings in a stratified 37-file byte-level sample, `docs/design/mr_rate_local_audit.md` §5.2), and NV-Generate-MR-Brain's own recommended-FOV table varies by up to ~45% per axis across (modality, plane) buckets — a single grid cannot represent this distribution well |
| Fixed grids per modality | Considered, superseded by the finer per-(modality, plane) table below (plane changes the FOV shape at least as much as modality does — compare T1w axial 240×240×174mm vs. T1w sagittal 176×250×250mm) |
| Fixed grids per modality **and** plane | **Chosen as the default** — see table below |
| A small set of geometry buckets | This *is* what the per-(modality, plane) table is — 15 buckets + 1 fallback, not one bucket per series |
| Variable shape/spacing conditioned like NVIDIA | This is architecturally what NV-Generate-MR-Brain itself does (spacing/dim are free conditioning inputs, `docs/design/nv_generate_mr_brain_audit.md` §2/§4) — the bucket table is this project's provisional discretization of that same idea into a small, reproducible default set, not a rejection of it; nothing in this Dataset prevents passing per-sample arbitrary `(shape, spacing)` later (the `target_spacing_mm`/`target_shape` fields are already carried through per-sample) |
| Native-volume or patch-based training | Rejected for v1: added complexity (variable-shape batching, patch sampling logic) not warranted for an initial baseline; NV-Generate-MR-Brain's own VAE training does use patches (`RandSpatialCropd`, `docs/design/nv_generate_mr_brain_audit.md` §5.4) but that is VAE-training-specific, not part of this Dataset/DataLoader layer's scope |

### Selected default preprocessing (the per-(modality, plane) geometry table)

**Source (VERIFIED):** `NV-Generate-CTMR/docs/inference.md`'s "Recommended
FOV for MR `rflow-mr-brain` model" table — the only shipped variant whose
diffusion U-Net was actually trained on MR-RATE
(`docs/design/nv_generate_mr_brain_audit.md` §6). Axis assignment rule (also from
that doc, quoted directly): *"set `dim` so the slice-stacking axis maps to
the smaller `dim[i]=128` (axial→z, sagittal→x, coronal→y)"*. This was
cross-checked (not merely trusted) against all 15 published rows: every row
has two equal larger FOV values plus one smaller value, and the smaller
value always falls on the anatomically-expected stacking axis — a
self-consistency check that would have failed if the rule were wrong.

Formula: `target_shape[i] = ceil_to_multiple(fov_mm[i] / spacing_mm[i],
divisible_by=16)`, rounding **up** (never down, so the resulting grid's
physical FOV never truncates below the documented median). Default
`spacing_mm = (1.0, 1.0, 1.0)` — matches NV-Generate-MR-Brain's own shipped
default inference spacing exactly (`configs/config_maisi_diff_model_rflow-mr-brain.json`,
`docs/design/nv_generate_mr_brain_audit.md` §2), not the contrastive loader's
1.0×0.5×0.5mm (tuned for a discriminative encoder's single fixed grid).

| Modality | Plane | Median FOV (mm), doc-cited | Target shape (voxels) | Resulting FOV (mm) |
|---|---|---|---|---|
| T1w | AXIAL | 174×240×240 | 176×240×240 | 176×240×240 |
| T1w | SAGITTAL | 250×176×250 | 256×176×256 | 256×176×256 |
| T1w | CORONAL | 240×240×200 | 240×240×208 | 240×240×208 |
| T2w | AXIAL | 158×240×240 | 160×240×240 | 160×240×240 |
| T2w | SAGITTAL | 240×162×240 | 240×176×240 | 240×176×240 |
| T2w | CORONAL | 200×200×180 | 208×208×192 | 208×208×192 |
| FLAIR | AXIAL | 175×250×250 | 176×256×256 | 176×256×256 |
| FLAIR | SAGITTAL | 250×176×250 | 256×176×256 | 256×176×256 |
| FLAIR | CORONAL | 250×250×200 | 256×256×208 | 256×256×208 |
| SWI | AXIAL | 145×230×230 | 160×240×240 | 160×240×240 |
| SWI | SAGITTAL | 230×140×230 | 240×144×240 | 240×144×240 |
| SWI | CORONAL | 230×230×155 | 240×240×160 | 240×240×160 |
| MRA | AXIAL | 158×220×220 | 160×224×224 | 160×224×224 (excluded by default modality policy) |
| MRA | SAGITTAL | 250×158×250 | 256×160×256 | 256×160×256 (excluded by default) |
| MRA | CORONAL | 240×240×179 | 240×240×192 | 240×240×192 (excluded by default) |
| *(fallback: unknown modality/plane, or OBLIQUE)* | | 256×256×256 (NVIDIA's own shipped default) | 256×256×256 | 256×256×256 |

All shapes/spacings above are `(D, H, W) = (S, R, A)`-ordered, matching every
other geometry parameter in `data.py`. `build_geometry_table()`
(`data_r2v.py`) computes this table from `NV_BRAIN_FOV_MM` at import/config
time — it is not hardcoded as 15 magic tuples, so changing `spacing_mm` or
`divisible_by` recomputes every bucket consistently.

Everything else in the preprocessing chain (orientation, normalization
formula, resample/crop-pad mechanics, posterior shift) is the same
regardless of bucket — only `target_shape`/`target_spacing` vary.

### `geometry_mode="fixed"` (configurable alternative)

Reverts to the contrastive loader's own single-grid strategy
(`target_shape=(256,384,384)`, `target_spacing=(1.0,0.5,0.5)mm`, i.e. FOV
256×192×192mm — reused, unchanged parameters). This is the only mode
compatible with the existing `.npz` cache mechanism (see "Cache and
performance behavior" below) and the only mode where `batch_size > 1` works
without a geometry-bucketing sampler.

---

## Compatibility with NV-Generate-MR-Brain

| Aspect | This Dataset (default) | NV-Generate-MR-Brain | Match? |
|---|---|---|---|
| Orientation | RAS canonical | `Orientationd(axcodes="RAS")` (`NV-Generate-CTMR/scripts/transforms.py:156`, `.../diff_model_create_training_data.py:58`) | **Exact match** (VERIFIED) — anatomical orientation only, see next row |
| **Array axis order (internal preprocessing, `data.py`)** | `(D, H, W) = (S, R, A)`, via explicit `.transpose(2, 0, 1)` (`data.py:205`) — required by the *original* contrastive pipeline's VJEPA encoders, not by NV-Generate-CTMR | `(X, Y, Z) = (R, A, S)`, i.e. MONAI's native post-`Orientationd` order with **no further permutation** anywhere in its code (`diff_model_create_training_data.py:158`, `diff_model_infer.py:141-145,254-256`) | These two differ by construction (see left cell) — this is why the next row exists |
| **Array axis order (`MRReportToVolumeDataset`'s returned `image`, added 2026-07-28)** | `(X, Y, Z) = (R, A, S)` — converted from the internal `(D,H,W)` via one `image.permute(0,2,3,1)` at the very end of `__getitem__`, plus a matching `_dhw_to_xyz` reindex of `target_shape`/`target_spacing_mm`/`native_*` | `(X, Y, Z) = (R, A, S)`, unchanged from the row above | **Exact match** (VERIFIED, real-sample smoke test in `docs/design/08_....md` Part B/C) — this Dataset's output is NV-Generate-CTMR-ready with **no adapter-side permute needed**. Full trace, worked example, and the reason `data.py` itself keeps `(D,H,W)` unmodified: `docs/design/08_dataset_recommendation_manifest_and_axis_order.md` §Part C |
| Interpolation | Trilinear | `mode="trilinear"` (`Resized`, same file) | **Exact match** |
| Intensity transform | `PercentileNormalizer(0.0, 99.5, 0.0, 1.0, clip=False)` | `ScaleIntensityRangePercentilesd(lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False)` | **Exact match** (VERIFIED) |
| Default spacing | 1.0×1.0×1.0mm | Shipped default `spacing=[1,1,1]` | **Exact match** |
| VAE downsample factor | N/A (Dataset layer only) | 4× per axis, both spatial dims (VERIFIED, `docs/design/nv_generate_mr_brain_audit.md` §1.1-1.2) — this Dataset's `divisible_by=16` default is stricter than the 4× the code actually enforces, giving headroom | Compatible (16 is a multiple of 4) |
| Modality codes | `modality` field carries `T1w`/`T2w`/`FLAIR`/`SWI` strings | Integer codes 9/10/11/20 (whole-brain) | **Not yet mapped** — a challenge adapter or training script must map this Dataset's string modality to NV-Generate-MR-Brain's integer `modality_mapping.json` codes; out of scope for this Dataset layer (no model-conditioning code is touched here) |
| Text/report conditioning | `report_text` field, raw string | **None exists** — exhaustive grep found zero text-encoder/cross-attention/report code anywhere in NV-Generate-CTMR (`docs/design/nv_generate_mr_brain_audit.md` §7) | **Explicit gap**, not this task's scope (see "Known limitations") |
| Skull state | Whole-brain (codes 9-20 equivalent) | Same whole-brain modality codes | Compatible |

---

## Persistent index/manifest representation

One CSV row per eligible (study, series) pair
(`MANIFEST_FIELDS` in `data_r2v.py`): `study_uid, series_id, image_path,
split, modality, plane, is_center_modality, native_shape,
native_spacing_mm, cache_index`.

- **Deterministic**: `discover_subjects` sorts every directory level it
  walks (`data.py:150-153,159,168`), so re-running `build_manifest_rows`
  against an unchanged `data_folder` reproduces byte-identical CSV output.
- **Resumable by construction, not by checkpointing**: since a full re-run is
  deterministic and idempotent, "resuming" a build is simply re-running it —
  no separate incremental-write/checkpoint mechanism was added, deliberately
  (would be unneeded complexity for a pure function of on-disk state).
- **Metadata-only except one header-only NIfTI read per eligible series**
  (`data.read_native_geometry`, no `get_fdata()` decode) — this is *not*
  sourced from the metadata CSV's own `array_shape`/`array_spacing_mm`
  columns even when a metadata CSV is given: those columns are computed via
  a plain `nib.load()` with **no RAS reorientation**
  (`data-preprocessing/src/mr_rate_preprocessing/mri_preprocessing/modality_filtering.py:311-319`,
  `load_image_properties`) — i.e. raw on-disk axis order, not this
  codebase's `(D, H, W) = (S, R, A)` convention, despite being named
  `ras_array_shape` before the release column-rename
  (`config_metadata_columns.json:395-397`). This is a genuine, verified
  discrepancy this design found and deliberately avoids relying on — see
  `SeriesMeta`'s docstring in `data_r2v.py` and
  `test_data_r2v.py::TestReadNativeGeometryOrientation` for the regression
  test that would catch this class of bug.
- **Split-agnostic**: each row carries its own split label (from
  `data.load_all_splits`); a `Dataset` constructed with `split="train"` vs.
  `"val"` filters the same manifest, no rebuild needed.
- **Report-source-agnostic**: the manifest has no report fields at all —
  report text is joined at `Dataset` construction time from whichever
  `ReportRecord`-yielding store is passed in.
- **Corrupt/unreadable series** (~0.43% of MR-RATE series,
  `logs/mr_rate_audit_metrics.json`): skipped with a `warnings.warn` during
  manifest build, not silently included with garbage geometry and not a
  hard crash of the whole build.

---

## Dataset return structure

See "Concise Data Contract" above for the field table. Design rationale for
choosing a plain `dict` over a tuple or a dataclass instance: named fields
prevent the exact kind of positional-tuple mistake `MRReportDataset`'s
3-tuple return risks (`volume_stack, selected_sentences, mask` — easy to
transpose), while a `dict` composes trivially with `collate_fn_r2v`'s
generic per-key stacking/listing, unlike a dataclass instance which would
need `dataclasses.fields()` introspection or field-by-field unpacking to do
the same. `ManifestRow`, `ReportRecord`, `GeometrySpec`, and `SeriesMeta` are
all genuine `@dataclass`es (structured, typed, IDE-discoverable) precisely
where the extra ceremony pays for itself (persistent/reusable objects, not
one-shot per-`__getitem__` return values).

## Collated batch structure

See "Concise Data Contract" above.

---

## Complete dimension trace

Axis-name legend used throughout this document and the code (defined once,
used consistently — never mixed with a generic image "H×W" convention):
**D** = Superior–Inferior, **H** = Right–Left, **W** = Anterior–Posterior
(this codebase's internal working order, all after RAS canonicalization).
**X** = Right–Left, **Y** = Anterior–Posterior, **Z** = Superior–Inferior
(NV-Generate-CTMR's own array axis order, also after RAS canonicalization —
see "Compatibility with NV-Generate-MR-Brain" below for why these two
orderings differ). **N** = number of series in a study (only relevant to
`MRReportDataset`, not this Dataset). **C** = channel. **B** = batch.

| Stage | Shape | Dimension meaning | Spacing order | Physical FOV (example: T1w axial) | Dtype | Transposed? | Batch/channel added? |
|---|---|---|---|---|---|---|---|
| Original NIfTI on disk | `(x, y, z)` (varies) | Whatever axis order the scanner/dcm2niix wrote (VERIFIED not RAS-canonical, `docs/design/mr_rate_local_audit.md` §5.2: RAS 52%/LAS 30%/SLA 9%/LIA 9%) | `(x, y, z)` mm, per `header.get_zooms()` | Varies per series (median ~174×240×240mm for T1w axial per NV-Generate-CTMR's own table) | `uint16` or `float32` (57%/42% split, `logs/mr_rate_audit_metrics.json`) | N/A (source) | No |
| Canonical (RAS) image | `(x', y', z')` = `(R, A, S)` | `nib.as_closest_canonical` output, axis order R,A,S | Reordered to match | Same physical extent as source (reorientation is axis permutation/flip only, no interpolation) | `float32` (after `get_fdata()`) | Possibly (axis permutation/flip) | No |
| After `.transpose(2,0,1)` | `(D, H, W) = (S, R, A)` | `data.py`'s own convention (`load_and_resample_nii:178-201`) — required by the *original* contrastive-pretraining pipeline's VJEPA encoders, not by NV-Generate-CTMR (see note below) | `(D, H, W)` mm | Same as canonical | `float32` | Yes (this is the `.transpose(2,0,1)` call) | No |
| Resampled | `(D, H, W)`, voxel count changed | Same axis meaning, now at `target_spacing` | `target_spacing_mm`, e.g. `(1.0,1.0,1.0)` | Approximately preserved (± rounding: `resize_array` uses `round(orig_shape[i]*scale[i])`) | `float32` | No | No |
| Crop/padded | `target_shape`, e.g. `(176, 240, 240)` = (D,H,W) for T1w axial | Same axis meaning, still `(D,H,W)`, now exactly `target_shape` | `target_spacing_mm` | Exactly `target_shape × target_spacing_mm` = 176×240×240mm | `float32` | No | No |
| **Final permute (`__getitem__`'s last step, added 2026-07-28)** | `(240, 240, 176)` = `(X,Y,Z)=(H,W,D)` for T1w axial | `image.permute(0,2,3,1)` on `[C,D,H,W]`; every other geometry field reindexed via `_dhw_to_xyz` | `target_spacing_mm` field, reindexed the same way | Same physical FOV, axes relabeled | `float32` | Yes (`(D,H,W)` → `(H,W,D)`) | No |
| `Dataset.__getitem__`'s `image` | `[1, 240, 240, 176]` | `+1` = channel, prepended before the `(X,Y,Z)` spatial dims | Carried in `target_spacing_mm` field, `[3]`, `(X,Y,Z)`-ordered | Same as above | `bfloat16` (cast at return) | No | Channel added |
| `DataLoader` batch (`collate_fn_r2v`) | `[B, 1, 240, 240, 176]` | `+B` = batch | `target_spacing_mm`: `[B, 3]` | Same per-sample (all samples in one batch share this bucket's geometry) | `bfloat16` | No | Batch added |
| Expected VAE input (NV-Generate-CTMR `AutoencoderKlMaisi`) | `[B, 1, 240, 240, 176]` | Same as the batch above — this Dataset's output is already channel-first, single-channel, `(X,Y,Z)`-ordered, matching the VAE's documented `in_channels=1` (`docs/design/nv_generate_mr_brain_audit.md` §1.1) and its own never-permuted `(X,Y,Z)` convention (`docs/design/08_....md` Part C) — **no further permutation needed at this boundary** | Same | Same | Model-dependent (fp16/fp32/bf16) | No | Already present |
| Expected VAE latent | `[B, 4, 60, 60, 44]` | `4` = `latent_channels` (VERIFIED, `config_network_rflow.json:12-38`); spatial dims = `target_shape / 4` in `(X,Y,Z)` order (VERIFIED 4× downsample on every axis, `docs/design/nv_generate_mr_brain_audit.md` §1.1-1.2 and `docs/design/08_....md` Part C's own T1w-axial worked example: 240/4=60, 176/4=44, both exact) | N/A (latent space) | N/A | Model-dependent | No | Already present |

**Why the (D,H,W)→(X,Y,Z) permute exists, and why it happens exactly once,
here:** `data.py`'s `(D,H,W)=(S,R,A)` convention is not an arbitrary choice —
it is load-bearing for the *original* contrastive-pretraining pipeline's
VJEPA/VJEPA2.1 video encoders, which hardcode "the axis right after channel
is the slice/temporal axis" in three independent places
(`ResidualTemporalDownsample`'s asymmetric `Conv3d(kernel=(3,1,1))`, VJEPA's
`tubelet_size` vs `patch_size` split, the sliding encoders'
`x.split(chunk_size, dim=2)`) — see
`docs/design/08_dataset_recommendation_manifest_and_axis_order.md`'s "Why
data.py's (D,H,W) reordering exists" section for the full trace with
file:line citations. That requirement does not apply to NV-Generate-CTMR,
which never permutes past its own `Orientationd(RAS)` `(X,Y,Z)` order (Part
C of the same doc) — so instead of leaving every future NV-Generate-CTMR
training/inference script to remember an external permute,
`MRReportToVolumeDataset.__getitem__` performs it once, internally, as its
very last step, immediately before returning the sample dict. `data.py`
itself is unmodified — the original contrastive pipeline still gets its
required `(D,H,W)` tensors unchanged.

**A note on what this repo's own docs call "256×384×384mm" (contrastive
default) vs. this design's per-bucket FOVs**: the prior dataloader audit
already corrected a plausible-looking mistake here — `target_shape` and
`target_spacing` are **both** `(D,H,W)`-paired internally, so e.g. the
contrastive default's physical FOV is 256mm(S–I)×192mm(R–L)×192mm(A–P),
**not** 256mm applied to every axis (`docs/design/mr_rate_dataset_and_dataloader_implementation.md`
§7.2). This design's per-bucket table is built the same way — every FOV
cited above is `target_shape[i] × target_spacing[i]` in the internal (D,H,W)
frame, verified per-axis, not assumed; only the *final* Dataset output
(from the table above) is reindexed to (X,Y,Z).

---

## Cache and performance behavior

- **`geometry_mode="fixed"`**: fully compatible with the existing
  `preprocess_volumes.py` `.npz` cache, unchanged — `MRReportToVolumeDataset`
  reads the same `<preprocessed_dir>/native_space/<study_uid>.npz` (`volumes:
  [N, D, H, W]`) and indexes it at `row.cache_index` (the series' position
  within `discover_subjects`' sorted `image_paths` for that study, captured
  at manifest-build time — matches `preprocess_volumes.py`'s own stacking
  order 1:1 by construction, since both call `discover_subjects` the same way).
- **`geometry_mode="per_modality_plane"` (default): live NIfTI only, `.npz`
  caching not supported.** This is a genuine, explicitly documented
  limitation, not an oversight: the existing cache format has one
  shape/spacing for an entire cache directory (`build_cache_manifest`'s
  `CACHE_CONFIG_KEYS`, `data.py`), whereas per-bucket geometry needs a
  different shape per (modality, plane). `MRReportToVolumeDataset.__init__`
  raises a clear `ValueError` if `use_preprocessed=True` is combined with
  this mode, rather than silently reading a mismatched cache.
  Extension path (not implemented): namespace the cache directory by bucket
  key (e.g. `<preprocessed_dir>/native_space__T1w__AXIAL/`), each built
  independently by pointing a **per-bucket-filtered** subject list at the
  existing, unmodified `preprocess_volumes.py` — deliberately not built now
  (would require `preprocess_volumes.py` to accept a series-level, not
  subject-level, filter/shape assignment, a larger change than this task's
  "smallest coherent adaptation" scope).
- **Manifest build cost**: one header-only NIfTI open per eligible series
  (no voxel decode) — cheap, but not free; for the full ~636K-series
  dataset this is ~636K header reads, embarrassingly parallel across a
  SLURM CPU array job (same pattern already used for `preprocess_volumes.py`,
  per `docs/design/fau_hpc_execution_profile.md`'s Profile/feasibility tables) —
  not run in this session (would touch real, un-extracted archive data; see
  "Tests and smoke-test results").

---

## Configuration reference

`R2VDatasetConfig` (`data_r2v.py`), one dataclass, no scattered constants:

| Field | Default | Notes |
|---|---|---|
| `split` | `"train"` | |
| `report_sections` | `("findings", "impression")` | ASSUMED default (see rationale above) |
| `geometry_mode` | `"per_modality_plane"` | or `"fixed"` |
| `geometry_spacing_mm` | `(1.0, 1.0, 1.0)` | only used in `per_modality_plane` mode |
| `geometry_divisible_by` | `16` | stricter than the 4× actually enforced by the VAE |
| `fixed_target_shape` / `fixed_target_spacing_mm` | `(256,384,384)` / `(1.0,0.5,0.5)` | only used in `fixed` mode; identical to `MRReportDataset`'s own defaults |
| `posterior_shift_mm` | `15.0` | unchanged from `MRReportDataset`'s default — a physical defacing-compensation constant, independent of the target grid |
| `normalizer` / `normalizer_kwargs` | `"percentile"` / NV-Generate-CTMR-matching kwargs | see "Compatibility" table |
| `series_selection` | `"all"` | or `"one_per_study_deterministic"` / `"one_per_study_random"` |
| `dtype` | `torch.bfloat16` | |
| `seed` | `0` | used by `one_per_study_random` and `GeometryBucketBatchSampler` |

`build_r2v_manifest.py` CLI flags map 1:1 onto `build_manifest_rows`'s
keyword arguments (`--metadata_csv`, `--splits_csv`, `--excluded_modalities`,
`--include_derived`, `--include_localizer`, `--dry_run`).

---

## Challenge-adapter boundary

```
MR-RATE filesystem  →  ManifestRow (persistent index)  →  Dataset sample (dict)  →  collated batch  →  [future challenge adapter]
```

- A future challenge adapter needs only to produce `ManifestRow` objects (or
  a manifest CSV in the documented schema) and something implementing the
  `ReportRecord`-yielding duck-typed interface (`__contains__`,
  `__getitem__`) — `MRReportToVolumeDataset`, `collate_fn_r2v`, and every
  downstream training-loop concern are completely decoupled from *how* those
  two inputs were produced. No speculative challenge filenames, directory
  layouts, or packaging assumptions are hardcoded anywhere in `data_r2v.py`.
- This mirrors the (not-implemented) `SubmissionAdapter`/`ChallengeExample`
  interface sketch already proposed in `docs/design/challenge_contract.md` — that
  sketch's `ChallengeExample` maps naturally onto one `ManifestRow` +
  `ReportRecord` pair; this design does not duplicate or contradict it.

---

## Tests and smoke-test results

**Synthetic tests (`contrastive-pretraining/tests/test_data_r2v.py`, 40+
tests) — all passing.** Cover: `series_id_from_path` (including the
non-matching-prefix fallback), manifest construction (eligible count,
exclusion of MRA/derived, multi-series studies, per-study eligible-count
variation, split labeling, native-geometry population, the no-metadata_csv
fallback-with-warning path), manifest CSV round-trip, `is_eligible` (every
branch), both `ReportRecord` stores (missing-report, malformed-JSONL-line,
empty-sentences, section-composition, unknown-section-raises), the geometry
table (hand-computed shape check, FOV-never-truncates invariant, fallback
bucket, divisibility), `GeometryPolicy` (fixed vs. per-bucket, unknown-pair
fallback, invalid-mode error), `MRReportToVolumeDataset` (split filtering,
missing-report exclusion, sample field/shape/dtype checks, report-section
selection), all three `series_selection` modes (including
reproducibility-across-construction and variation-across-epochs for the
random mode), `collate_fn_r2v` (batch size 1 and >1, mismatched-shape error
message, empty-batch error, real `DataLoader` end-to-end),
`GeometryBucketBatchSampler` (no cross-bucket mixing, full coverage without
drop_last), modality-balance weighting, `.npz` cache compatibility
(`geometry_mode="fixed"` round-trip against a real `preprocess_nii`-built
cache, correct `cache_index` retrieval, and the explicit rejection of
`use_preprocessed=True` with `geometry_mode="per_modality_plane"`), and a
dedicated NIfTI-orientation regression test (`data.read_native_geometry`
against a synthetic NIfTI with a non-trivial, hand-constructed axis
permutation, cross-checked against `load_and_resample_nii`'s independently
established convention, plus a `get_fdata()`-must-not-be-called check).

**Backward compatibility:** the full pre-existing test suite (106 tests
across `test_imports.py`, `test_data.py`, `test_preprocess_cache.py`,
`test_pooling.py`, `test_mr_rate_model.py`, `test_fusion_modes.py`,
`test_vision_encoder.py`) plus this session's additions (2 new
`PercentileNormalizer` `clip=` tests in `test_data.py`, the new
`test_data_r2v.py` file, and a `TestBackwardCompatibility` class that
constructs a real `MRReportDataset` alongside `data_r2v` imports) —
**193 tests, 0 failures** — run via
`/apps/python/3.12-conda/envs/pytorch2.5.1/bin/python3 -m pytest --no-cov`
(the environment the prior audit session also used; `pytest`/`pytest-cov`
were not pre-installed there and were added via `pip install --user` during
this session, with the user's explicit permission after that network access
was flagged).

**Real-data smoke test — partial, and here is exactly why:**
- **Report side: run successfully against real data.**
  `contrastive-pretraining/data/findings_sentences.jsonl` (already extracted,
  not an archive) is real, checked-into-git report-derived sentence data —
  `SentenceJSONLReportStore` loaded **97,887 real records** from it (a count
  consistent with the documented `has_report=99.86%` of ~98,334 studies) and,
  over a 50-record sample, produced non-empty composed text for all 50
  (length range 460–1657 characters). **No report text or identifiers were
  printed** — only counts and length statistics.
- **Image side: not run, and could not be run without violating an explicit
  task constraint.** Both real local MRI roots
  (`/hnvme/workspace/b180dc29-MR-RATE`, `/hnvme/workspace/y100dc19-MR-Rate-raw`)
  are still packaged as un-extracted archives (`batchNN.tar`-of-per-study-zips,
  and WebDataset `shard-*.tar` respectively — independently confirmed
  unchanged since the prior audit, `docs/design/mr_rate_local_audit.md` §3).
  `discover_subjects()` (and therefore `build_manifest_rows`) requires a real
  extracted directory tree and returns 0 subjects against either root as-is
  (already verified live by the prior audit). Extracting either root, even
  partially, was judged out of scope: the task's own non-goals list
  "preprocessing the complete local dataset" and "do not perform a full
  dataset scan," and a **prior session in this same project already had one
  accidental real-identifier leak into a transcript while peeking inside
  these exact tar archives** (`docs/design/audit_progress.md`'s "Known incident"
  section) — repeating that kind of archive inspection for a smoke test
  whose main value (confirming the shared preprocessing primitives behave
  correctly on realistic geometry/intensity) is already covered by (a) the
  prior audit's own extensive live execution against synthetic data built to
  mirror real characteristics (`docs/design/mr_rate_dataset_and_dataloader_implementation.md`
  §11) and (b) this session's own targeted orientation/permutation regression
  test, was judged not worth the risk. **Reported here explicitly, per the
  task's own instruction to report tests that could not run and why**, rather
  than silently skipped.
- **Notable, out-of-scope observation:** `findings_sentences.jsonl` is real,
  presumably de-identified but still patient-report-derived text, already
  committed to this git repository's history (confirmed not `.gitignore`d,
  `git status` shows it clean/tracked). No action was taken on this (out of
  scope for this task), but it's worth the user's attention independent of
  this dataset/dataloader work.

---

## Known limitations

1. **`geometry_mode="per_modality_plane"` has no cache support** (see "Cache
   and performance behavior") — live-NIfTI-only in that mode for this version.
2. **No brain-vs-spine filtering possible** — `body_region` is unreleased.
3. **No contrast-enhancement conditioning** — not available in the release at all.
4. **No per-sequence report attribution** — the study-level report ↔
   series-level image label-noise risk is mitigated (via modality/plane
   conditioning) but not eliminated; an LLM-based per-sequence attribution
   pass is a natural future step, out of scope here.
5. **`build_manifest_rows` requires read access to every eligible series'
   NIfTI header** — not zero-I/O; acceptable but not free at full-dataset scale.
6. **The geometry table's axis-assignment rule was verified
   self-consistent across all 15 published rows, but the underlying
   NV-Generate-CTMR axis convention (whether its own internal array order
   truly matches this codebase's RAS interpretation beyond the diagonal
   affine `save_image()` writes) is INFERRED, not independently reproduced
   against the installed `monai` source** (which is not vendored locally,
   per `docs/design/nv_generate_mr_brain_audit.md` §0) — flagged, not re-derivable
   without that source.
7. **No end-to-end real-image smoke test was run** (see above) — the live
   NIfTI path is verified correct via its reused, already-tested primitives
   plus a dedicated synthetic orientation test, not via a real MR-RATE file.
8. **Not tested under `torch.utils.data.DistributedSampler`** — nothing in
   this Dataset's design should conflict with it (ordinary map-style
   dataset, `__len__`/`__getitem__` only), but combining `DistributedSampler`
   with `GeometryBucketBatchSampler` correctly (so different ranks see
   correctly-partitioned, still-bucket-homogeneous batches) was not
   implemented or tested — flagged as a documented, not-yet-solved extension
   point for multi-GPU training.

## Open decisions that depend on the final challenge rules

All nine items already catalogued in `docs/design/challenge_contract.md`'s
assumption ledger (A1–A9) still apply unchanged — this design's manifest/
Dataset boundary was built specifically so that resolving any of them (e.g.
case granularity, whether modality/plane are given vs. must be inferred,
output serialization convention) only requires changing the challenge
adapter layer described above, not `data_r2v.py` itself. Additionally,
specific to this design:
- Whether `report_sections=("findings","impression")` is the right default
  primary conditioning text, or whether the real challenge/task expects the
  raw, unsegmented `report` field instead.
- Whether the 1.0mm-isotropic-spacing, per-(modality,plane)-bucket geometry
  table should be revisited once real challenge geometry requirements (if
  any) are published — `data_schema: {}` was still empty as of the most
  recent locally-available platform artifact.
- Whether MRA should really be excluded, if a future challenge phase
  requires generating it despite NV-Generate-MR-Brain's current lack of support.

---

## Source-code index

**New files:**
- `contrastive-pretraining/scripts/data_r2v.py` — geometry policy (`GeometrySpec`,
  `GeometryPolicy`, `build_geometry_table`, `NV_BRAIN_FOV_MM`), report stores
  (`ReportRecord`, `StructuredReportStore`, `SentenceJSONLReportStore`),
  metadata/eligibility (`SeriesMeta`, `MetadataStore`, `series_id_from_path`,
  `is_eligible`), manifest (`ManifestRow`, `build_manifest_rows`,
  `write_manifest_csv`, `read_manifest_csv`), config (`R2VDatasetConfig`),
  the dataset (`MRReportToVolumeDataset`), sampling
  (`compute_modality_balance_weights`, `get_modality_balanced_sampler`), and
  collation (`collate_fn_r2v`, `GeometryBucketBatchSampler`).
- `contrastive-pretraining/scripts/build_r2v_manifest.py` — CLI wrapper.
- `contrastive-pretraining/tests/test_data_r2v.py` — new test suite.

**Modified files (additive/backward-compatible only):**
- `contrastive-pretraining/scripts/data.py` — `PercentileNormalizer.__init__`
  gained `clip=True` (default preserves prior behavior); new functions
  `read_native_geometry` (L203-220), `load_all_splits` (L222-235),
  `load_split_uids` (L238-247, now built on `load_all_splits`).
- `contrastive-pretraining/tests/test_data.py` — two new tests in
  `TestPercentileNormalizer` for the `clip=` keyword.
- `contrastive-pretraining/README.md` — repository-structure listing, test
  table, and a new "Report-to-Volume Dataset" section.

**Read, not modified (cited above with line numbers):**
- `contrastive-pretraining/scripts/data.py` (`discover_subjects`,
  `load_and_resample_nii`, `resize_array`, `Normalizer` classes, `crop_or_pad`,
  `preprocess_nii`, `build_cache_manifest`/`validate_cache_manifest`,
  `MRReportDataset`, `collate_fn`)
- `contrastive-pretraining/scripts/data_inference.py` (`collate_fn_infer`)
- `contrastive-pretraining/scripts/preprocess_volumes.py` (whole file, reused unchanged)
- `contrastive-pretraining/scripts/mr_rate_trainer.py:430-435` (tokenization convention)
- `contrastive-pretraining/tests/conftest.py`, `tests/test_data.py` (existing
  fixture/test conventions this new suite follows)
- `data-preprocessing/src/mr_rate_preprocessing/configs/config_metadata_columns.json`
  (released metadata schema: `column_order_prefix`, `column_rename_map`, `columns_to_drop`)
- `data-preprocessing/src/mr_rate_preprocessing/mri_preprocessing/modality_filtering.py:301-319,352-369,488-501`
  (`load_image_properties`'s pre-reorientation axis order; `construct_modality_name`'s series_id format)
- `data-preprocessing/src/mr_rate_preprocessing/mri_preprocessing/dcm2nii.py:71-80`,
  `brain_segmentation_and_defacing.py:169,176`, `zip_and_upload.py:70-77,129`,
  `prepare_metadata.py:587-590,610-618` (filename↔series_id chain of evidence)
- `NV-Generate-CTMR/docs/inference.md` (recommended-FOV tables, modality codes)
- `NV-Generate-CTMR/scripts/transforms.py:42-71` (MRI intensity transform)
- `docs/design/mr_rate_local_audit.md`, `logs/mr_rate_audit_metrics.json`,
  `docs/design/report2volume_gap_analysis.md`, `docs/design/recommended_next_steps.md`,
  `logs/proposed_model_manifest_schema.json`,
  `docs/design/mr_rate_dataset_and_dataloader_implementation.md`,
  `logs/mr_rate_dataset_contract.json`, `docs/design/nv_generate_mr_brain_audit.md`,
  `docs/design/challenge_contract.md`, `docs/design/fau_hpc_execution_profile.md`,
  `docs/design/audit_progress.md` (all prior-session deliverables this design cites)
