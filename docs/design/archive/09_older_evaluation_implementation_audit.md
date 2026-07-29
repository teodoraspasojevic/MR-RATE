# Audit: the older evaluation implementation (`~/NV-Generate-CTMR`)

Read-only audit, written before porting anything. Covers the `evaluation/` package built in a
sibling repository during a prior work session, **including this session's own earlier additions
to it** (`evaluate_r2v.py`, `diversity_diagnostics.py`, `save_representative_examples.py`), since
those were built against the wrong repository per the clarified instructions and are in scope for
the same audit. Nothing in this document is a plan for the new implementation — see
[`10_evaluation_geometry_contract_and_shape_mismatch_policy.md`](10_evaluation_geometry_contract_and_shape_mismatch_policy.md)
for that.

**Location**: `/home/hpc/y100dc/y100dc19/NV-Generate-CTMR/evaluation/` (a separate git repository,
`teodoraspasojevic/NV-Generate-CTMR`, not a submodule of this repo). Its own `EVALUATION_STATUS.md`
and `docs/evaluation_results_report.md` document what it already computed (full-test-split VAE
reconstruction + DM+VAE generation, n=5,536, job IDs and result paths recorded there) — those
results stand as historical evidence but **are not treated as final** by this audit; see §"Trust
in previously produced metric values" below.

## 1. Available metrics

| Metric | File:function | Math source |
|---|---|---|
| MAE / MSE / PSNR / NCC | `metrics.py: mae/mse/psnr/ncc` | plain numpy |
| SSIM (3D whole-volume, 2D per-plane mean) | `metrics.py: ssim_3d/ssim_2d_mean` | `skimage.metrics.structural_similarity` |
| Relative intensity error, edge-preservation, Laplacian-variance ratio, HF-energy ratio | `metrics.py` | numpy + `scipy.ndimage` |
| Intensity percentiles, through-plane consistency | `metrics.py` | numpy, descriptive only |
| MedicalNet-feature Fréchet distance (3D) | `distribution_metrics.py: frechet_distance_with_diagnostics` + `MedicalNetFeatureExtractor` | `monai.metrics.FIDMetric` + `monai.networks.nets.resnet10` (Med3D, Chen et al. 2019) |
| ImageNet-slice FID / axial-slice FVD-style | same `frechet_distance_with_diagnostics` + `InceptionFeatureExtractor` | `monai.metrics.FIDMetric` + `torchvision.models.inception_v3` |
| Inception Score | `distribution_metrics.py: inception_score` | plain numpy (Salimans et al. 2016) |
| Precision / Recall / Density / Coverage | `distribution_metrics.py: precision_recall_density_coverage` (added this session) | plain numpy k-NN (Kynkäänniemi et al. 2019 + Naeem et al. 2020) |
| Report-image similarity ("CLIPScore-like") | `evaluate_r2v.py: report_image_similarity` (added this session) | pluggable, always returns `available=False` — no model exists |

## 2. Expected input files and tensors

- Reads real volumes exclusively through `data.mrrate.loader.resolve_data_dict`/`load_mrrate_sample`
  (its OWN separate dataset layer, not this repo's `data_r2v.py`) — a manifest row + a staged
  directory or live `/anvme` archive → a `nibabel`-loadable `.nii.gz` path.
- VAE/generated tensors: `torch.Tensor`, produced in-process by `autoencoder.encode/decode` or
  `scripts.diff_model_infer.run_inference` — never round-tripped through a saved file during
  the main evaluation loop (only `save_representative_examples.py`, added this session, writes
  `.nii.gz` for a curated subset).
- R2V predictions (`evaluate_r2v.py`, added this session): a CSV (`case_id,sequence,
  prediction_path`) pointing at externally-saved `.nii.gz` files, loaded via plain `nib.load`.

## 3. Expected axis order

Tensor axis order: `(D, H, W)` internally in `data.mrrate.loader`'s own preprocessing, reindexed by
`Orientationd(axcodes="RAS")` (MONAI transform, `evaluate_vae.py:120`) so that by the time a
volume reaches any metric function it is a plain `(X, Y, Z)`-shaped array with axis
0=Right-Left (sagittal), axis 1=Anterior-Posterior (coronal), axis 2=Superior-Inferior (axial) —
`visualize.py`'s and `distribution_metrics.py`'s slice-selection functions hardcode `axis=2` for
"axial." **Verified directly against a real staged file this session** (`nib.aff2axcodes` on
`.../staged/smoke/.../*_atlas_t1w-raw-axi.nii.gz` → `('R','A','S')` already on disk), not merely
assumed from the docstring.

## 4. Intensity preprocessing

NVIDIA's own `scripts.transforms.define_fixed_intensity_transform("mri")` (imported and reused
unchanged, `evaluate_vae.py:65,115`) — a `ScaleIntensityRangePercentilesd`-based transform,
producing values nominally in `[0, 1]`. `metrics.py`'s `data_range=1.0` default assumes this.

## 5. Feature encoders and checkpoints

| Encoder | Checkpoint | Provenance recorded? |
|---|---|---|
| MedicalNet ResNet-10 (Med3D) | `pretrained/medicalnet/resnet_10_23dataset_statedict.pth`, sha256 `2d54af72...` | Yes — `medicalnet_checkpoint_provenance.json` (source URL, sha256, architecture) |
| torchvision Inception-v3 | `Inception_V3_Weights.IMAGENET1K_V1` (downloaded, not vendored) | Yes — version string only, no local sha256 (HF/torchvision-managed) |

## 6. How it finds ground-truth/generated pairs

- **VAE-only**: paired *by construction* — `encode(x)` then `decode()` of the same in-memory `x`
  the loader just produced for that `(case_id, sequence)` manifest row. No separate identifier
  matching is needed or performed.
- **DM+VAE**: explicitly **not paired** — generation is unconditional (metadata/class-label only),
  and the code never attempts to match a generated volume to any specific real case for voxelwise
  purposes (confirmed, this is correct and should be preserved as a policy, not a gap).
- **R2V evaluator** (added this session): pairs by `(case_id, sequence)` string equality against
  the manifest row — **does not use a stable series identifier check beyond that string match**,
  does not reject duplicate predictions for the same key (last-one-wins if a predictions CSV had
  two rows for the same case/sequence — never checked), and does not check split/modality
  consistency between the prediction and the manifest row it matched.

## 7. Batch handling

No batching at all — every entry point (`evaluate_vae.py`, `evaluate_dm_vae.py`, `evaluate_r2v.py`)
loops one `(case, sequence)` at a time, encodes/decodes/generates one volume, computes metrics, and
moves on. There is no `DataLoader`, no collate function, no batch dimension anywhere in the
per-case tensors (`x.unsqueeze(0)` only adds a size-1 batch dim immediately before the model call).

## 8. How it aggregates metrics

`aggregate()`/`aggregate_dm()` (`evaluate_vae.py:337`, `evaluate_dm_vae.py:260`): group rows by
sequence and by `"overall"`, mean/std per metric, dropping non-finite values from that one metric
only (`isinstance(x,(int,float,np.floating)) and np.isfinite(x)`). `aggregate_full_test.py` pools
raw feature vectors across Slurm-array shards and recomputes FID/diversity **once** over the full
population, not an average of per-shard numbers — this pattern (correct, reusable) is described in
§17.

## 9. FID support

Yes — 3D (MedicalNet-feature) FID over the whole volume, and a slice-level 2D variant. **Not
literal 2.5D FID** in the sense of separately-scored axial/sagittal/coronal planes combined by a
documented weighting rule — only axial mid-slices/axial-slice-sequence features are used
(`distribution_metrics.py: axial_mid_slice`, `non_empty_axial_slices`, both hardcode `axial_axis=2`
and never touch sagittal/coronal). This is a real gap relative to what a genuine 2.5D FID
implementation should provide (see §"conceptually useful, should be reimplemented" below).

## 10. Clinical or image-text metrics

None with real backing. MedicalNet-feature FID is the closest thing to a "clinical" proxy and is
explicitly self-labeled "Category B: conditionally interpretable, not validated" in its own
docstring. `report_image_similarity()` (added this session) is a pluggable interface that always
returns `available=False` — correctly does not fabricate a score.

## 11. Output formats

Per-run: `per_case_metrics.csv`, `failures.csv`, `per_sequence_metrics.json`,
`aggregate_metrics.json`, `distribution_metrics.json`, `environment.json` (checkpoint sha256s,
library versions, GPU), `run_config.json`, `wandb_configuration.json`, `distribution_features/
{features.npz, feature_index.csv}`, `visualizations/*.png`. `save_representative_examples.py`
(added this session) additionally writes `volumes/*.nii.gz` + `selection_manifest.json`.

## 12. Tests and Slurm scripts

- Tests: plain assert-based scripts (no pytest), run via `python -m evaluation.test_*`. This
  session added `test_diversity_diagnostics.py` (7 tests) and `test_evaluate_r2v.py` (14 tests,
  including one true end-to-end `main()` dry run) — both pass, and the dry run caught two real
  bugs (a numpy-bool JSON-serialization crash, a missing `import nibabel`) before they could fail
  a real job. No geometry-focused regression suite exists for the *shape-mismatch* policy itself
  — `test_evaluate_r2v.py`'s geometry tests check that `validate_geometry`/`center_crop_or_pad`
  behave as *implemented*, not that the implemented policy is the *correct* one (see §"invalid and
  should not be retained" below — the policy itself needed correcting, and the tests faithfully
  tested the wrong policy).
- Slurm: 12 sbatch scripts (smoke/representative/full-array/aggregate per pipeline + a new
  `nvidia_mri_vae_save_examples.sbatch`), account `y100dc`, QoS `mq_health`, partition `h200` —
  same NHR@FAU account this repository also has access to (confirmed via
  `docs/design/fau_hpc_execution_profile.md`), so the *infrastructure* conventions (account/QoS/
  bind-mount pattern) are directly reusable even though the scripts themselves are not.

## 13. Dependencies and licenses

`NV-Generate-CTMR`'s own source is Apache 2.0 (`LICENSE`); model weights are under
NVIDIA Open Model License (CT/MR-Brain) or NVIDIA Non-Commercial (MR) — see its README's license
table. `evaluation/`'s own files (this session's additions and the prior session's) carry no
separate license header but live inside the Apache-2.0-licensed repository. **Nothing in
`evaluation/` is verbatim-copied NVIDIA source** — it is new code written against NVIDIA's public
`scripts/` API (`load_models`, `run_inference`, etc.), so porting its *logic* into this repository
carries no NVIDIA copyright-attribution obligation; only the vendored `NV-Generate-CTMR/scripts/`
copy already present in *this* repo (added by commit `cf5cf1f`, its own `LICENSE`/`LICENSE.weights`
intact) needs attribution, and that copy is untouched by this work.

Key package dependencies observed: `torch`, `monai>=1.5.0` (for `resnet10`, `FIDMetric`,
`MaisiDownsample`), `torchvision` (Inception-v3), `nibabel`, `scipy`, `scikit-image`, `wandb`
(optional, degrades to no-op). None of these are currently in
`contrastive-pretraining/requirements.txt` — installing `monai` there is new-but-necessary.

## 14. Assumptions that do not hold for this repository's R2V pipeline

| Assumption in the old code | Reality in `data_r2v.MRReportToVolumeDataset` |
|---|---|
| Data loaded via `data.mrrate.loader` (its own tar/zip archive layout, `/anvme`-rooted) | This repo's R2V data loads via `data_r2v.py` + `r2v_storage.py` (a *different* archive-backed layout — `batchNN.tar` → per-study `.zip`, or WebDataset `shard-*.tar` — both already implemented here, not to be redesigned) |
| One `(case_id, sequence)` = manifest row, resolved fresh per call | `MRReportToVolumeDataset` samples are pre-selected at construction (`series_selection` policy) and served through a real `torch.utils.data.Dataset.__getitem__` + `DataLoader` |
| Target shape/spacing = whatever the model's own fixed config says (`dim=[256,256,256]`) | This repo's Dataset has its *own* geometry-bucket system (`GeometryPolicy`, `NV_BRAIN_FOV_MM`, `geometry_mode="per_modality_plane"` default) producing a *different*, modality/plane-dependent `target_shape`/`target_spacing_mm` per sample — the old code's implicit "one fixed shape" assumption does not hold here |
| Real image tensor and geometry fields are separate ad hoc variables the eval script computes itself | This repo's Dataset already returns `target_shape`/`target_spacing_mm`/`native_shape`/`native_spacing_mm`/`native_fov_mm`/`study_key`/`series_key` as first-class sample-dict fields — duplicating that computation in evaluation code would violate "do not duplicate these definitions" |
| `float32` tensors throughout | This repo's Dataset defaults to `torch.bfloat16` (`R2VDatasetConfig.dtype`) — every metric function needs an explicit `.float()` cast before any numpy conversion |
| Report = one row, one series (VAE-only/DM+VAE never touch reports at all) | This repo's R2V dataset can and does repeat the *same* report across *multiple* series rows (`series_selection="all"`, the default) — a report-to-volume evaluator must handle one-report-to-many-series correctly, not assume a 1:1 report:series mapping |
| Axis order `(X,Y,Z)` = `(R,A,S)`, `channel` implicit | Same convention, confirmed independently for this repo's Dataset output (`_dhw_to_xyz`, `image.permute(0,2,3,1)`) — **this one assumption does carry over correctly**, a rare case of full compatibility |

## 15. Classification of every component

| Component | Classification | Notes |
|---|---|---|
| `metrics.py` (mae/mse/psnr/ncc/ssim/edge-preservation/laplacian/hf-energy/intensity-percentiles/through-plane-consistency) | **Reusable after adaptation** | Pure numpy/scipy/skimage, dataset-agnostic — only needs float32 casting before the call site, not internal changes |
| `distribution_metrics.py`'s Fréchet-distance math (`frechet_distance_with_diagnostics`) and feature extractors (`MedicalNetFeatureExtractor`, `InceptionFeatureExtractor`) | **Reusable after adaptation** | Correct, careful (bootstrap CI, rank-deficiency detection) — needs float32 casting and a documented feature/geometry fingerprint added to the cache (was missing — see below) |
| `distribution_metrics.py`'s `precision_recall_density_coverage`/`compute_diversity_metrics` (added this session) | **Reusable unchanged** | Pure numpy, no dataset assumptions at all |
| `evaluate_vae.py`'s `required_spatial_divisor` + `DivisiblePadd(method="end")` + `crop_to_original` pattern (`:93-134`) | **Conceptually correct, reimplement against the new Dataset** | This is *already* the right kind of "decoder-boundary" correction (deterministic, end-only padding with an exact inverse-crop) — but it operates on the old loader's raw-NIfTI-then-transform path; the new implementation needs the equivalent logic driven by `target_shape` already coming out of `data_r2v`'s Dataset, with the padding amount explicitly recorded per sample (the old code never recorded it either — see §16) |
| `visualize.py` (paired + unpaired panels) | **Reusable after adaptation** | Same axis convention; needs float32 casting and no assumption about a `mask` argument (this repo's R2V Dataset has no per-sample brain-mask field) |
| `aggregate_full_test.py`'s pool-then-recompute-once pattern for sharded FID | **Reusable after adaptation** | The *pattern* (never average per-shard FID; pool raw features once) is correct and important — the file itself is tied to the old sharding/manifest format and should be reimplemented against this repo's manifest/shard conventions |
| `sharding.py: shard_rows` | **Reusable unchanged** | Three-line, dataset-agnostic contiguous-block splitter |
| `wandb_logging.py` | **Reusable unchanged** | Fully dataset-agnostic wrapper, never crashes the caller |
| `evaluate_r2v.py`'s prediction-manifest schema concept (`case_id,sequence,prediction_path`) | **Conceptually useful, reimplement** | The *shape* of the idea (a CSV mapping identifiers to saved prediction files) is right; the identifier needs to be this repo's real `(study_key, series_key)`, and duplicate/ambiguous detection needs adding (currently absent — see §16) |
| `evaluate_r2v.py`'s `validate_geometry()` (shape/spacing/affine boolean check) | **Conceptually useful, reimplement** | Checking shape/spacing/affine is the right *idea*; the new version needs the full `GeometryRecord` contract (orientation, axis meaning, crop/pad provenance, valid-region bounds — none of which the old function checks) |
| `evaluate_r2v.py`'s `center_crop_or_pad()` used via `--center-crop-pad` | **Invalid, should not be retained as-is** | See §16 — this is exactly the "blind center crop" pattern the new policy must reject by default |
| Inception Score as the primary diversity signal | **Reimplement/deprioritize** | Already superseded in practice by precision/recall/density/coverage (added this session); keep Inception Score only as a legacy/comparison number, never as the primary mode-collapse diagnostic |
| 2.5D FID (as actually implemented — axial-only) | **Conceptually useful, reimplement properly** | Needs genuine per-plane (axial+sagittal+coronal) scoring with a documented, tested slice-selection and volume-vs-slice-weighting rule (§9) |
| No batching anywhere | **Reimplement** | This repo's Dataset/DataLoader/`GeometryBucketBatchSampler`/`collate_fn_r2v` already exist and support real batches within a geometry bucket — the new evaluator should use them, not loop one-by-one like the old code |

## 16. Every location where the old code compares volumes despite incompatible geometry

This is the section the corrected implementation must not repeat.

1. **`evaluate_r2v.py:150-172`, `center_crop_or_pad()`, invoked from `evaluate_one_prediction`
   (`:243-246`) whenever `--center-crop-pad` is passed and shapes differ but spacing/affine
   match.** This crops/pads *purely to make array sizes equal*, with **no provenance** for *why*
   the shapes differ — it does not know (and does not ask) whether the difference is a known,
   invertible encoder/decoder padding artifact or an arbitrary mismatch. This is precisely the
   "blind center crop... because it makes array sizes equal" failure mode the corrected policy
   must reject by default. **Every metric value in this session's own `evaluate_r2v.py` test
   suite that exercises this path (`test_evaluate_one_prediction_shape_mismatch_excluded_by_default`,
   the "with `--center-crop-pad`" branch) is validating the OLD, now-rejected behavior — it is not
   a metric result that was ever used for a reported number (no real R2V checkpoint exists), but
   the *code path* itself must not be ported as-is.**
2. **`evaluate_r2v.py:132` (`GEOMETRY_AFFINE_TOL = 1e-2`) used identically regardless of what the
   affine values actually represent.** A 1e-2 tolerance on raw affine matrix entries conflates
   "spacing is off by 1%" with "origin is off by 1cm" — the tolerance is not decomposed into
   separate, independently-justified rotation/spacing/origin tolerances, so two affines that
   differ only in a large origin shift but agree on spacing/rotation could either pass or fail
   this check somewhat arbitrarily depending on the shift's magnitude relative to spacing. The new
   policy must check spacing, orientation/rotation, and origin/FOV-overlap as **separate,
   independently-tolerable** quantities (see design doc §"GeometryRecord fields").
3. **`evaluate_vae.py`/`evaluate_dm_vae.py` themselves do NOT have this bug** — worth stating
   explicitly since the audit's job is also to confirm what's already correct: `crop_to_original`
   only ever inverts a padding amount the same function call itself computed and applied
   (`DivisiblePadd(method="end")` → `slice(0, original_size)`), so there is no case in either file
   where two independently-sourced volumes of different provenance are silently resized to match.
   This pattern should be **preserved**, not replaced, in the new VAE-reconstruction evaluator.
4. **No world-coordinate resampling path exists anywhere in the old code.** When geometry does not
   match and is not a known decoder-boundary case, the old `evaluate_r2v.py` only had two outcomes:
   silently crop/pad (bug #1) or exclude entirely. There was no intermediate "same anatomy/FOV,
   different grid → resample onto the target grid using the real affines, label the result
   `world_aligned`" option — this is a missing capability, not an active bug, but the absence
   means the old code could never distinguish "these are genuinely incompatible" from "these
   describe the same physical volume on a different voxel grid," which the new policy must.

## 17. Trust in previously produced metric values

Per the explicit instruction not to trust old spatially-invalid comparisons: the historical
full-test-split numbers in `~/NV-Generate-CTMR/results/PHASE_C_FINAL_REPORT.md` and
`docs/evaluation_results_report.md` (VAE-only PSNR/SSIM/FID, DM+VAE FID/diversity) were **all**
computed via the VAE-only/DM+VAE paths described in §16 point 3 above — i.e., they never exercised
the buggy `center_crop_or_pad` path at all (VAE-only never calls it; DM+VAE never computes paired
metrics against real volumes in the first place). **Those specific historical numbers are not
invalidated by the shape-mismatch bug** (bug #1 only affects the not-yet-used R2V evaluator, which
has never scored a real checkpoint). They remain historically informative but are **not this
repository's evaluation** — this repository re-derives its own numbers against the real
`data_r2v` Dataset and vendored NVIDIA code from scratch (§"Recomputed evaluation" in the final
report), not by importing or reusing those old CSV/JSON files.
