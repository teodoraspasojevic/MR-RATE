# Evaluation geometry contract and shape-mismatch policy

Full design reference for `contrastive-pretraining/evaluation/geometry.py`. See
`contrastive-pretraining/evaluation/README.md` for the beginner-friendly summary and how-to-run
instructions; this document is the complete policy.

## 1. Why this exists

The older evaluation implementation (audited in
[`09_older_evaluation_implementation_audit.md`](09_older_evaluation_implementation_audit.md) §16)
had one real bug: `center_crop_or_pad()` would resize a prediction to match a target's array shape
whenever shapes differed but spacing/affine loosely matched, with **no proof** that the shape
difference came from a known, invertible cause. Array-shape equality was being treated as
sufficient evidence of physical alignment. It is not: two arrays can have equal shapes and
represent completely different anatomy (different FOV, different crop), and two arrays can have
different shapes and represent the *same* physical volume (different resampling grids). This
document specifies the replacement policy, implemented in `geometry.py` and enforced by every
`evaluate_*.py` entry point.

## 2. The `GeometryRecord` contract

One record per volume (real Dataset sample, VAE reconstruction, NVIDIA generation, or an
externally-saved prediction), with these fields (`geometry.GeometryRecord`):

| Field | Meaning | Source when built from a Dataset sample | Source when built from a `.nii.gz` file |
|---|---|---|---|
| `shape` | voxel dimensions | `sample["target_shape"]` or `["native_shape"]` | `nib.load(...).shape` |
| `axis_order` | axis labels | `("X","Y","Z")`, this repo's Dataset guarantee | opaque (`axis0/1/2`) -- meaning comes from `orientation` |
| `anatomical_axis_meaning` | what each axis is | `("R-L","A-P","S-I")`, the Dataset's guarantee | derived from `nib.aff2axcodes(affine)` -- **read, not assumed** |
| `spacing_mm` | voxel spacing | `sample["target_spacing_mm"]` | `nib.header.get_zooms()` |
| `orientation` | e.g. `"RAS"` | `"RAS"` (Dataset guarantee, see §3) | `"".join(nib.aff2axcodes(affine))` |
| `affine` | 4x4 NIfTI affine | `None` -- **not available from this Dataset's sample dict** (see §3) | the file's real affine |
| `modality`, `acquisition_plane` | conditioning fields | `sample["modality"]`/`["acquisition_plane"]` | caller-supplied (from the prediction manifest or paired target) |
| `crop_pad` | provenance of any shape change already applied to reach this state | `None` initially; set via `with_crop_pad_applied()` after a proven pad/crop | `None` |
| `preprocessing_version` | which evaluation code version produced this | `EVALUATION_VERSION` constant | `"external_prediction"` (or caller-supplied) |
| `study_key`/`series_key` | identifiers | `sample["study_key"]`/`["series_key"]` | from the prediction manifest / pairing |

`fingerprint()` returns a JSON-safe subset (shape/axis_order/spacing/orientation/modality/plane/
preprocessing_version/geometry-contract-version) used to key feature caches
(`feature_cache.py`) -- deliberately excludes the affine matrix and identifiers.

**Known, honest limitation**: this Dataset's `__getitem__` never returns an affine or a target
FOV field (verified directly, `docs/design/09_....md` §1 "fields explicitly absent"). RAS
orientation is a *guarantee of construction* (`nib.as_closest_canonical`, never bypassed in
`data.py`), not a per-sample value to re-check -- so a `GeometryRecord.from_dataset_sample()`
record's `affine=None` is correct, not a gap to silently paper over. Paired VAE-reconstruction
comparisons never need the affine (see §4); anything that DOES need a real affine (an externally
saved prediction) is built via `from_nifti()` instead, which always has one.

## 3. Why RAS + `(X,Y,Z)`-order compatibility "just works" here

Verified directly against real data this session (not assumed): `nib.aff2axcodes` on a real
staged/streamed MR-RATE volume returns `('R','A','S')`, and `data_r2v.py`'s own module docstring
independently states the same convention for its `image` tensor output
(`axis0=Right-Left, axis1=Anterior-Posterior, axis2=Superior-Inferior`, after
`Orientationd`-equivalent canonicalization in `data.py`). `distribution_metrics.PLANE_AXES` and
`geometry.DATASET_ANATOMICAL_AXIS_MEANING` both import this same convention rather than
re-deriving it (`test_evaluation_distribution_metrics.py::test_plane_axes_matches_dataset_axis_order`
checks they never drift apart).

## 4. The four possible outcomes of `compare_geometry(target, prediction)`

Checked in this order (see `geometry.compare_geometry`):

1. **`INCOMPATIBLE`** immediately if `modality`, `acquisition_plane`, or `anatomical_axis_meaning`/
   `orientation` differ. These are categorical, never a tolerance question.
2. **`STRICT_MATCH`**: shape equal, spacing equal (within `spacing_tol_mm`, default 1mm-scale
   1e-3), and -- if both sides have a real affine -- rotation equal (within `rotation_tol_deg`,
   default 1°) and origin equal (within `origin_tol_mm`, default 1mm). Compute paired metrics
   directly.
3. **`DECODER_BOUNDARY_CORRECTABLE`**: shape differs, but spacing/orientation/rotation/origin all
   agree. This is the "VAE padding" shape of problem. `compare_geometry` itself NEVER resolves
   this -- it only classifies. The caller must supply independently-verifiable provenance:
   - `evaluate_vae.py` has this by construction: it computed the padding itself
     (`geometry.pad_to_divisible`), so it can invert it exactly
     (`geometry.crop_using_record`) and re-check `STRICT_MATCH` afterward as a final assertion.
   - `evaluate_r2v.py` does NOT control how an external prediction was produced. By default it
     **excludes** a `DECODER_BOUNDARY_CORRECTABLE` case. `--known-padding-divisor N` is the only
     escape hatch, and only works if the prediction's actual shape exactly equals what
     `pad_to_divisible(target_shape, N)` predicts -- otherwise it still refuses
     (`test_evaluate_r2v_integration.py::test_wrong_known_padding_divisor_still_refuses_to_guess`).
     There is no other way to "fix" a shape mismatch in this evaluator; blind center-crop/pad
     was deliberately removed (see §1).
4. **`WORLD_ALIGNED_ELIGIBLE`**: shapes/spacing differ (or origin/rotation differ) but both sides
   have a real affine and the world-space FOV overlap (`_fov_overlap_fraction`, a corner-to-corner
   bounding-box intersection in world coordinates) is >= `min_fov_overlap_fraction` (default
   0.98 -- deliberately strict; see the note on cubed volumetric overlap below). By default,
   `evaluate_r2v.py` **excludes** these too. `--allow-world-aligned` opts in:
   `geometry.resample_world_aligned()` uses `scipy.ndimage.affine_transform` with `order=1`
   (trilinear -- the correct interpolation for continuous MRI intensity, never nearest-neighbor)
   to resample the prediction onto the target's exact grid, and the result is written to a
   **separate** `per_case_metrics_world_aligned.csv` file -- `evaluate_r2v.py` never merges
   `strict` and `world_aligned` rows into one aggregate (see `aggregate_metrics.json`'s two
   top-level keys).
5. Anything else (affine missing on one side with a shape/spacing mismatch, or FOV overlap below
   threshold) is **`INCOMPATIBLE`**.

**Note on `min_fov_overlap_fraction`**: this is a *volumetric* (3D) fraction, not linear -- shifting
all three axes by a fraction `f` of the FOV shrinks the overlap by roughly `(1-f)^3`, not `(1-f)`.
A 0.98 threshold is reachable with realistic sub-mm-to-few-mm misalignments on real brain FOVs
(~150-250mm) but is not a "roughly in the right place" threshold -- it means "the same grid to
within measurement noise." See `test_evaluation_geometry.py` for the exact arithmetic used to
pick test cases around this threshold.

## 5. What each evaluator actually does with this

- **`evaluate_vae.py`**: paired by construction (same in-memory tensor, `encode` then `decode`).
  Computes `pad_to_divisible(target_shape, required_divisor)`; if non-trivial, pads before
  `encode`, crops back with the exact same `CropPadRecord` after `decode`
  (`crop_using_record`), then calls `compare_geometry` as a **final safety assertion** (expected
  to always be `STRICT_MATCH` -- if it's not, that's a bug in the padding logic, and the row is
  recorded as a failure rather than silently scored). In practice, on this repository's real
  Dataset, `R2VDatasetConfig`'s default `geometry_divisible_by=16` already matches the VAE's own
  required divisor (verified: real T1w/SAGITTAL bucket shape `(176,256,256)` is already divisible
  by 16), so padding is usually a no-op -- the mechanism exists and is tested regardless, for any
  bucket/config combination where it isn't.
- **`evaluate_generation.py`**: never calls `compare_geometry` between real and generated volumes
  at all -- there is no target to pair against (Mode B: metadata-conditioned unconditional
  generation). Enforced by `test_evaluation_unconditional_policy.py`'s static check that no
  paired-metric function is ever called in that module.
- **`evaluate_r2v.py`**: the full four-outcome policy above, driven by real pairing
  (`pairing.py`) first (identifier-based; see that module for the missing/duplicate/ambiguous/
  mismatched-modality-or-plane/split-mismatch rejection rules, which are a *separate* concern
  from geometry -- a correctly-paired item can still fail geometry, and a geometrically-compatible
  file is worthless if it was paired to the wrong patient).

## 6. Distributional comparisons never require a shared voxel grid

For `evaluate_generation.py` (and the distribution-metrics half of `evaluate_r2v.py`), real and
generated/predicted volumes are never resampled onto a common grid before feature extraction --
each volume is independently fed through the same feature extractor (MedicalNet 3D global-pool,
or Inception-v3 on a 299x299 bilinear resize), which is metric *preprocessing*, not a geometric
alignment claim (see `distribution_metrics.py`'s module docstring). `test_evaluation_distribution_metrics.py::test_medicalnet_extract_accepts_varying_input_shapes`
confirms this explicitly: two volumes of very different native shape produce equal-dimensional
feature vectors, and `CaseFeatures.sequence` (not shape) is what buckets a given case for
aggregation. What IS still enforced: identical preprocessing for both populations, and the same
modality/plane bucket is never silently mixed into a different one's population (see
`evaluate_generation.py`'s per-sequence loop).

## 7. Intensity domain

`R2VDatasetConfig`'s default `normalizer_kwargs` (`clip=False`) was deliberately chosen (see
`data_r2v.py`'s own docstring, and `docs/design/06_....md`) to match NV-Generate-CTMR's own
`ScaleIntensityRangePercentilesd(clip=False)` -- so the Dataset's `image` tensor and the VAE's
expected input are already in the same intensity convention by construction, verified directly
against real data this session (`image.min()≈0, image.max()≈1.83` -- unclipped, as documented,
not `[0,1]`). `metrics.py`'s `data_range=1.0` PSNR reference is therefore a fixed comparability
scale, not a claimed maximum -- documented in that module's own docstring rather than silently
assumed. `evaluate_r2v.py` additionally checks (`intensity_rescaled_before_comparison`, via each
side's 99.5th percentile) whether a prediction appears to be in a different intensity convention
entirely (e.g. a `[0,1000]`-scaled NVIDIA-style output) and rescales defensively -- logged, not
silent -- before computing any paired metric.

## 8. Caching and old-result migration

`feature_cache.FeatureCache` keys every saved feature array set by a fingerprint covering split,
manifest sha256, filters, geometry fingerprint, and encoder config (checkpoint sha256 included).
Any field change is a cache miss, never a silent reuse (`test_evaluation_feature_cache.py`). A
cache produced by the older implementation is automatically incompatible: its fingerprint schema
never had an `evaluation_package`/`cache_schema_version` key at all, so `FeatureCache.load()`
treats it as a miss rather than crashing or reusing it
(`test_old_format_cache_missing_fingerprint_sidecar_is_a_miss_not_a_crash`). Every result
directory this evaluator writes carries `evaluation_version = "mr_rate_evaluation_v1"` in its
`run_config.json` -- old results from `~/NV-Generate-CTMR` carry no such field and are never
mistaken for this evaluator's own output (see the results report for exactly which old numbers
are/aren't still trustworthy).

## 9. Cross-experiment consistency: same cohort, same shape (`cohort.py`)

A geometrically-valid comparison *within* one evaluator run is not the same thing as two
different evaluator runs (VAE reconstruction vs. unconditional generation) being comparable *to
each other*. Two additional, independent guarantees are needed for that, both implemented in
`evaluation/cohort.py` (not part of the per-pair `compare_geometry` policy above, but load-bearing
for the same reason: an apples-to-oranges comparison is just as misleading as a geometrically
invalid one):

- **`select_cohort()`**: picks up to N cases per sequence via a stable `(study_uid, series_id)`
  sort followed by a per-sequence-fresh `RandomState(seed)` draw. This makes the selected cohort a
  pure function of (dataset content, sequences, N, seed) -- independent of manifest row order,
  dict-iteration order, or which other sequences were also requested. Before this function
  existed, `evaluate_vae.py` and `evaluate_generation.py` each pre-subsampled raw manifest rows
  with a *different* `size` argument to the same `numpy.random.RandomState`, which (by how
  `RandomState.choice` consumes its stream) produced entirely different draws even at identical
  `--seed` -- verified as a real, silent inconsistency before this fix, not a hypothetical one.
- **`default_fixed_geometry()`**: both `evaluate_vae.py`'s real+reconstruction pair and
  `evaluate_generation.py`'s real reference population now default to `geometry_mode="fixed"` at
  NVIDIA's own native generation shape/spacing (read once from `diffusion_unet_inference`'s
  `dim`/`spacing` -- the same value `evaluate_generation.py` already needed for its generated
  volumes' own shape, so there is exactly one source of truth, never a hardcoded duplicate). The
  Dataset's per-modality/per-plane FOV buckets (`geometry_mode="per_modality_plane"`, still
  available as an explicit opt-out) remain a legitimate choice for studying the VAE in isolation
  at each modality's tightest-fitting native FOV, but a real, documented tradeoff: it no longer
  matches the generated volumes' fixed shape, so runs in that mode are not cross-comparable with a
  `generation` run and both scripts log a warning when it's selected.

`tests/test_evaluation_cohort.py` and `tests/test_evaluation_cross_evaluator_consistency.py`
verify both properties directly -- the latter drives `evaluate_vae.main`/`evaluate_generation.main`'s
own source to confirm the default is actually wired up, not just documented.
