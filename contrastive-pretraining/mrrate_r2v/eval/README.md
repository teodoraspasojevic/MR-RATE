# `mrrate_r2v.eval` — scoring a generated volume

One pipeline, three tasks, one result layout.

Full pipeline context: [`docs/R2V.md`](../../../docs/R2V.md). This file is about the evaluation
layer itself — use it when you want to know what a metric means, why a case was excluded, or how to
add a task.

---

## The shape of it

```
manifest ──► Dataset (the one training uses) ──► generate ──► score ──► results/
                                     one case at a time, nothing stored
```

`LiveEvaluator.run` in [`live.py`](live.py) is the path. `cli/evaluate.py` and the tests both call
it; there is no second implementation to drift.

**There is no cohort and no prediction set.** Evaluation builds the same
`MRReportToVolumeDataset` that `cli.train_r2v` builds, from the same manifest and the same
`R2VDatasetConfig`, and streams generate-then-score one case at a time. Volumes are never written.
That is the point: the evaluator *cannot* preprocess differently from training, because there is
only one description of the preprocessing and both read it from the same flags.

> The old pipeline froze a cohort directory (`cli.preprocess`), wrote a prediction set
> (`cli.predict_*`), and scored the two against each other with a `cohort_id` handshake. It worked,
> but it had three artifacts that could drift, and one of them did: training ran at
> `posterior_shift_mm=15` while every cohort was built at `0`, displacing 15.8% of test cases with
> nothing able to see it. `run_evaluation` in [`runner.py`](runner.py) remains for reading results
> produced before the change.

### What it does, in order

1. **Selects cases deterministically, with no RNG.** Ordered by `(study_uid, series_id)` within
   each (modality, plane) bucket, round-robined across buckets. Prefix-stable, so `--n-per-bucket`
   returns the first N of the full run rather than a different sample.
2. **Generates**, seeded per case with `stable_seed(--seed, case_id)` — a function of the case, not
   of iteration order, so a rerun, a resume or a different world size reproduces every volume.
3. **Checks the intensity space on the first volume** and refuses the whole run if it is
   `postprocess_mr`'s int16 `[0, 1000]` instead of the percentile `~[0, 1]` the ground truth is in.
   Every metric consumes a 1000x-offset pair happily, so this cannot be caught by reading output.
4. **Checks geometry per case** before any voxelwise metric. A mismatch is excluded with a reason,
   never resized to fit.
5. **Computes exactly the metric groups [`tasks.py`](tasks.py) declares** for the task.
6. **Releases the volumes** and keeps only the metric row and the feature vectors.
7. **Writes one canonical layout** — the same files for every task.

### What replaces `cohort_id`

`run_id` — `run_fingerprint()` hashes the ordered case list together with every preprocessing
setting, the task, the sample cap, the seed and the model checkpoint. Equal `run_id` means the same
cases at the same geometry under the same preprocessing, which is exactly the guarantee `cohort_id`
carried. It is *computed* rather than stored, so unlike `cohort_id` it cannot go stale relative to
anything. It appears in `summary.json` and `run_manifest.json` (and, for backward readability, also
under the old `cohort_id` key).

---

## Tasks

| `--task` | Ground truth per prediction? | Metric groups |
|---|---|---|
| `reconstruction` | yes, that exact input volume | fidelity, perceptual, distribution |
| `report2volume` | yes, the series the report describes | fidelity, perceptual, distribution, report_alignment |
| `generation` | **no** | distribution |

`generation` gets no voxelwise metrics because no real patient corresponds to a generated volume —
comparing pixel by pixel against an arbitrary real scan measures "how different are two random
brains." This lives in `tasks.py` as a property of the task, so it cannot be bypassed by forgetting
a flag. `test_eval_tasks_and_runner.py` asserts it.

### Adding a task

Add a `TaskSpec` to `TASKS` in `tasks.py`. If it needs a new metric family, add a group to
`METRIC_GROUPS` (with `needs_pair` set correctly) and its metric names to `GROUP_METRIC_NAMES`, then
compute them in `runner.compute_paired_metrics`. The runner and CLI pick it up with no changes.

---

## The modules

| Module | Owns | Needs |
|---|---|---|
| `tasks.py` | which metrics are valid for which task | stdlib |
| `runner.py` | the pipeline | numpy |
| `geometry_contract.py` | may these two volumes be compared? | numpy, nibabel |
| `paired.py` | voxelwise + detail metrics on one pair | numpy, scipy (skimage for SSIM, lazily) |
| `distribution.py` | FID, Inception Score, precision/recall/density/coverage | torch, torchvision |
| `features.py` | fingerprint-gated feature cache | numpy |
| `aggregate.py` | per-bucket, per-sequence and overall means | numpy |
| `summary_csv.py` | the CSV deliverable: per bucket, per modality, two overall rows | numpy |
| `anatomy.py` | anatomical plausibility, real population vs produced | numpy, scipy |
| `figures.py` | example slice montages, optional NIfTI export | PIL, nibabel |
| `pairing.py` | identifier matching, for importing external NIfTIs | stdlib |
| `wandb_logging.py` | optional W&B, degrades to a no-op | — |

`__init__.py` re-exports **nothing**, so a heavy dependency in one module never blocks another.
Import what you need:

```python
from mrrate_r2v.eval import paired as M
M.psnr(gt_array, pred_array)          # no cohort, no model, no torch
```

---

## Why shape equality is not enough

The single rule `geometry_contract.py` exists to enforce: **equal `.shape` is never treated as proof
that two volumes occupy the same physical space.**

- **Same shape, different anatomy.** Crop one volume from two different corners and both crops are
  the same size while showing different brain regions. Only the affine catches this.
- **Different shape, same anatomy.** Resample one scan at two spacings and you get two shapes that
  are still legitimately the same scan — just not comparable voxel-by-voxel yet.

So `compare_geometry` checks shape, spacing, orientation, affine rotation, and origin/FOV overlap as
*separate* quantities and returns one of four verdicts:

| Verdict | Meaning | What happens |
|---|---|---|
| `STRICT_MATCH` | same grid | metrics computed |
| `DECODER_BOUNDARY_CORRECTABLE` | shape differs, everything else agrees | corrected only if the caller can *prove* the padding it applied (`CropPadRecord`); otherwise excluded |
| `WORLD_ALIGNED_ELIGIBLE` | different grid, same anatomy | resampled only on explicit opt-in, and reported separately |
| `INCOMPATIBLE` | anything else | excluded, reason recorded |

The implementation this replaced resized two volumes whenever their shapes merely differed, with no
check for *why*. A score computed that way looks precise and proves nothing.

Where padding is legitimately needed — the NVIDIA VAE requires axes divisible by a model-derived
divisor — `cli/evaluate.py:reconstruct` pads at the end of each axis before encoding and crops back the *exact*
recorded amount after decoding, so the reconstruction returns on the cohort's own grid.

---

## What each metric requires of its input

Every metric assumes something about intensity range, spatial extent, and background. Where those
assumptions are violated the number is still *computed* — it is just not measuring what its name
says. This table is the contract; the two sections after it record what our volumes actually look
like and where the mismatches are.

| Metric | Needs | Why | Our volumes satisfy it? |
|---|---|---|---|
| `mae`, `mse` | same grid; any range | scale-dependent, so only comparable within one intensity convention | yes |
| `psnr` | **a true peak equal to `data_range`** | `10·log₁₀(range²/MSE)` — a wrong peak biases the result | **no** — see below |
| `ncc` | same grid | scale/offset invariant by construction | yes |
| `ssim_3d`, `ssim_2d_mean` | `data_range` ≈ true peak; **non-degenerate local variance** | SSIM's luminance/contrast terms collapse on constant regions | **no** — padding is constant |
| `edge_preservation`, `laplacian_variance`, `hf_energy` | same grid; comparable sharpness convention | ratios of derivative energy | yes |
| MedicalNet 3D FID | **per-volume z-score over positive voxels** (Med3D's own preprocessing), foreground-cropped | trained in that domain; GAP dilutes over shared empty space | **no** — see the failure section |
| 2.5D Inception FID | ImageNet mean/std, 299×299, 3-channel | Inception-v3's training domain | **yes** — `to_inception_input` does exactly this |
| Inception Score | same as above | softmax over ImageNet classes | yes |
| precision/recall/density/coverage | a feature space where between-sample distance exceeds within-sample perturbation | k-NN balls must be wider than the effect being measured | **no** on MedicalNet features |
| anatomical plausibility | brain foreground identifiable; L-R axis known | symmetry/ICV need the anatomy, not the padding | yes |

### What our volumes actually are

Measured on the real `test_v1` cohort (40–60 volumes sampled, `--geometry-mode fixed`):

| | GT (cohort) | VAE reconstruction | Diffusion generated |
|---|---|---|---|
| shape / dtype | 256³ float32 | 256³ float32 | 256³ float32 |
| p50 | 0.0000 | **0.0583** | 0.0000 |
| p99.5 | 0.906 | 0.865 | 1.012 |
| max (mean) | 1.913 | 1.804 | 1.905 |
| max (range over volumes) | **[1.21, 4.34]** | [1.17, 4.54] | [1.59, 2.27] |
| % voxels > 1.0 | 0.16 | 0.08 | 0.61 |
| % voxels < 0 | 15.8 (all ≈ −1e-8, fp noise) | 0.25 | 0.00 |
| **% bit-exact 0** | **52.0** | **0.0** | **64.8** |

Three mismatches follow from this table:

1. **`data_range=1.0` is not the peak.** The normalizer is percentile `[0, 99.5] → [0, 1]` with
   `clip=False`, so 1.0 is where the 99.5th percentile lands, not a ceiling. True maxima average
   1.91 and reach 4.34. Because the assumed range is too *small*, **reported PSNR is pessimistic by
   ~5.8 dB on average** (per-volume 1.7–12.8 dB). It is still a valid *relative* measure — every
   volume uses the same convention — but do not compare it to a published PSNR computed on clipped
   [0,1] data. To make it absolute, set `clip=True` in the normalizer, which is a new `cohort_id`.
2. **The VAE does not reproduce the zero background.** GT background is bit-exact 0 over 52% of the
   volume; the reconstruction's is ≈0.058 with 0% exact zeros. That is a global offset over half
   the volume, and it inflates every whole-volume metric.
3. **The generated volumes are *more* empty than the GT** (64.8% vs 52.0% exact zeros) and have a
   higher upper tail (0.61% of voxels above 1.0 vs 0.16%).

### Where the foreground mask is used, and where it is not

There are two independent masking mechanisms, easily confused:

| Mechanism | Where | What it does |
|---|---|---|
| `foreground_mask_from_intensity(gt, percentile=1.0)` | `runner.compute_paired_metrics`, computed **once per case from the ground truth** | a boolean mask, `gt > p1(gt)`. Passed to `mae/mse/psnr/ncc/relative_intensity_error/edge_preservation/laplacian_variance` → these are the `_fg` columns. Covers ~34% of the volume. |
| `min_fg_frac` slice filter | `ssim_2d_mean` and `distribution.non_empty_slices` | drops whole 2D *slices* whose foreground fraction is below 1%. Not a voxel mask. |

**It is always derived from the ground truth, never the prediction** — otherwise a degenerate
prediction could select its own easier evaluation region (`test_foreground_mask_comes_from_ground_truth_only`).

Metrics with **no** mask at all: `mae_whole`, `mse_whole`, `psnr_whole`, `ncc_whole`,
`ssim3d_whole`, `hf_energy_ratio` (FFT is not meaningfully maskable), and every distribution metric
(feature extractors see the whole volume).

Note the mask is an intensity heuristic, **not** a brain mask — the R2V cohort carries no
per-sample HD-BET mask. It excludes padding and air, but also includes skull, scalp and neck fat.

### Does padding affect every metric? No — here is exactly which

`crop_or_pad` pads with **bit-exact 0**. Native air after normalization is ≈1e-4, so the two are
distinguishable. Padding averages **52% of a cohort volume**, and for axial acquisitions (81% of the
cohort) **103 of 256 Z-slices** are pure padding (range 6–116).

| Metric | Affected? | How |
|---|---|---|
| `*_fg` variants | **no** | the mask excludes padding by construction |
| `mae_whole`, `mse_whole`, `psnr_whole` | **yes, mildly** | padding is trivially easy, but the VAE's non-zero background partly cancels the free win: PSNR 25.35 dB whole vs 26.21 dB non-padded (+0.9 dB) |
| `ssim3d_whole` | **yes, severely** | SSIM's local windows have zero variance on bit-constant regions, so any prediction noise drives local SSIM to ~0 across half the volume. **Treat `ssim3d_whole` as uninformative on this data** and prefer `ncc_fg` / per-plane `ssim2d_*` |
| `ssim_2d_mean` | **no** | background-only slices are excluded by `min_fg_frac` |
| `hf_energy_ratio` | **yes, mildly** | zero regions contribute no high-frequency energy to either side |
| MedicalNet FID / PRDC | **yes, strongly** | GAP over a region every subject shares dilutes between-subject variation (~4× in feature spread) |
| 2.5D Inception FID | **no** | `non_empty_slices` skips empty slices before feature extraction |
| anatomical plausibility | **no** | computed on the foreground only |

---

## The metrics

### Fidelity — how close, voxel by voxel

`mae`, `mse`, `psnr`, `ncc`, `ssim_3d`, `relative_intensity_error`, each with a `_fg`
foreground-restricted variant.

The foreground mask is derived from the **ground truth only**. Deriving it from the prediction would
let a mostly-empty prediction choose its own easier evaluation region;
`test_foreground_mask_comes_from_ground_truth_only` pins this.

Two things to know before quoting a number:

- **Intensities are not clipped to [0, 1].** The default percentile normalizer leaves values above
  the 99.5th percentile above 1.0. `data_range=1.0` is a fixed reference scale for cross-sample
  comparability, not a real maximum — so PSNR here is relative, not absolute.
- **Geometry is not checked inside these functions.** They only ever see arrays. The caller must
  have gotten `STRICT_MATCH` first; `runner.py` does.

### Perceptual — is detail surviving

`edge_preservation_ratio`, `laplacian_variance_ratio`, `high_frequency_energy_ratio`, `ssim_2d_mean`
per plane. Around 1.0 means preserved; below means blurred. 0.6-0.9 is common for a compressive VAE.

Per-plane SSIM excludes background-only slices — an empty slice's SSIM is uninformative and would
bias the mean toward whatever a near-constant comparison scores. Check `n_slices_used` alongside the
mean.

### Anatomical plausibility — does it look like a *brain*

[`anatomy.py`](anatomy.py). FID and PSNR can both look acceptable for a volume that is
anatomically wrong. These five measures check properties that hold for essentially every real head
MRI, and are compared **real population vs produced population** with a two-sample KS test — so the
group is valid for unconditional generation as well as for paired tasks.

| Measure | Real brains | Catches |
|---|---|---|
| `lr_symmetry_ncc` | ~0.85–0.95 | implausible asymmetry, e.g. a missing hemisphere. Evaluated over the **union** of foreground and its mirror — the intersection would be blind to exactly this failure |
| `intracranial_fraction` | stable per (modality, plane) | heads too large or small for the FOV |
| `tissue_contrast_separation` | > ~1.0 | grey/white matter at indistinguishable intensities (2-component 1-D GMM on foreground) |
| `foreground_compactness` | ~0.5–0.8 | scattered noise or disconnected fragments |
| `background_purity` | ~1.0 | haze in the air. **The VAE here scores low: it fills the background with ~0.058 instead of 0** |

Requires no model weights and no torch — numpy + scipy only, so it runs on CPU and stays available
with `--skip-metric-groups distribution` for a cheap sanity pass on a generator.

**Named "anatomical plausibility", not "clinical".** Every measure is computed from voxel
intensities with no segmentation network and no radiologist input. None of them can tell you whether
a volume shows the pathology its report describes — that needs the image-text model this project
does not have, or a validated segmentation model. Calling them clinical would oversell them.

### Distribution — do the populations match

| Metric | Reads as |
|---|---|
| MedicalNet FID (3D) | lower is better. Generation will be much higher than reconstruction — expected, not a bug. MedicalNet was trained on many organs, not brain MRI specifically. |
| 2.5D Inception FID | lower is better, computed on slices across all three planes. Not comparable in scale to the MedicalNet number — different feature space. Inception-v3 has never seen a medical image. |
| Inception Score | higher is better, but not very informative for brain MRI. Prefer the next row. |
| Precision / Recall / Density / Coverage | precision = do outputs look real; recall/coverage = do they span the real range. High precision + low recall is classic mode collapse. Needs 50+ per group to be stable. |

> **Known failure: MedicalNet FID and PRDC are not trustworthy on this data.** Measured on the real
> cohort, `FID(T1w real, T2w real) = 0.0009` while `FID(T1w real, its own VAE reconstruction) =
> 0.0111` — the features rank two grossly different contrasts as **12× more similar** than a volume
> versus its own reconstruction. The same collapse makes reconstruction PRDC come out exactly 0.
> The 2.5D Inception backbone passes the same test (138.2 vs 78.8, ratio 1.75×). Read the 2.5D
> numbers; treat MedicalNet's as diagnostic only. Full analysis in the section below.

The Fréchet distance is computed in `distribution.frechet_distance` — float64 throughout, with
epsilon regularization retried on a near-singular covariance product (routine when the sample is
smaller than the feature dimension). Every FID comes with a rank-deficiency flag and a bootstrap CI
so one number is never read as more precise than the sample supports.

It is computed here rather than via `monai.metrics.FIDMetric`: that path passes a `disp=` argument
scipy removed in 1.17, so it raises on any current scipy. Keeping it in-package also means an
evaluation run needs no monai.

For `generation` there is no pairing, so real and produced volumes are lined up **within a bucket**
and truncated to `min(n_real, n_gen)` — Fréchet distance needs two populations of comparable size,
not a correspondence. The truncation is logged.

Bucket, not sequence, and this matters: only within a (modality, plane) bucket do all volumes share
a geometry. Grouping by sequence would compare an axial real volume against a sagittal generated
one and attribute the difference to the model.

### Intra-set SSIM — the mode-collapse probe

Mean pairwise SSIM *within* each population, reported separately for real and produced. The number
that matters is the **difference**: a generator whose intra-set SSIM sits clearly above the real
data's is producing less variety than the data it was trained on. This is the convention the 3D
brain-MRI generation literature uses (average pairwise MS-SSIM over generated samples).

Implemented on mid-axial slices with 2D SSIM rather than a true 3D MS-SSIM: pairwise over N volumes
is O(N²), and at 200 volumes per bucket a full 3D pass would dominate the whole evaluation.
`max_pairs=200` bounds it with a deterministic sample of pairs. Mixed shapes are refused, not
averaged — compare within a bucket.

### Report alignment

Currently **unavailable**: no validated MRI image-text model exists in this project. Every case
records `report_image_similarity_available=False` with a concrete reason. It is never silently
substituted with a different model — a number attributed to report alignment would then be measuring
something else. To wire one in, pass an object with `.score(report_text, volume) -> float` as
`EvaluationInputs.report_image_model`.

### Report consistency (the blinded classifier)

`report_classifier.py` + `report_labels.py`. **This is the group that answers "does the volume say
what the report said"**, and it is the local stand-in for the VLM3D challenge's Blinded Classifier
Consistency metric.

A frozen MedicalNet ResNet-10 — the same 512-d backbone this package's 3D FID already uses, so the
features cost nothing extra — feeds a ~140k-parameter head fitted on **real train-split volumes
only** (`cli.train_report_classifier`). At evaluation the head runs blind on the generated volumes
and its verdict is compared against the 14 merged clinical labels of the report each volume was
conditioned on.

Four things decide whether the number means anything, and all four are enforced in code:

| | why |
|---|---|
| the classifier sees the **image and nothing else** — no bucket, no modality | a head given the bucket scores well from `SWI AXIAL → hemorrhage is common` without looking at a voxel, and would rate a degenerate generator just as highly |
| every score is reported next to `real_reference` — the same classifier on the **real** volumes | 0.58 against a real ceiling of 0.61 and 0.58 against 0.95 are opposite conclusions |
| a label is `usable` iff the classifier can do it **on real data** (`MIN_REAL_AUROC`) | deciding usability from the generated score would let a model promote whichever labels flattered it |
| unlabelled cases are excluded, never imputed negative | "not classified" ≠ "the report says no"; imputing deflates every prevalence |

Outputs: `report_consistency.json` (per label + per case + provenance),
`report_consistency_per_label.csv` (**the table to quote**), and
`report_consistency_per_case.csv` (the input to a case-level permutation test). All three are
written for every task — a task that does not declare the group says so inside the file rather than
omitting it, because the result layout is identical across tasks by invariant.

Without `--report-classifier` the group records `available=False` with a reason and every other
metric still runs.

### W&B reporting

`wandb_evaluation.py` is the R2V-specific assembly on top of `wandb_logging.WandbRun`, which stays
dataset-agnostic (it was ported unchanged). Two deliverables:

- **`metrics/all`** — one `wandb.Table` holding every metric family that ran, per bucket then
  aggregate, ending in provenance rows including `train_samples_seen`. The same table is printed by
  `cli.evaluate` so the Slurm log carries the full result. `metrics_table()` builds it; every cell
  is a finite float or None, because a numpy scalar or a NaN breaks `wandb.Table`.
- **A few example panels** rendered by `figures.validation_panel_html` — the *same* renderer the
  training loop uses, so an evaluation panel and a training panel are directly comparable. A
  `_PanelCase` shim presents a cohort case in the shape that renderer already accepts rather than
  refactoring it.

`select_panel_cases()` picks the worst, best and median by a per-case metric, round-robin over
buckets so a small `n_panels` still spans several anatomies, and labels each pick with its reason.
Panels are withheld unless `log_reports=True` — they embed report text.

---

## Making it fast

**It is CPU-compute-bound, not I/O-bound.** Measured on real 256³ volumes off Lustre:

| | Time per case | Share |
|---|---|---|
| reading both `.npy` volumes (134 MB) | 39 ms @ 3.4 GB/s | **0.5%** |
| metric compute | 6,750 ms | 99.5% |

So **staging the cohort to node-local `$TMPDIR` is not worth doing** — it would optimize 0.5% of
the work, a ceiling of 1.005×. Lustre already reads these volumes faster than the metrics can
consume them.

What does help is `--workers`, because per-case scoring is embarrassingly parallel. Measured on 8
real case pairs:

| `--workers` | s/case | speedup |
|---|---|---|
| 1 | 6.75 | 1.00× |
| 2 | 3.37 | 2.00× |
| 4 | 1.72 | 3.93× |
| 8 | 0.92 | 7.32× |

Results are byte-identical at every worker count (`test_parallel_scoring_gives_identical_results_to_serial`
asserts it) — `--workers` is purely a wall-clock knob. `slurm/05_evaluate.sbatch` passes
`$SLURM_CPUS_PER_TASK` and asks for 32 CPUs.

**Two caveats.** `--workers` only parallelizes *paired* scoring, so it does nothing for
`--task generation`, which has no paired metrics. And distribution metrics run their feature
extractors serially on the GPU in the parent process — that cost is unaffected by `--workers`.

Where the per-case time goes, if you want to cut it further:

| Metric | Cost | Share |
|---|---|---|
| `high_frequency_energy_ratio` (2× 3D FFT) | 1,590 ms | 24% |
| `ssim_3d` | 2,005 ms | 30% |
| `ssim_2d_mean` × 3 planes | 1,664 ms | 25% |
| `laplacian_variance_ratio` | 753 ms | 11% |
| `edge_preservation_ratio` | 432 ms | 6% |
| mae/mse/psnr/ncc (whole + fg) | 367 ms | 5% |

Dropping the `perceptual` group would roughly halve it, at the cost of the detail-preservation
diagnostics. The FFT metric's frequency-radius mask is already cached by shape (it was being
rebuilt per volume — three 134 MB float64 arrays, 239 ms each time).

## Reading a result directory

| File | What it is |
|---|---|
| `metrics_per_bucket.csv` | **start here.** One row per (modality, plane): the shape and spacing it was scored at, the sample counts, and every metric. |
| `metrics_summary.csv` | per modality, then `overall_macro` and `overall_weighted` |
| `per_case_metrics.csv` | one row per scored case |
| `summary.json` | the same numbers machine-readably, plus which metric groups ran and why |
| `distribution_metrics.json` | population metrics, when computed |
| `anatomy_metrics.json` | anatomical plausibility, real population vs produced |
| `excluded_cases.json` | every unscored case, with a specific reason |
| `run_manifest.json` | `cohort_id`, task, checkpoint hashes, contract versions |
| `figures/` | example orthogonal-slice montages -- **look at these**; a number tells you a volume is worse, a picture tells you how |

### Why two overall rows

`overall_macro` is the unweighted mean across buckets; `overall_weighted` weights by
`population_bucket_counts`, the eligible-population frequencies recorded in `cohort.json`. The
cohort is sampled to equal size per bucket so that per-bucket FID is stable, which means cohort
counts are a sampling artefact and must never be used as weights -- doing so would silently turn
the weighted aggregate back into the macro one. Quote macro when comparing models, weighted when
claiming what a clinical population would see.

Always compare `n_scored` with `n_cohort` per bucket. Exclusion categories you may see:

| Category | Meaning |
|---|---|
| `missing_prediction` | a cohort case the model produced nothing for |
| `geometry_incompatible` | prediction grid does not match the cohort's; the reasons list says which check failed |
| `no_matching_case` | prediction names a `case_id` not in this cohort |
| `duplicate_case` | two predictions for the same case |
| `unpaired_item` | a paired task got an item with no `case_id` |

A good FID does not prove clinical correctness, and a good reconstruction does not predict that a
report-conditioned model will score well — each stage measures something different.

---

## Example figures

`--save-figures N` (default 3) writes N montages per **bucket** into `<results>/figures/`, rendered
by [`figures.py`](figures.py) with PIL:

    paired tasks   rows = ground truth / prediction / |difference|
    generation     rows = generated / an unpaired real reference from the SAME bucket
                          (captioned as NOT a counterpart)
    columns        sagittal / coronal / axial mid-slices, superior up

Cases are chosen at evenly-spaced *metric ranks*, not arbitrarily, so `N=3` gives you the worst,
median, and best case by `psnr_fg` -- the worst one is where failure modes are visible. Filenames
are `{modality}__{plane}_rank{0..N}_{case_id}.png`, so rank0 is always the worst.

Ground truth and prediction share one intensity window so the rows are genuinely comparable; the
difference row is scaled to its own max, which is printed in the label (an auto-scaled difference
image with no stated range is easy to misread).

`--save-nifti-cases N` additionally exports gt/prediction/absdiff as `.nii.gz` for the first N
figured cases per bucket, to open in a real viewer. The affine is synthesized from spacing -- it
carries orientation and voxel size but no true patient-space origin.

Figure generation is never fatal: a rendering failure is logged and the metrics still get written.

## Why MedicalNet FID fails here — and how to check any FID backbone

**Validity test.** A feature space usable for FID must rank a *different modality* as further away
than *a volume versus its own reconstruction*. Measured on the real cohort, 30 volumes per group:

| Backbone | FID(T1w, T2w) | FID(T1w, T1w recon) | ratio | verdict |
|---|---|---|---|---|
| MedicalNet, as shipped | 0.0009 | 0.0111 | **0.09×** | **invalid** |
| MedicalNet + Med3D z-score | 0.1052 | 2.3648 | **0.04×** | **still invalid** |
| 2.5D Inception (control) | 138.2 | 78.8 | **1.75×** | valid |

Three causes, in order of size:

1. **The VAE's background floor dominates the measurement.** The reconstruction fills the air with
   ~0.055 where ground truth is bit-exact 0 over 52% of the volume. Forcing that background back to
   zero drops the "reconstruction distance" from 0.0111 to **0.0018 — an 84% reduction**. Most of
   what MedicalNet FID was reporting is that one global offset, not image quality.
2. **The features barely discriminate at all.** Between-subject std is 0.0015 against a mean
   magnitude of 0.166 — a relative spread of **1.08%** — with **160 of 512 dims exactly
   zero-variance** (dead units).
3. **No input normalization.** Med3D z-scores each volume over its strictly-positive voxels; we fed
   raw percentile-normalized volumes. Fixing this raises relative spread to 9.49% (8.8×) and
   cropping the padding adds ~4× more — but **it does not make the ratio pass**, because cause 1
   scales up with it.

The weights load correctly (MONAI's `_resnet` uses `strict=True`); this was never a loading bug.

`MedicalNetFeatureExtractor` now applies Med3D's normalization and foreground cropping by default
(`normalize=`, `crop_to_foreground=`), which is strictly better than before. **That alone does not
make the metric valid** — the honest fixes are to give the VAE a zero background, or to use a
backbone that passes the validity test above. Run that test against any new backbone before
trusting a single FID number from it.

## How this compares to published protocols

Surveyed against current 3D brain-MRI generation papers
([Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis](https://arxiv.org/html/2305.18453v5),
[3D MedDiffusion](https://arxiv.org/html/2412.13059v1)):

| Standard practice | Here |
|---|---|
| 3D-FID from a 3D ResNet pretrained on medical data | yes — but our backbone fails the validity test above, so read the 2.5D variant |
| MS-SSIM, **intra-set** (mean pairwise similarity *among generated samples*) as a mode-collapse probe | **missing** — we have paired SSIM and PRDC, not intra-set MS-SSIM. The closest published convention we do not implement |
| MMD with a Gaussian kernel on medical features | **missing** |
| LPIPS | missing (2D, perceptual) |
| Intensity normalized to a bounded range ([-1,1] or [0,1]) before metrics | **we do not clip** — see the PSNR caveat above. Most papers clip; our numbers are not directly comparable to theirs |
| Fixed isotropic resample (commonly 128³ @ 1.5 mm) | 256³ @ 1.0 mm — higher resolution than the common convention |
| Downstream-task validation (segmentation Dice on synthetic data) | missing; would need a segmentation model |
| Anatomical/clinical consistency metrics | **rare in the literature** — most papers report none. Our `anatomy` group goes beyond the surveyed protocols |

Two gaps worth closing if you want protocol comparability: **intra-set MS-SSIM** and **clipped
intensities**. The second changes the cohort, so it is a new `cohort_id`.

## Privacy

`study_uid`/`series_id` are anonymized but still identifiers. They are used for matching and never
written into `cohort.json`, `summary.json`, or any results file — those carry `case_id` (a hash) and
`cohort_id`. Quote the `cohort_id` in a paper, not a case list.

---

## Testing

```bash
python -m pytest tests/test_eval_tasks_and_runner.py tests/test_cohort_contract.py \
                 tests/test_eval_geometry_contract.py tests/test_eval_paired_metrics.py \
                 tests/test_eval_distribution.py tests/test_eval_features.py \
                 tests/test_eval_pairing.py -v --no-cov
```

CPU, synthetic data, seconds. `test_eval_tasks_and_runner.py` runs the whole pipeline end to end.
