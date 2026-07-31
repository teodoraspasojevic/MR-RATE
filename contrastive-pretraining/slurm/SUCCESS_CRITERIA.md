# Success criteria for the R2V evaluation run

What each job must satisfy before its output is treated as a result. A job that "finished" without
meeting these is a failed job — fix and resubmit rather than reading the numbers.

Checked with:

```bash
python3 slurm/check_run.py --cohort <cohort dir> \
    [--pred-vae <dir>] [--pred-gen <dir>] [--results-recon <dir>] [--results-gen <dir>]
```

Exit 0 only if every applicable check passes. Absent inputs are reported SKIP, never PASS.

## Stage 1 — cohort (`02_preprocess.sbatch test_v1 200`)

| # | Criterion | Why it can fail |
|---|---|---|
| C1 | Exactly **10 buckets**, each with **200 cases** → 2,000 total | a wrong `--series-selection` collapses the planes; measured T2w SAGITTAL = 6 with `one_per_study_per_sequence` |
| C2 | Every bucket's shape matches the Option-B table and **every axis is divisible by 32** | generation refuses a non-div-32 shape, so this must be caught here |
| C3 | Each bucket's `shape × spacing` reproduces NVIDIA's published FOV to < 0.5 mm | the whole point of deriving spacing as FOV/shape |
| C4 | `cohort.json` records `posterior_shift_mm: 0`, `normalizer: percentile`, `seed: 42` | these are hashed into `cohort_id`; a silent default change makes runs incomparable |
| C5 | `population_bucket_counts` present and non-empty for all 10 buckets | without it the weighted aggregate silently degenerates to the macro one |
| C6 | 10 `.npz` archives, `verify_complete()` returns empty | a partial cohort must never be scored as if it were full |
| C7 | Volume intensities: p99.5 in [0.5, 2.0], min ≥ −1e-3, no NaN/Inf | percentile normalizer does not clip; a broken read shows up here. The −1e-3 floor is not slack: spline resampling undershoots at sharp edges, so background voxels land a hair below zero (measured worst −6.2e-05, typically ~1e-7). A real sign error would be a fraction of the [0, 1] scale. |

## Stage 2a — VAE reconstruction (`03_predict_vae.sbatch`)

| # | Criterion |
|---|---|
| R1 | `n_volumes == 2000`, `failures == []` |
| R2 | `cohort_id` in `predictions.json` equals the cohort's |
| R3 | Every item's `shape` equals its case's shape (no resize anywhere) |
| R4 | 10 archives, `verify_complete()` empty |

## Stage 2b — generation (`04_predict_generation.sbatch test_v1 0`)

| # | Criterion |
|---|---|
| G1 | `n_volumes == 2000`, `failures == []` |
| G2 | `cohort_id` in `predictions.json` equals the cohort's |
| G3 | Each generated volume's shape equals its bucket's shape |
| G4 | 10 archives, `verify_complete()` empty |
| G5 | `requested_geometry_per_bucket` matches the cohort's geometry for all 10 buckets |
| G6 | Volumes are not degenerate: per-volume std > 0.01 and foreground fraction in [0.05, 0.95] |

## Stage 3 — evaluation (`05_evaluate.sbatch`, twice)

| # | Criterion | Note |
|---|---|---|
| E1 | `n_scored == 2000` for reconstruction, `n_excluded == 0` | a geometry exclusion means a real bug upstream, not a tolerable loss |
| E2 | `n_scored == 0` for generation, and **no** voxelwise column in its CSV | structural: no real patient behind a generated volume |
| E3 | Both CSVs written, 10 data rows in `metrics_per_bucket.csv`, 6 in `metrics_summary.csv` | 4 modalities + macro + weighted |
| E4 | `overall_macro != overall_weighted` on at least one metric | if equal, the population weights were not applied |
| E5 | Reconstruction `psnr_fg` > 20 dB, `edge_preservation_fg` > 0.8, and `ssim3d_whole` > 0.30 in every bucket | **Revised 2026-07-30 after the first run.** The original form required `ssim3d_whole` > 0.5 and failed in 6 of 10 buckets (0.354–0.497) while PSNR (24.8–28.8 dB) and NCC (0.978–0.991) were excellent. The 0.5 was a guess made before any measurement. Verified before revising: the rank-0 *worst* case of the *worst* bucket (FLAIR CORONAL, psnr_fg 23.08) preserves sulci, ventricles, cerebellum, brainstem and the small FLAIR white-matter hyperintensities, with a diffuse fine-edge difference image and no structural error. SSIM3D correlates with `edge_preservation_fg` at **r = 0.735**, i.e. the low values are the texture term responding to mild, quantified blur (0.85–0.92) — worst on high-texture FLAIR, best on smooth T2w (0.693). `edge_preservation_fg` replaces it as the direct blur measure; the 0.30 SSIM floor only catches a genuinely broken reconstruction. |
| E6 | Reconstruction `ncc_fg` > 0.9 in every bucket | catches an axis-order scramble, which SSIM alone can miss |
| E7 | Generation FID > reconstruction FID on **both** aggregate rows, and in at least half the buckets | **Revised 2026-07-30 after the first run.** The original form demanded the ordering in *every* bucket and failed in 2 (FLAIR CORONAL, T1w CORONAL). The premise was wrong: reconstruction FID is not ≈0. It is a roughly constant **floor of 8.4–18.9** (CV 0.26) set by the encoder's information loss — a systematic feature-space shift is precisely what FID measures. Generation FID varies 6× by bucket (12.8–77.2, CV 0.57). So where the diffusion model is strong its samples can legitimately sit closer to the real feature distribution than the VAE's own reconstructions of real data, which is not a contradiction: generated volumes pass through the **decoder only**, reconstructions through **encoder + decoder**. Measured aggregate: 34.38 vs 13.46 macro (**2.55×**), 34.20 vs 13.33 weighted (2.57×). A bucket where generation beats the reconstruction floor is worth noting, not a failure; the ordering inverting in a majority of buckets would be. |
| E8 | FID validity: FID(this bucket's real vs another modality's real) > FID(real vs own reconstruction) | if it fails, the feature extractor cannot tell contrasts apart and its FID is meaningless — this is exactly how MedicalNet was caught |
| E9 | Figures written for all 10 buckets, and visually a brain | the last line of defence a number cannot provide |

## Known-acceptable outcomes (not failures)

- **T2w SAGITTAL / CORONAL and any low-`nvidia_train_n` bucket scoring badly on generation.**
  NVIDIA trained those on 551 / 125 images and says quality is not guaranteed. Kept in the
  aggregates, flagged by the `nvidia_low_train_n` column.

  **Observed on the 2026-07-30 run (job 664982):** E9's visual half fails for exactly one bucket,
  **T2w CORONAL** — the generated volumes are a textured blob inside a skull-like envelope with no
  ventricles and no grey/white structure. Not a pipeline fault: the metrics independently ranked it
  worst of all ten buckets (Inception 2.5D FID 77.2 against a 12.8–51.8 range, and the largest
  diversity loss at intra-set SSIM +0.220). T2w AXIAL and T2w SAGITTAL both produce good brains.

  Two explanations are **confounded** in this run and it is not yet known which drives it:
  1. T2w CORONAL has the lowest training count (125). But T2w AXIAL has 195 — barely more — and
     works, which weakens this on its own.
  2. T2w CORONAL is the only bucket whose requested shape is **cubic** (192³, min/max axis 1.000);
     every other bucket is 0.625–0.750. Since `rflow-mr-brain` conditions on modality and spacing
     but *not* on plane, the plane is implied by the anisotropy of the request, so a cubic request
     may simply be out of distribution.

  Disambiguating probe (~30 s of GPU, not yet run): generate T2w at T1w CORONAL's geometry
  (256×192×256, aspect 0.750). A brain implies (2) and the bucket geometry could be changed to
  recover a usable number; another blob implies (1) and the bucket stays flagged as unlearned.
- **MedicalNet FID failing E8 while 2.5D Inception passes.** Measured across 16 preprocessing
  configurations: MedicalNet's best validity ratio was 1.00×, Inception's 1.75×. Report both, lead
  with Inception, state the MedicalNet caveat.
- **`report_image_similarity_available == False`.** No validated MRI image-text model exists in
  this project; recorded as unavailable with a reason rather than faked.
- **PSNR looking pessimistic by ~5.8 dB** against papers that clip to [0, 1]. The percentile
  normalizer does not clip, so `data_range=1.0` is a reference scale, not a maximum.
