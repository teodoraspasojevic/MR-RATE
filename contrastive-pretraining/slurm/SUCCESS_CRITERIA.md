# Success criteria for the R2V evaluation run

What a job must satisfy before its output is treated as a result. A job that "finished" without
meeting these is a failed job — fix and resubmit rather than reading the numbers.

Checked with:

```bash
python3 slurm/check_run.py --results <results dir> [--results <another> ...]
```

Exit 0 only if every applicable check passes. Absent inputs are reported SKIP, never PASS.

## What changed (2026-08-10)

**The cohort and prediction stages are gone**, and so are the criteria that policed them.
`cli.evaluate` builds the dataset, generates and scores in one pass, so there is no frozen cohort
to verify and no prediction set to match against it. The properties the old `C*`, `R*` and `G*`
checks enforced are now either impossible to violate or checked inside the run and recorded in
`run_manifest.json`:

| Old check | Where it lives now |
|---|---|
| C1 bucket counts | `run_manifest.json:bucket_counts`; selection is deterministic and bucket-interleaved by construction |
| C2 div-32 shapes | asserted per case in `cli.evaluate:build_generation`, and by `test_eval_live.py::test_every_bucket_shape_is_a_multiple_of_32` against the shipped geometry table |
| C3 FOV reproduces NVIDIA's | property of `build_geometry_table`, tested in `test_data_dataset.py` |
| C4 preprocessing recorded | `run_manifest.json:geometry`, hashed into `run_id`, **and** compared against the adapter's recorded training values before any GPU work |
| C5 population counts | `run_manifest.json:population_bucket_counts` |
| C6/R4/G4 archive completeness | no archives exist; a case that fails to generate is an explicit `generation_failed` exclusion |
| R2/G2 `cohort_id` handshake | `run_id`, computed from the run rather than stored — it cannot go stale |
| R3/G3 no resize | `check_case_geometry` per case; a mismatch is excluded with a reason, never resized |

## Stage 1 — the run itself

| # | Criterion | Why it can fail |
|---|---|---|
| S1 | The job wrote `summary.json` | a killed job (walltime) leaves a partial directory; a full-split run needs `--time` well past 8 h |
| S2 | `run_id` is recorded | without it the results cannot say which cases produced them |
| S3 | `metric_groups_skipped` is empty, or its contents are intended | a skipped group is recorded, never silent, but it still means the run is not comparable to a full one |

## Stage 2 — evaluation (`05_evaluate.sbatch <task> <tag> [adapter.pt]`)

| # | Criterion | Note |
|---|---|---|
| E1 | `n_scored == n_cohort_cases` for reconstruction, `n_excluded == 0` | a geometry exclusion means a real bug, not a tolerable loss. The count is no longer a literal 2,000: the scale is a run parameter (`--n-per-bucket`, or the whole split), so the check is *all selected cases were scored* |
| E2 | `n_scored == 0` for generation, and **no** voxelwise column in its CSV | structural: no real patient behind a generated volume |
| E3 | Both CSVs written, 10 data rows in `metrics_per_bucket.csv`, 6 in `metrics_summary.csv` | 4 modalities + macro + weighted |
| E4 | `overall_macro != overall_weighted` on at least one metric — **unless the whole split was evaluated**, where they legitimately coincide | with `--n-per-bucket` the cohort is balanced by construction and the population weights must move the number. On a full-split run the population counts *are* the scored counts, so equality is correct and the checker asserts equality instead |
| E5 | Reconstruction `psnr_fg` > 20 dB, `edge_preservation_fg` > 0.8, and `ssim3d_whole` > 0.30 in every bucket | **Revised 2026-07-30 after the first run.** The original form required `ssim3d_whole` > 0.5 and failed in 6 of 10 buckets (0.354–0.497) while PSNR (24.8–28.8 dB) and NCC (0.978–0.991) were excellent. The 0.5 was a guess made before any measurement. Verified before revising: the rank-0 *worst* case of the *worst* bucket (FLAIR CORONAL, psnr_fg 23.08) preserves sulci, ventricles, cerebellum, brainstem and the small FLAIR white-matter hyperintensities, with a diffuse fine-edge difference image and no structural error. SSIM3D correlates with `edge_preservation_fg` at **r = 0.735**, i.e. the low values are the texture term responding to mild, quantified blur (0.85–0.92) — worst on high-texture FLAIR, best on smooth T2w (0.693). `edge_preservation_fg` replaces it as the direct blur measure; the 0.30 SSIM floor only catches a genuinely broken reconstruction. |
| E6 | Reconstruction `ncc_fg` > 0.9 in every bucket | catches an axis-order scramble, which SSIM alone can miss |
| E7 | Generation FID > reconstruction FID on **both** aggregate rows, and in at least half the buckets | **Revised 2026-07-30 after the first run.** The original form demanded the ordering in *every* bucket and failed in 2 (FLAIR CORONAL, T1w CORONAL). The premise was wrong: reconstruction FID is not ≈0. It is a roughly constant **floor of 8.4–18.9** (CV 0.26) set by the encoder's information loss — a systematic feature-space shift is precisely what FID measures. Generation FID varies 6× by bucket (12.8–77.2, CV 0.57). So where the diffusion model is strong its samples can legitimately sit closer to the real feature distribution than the VAE's own reconstructions of real data, which is not a contradiction: generated volumes pass through the **decoder only**, reconstructions through **encoder + decoder**. Measured aggregate: 34.38 vs 13.46 macro (**2.55×**), 34.20 vs 13.33 weighted (2.57×). A bucket where generation beats the reconstruction floor is worth noting, not a failure; the ordering inverting in a majority of buckets would be. |
| E8 | FID validity: FID(this bucket's real vs another modality's real) > FID(real vs own reconstruction) | if it fails, the feature extractor cannot tell contrasts apart and its FID is meaningless — this is exactly how MedicalNet was caught |
| E9 | Figures written for all 10 buckets, and visually a brain | the last line of defence a number cannot provide |
| ER1 | `n_scored == n_cohort_cases` for report2volume, `n_excluded == 0` | as E1: one generated volume per case on that case's own grid, so an exclusion means the sampler emitted a grid that was not requested |
| ER3 | `macro_auroc_usable_labels > 0.5` over at least one usable label | 0.5 is what an image-blind guesser gets, so this is the floor, not the target. Judge the value against `real_reference.macro_auroc_usable_labels` in the same file — the ceiling this classifier reaches on the **real** volumes. A SKIP here (no classifier passed) is not a pass. |

**Expect report2volume fidelity to be far below reconstruction**, and do not apply E5's thresholds
to it. A report constrains pathology and gross anatomy, not voxel positions; training-time SSIM for
all four arms sat at 0.34–0.36 against an autoencoder ceiling of 0.915. Compare arms against each
other **at the same `run_id`**, never against a reconstruction number.

**Comparing arms:** four runs are comparable exactly when their `run_id` values agree on everything
except the model — which `slurm/final_eval/` guarantees by construction, since the four `run_*.sh`
scripts set `R2V_CONFIG` and nothing else. If two `run_id`s differ, check `run_manifest.json`
(`geometry`, `split`, `n_per_bucket`, `seed`) before reading the numbers side by side.

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
