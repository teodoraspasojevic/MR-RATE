# Report-to-Volume: the complete guide

Everything about generating brain MRI volumes from radiology reports and scoring the result.
This is the only document you need; the per-module READMEs
([data](../contrastive-pretraining/mrrate_r2v/data/README.md),
[eval](../contrastive-pretraining/mrrate_r2v/eval/README.md)) go deeper on their own area.

For the *conditioning* side — which text encoder, which report format, and where modality/spacing
should come from — see **[TEXT_ENCODERS.md](TEXT_ENCODERS.md)** and the
[textenc](../contrastive-pretraining/mrrate_r2v/textenc/README.md) /
[textbench](../contrastive-pretraining/mrrate_r2v/textbench/README.md) READMEs.

**Training a report adapter? Start at
[`textenc/README.md` Part 4](../contrastive-pretraining/mrrate_r2v/textenc/README.md) for the four
named configurations (`--conditioning`), then [TEXT_ENCODERS.md §9](TEXT_ENCODERS.md) for the study
behind them.** Between them they cover the configurations and their exact conditioning shapes, how each is run
(single- and multi-GPU), validation and its two metrics, W&B and the interactive panel, checkpoint
contents, the measured H200 batch sizes, and the per-bucket geometry table.

All code lives in one package: `contrastive-pretraining/mrrate_r2v/`. Run everything from
`contrastive-pretraining/`.

---

## 1. The pipeline in one picture

```
  raw MR-RATE storage
  (un-extracted tars)
         │
         │  stage 0  ── cli.build_manifest ──────────────────────►  manifest.csv
         │                                              "what series exist and where"
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  ONE Dataset.  MRReportToVolumeDataset + R2VDatasetConfig, built by one       │
  │  function from one set of flags. Training and testing both use it.           │
  └──────────────────────────────────────────────────────────────────────────────┘
         │                                                    │
         │  --split train                                     │  --split test
         ▼                                                    ▼
  cli.train_r2v                                        cli.evaluate --task ...
    loader → encode → UNet → backward                    loader → encode → UNet
    → adapter_last.pt                                    → sample → score → release
                                                                     │
                                                                     ▼
                                                              RESULTS DIRECTORY
                                                              metrics_per_bucket.csv
                                                              metrics_summary.csv + 6 more
```

**Train and test are the same program up to the point where one trains and the other infers.**
That is the whole design. Nothing is frozen to disk between preprocessing and metrics: no cohort
directory, no prediction set, no `.npy` on disk. A case is preprocessed, generated, scored and
released, one at a time.

Three properties follow, and they are the reason it has this shape:

| Property | How it is guaranteed |
|---|---|
| **Test preprocesses exactly like train** | There is one `R2VDatasetConfig` and one `build_dataset`. Not a convention — there is no second code path that *could* differ. |
| **Evaluation is always the same** | One `LiveEvaluator.run`. Every task, every model. `eval/tasks.py` decides the metric set from `--task` and nothing else does. |
| **The same cases, every time** | Case selection uses **no RNG**: ordered by `(study_uid, series_id)` within each (modality, plane) bucket, round-robined across buckets. Prefix-stable, so `--n-per-bucket` is a prefix of the full run. |
| **The same volumes, every time** | Sampler noise is `stable_seed(--seed, case_id)` — a function of the case, not of iteration order. A rerun, a resume, a `--n-per-bucket`, or a different world size all reproduce the same volume for the same case. |

And the gate that replaces the old `cohort_id` handshake: `run_id` hashes the ordered case list
with every preprocessing setting, the task, the sample cap, the seed and the model checkpoint.
Equal `run_id` means the same cases at the same geometry under the same preprocessing. On top of
that, `cli.evaluate --task report2volume` compares its own `--posterior-shift-mm` / `--normalizer` /
`--geometry-mode` against what the adapter recorded at training time and **refuses to run** on a
mismatch, before any GPU work.

> **Why this replaced the cohort pipeline (2026-08-10).** The old design froze a cohort directory,
> wrote a prediction set, and scored the two against each other with a `cohort_id` handshake. It
> had three artifacts that could drift, and one did: `cli.train_r2v` had no `--posterior-shift-mm`
> flag, so every run took the dataclass default of 15 mm while `cli.preprocess` hardcoded 0.
> Measured: **15.8% of test cases** displaced, correlation 0.63–0.85 on those, concentrated in the
> coronal and sagittal buckets. Nothing could surface it, because the value existed on only one
> side. Removing the intermediate artifacts removes the class of bug, and costs ~365 GB of disk per
> experiment set as a bonus.

### How much to evaluate

| Scale | Cases | Wall clock (1×H200) | Use |
|---|---|---|---|
| `--n-per-bucket 8` | 80 | ~10 min | wiring check; no metric means anything and the run says so |
| `--n-per-bucket 200` | 2,000 | ~4 h | the scale every earlier result was produced at |
| unset (**default**) | 29,027 | ~60 h | the entire test split |

Measured 6.7 s/case at 30 inference steps, plus ~1 s/case of scoring. **CTFlow** (Wang et al.,
ICCV 2025 VLM3D, the CT-RATE report-to-volume model) evaluates on the *whole* CT-RATE validation
set — 3,000 volumes, partitioned across 64 workers, no subsampling — which is the precedent for
making the full split the default here. Note that CTFlow does not seed its sampler at all
(`torch.randn` with no generator), so its individual volumes are not reproducible; ours are.

---

## 2. Quickstart

```bash
cd contrastive-pretraining
```

### First time on a machine — build the manifest (once per storage location)

```bash
python -m mrrate_r2v.cli.build_manifest --source shards_parquet \
    --shards-root      /hnvme/workspace/y100dc19-MR-Rate-raw \
    --out-csv          /hnvme/workspace/y100dc19-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv \
    --out-report-index-csv /hnvme/workspace/y100dc19-MR-Rate-raw/r2v_manifest/report_index_shards_native.csv \
    --verify-sample 20
```

### Then, per experiment set — freeze a cohort (once)

```bash
python -m mrrate_r2v.cli.preprocess \
    --manifest-csv     .../manifest_shards_native.csv \
    --report-index-csv .../report_index_shards_native.csv \
    --split test --sequences T1w T2w FLAIR SWI --n-per-bucket 200 \
    --out /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/cohorts/test_v1
```

Note the `cohort_id` it prints. That is the identifier of your experiment set.

### Then, per model — predict and evaluate

```bash
COHORT=.../cohorts/test_v1

python -m mrrate_r2v.cli.predict_vae --cohort $COHORT \
    --checkpoint .../models/autoencoder_v1.pt --out .../predictions/vae_v1

python -m mrrate_r2v.cli.evaluate --task reconstruction \
    --gt $COHORT --pred .../predictions/vae_v1 --out .../results/vae_v1
```

Read `.../results/vae_v1/metrics_per_bucket.csv` and `metrics_summary.csv`.

### On the cluster

```bash
sbatch slurm/01_smoke_test.sbatch                        # whole pipeline, 2 cases. Do this first.
sbatch slurm/02_preprocess.sbatch test_v1 200            # cohort: 200 per (modality, plane)
sbatch slurm/03_predict_vae.sbatch test_v1 vae_v1
sbatch slurm/04_predict_generation.sbatch test_v1 0 gen_v1   # 0 = match the cohort's counts
sbatch slurm/05_evaluate.sbatch reconstruction test_v1 vae_v1
sbatch slurm/05_evaluate.sbatch generation     test_v1 gen_v1
```

Paths and the apptainer invocation live in `slurm/_common.sh` — edit them in that one place.
Set each job's `--time` from the smoke test's measured per-case rate rather than guessing.

---

## 3. The three tasks

`--task` is the only thing that decides which metrics run.

| `--task` | The model was given | Ground truth? | Metrics computed |
|---|---|---|---|
| `reconstruction` | a real volume, to encode and decode | yes, that exact volume | fidelity, perceptual, distribution, anatomy |
| `report2volume` | a report | yes, the series that report describes | fidelity, perceptual, distribution, anatomy, report alignment, **report consistency** |
| `generation` | a modality label only | **no** | distribution, anatomy |

**Why `generation` gets no voxelwise metrics.** An unconditional generator is told "make a T1w
brain" and nothing about any patient. Computing MAE or SSIM against an arbitrary real scan would
measure "how different are two random brains," not generation quality. This is enforced in
[`eval/tasks.py`](../contrastive-pretraining/mrrate_r2v/eval/tasks.py) as a property of the task,
not as a flag you could forget.

**Expect `report2volume` fidelity to be much weaker than `reconstruction`.** A report constrains
pathology and gross anatomy, not voxel positions. Low PSNR there is not necessarily a bug; compare
against other report-to-volume runs on the same cohort, never against a reconstruction number.

---

## 4. Controlling the experiment

Every knob below belongs to `cli.preprocess` and is recorded in `cohort.json`. That is the point:
changing any of them produces a different `cohort_id`, so results across the change can never be
silently mixed.

### How many samples

```bash
--n-per-bucket 200        # 200 cases per (modality, plane) bucket -> 10 buckets = 2000 cases
                          # 0 = every eligible case
--sequences T1w T2w FLAIR SWI
--split test              # train | val | test
--seed 42
```

**A bucket is one (modality, plane) pair** — `T1w__AXIAL`, `FLAIR__SAGITTAL`, and so on. It is the
unit of everything downstream: one geometry, one archive on disk, one row in the results CSV, one
FID. Sampling per bucket rather than per sequence is what makes that work: the real plane mix is
~81% axial, so a per-sequence draw would leave the coronal buckets with a handful of cases and an
unusable per-bucket FID.

The price is that the cohort no longer mirrors the real plane distribution — which is why the
results carry **two** overall numbers (see §5), and why `cohort.json` separately records
`population_bucket_counts`, the *eligible-population* frequencies used as weights.

Selection is deterministic given `--seed`, and asking for one bucket picks exactly the same cases as
asking for ten — bucket order can never shift another bucket's draw. See
`cohort.select_cohort_buckets`.

### Which series of a study

```bash
--series-selection one_per_study_per_bucket   # default
```

| Value | One sample per… | Use it for | Cases on the real test split |
|---|---|---|---|
| `one_per_study_per_bucket` | **(study, modality, plane)** | **evaluation** — one independent observation per bucket | 21,148, every bucket populated |
| `one_per_study_per_sequence` | (study, sequence) | a single-plane study only | 17,583, but the planes collapse — see below |
| `all` | eligible series | training, where you want maximum data | 29,027 |
| `one_per_study_deterministic` | study, across all sequences | a single-sequence cohort only | 4,893, but **4,861 of them T1w** |
| `one_per_study_random` | study, redrawn each epoch | training | 4,893/epoch, same modality skew |

Three traps this table exists to prevent:

- **`one_per_study_per_sequence` collapses the PLANES.** It looks equivalent to the default, and is
  not: when a study has the same sequence in several planes it prefers the center-modality series,
  which on MR-RATE is axial T1w. Measured on a 200-per-bucket request it produced T2w SAGITTAL = 6
  cases and T1w CORONAL = 16 — silently, since a per-sequence count still looks healthy.
- **`all` pseudo-replicates.** Near-duplicate series from one session are not independent
  observations — measured mean 1.96, max 13 per (study, sequence). Plain means overweight
  multi-series studies and plain std/CIs come out falsely narrow, because the aggregation does not
  model clustering. It also biases FID by shrinking the real distribution's apparent spread.
- **`one_per_study_deterministic` collapses to T1w.** It picks one series per *study*, preferring the
  center-modality series. A 4-sequence request therefore yields ~99% T1w and leaves T2w/FLAIR/SWI
  essentially unevaluated — silently.

### What `--n-per-bucket` counts

A **case** = one (study, modality, plane) triple = exactly one series. `cases == series` always,
while distinct *studies* is smaller: one patient contributes one case per bucket they have.

| `--n-per-bucket` | buckets | cases (= series) |
|---|---|---|
| 200 | 10 | 2,000 |
| 0 (all) | 10 | 21,148 |

Some patients contribute to more than one bucket, so quote the **per-bucket** rows as primary — the
two overall rows are mildly clustered.

### Field of view

```bash
--geometry-mode per_modality_plane               # default
--geometry-mode fixed --fixed-shape 256 256 256 --fixed-spacing-mm 1 1 1
```

| Mode | What you get | Use when |
|---|---|---|
| `per_modality_plane` | each bucket on NVIDIA's published recommended FOV for that (modality, plane) | **the default, and what you want.** Both the cohort and the generator use each anatomy's natural FOV. |
| `fixed` | every volume on one grid | a deliberate single-geometry study only |

**How a bucket's grid is derived.** NVIDIA publishes a recommended *field of view* per (modality,
plane). The shape is the nearest multiple of **32** to that FOV, and the spacing is then
`FOV / shape`. So the FOV matches NVIDIA's exactly, and the spacing — which is a real conditioning
input to the diffusion UNet, not just metadata — comes out as whatever makes it so.

Why 32 and not 16: the UNet has 4 levels, the latent is `shape / 4`, and the latent must be
divisible by 8. A div-16-but-not-32 shape raises a skip-connection size mismatch mid-sampling
(verified: `(240, 240, 176)` fails with "Expected size 14 but got size 15"). `cli.predict_generation`
refuses such a shape up front rather than padding, since padding would change the FOV.

| Bucket | shape (X, Y, Z) | spacing (mm) | FOV (mm, = NVIDIA's) |
|---|---|---|---|
| T1w AXIAL | 256 × 256 × 160 | 0.9375 · 0.9375 · 1.0875 | 240 × 240 × 174 |
| T1w SAGITTAL | 192 × 256 × 256 | 0.9167 · 0.9766 · 0.9766 | 176 × 250 × 250 |
| T1w CORONAL | 256 × 192 × 256 | 0.9375 · 1.0417 · 0.9375 | 240 × 200 × 240 |
| T2w AXIAL | 256 × 256 × 160 | 0.9375 · 0.9375 · 0.9875 | 240 × 240 × 158 |
| T2w SAGITTAL | 160 × 256 × 256 | 1.0125 · 0.9375 · 0.9375 | 162 × 240 × 240 |
| T2w CORONAL | 192 × 192 × 192 | 1.0417 · 0.9375 · 1.0417 | 200 × 180 × 200 |
| FLAIR AXIAL | 256 × 256 × 160 | 0.9766 · 0.9766 · 1.0938 | 250 × 250 × 175 |
| FLAIR SAGITTAL | 192 × 256 × 256 | 0.9167 · 0.9766 · 0.9766 | 176 × 250 × 250 |
| FLAIR CORONAL | 256 × 192 × 256 | 0.9766 · 1.0417 · 0.9766 | 250 × 200 × 250 |
| SWI AXIAL | 224 × 224 × 160 | 1.0268 · 1.0268 · 0.9062 | 230 × 230 × 145 |

This replaced a single 256³ @ 1 mm grid, which padded a measured **52%** of each volume with
background; at the per-bucket FOVs it is **9.7%**.

**`--fixed-shape` and `--fixed-spacing-mm` are `(X, Y, Z)`** = (Right-Left, Anterior-Posterior,
Superior-Inferior) — the same order the Dataset returns, the cohort records, and the NVIDIA model
uses. Internally the preprocessing works in `(D, H, W)`; `cli.preprocess` converts at the boundary
via `geometry.xyz_to_dhw`. You never see `(D, H, W)` from outside the package: every file in a
cohort directory is `(X, Y, Z)`.

Because shapes differ between buckets, use `GeometryBucketBatchSampler` for batch_size > 1, and
never compare a per-bucket run's numbers with a fixed-mode run's.

### Intensities

```bash
--normalizer percentile          # percentile (default) | zscore | minmax
--posterior-shift-mm 0           # default 0 for R2V cohorts
```

`--posterior-shift-mm` compensates for defacing, which removes anterior tissue and so pulls the
intensity centroid backwards.

**An evaluation cohort must use whatever value the run it scores was TRAINED with — for the final
four that is 15, not 0.** `cli.train_r2v` had no `--posterior-shift-mm` flag until 2026-08-10, so
every run including the final four silently took `R2VDatasetConfig`'s default of **15.0**, while
`02_preprocess.sbatch` hardcoded **0**. Nothing surfaced the divergence: the flag simply did not
exist on one side.

Measured effect (two 120-case cohorts differing only in this value): `crop_or_pad` clamps the shift
away unless a volume's resampled A-P extent exceeds its bucket target, so **15.8% of test cases
differ**, with correlation **0.63–0.85** on the affected ones. It is not uniform — T2w SAGITTAL and
FLAIR CORONAL 5/12 each, T2w CORONAL 3/12, three buckets 0/12 — so scoring at 0 would have
penalised the coronal and sagittal buckets specifically, for a displacement the model did not cause.

`test_v2` (shift 0) is therefore superseded by **`test_v3` (shift 15)**. The earlier note that 0
"won 8 of 10 buckets" was measured on cohort construction alone and never applied to training; it is
not a reason to score a model on a grid it never saw. Both CLIs now expose the knob, the training
flag's default is read off the dataclass so the two cannot drift, and the value is hashed into
`cohort_id` so the two can never be mixed. Pinned by
`test_r2v_inference_contracts.py::test_training_and_cohort_preprocessing_defaults_are_reachable_and_agree`.

Heads up: the default percentile normalizer does **not** clip. Values above the 99.5th percentile
exceed 1.0. PSNR's `data_range=1.0` is therefore a fixed reference scale, not a true ceiling —
treat PSNR as relative, not absolute.

### Conditioning text

```bash
--report-sections findings impression      # default
```

Choose from `raw`, `clinical_information`, `technique`, `findings`, `impression`. The whole selected
text is used every time — never truncated or randomly subsampled.

---

## 5. Reading the results

Same layout for every task:

| File | What it is |
|---|---|
| `metrics_per_bucket.csv` | **read this first.** One row per (modality, plane): the shape and spacing it was scored at, the sample counts, and every metric. |
| `metrics_summary.csv` | the aggregates — one row per modality, then `overall_macro` and `overall_weighted` |
| `per_case_metrics.csv` | one row per scored case |
| `summary.json` | the same numbers machine-readably, plus which metric groups ran and why |
| `distribution_metrics.json` | FID and diversity metrics, when computed |
| `anatomy_metrics.json` | anatomical plausibility of the produced population against the real one |
| `excluded_cases.json` | every case that was *not* scored, with a specific reason |
| `run_manifest.json` | exactly what ran: `cohort_id`, task, checkpoint hashes, versions |
| `figures/` | example slice montages (ground truth / prediction / difference), worth a look before trusting any number |

### The two overall rows

`metrics_summary.csv` ends with two rows that will not agree, deliberately:

| Row | What it means |
|---|---|
| `overall_macro` | unweighted mean across buckets — every anatomy counts equally |
| `overall_weighted` | weighted by `population_bucket_counts`, i.e. what the real test split looks like |

The cohort is sampled to **equal size per bucket**, so cohort counts carry no information about how
common a bucket is and must never be used as weights — that would silently turn the weighted
aggregate back into the macro one. Quote `overall_macro` when comparing models, `overall_weighted`
when claiming what a clinical population would see.

`metrics_per_bucket.csv` also carries `nvidia_train_n` — NVIDIA's own published count of training
images for that bucket — and an `nvidia_low_train_n` flag. Three T2w buckets were trained on
195/125/551 images and NVIDIA states quality is not guaranteed there. They are **kept** in every
aggregate; the column is there so a weak number can be read as coverage rather than failure.

Always check `n_scored` against `n_cohort` per bucket. If they differ, `excluded_cases.json` says
why for every case. Nothing is ever silently dropped.

### The metrics

| Metric | Tells you | Direction | Reasonable range here | Watch out |
|---|---|---|---|---|
| MAE / MSE (fg) | average per-voxel error | lower | compare within your own runs | foreground mask comes from the ground truth only |
| PSNR (fg) | error on a log scale | higher | ~25-30 dB is a faithful VAE reconstruction; single digits means something is broken | reference scale is fixed at 1.0, not a real max |
| SSIM (3D + per plane) | does it look like the same image | higher, max 1.0 | > 0.5 typical for a lossy 3D VAE, > 0.8 strong | inflated on nearly-empty slices — check `n_slices_used` |
| Edge preservation, Laplacian variance, HF energy | is detail surviving or blurring | ~1.0 = preserved | 0.6-0.9 common for a compressive VAE | secondary; read alongside PSNR/SSIM |
| MedicalNet FID | do the real and produced *populations* match, in 3D medical features | lower | generation will be much higher than reconstruction — expected, not a bug | MedicalNet was trained on many organs, not brain MRI specifically |
| 2.5D Inception FID | same idea, on 2D slices across all three planes | lower | not comparable in scale to the MedicalNet number | Inception-v3 has never seen a medical image |
| Inception Score | generic "confident and diverse" | higher | not very informative for brain MRI | prefer precision/recall below |
| Precision / Recall / Density / Coverage | precision = do outputs look real; recall/coverage = do they span the real range | higher | high precision + low recall = mode collapse | needs 50+ per group to be stable |
| Intra-set SSIM (real vs produced) | mode collapse, the standard generation-literature probe | compare the two, not the absolute value | produced clearly **above** real = less variety than the data | mid-axial slices only, capped at 200 pairs per bucket |
| Anatomy (L-R symmetry, intracranial fraction, tissue contrast, background purity) | does the output look like a brain at all | closer to the real column | a large KS statistic means the populations differ on that measure | heuristic masks, not a segmentation |
| Report-image similarity | does the volume match the report | higher | **unavailable** | no validated MRI image-text model exists in this project; recorded as unavailable with a reason, never faked |
| **Blinded classifier consistency** | does a classifier trained only on *real* volumes read, off a generated one, the findings its conditioning report described | higher | above 0.5; judge against the real-volume reference in the same table | needs `--report-classifier`. Per-label AUROC is only meaningful where the classifier is `usable` — i.e. where it can do that label on real data |

### Speed

Evaluation is CPU-compute-bound: decompressing and reading both volumes takes ~0.3 s against
~6.7 s of metric compute, so I/O is a few percent of the work. **Staging a cohort to node-local
`$TMPDIR` is therefore not worth doing** — the filesystem already delivers 3.4 GB/s here, well past
what the metrics can consume.

Use `--workers` instead. Per-case scoring is embarrassingly parallel and scales near-linearly
(measured 7.32× at 8 workers), with byte-identical results at any worker count.
`slurm/05_evaluate.sbatch` already passes `$SLURM_CPUS_PER_TASK` with 32 CPUs requested.

It does not help `--task generation`, which has no paired metrics, and it does not speed up the
distribution-metric feature extractors (serial, on GPU). Full breakdown:
[eval README](../contrastive-pretraining/mrrate_r2v/eval/README.md#making-it-fast).

### Comparing two runs

Check that both `run_manifest.json` files show the same `cohort_id`. If they do, the two models saw
identical cases at identical FOV with identical preprocessing, and the numbers are directly
comparable. If they differ, they are not — and `cli.evaluate` would have refused to produce them
against a mismatched cohort in the first place.

---

## 6. Scoring a checkpoint that already wrote NIfTI files

You do not need `predict_r2v` for this.

```bash
python -m mrrate_r2v.cli.import_predictions \
    --cohort $COHORT --predictions-csv /path/to/predictions.csv \
    --out .../predictions/external_v1

python -m mrrate_r2v.cli.evaluate --task report2volume \
    --gt $COHORT --pred .../predictions/external_v1 --out .../results/external_v1
```

CSV schema (`study_key`/`series_key` must match the manifest's `study_uid`/`series_id` exactly —
never a filename or row position):

```csv
study_key,prediction_path,series_key
STUDY_0001,/path/vol_0001.nii.gz,T1w-PRE-AXI
```

`series_key` may be omitted only when the study has exactly one case in the cohort. Ambiguous,
duplicated, or unmatched rows are rejected into `import_report.json` with a reason — never guessed.

Matching happens once, here, and is recorded. The evaluator then works from `case_id` like any
other prediction set.

---

## 7. The report-to-volume model

A frozen NV-Generate-MR-Brain denoiser plus a trained report adapter. Two extra stages:

```
cli.train_r2v     →  adapter_*.pt        (frozen base + RadBERT; only the adapter learns)
cli.generate_r2v  →  volume + manifest   (one report, or a whole cohort)
cli.predict_r2v   →  PREDICTION dir      (a cohort, in the format cli.evaluate scores)
```

`slurm/06_train_r2v.sbatch` and `slurm/07_generate_r2v.sbatch` are the runnable versions of both.
They use `$SIF_IMAGE_TEXT` (`nvidia+redbert.sif`), not the base image, because the base image has no
transformers.

What is trainable: `context_proj`, the five cross-attention adapters, and `null_context` — **8,080,000
of 188,580,868 parameters (4.28%)**. `models/adapter.py` asserts that before the first optimizer step
and refuses to start otherwise; the counts are logged every run.

Three things worth knowing before changing anything here:

- **The loss is NVIDIA's**: `torch.nn.L1Loss()` on the rectified-flow velocity target `x0 − ε`
  (`diff_model_train.py:328,478`). Not MSE, not EDM-weighted, and no auxiliary report/image
  alignment term. The adapter earns its keep on the original generative objective.
- **`scale_factor` comes from the base checkpoint** (0.970450), not from `1/std(z)` of the first
  batch. Official recomputes it because it trains from scratch; a frozen denoiser cannot follow a
  rescaled latent space. `--scale-factor recompute` restores the official behaviour.
- **There is no EMA**, because there is none in the official code — a search for `ema` across
  `NV-Generate-CTMR/scripts/*.py` and `configs/*.json` returns nothing, and the released checkpoint
  carries only `unet_state_dict`/`optimizer_state_dict`/`scheduler_state_dict`. Adding one would be
  inventing a behaviour to preserve, so nothing here does.

Guidance is hierarchical, with the report as an increment on top of NVIDIA's modality term:

```
D_guided = D_00 + s_modality * (D_m0 − D_00) + s_report * (D_mr − D_m0)
```

`--report-guidance-scale 0` collapses this to `diff_model_infer.py:207` exactly — that equality is
asserted numerically in `tests/test_r2v_conditioning.py`, so report guidance cannot silently change
what the original model does.

For a different text encoder, add it to `TEXT_EMBEDDERS` in
[`text.py`](../contrastive-pretraining/mrrate_r2v/text.py) and pass `--text-encoder <name>`. The
trainable projection is built from `embedder.output_dim`, so no other file changes.

### The training recipe

Sizes, from the run logs: **575,187** train and **23,356** val (study, series) samples. 8,080,000
trainable parameters. One H200 node = 4 GPUs; the `h200` partition caps a job at **24 h**.

| | value | why |
|---|---|---|
| optimizer | `Adam`, betas (0.9, 0.999), eps 1e-8, wd 0 | `training.py:104` = `diff_model_train.py:198`; the recipe the checkpoint was trained with |
| schedule | `PolynomialLR`, power 2, no warmup | `training.py:109` = `diff_model_train.py:220` |
| lr | **sweep pending** — jobs 711945-49 | every earlier sweep is void, see below |
| precision | bf16 autocast | fp16 overflows this model (`TrainingConfig.amp_dtype`) |
| grad clip | 1.0 | measured grad norms are 0.002–0.017, so it never binds — it is a NaN backstop, not regularisation |
| batch / GPU | 8 | **not 4** — see below; 4 halves throughput and the memory is there |
| effective batch | 256 | `batch x grad_accum x world_size`, **not rescaled by node count** — at 8 GPUs pass `R2V_GRAD_ACCUM=4`, or EB silently doubles |
| epochs | 2 (~4,500 optimizer steps) | 8.08M parameters against 575k volumes; no overfitting has been observed or is expected inside 2 epochs |
| nodes x GPUs | 2 x 4 | ~11 h for 2 epochs at batch 8, so the whole run fits one job under the 24 h cap |
| report/modality dropout | 0.10 / 0.10 | enables classifier-free guidance at inference |

**Throughput is bounded by the frozen VAE encode, not by the adapter.** `vae_encode_seconds` is
90.6% of `step_seconds` at batch 8 (`cache/r2v/benchmarks/h200_688301.json`). Measured **20.2
volumes/s** on one node at batch 8 (job 710049: 350 micro-steps in 556 s, 4 ranks) → **~7.9 h/epoch
on one node**. Two-node scaling measured at 73% on a 200-step run (`scale3_1node` vs
`scale3_2node`), which is a floor: that run syncs every micro-step, the recipe above syncs every 4.

**Batch 4 costs half the throughput, and the configs default to it.** Job 711945 (configuration B,
4 ranks x batch 4) runs optimizer steps 25 → 50 in 641.3 s — 400 micro-steps, startup excluded —
= 1.60 s/micro-step = **9.98 volumes/s**, against 20.2 at batch 8, so 16 h/epoch instead of 7.9.
At effective batch 256 that is 25.7 s per optimizer step, against ~12.7 s at batch 8. The per-volume cost at batch 4 and 8 is within 1% only at the 256³
fallback bucket (`h200_688301.json`); at the real per-bucket geometries the volumes are smaller, so
batch 4 leaves the GPU idle and fixed per-step overhead dominates. Memory is not the constraint:
configuration B at batch 4 peaks at 35.6 GB of 140 (job 711503), and the 256³ worst case at batch 8
was 85–87 GB. Raise `R2V_BATCH_SIZE` to 8 and halve `R2V_GRAD_ACCUM` to hold the effective batch;
drop back to 4 only if a validation pass OOMs.

**Every learning-rate sweep before 2026-08-07 is void.** `lrsweep_*` finished with `final_loss:
NaN`; `lrsweep2_*` never wrote a summary; `lrsweep3_*` looks clean in `train_summary.json` but its
log reads `skipped 828` at optimizer step 600 — **58% of its optimizer steps were discarded** for
non-finite gradients, and not uniformly: the cause is the cuDNN SDPA backend returning a non-finite
`grad_q` at latent 48³ in bf16 (`verify_fix_695257` E1), which hits only the buckets with a 48 in
their latent. Those runs trained on a biased subset of the buckets. The fix is the backend guard in
`report_conditioned_unet.py:71-100` (FLASH + EFFICIENT + MATH, cuDNN excluded), verified at 0/1200
non-finite across all 16 bucket geometries (E3), 0/200 skipped in the real pipeline (E5), and 0
skipped over 200 optimizer steps on 4 GPUs (job 710049).

A second defect made those sweeps compare the wrong thing: `fit` passed a **micro**-step count as
the `PolynomialLR` horizon while stepping the scheduler once per **optimizer** step, so the decay
ran `grad_accumulation_steps` times too slowly — job 690962 reached 64% of its base LR at its last
step instead of ~0, and at accum 16 the LR would have been constant to within 12%. Fixed at
`training.py:557-565`, pinned by
`test_r2v_training.py::test_the_schedule_horizon_is_counted_in_optimizer_steps_not_micro_steps`.

### The four-configuration result (2026-08-09)

Jobs 714497-500, `slurm/final/`: lr 1e-3, effective batch 256, 2 epochs (4,493 optimizer steps),
2 nodes x 4 H200, **0 skipped steps in all four**. A, B and D completed; C reached step 4200 and was
killed by a host-RAM OOM during its N=512 pass, after that pass and its checkpoint were written.
All numbers below are the step-4200 full validation at N=512.

| | SSIM | FVD | 2.5D FID | **sensitivity** | trainable |
|---|---|---|---|---|---|
| A `cxr_bert_cls` | 0.34734 | 16.34 | 24.89 | **+0.0418** | 8.08M |
| B `cxr_bert_tokens` | 0.34228 | 16.52 | **24.36** | +0.00179 | 8.08M |
| C `radbert_tokens` | 0.35201 | 16.65 | 24.70 | -0.00124 | 8.08M |
| D `report2ct_style` | **0.35738** | **16.04** | 26.13 | +0.00506 | 11.75M |

**Image quality does not separate these four; conditioning separates them by ~23x.** SSIM spans
0.342-0.357, FVD 16.0-16.7, FID 24.4-26.1 -- all within a few percent, and the winner differs by
metric. `ssim_advantage` (does the *correct* report pull the generation toward the *right* patient)
is the only axis with a real gap, and only configuration A has one: +0.0348 / +0.0332 / +0.0418
across its three measurements, the last on 512 cases. B and C sit at zero to within noise on every
measurement including at N=512.

**This inverts the prediction the B and C arms were built on.** The reasoning -- recorded in
`slurm/configs/` -- was that A's single pooled token makes the cross-attention degenerate (softmax
over one key is constant, so `to_q`/`to_k` receive no gradient and 33.8% of the adapter is inert),
so keeping the token axis had to condition better. It does not. A plausible mechanism, untested: the
CXR-BERT CLS vector was trained by a sentence-level CLIP objective and is already a usable summary,
whereas the token path must learn what to attend to from scratch within 4,493 steps on an 8M
adapter. Read this as "the token-sequence adapters did not converge to using their text here", not
as "token attention cannot work".

**Do not read D's SSIM/FVD lead as a conditioning win**: D trains 11.75M parameters against 8.08M
(a 2560-wide fusion needs a wider `ContextProjection`), so it is not capacity-matched, and its
sensitivity is 8x below A's.

**Configuration E (`report2ct_style_meta`) was added after this table and has not run yet.** It is
D plus a third conditioning token holding `[MODALITY] .. [PLANE] .. [SPACING] ..` as text -- the
prefix A, B and C carry at the head of their joined string and D structurally could not, since it
never composes one. Same three encoders, same masked-mean pooling, same feature-axis order, same
trainable count as D; findings and impression keep sequence indices 0 and 1, so **D vs E is a
one-token difference and nothing else**. The information is not new to the model (modality is
already a class label, spacing already a `spacing_tensor`, plane implied by the bucket geometry) --
only its entry point is, as a cross-attention key the adapter can weight per voxel -- so a null
result is an honest possible outcome rather than a wiring bug. Run it with
`slurm/final/run_E_report2ct_style_meta.sh` and score it with
`slurm/final_eval/run_E_report2ct_style_meta.sh`; both source the same common files A-D used, so
the arm is comparable by construction.

Caveats that bound all of it: report-volume *semantic* fidelity is still unmeasured (see the module
docstring in `validation.py`), so `ssim_advantage` is a structural stand-in; one seed per arm; and
FVD/FID at N=512 are still rank-deficient against 512- and 2048-d features.

### How often to validate, and on how many cases

Every validation case costs a full diffusion sampling run, so this is a real budget. Measured at
N=16 on 4 ranks (`lrsweep_1e-5`, validation 1): 180.7 s total = ~72 s fixed overhead + **1.50
s/case** (generate + features + SSIM) + **21.0 s/case** for the condition-sensitivity swap, which is
by far the most expensive part and belongs on its own rarer schedule.

That per-case model **underestimates a real pass by about 3x** and should not be used for budgeting.
`val/seconds` from the sweep's own checkpoints is the number to trust: **572-765 s per pass at
N=64** (jobs 711945, 713012), of which the recorded components account for only ~20%. The remainder
is not the Frechet `sqrtm` (1.3 s at 512-d, 25 s at 2048-d across three planes, timed directly) —
it is almost certainly re-reading and preprocessing the validation volumes every pass, so the cost
scales with N.

| | N | every | measured / budgeted cost |
|---|---|---|---|
| quick pass (SSIM curve + FVD/FID trend) | 128 | 600 optimizer steps | ~15-20 min; ~7 points over a 2-epoch run, ~2 h against ~11 h of training |
| condition sensitivity | 8 | 1800 optimizer steps | 192-282 s measured |
| full pass | 512 | once, late | expensive; the calibrated numbers come from `cli.evaluate` instead |

**Raise the process-group timeout when you touch any of this.** Rank 0 computes every distribution
metric alone after the `all_gather`, so the other ranks sit inside a collective for minutes. At
NCCL's 600 s default that is indistinguishable from a hang: job 713012 completed all 600 optimizer
steps and was then killed in its final validation by
`WorkNCCL(SeqNum=240014, OpType=ALLGATHER) ran for 600011 milliseconds before timing out`.
`cli/train_r2v.py` now passes `timeout=timedelta(hours=1)` to `init_process_group`.

**The reference values, measured** (`cli.validation_reference`, val split, N=64, seed 0, job 713646):

| | value | reading |
|---|---|---|
| `ssim_autoencoder` | **0.910 ± 0.035** | the frozen VAE round-trip. Every generated volume comes out through this decoder, so this is the practical structural ceiling — and models sitting at ~0.33 are nowhere near it. **Low SSIM here is not a decoder artefact.** |
| `fvd_real_vs_real` | **30.01** | two disjoint halves of the *real* set; true answer 0 |
| `fid_2p5d_real_vs_real` | **21.14** | as above |
| `ssim_identity` / `shift_1vox` / `blur_sigma1` / `noise_sigma0p05` | 1.000 / 0.875 / 0.914 / 0.572 | SSIM responds to damage, in a sensible order |

Two consequences. First, the `validation.py` warning that quoted a real-vs-real floor of "~6100" and
declared the curve unusable was wrong by two orders of magnitude, and is now corrected against this
measurement. But the honest reading is still cautious: measured **at the same N=64**, job 711945
scored FVD 31.3 against a floor of 30.0 — the model is essentially *at* the FVD measurement floor,
so that metric has almost no dynamic range here. FID (36.0 against a floor of 21.1) has more. SSIM
remains the metric to rank on. (Model FVD of 43–58 quoted elsewhere was measured at N=16; Frechet
values at different N are not comparable, which is exactly what `rank_level` exists to record.)
Second, `real_vs_real_baseline` splits N in half, so an N=64 reference
describes a **32-per-side** floor while the curve compares 64 against 64 — the reference must be run
at twice the validation N to be like-for-like (N=256 for the final runs' N=128).

**Read SSIM as the curve and FVD/FID as a trend.** `rank_status` (`validation_metrics.py:85`) calls
a Frechet distance well-conditioned only at `N >= 2 x feature_dim` — 1024 volumes for FVD, 4096 for
2.5D FID. Anything affordable in-loop is `rank_level: 0`, comparable only against other values at
the same N with the same extractor, never across runs with different N. Calibrated distribution
numbers come from `cli.evaluate` over a real cohort, not from the training loop.

What makes N=128 enough is that the case list and seed are **fixed** (`select_validation_cases` is
deterministic in `config.seed` and prefix-stable), so consecutive points are paired and the curve
tracks the model rather than the sample. `cli.validation_reference` gives the real-vs-real noise
floor those curves are read against; run it once at the same N and seed and pass
`--validation-reference`.

**There is no held-out validation *loss*.** Grepping `val/loss` across `mrrate_r2v/` returns
nothing, so the only validation signal is generative and therefore expensive. The same L1 velocity
objective on val batches costs one forward pass — roughly 100x less per case than sampling — and
would give a true train/val overfitting curve at every 100 steps for a negligible fraction of the
budget. It is the single highest-value addition to the validation path.

---

## 7b. Evaluating a trained adapter (the four-arm test run)

Training produces `adapter_*.pt`; this is how those become numbers. Everything here postdates
2026-08-09 — see "What a pre-2026-08-09 evaluation is worth" at the end of this section.

### The three artifacts you need first

```bash
cd contrastive-pretraining

# 1. the evaluated cohort -- test split, at the report format AND posterior shift training used
sbatch slurm/02_preprocess.sbatch test_v3 200

# 2 + 3. the blinded classifier's own data -- train and val splits, never test
R2V_SPLIT=train sbatch slurm/02_preprocess.sbatch clf_train_v2 500
R2V_SPLIT=val   sbatch slurm/02_preprocess.sbatch clf_val_v2   100
sbatch slurm/14_train_report_classifier.sbatch clf_train_v2 clf_val_v2 report_classifier_v2
```

`02_preprocess.sbatch` now passes `--report-format findings_impression_meta` and writes
`report_sections.json`. Both matter and neither is a default:

- **`--report-format`**: A, B and C were trained on the order-agnostic `*_meta` spec, whose
  conditioning text carries a `[MODALITY]/[PLANE]/[SPACING]` prefix. A cohort built without it holds
  text with no prefix — out of distribution for three of the four arms.
  `assert_report_format_matches` refuses that pairing rather than scoring it, so the symptom is a
  hard exit. A cohort freezes **one** format, so build a second with `impression_findings_meta` if
  you want to measure the order-robustness the two-format training was for.
- **`report_sections.json`**: configurations D and E encode findings and impression as separate
  cross-attention tokens and cannot recover them from the joined string. `cli.predict_r2v` refuses
  up front on a cohort that has no sections rather than conditioning D on one token instead of two.
  E additionally needs the `acquisition` section; the Dataset composes it per case, and the
  cohort path fills it in from the case's own modality/plane/spacing
  (`cli/generate_r2v.py` -> `formats.with_acquisition_section`).

Neither exists in the old `test_v1` (built 2026-07-30), which is why the cohort is rebuilt rather
than reused.

### Then the four runs

```bash
slurm/final_eval/run_A_cxr_bert_cls.sh
slurm/final_eval/run_B_cxr_bert_tokens.sh
slurm/final_eval/run_C_radbert_tokens.sh
slurm/final_eval/run_D_report2ct_style.sh
slurm/final_eval/run_E_report2ct_style_meta.sh
```

Each submits **two** jobs — `13_predict_r2v.sbatch` then `05_evaluate.sbatch`, chained with
`--dependency=afterok` so a failed prediction set can never be scored. The four `run_*.sh` scripts
set `R2V_CONFIG` and nothing else; every other parameter lives in
`slurm/final_eval/_final_eval_common.sh`, exactly as `slurm/final/_final_common.sh` does for
training. "The only difference between the four evaluations is the conditioning mechanism" is
enforced by construction rather than by four command lines staying in sync.

`SMOKE=1 slurm/final_eval/run_A_cxr_bert_cls.sh` runs eight cases end to end (~30 min) and proves
the conditioning rebuilds, the report format is accepted, and the intensity space is right. Do that
once per arm before committing four long jobs.

**Checkpoint selection**: `adapter_last.pt` where it exists, `adapter_step0004200.pt` for C — whose
training job was killed by a host-RAM OOM in its final validation. So C is compared at 4,200
optimizer steps against 4,493 for the others; quote that caveat wherever the four-way table appears,
or set `CHECKPOINT_KIND=step4200` for the step-matched comparison.

### 13 vs 07 — only one of them produces numbers

| | writes | scoreable |
|---|---|---|
| `13_predict_r2v.sbatch` (`cli.predict_r2v`) | `predictions.json` + per-bucket `.npz`, in the cohort's percentile-normalised space | **yes** |
| `07_generate_r2v.sbatch` (`cli.generate_r2v`) | `.nii.gz` + a generation manifest, in NVIDIA's int16 `[0, 1000]` range | no — use it to *look* at a sample |

### W&B: one table, a few panels

`slurm/final_eval/` logs to W&B by default (`R2V_WANDB=online`, project `mr-rate-r2v-eval`, one
group for the four arms, run named after the arm). A hand-run `05_evaluate.sbatch` stays offline
unless you set `R2V_WANDB`.

| What | Where | Note |
|---|---|---|
| **`metrics/all`** | a `wandb.Table` panel | **every metric that ran**: per bucket, then per modality, then `overall`, then distribution / diversity / mode-collapse / anatomy / report-consistency, then provenance rows — including **`train_samples_seen`**, the volumes the optimizer actually consumed |
| headline scalars | run summary + run table | `psnr_fg`, `ssim3d_whole`, `mae_fg`, `ncc_fg`, both FIDs, the report-consistency macro AUROC and its real-volume ceiling, `train_samples_seen`. These become sortable columns, so the four-arm comparison *is* the W&B run table |
| example panels | `examples/<bucket>/<case_id>` | interactive ground-truth vs generated with a slice slider and the report text, rendered by the **same** `figures.validation_panel_html` the training loop uses |

The same table is printed to stdout at the end of the job, so the Slurm log carries the full result
rather than three scalars.

**Panels are a sample, never the cohort.** `--wandb-panels` defaults to 6; 2,000 interactive panels
is ~1 GB of base64 and an unusable workspace (measured: ~1.2 MB each). Cases are the worst, best and
median by `--wandb-rank-metric` (default `psnr_fg`), spread across buckets, and each panel records
*why* it was picked — an unlabelled panel invites reading a best case as typical.

**Panels embed patient report text**, so they are gated behind `--wandb-log-reports`
(`R2V_WANDB_REPORTS=1`, the default in `final_eval/`) exactly as in `cli.train_r2v`. Keep that
project private, or set `R2V_WANDB_REPORTS=0` — the metrics table is logged either way and only the
panels are withheld.

**Why the training-sample count needs `cli.predict_r2v` to record it.** The evaluator reads only
`.npy` and `.json` by design and must never open a checkpoint, so `predict_r2v` writes
`model.training` into `predictions.json` (`models/adapter.py:training_provenance`). `samples_seen`
is `optimizer_step x effective_batch_size` — 1,150,208 for these runs — and is *not* the dataset
size; `samples_per_epoch` (~575k, matching the logged train split) is reported separately rather
than conflated. `effective_batch_size` needs `world_size`, which lives in `train_summary.json`, not
in the checkpoint: configuration C's job died before writing one, so `final_eval/` passes
`--train-world-size 8` (the launch geometry recorded in `slurm/final/_final_common.sh`) and the
prediction set records `world_size_source` so a supplied value is never mistaken for a read one.

### The acceptance gate

```bash
python3 slurm/check_run.py --cohort <cohort> \
    --pred-r2v <predictions> --results-r2v <results>
```

Four report-to-volume criteria: `ER1` every case scored and none excluded, `ER2` the predictions are
in the cohort's intensity space (checked from the numbers, not trusted), `ER3` blinded-classifier
consistency above chance on the usable labels, `ER4` the classifier was not fitted on the test
split. Run it before reading any number as a result.

### What a pre-2026-08-10 evaluation is worth

Nothing — and there was no such evaluation, because two of these five defects made
`cli.predict_r2v` refuse to start for three of the four arms. The other three were silent:

1. **`cli.predict_r2v` wrote volumes in NVIDIA's int16 `[0, 1000]` range** against a `[0, 1]`
   ground truth. Every paired metric consumes a 1000x-offset pair and returns a plausible number.
   `predict_generation` had always divided by 1000 and `validation.py` had a guard; the evaluated
   path had neither. It now generates with `postprocess=False` and asserts the space on case 1.
2. **Configurations B and C could not be rebuilt at inference.** `rebuild_embedder` had no branch
   for `kind="tokens"`, so those adapters fell through to a path that defaults to RadBERT — and
   since CXR-BERT and RadBERT are both 768-wide, every downstream shape check passed. Given
   `--text-checkpoint` (which `07_generate_r2v.sbatch` hardcoded) a CXR-BERT arm silently loaded
   RadBERT. `load_adapter_checkpoint` is now also given the live embedder's identity, so
   `assert_conditioning_compatible` actually runs.
3. **Configuration D could not run at all**, since nothing passed it per-section text.
4. **`assert_report_format_matches` compared nothing.** It read `cohort.spec.geometry_fingerprint`,
   an attribute a parsed `cohort.json` — a plain dict — never has, so the cohort's format read as
   `None` unconditionally. That is not a benign no-op: it passes silently when the adapter records
   no format, and refuses *every* cohort when it records one. All four final adapters record one.
   Its unit test survived because the test's stub cohort exposed the same wrong attribute.
5. **The format check was applied to configuration D**, which never reads the joined string at all
   and records `report_format=None` by construction — so it matched no cohort and D was refused
   regardless. The check now takes the embedder and exempts a sectioned configuration; what D
   actually needs (the cohort *has* sections) is checked separately, before any sampling.

Pinned by `tests/test_r2v_inference_contracts.py`, which exists because three of these produced
complete, valid-looking output and the other two were only reachable by running the thing.

---

## 8. Testing

```bash
cd contrastive-pretraining
python -m pytest                          # everything, with coverage
python -m pytest --no-cov -q              # faster
python -m pytest tests/test_cohort_contract.py tests/test_eval_tasks_and_runner.py -v
```

**The R2V tests are not in version control, and CI does not run them.** `.gitignore` ignores
`/contrastive-pretraining/tests`, so only the 8 original *contrastive* test files are tracked; all
33 R2V test files — including the two invariant files below — exist on the working machine only.
This is deliberate and left as is, but it has two consequences worth knowing: the suite will not
survive a fresh clone, and a change that breaks a load-bearing invariant will pass CI. Run
`python -m pytest` locally before trusting any R2V change; three of the five defects fixed on
2026-08-10 were test-visibility failures.

No GPU, no real data, no checkpoints — a few seconds. Two files are worth knowing about:

- `test_cohort_contract.py` — the comparability guarantees. That `cohort_id` changes when the seed,
  FOV, normalizer, or case list changes; that a mismatched prediction set is refused; that
  `cohort.json` carries no patient identifiers.
- `test_eval_tasks_and_runner.py` — that `generation` never produces a voxelwise metric, that the
  result layout is identical across tasks, and that a shape-mismatched prediction is excluded rather
  than resized.

---

## 9. Where things are

```
contrastive-pretraining/
  mrrate_r2v/
    cohort.py          the frozen ground-truth contract
    predictions.py     its mirror on the prediction side
    data/              manifest -> Dataset -> preprocessed volume   (README.md)
    eval/              cohort + predictions + task -> metrics       (README.md)
    models/nvidia.py   the only place vendored NVIDIA code is imported
    models/report_conditioned_unet.py
                       the pretrained diffusion UNet + report cross-attention adapters,
                       and the strict pretrained-checkpoint loader
    models/adapter.py  what is trainable (asserted, not assumed) + the adapter checkpoint format
    text.py            the replaceable text-encoder seam: RadBERT, a test mock, one registry,
                       plus encode_reports (the one dispatch seam) and rebuild_embedder
    textenc/           the encoder zoo + report formats + fusion            (README.md)
                       conditioning.py = the named configurations (textenc/README.md Part 4)
    textbench/         encoder x format selection benchmark; never imported by the trainer
                       (README.md; results and rationale in docs/TEXT_ENCODERS.md)
    conditioning.py    modality ids, NVIDIA's own modality dropout, report dropout, CFG
    training.py        adapter training; mirrors NV-Generate-CTMR/scripts/diff_model_train.py
    validation.py      step-based validation during training: FID + alignment proxy, DDP-safe
    sampling.py        report-to-volume sampling; mirrors scripts/diff_model_infer.py
    cli/               the entry points (nine generation + four text-encoder + one benchmark)
  scripts/             the contrastive-pretraining pipeline (separate; see its own README)
  slurm/               _common.sh + the numbered job scripts
  tests/
docs/
  R2V.md               this file
  TEXT_ENCODERS.md     report analysis, the encoder zoo, formats, metadata conditioning
  design/archive/      the audits and design records behind these decisions
  nhr_official_docs/   FAU cluster documentation
```

`scripts/data.py` is shared: the R2V Dataset imports its volume preprocessing unchanged, so both
pipelines can never drift apart on how a volume is prepared. Everything else in `scripts/` belongs
to the contrastive model and is untouched by this pipeline.

---

## 10. Storage and hygiene

Cohorts and prediction sets are large, but bundled and compressed: **~14 MB per volume** at the
per-bucket FOVs (2.91× lossless, since a padded volume is ~50% exact zeros). A 10-bucket,
200-per-bucket cohort is ~28 GB, and each prediction set the same again.

Volumes live in **one archive per bucket** (`volumes/<modality>__<plane>.npz`), so a cohort is ~10
files rather than ~2,000. That is not cosmetic: `/hnvme` enforces a **file-count** quota (61k soft),
and one file per volume put three artifact directories over it. An `.npz` is a zip of `.npy`
members and zip members are individually readable, so random access is still
`VolumeReader(root).read(bucket, case_id)` with nothing to unpack first.

- Write them to a workspace (`/hnvme/workspace/...`), never to git or `$HOME`.
- `cohort.json` and every results file are identifier-free and safe to copy or share.
- `index.csv` is the only file containing real (anonymized) `study_uid`/`series_id` values. Keep it
  in the workspace and do not paste identifiers into logs, issues, or papers — quote the
  `cohort_id` instead.

---

## 11. Background

Design records, audits, and the evidence behind these choices are in
[`docs/design/archive/`](design/archive/) — see its
[INDEX.md](design/archive/INDEX.md). You do not need them to use the pipeline; read them when you
want to know *why* a default is what it is.
