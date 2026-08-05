# Report-to-Volume: the complete guide

Everything about generating brain MRI volumes from radiology reports and scoring the result.
This is the only document you need; the per-module READMEs
([data](../contrastive-pretraining/mrrate_r2v/data/README.md),
[eval](../contrastive-pretraining/mrrate_r2v/eval/README.md)) go deeper on their own area.

For the *conditioning* side — which text encoder, which report format, and where modality/spacing
should come from — see **[TEXT_ENCODERS.md](TEXT_ENCODERS.md)** and the
[textenc](../contrastive-pretraining/mrrate_r2v/textenc/README.md) /
[textbench](../contrastive-pretraining/mrrate_r2v/textbench/README.md) READMEs.

**Training a report adapter? Start at [TEXT_ENCODERS.md §9](TEXT_ENCODERS.md#9-the-three-supported-conditioning-configurations).**
It covers the three supported configurations and their exact conditioning shapes, how each is run
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
  │  stage 1   cli.preprocess                                                    │
  │            picks the cases, preprocesses the volumes                         │
  │                                                                              │
  │            ────►  COHORT DIRECTORY  ◄──── the contract                       │
  │                   cohort.json  volumes/*.npy  reports/*.txt  index.csv       │
  └──────────────────────────────────────────────────────────────────────────────┘
         │                                                    │
         │  stage 2                                           │  stage 3
         ▼                                                    ▼
  cli.predict_vae          ────►  PREDICTION DIRECTORY  ────►  cli.evaluate
  cli.predict_generation          predictions.json                --task <task>
  cli.predict_r2v                 volumes/*.npy                   --gt <cohort>
  cli.import_predictions                                           --pred <predictions>
                                                                     │
                                                                     ▼
                                                              RESULTS DIRECTORY
                                                              metrics_per_bucket.csv
                                                              metrics_summary.csv + 5 more
```

Three properties follow from this shape, and they are the reason it exists:

| Property | How it is guaranteed |
|---|---|
| **Evaluation is always the same** | One `run_evaluation()` function. Every task, every model. |
| **GT and predictions are always matched the same way** | By `case_id`, assigned once in stage 1. The evaluator never infers a pairing. |
| **Same FOV, same sample count** | Both live in `cohort.json`. Stages 2 and 3 read them; neither can choose its own. |

And one hard gate: a prediction directory records the `cohort_id` it was produced against.
`cli.evaluate` **refuses to run** if it does not match the `--gt` cohort. You cannot accidentally
compare two models that saw different data.

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
| `reconstruction` | a real volume, to encode and decode | yes, that exact volume | fidelity, perceptual, distribution |
| `report2volume` | a report | yes, the series that report describes | fidelity, perceptual, distribution, report alignment |
| `generation` | a modality label only | **no** | distribution only |

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
| `all` | eligible series | training, where you want maximum data | 34,453 |
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
intensity centroid backwards. It belongs to the contrastive pipeline; for R2V the default is **0**,
measured: shifting by 15 mm lost on 8 of the 10 buckets.

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

---

## 8. Testing

```bash
cd contrastive-pretraining
python -m pytest                          # everything, with coverage
python -m pytest --no-cov -q              # faster
python -m pytest tests/test_cohort_contract.py tests/test_eval_tasks_and_runner.py -v
```

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
                       conditioning.py = the three supported configurations (TEXT_ENCODERS.md §9)
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
