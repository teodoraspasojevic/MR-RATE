# Report-to-Volume: the complete guide

Everything about generating brain MRI volumes from radiology reports and scoring the result.
This is the only document you need; the per-module READMEs
([data](../contrastive-pretraining/mrrate_r2v/data/README.md),
[eval](../contrastive-pretraining/mrrate_r2v/eval/README.md)) go deeper on their own area.

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
                                                              summary.json + 4 more
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
    --split test --sequences T1w T2w FLAIR SWI --n-per-sequence 200 \
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

Read `.../results/vae_v1/summary.json`.

### On the cluster

```bash
sbatch slurm/01_smoke_test.sbatch                        # whole pipeline, 2 cases. Do this first.
sbatch slurm/02_preprocess.sbatch test_v1 200            # cohort: 200 per sequence
sbatch slurm/03_predict_vae.sbatch test_v1 vae_v1
sbatch slurm/04_predict_generation.sbatch test_v1 100 gen_v1
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
--n-per-sequence 200      # 200 cases per sequence
                          # (omit entirely for every eligible case)
--sequences T1w T2w FLAIR SWI
--split test              # train | val | test
--seed 42
```

Selection is deterministic given `--seed`, and asking for one sequence picks exactly the same cases
as asking for four — sequence order can never shift a draw. See `cohort.select_cohort`.

### Which series of a study

```bash
--series-selection one_per_study_per_sequence   # default
```

| Value | One sample per… | Use it for | Cases on the real test split |
|---|---|---|---|
| `one_per_study_per_sequence` | **(study, sequence)** | **evaluation** — one independent observation per sequence | 17,583 (T1w 4863 · T2w 4795 · FLAIR 4550 · SWI 3375) |
| `all` | eligible series | training, where you want maximum data | 34,453 |
| `one_per_study_deterministic` | study, across all sequences | a single-sequence cohort only | 4,893, but **4,861 of them T1w** |
| `one_per_study_random` | study, redrawn each epoch | training | 4,893/epoch, same modality skew |

Two traps this table exists to prevent:

- **`all` pseudo-replicates.** Near-duplicate series from one session are not independent
  observations — measured mean 1.96, max 13 per (study, sequence). Plain means overweight
  multi-series studies and plain std/CIs come out falsely narrow, because the aggregation does not
  model clustering. It also biases FID by shrinking the real distribution's apparent spread.
- **`one_per_study_deterministic` collapses to T1w.** It picks one series per *study*, preferring the
  center-modality series, which on MR-RATE is the T1w one. A 4-sequence request therefore yields
  ~99% T1w and leaves T2w/FLAIR/SWI essentially unevaluated — silently.

### What `--n-per-sequence` counts

A **case** = one (study, sequence) pair = exactly one series. So `cases == series` always, while
distinct *studies* is smaller because one patient contributes one case per sequence they have:

| `--n-per-sequence` | cases (= series) | distinct studies |
|---|---|---|
| 200 | 800 | 746 |
| 500 | 2,000 | 1,726 |
| 1000 | 4,000 | 2,940 |
| omitted | 17,589 | 4,893 |

At 500/sequence, ~14% of cases share a patient with another case, so quote the **per-sequence**
numbers as primary — the `overall` row is mildly clustered.

### Field of view

```bash
--geometry-mode fixed                            # default
--fixed-shape 256 256 256 --fixed-spacing-mm 1 1 1
```

| Mode | What you get | Use when |
|---|---|---|
| `fixed` | every volume on one grid | **comparing models.** Batching works, and generated volumes share the grid. |
| `per_modality_plane` | a per-(modality, plane) grid sized from NVIDIA's published median FOVs | training, or studying one model at each anatomy's natural FOV |

With `fixed` and no explicit shape, the default is read from the NVIDIA model config, so a cohort
and that generator's output land on the same grid without a hardcoded constant that could drift.

**`--fixed-shape` and `--fixed-spacing-mm` are `(X, Y, Z)`** = (Right-Left, Anterior-Posterior,
Superior-Inferior) — the same order the Dataset returns, the cohort records, and the NVIDIA model
uses. Internally the preprocessing works in `(D, H, W)`; `cli.preprocess` converts at the boundary
via `geometry.xyz_to_dhw`. You never see `(D, H, W)` from outside the package: every file in a
cohort directory is `(X, Y, Z)`.

The cost of `fixed`: a sagittal scan is naturally taller and narrower than an axial one, so forcing
one cube pads some volumes with extra background. That is a real trade-off, taken deliberately —
without it, two experiments' fidelity numbers do not live in the same space.

With `per_modality_plane`, shapes differ between buckets. Use `GeometryBucketBatchSampler` for
batch_size > 1, and do not compare the numbers with a fixed-mode run.

### Intensities

```bash
--normalizer percentile          # percentile (default) | zscore | minmax
--posterior-shift-mm 15.0
```

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

Five files, same five for every task:

| File | What it is |
|---|---|
| `summary.json` | **read this first.** Per-sequence and overall means, plus which metric groups ran and why. |
| `per_case_metrics.csv` | one row per scored case |
| `distribution_metrics.json` | FID and diversity metrics, when computed |
| `excluded_cases.json` | every case that was *not* scored, with a specific reason |
| `run_manifest.json` | exactly what ran: `cohort_id`, task, checkpoint hashes, versions |
| `figures/` | example slice montages (ground truth / prediction / difference), worth a look before trusting any number |

Always check `n_scored` against `n_cohort_cases` in `summary.json`. If they differ,
`excluded_cases.json` says why for every case. Nothing is ever silently dropped.

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
| Report-image similarity | does the volume match the report | higher | **unavailable** | no validated MRI image-text model exists in this project; recorded as unavailable with a reason, never faked |

### Speed

Evaluation is CPU-compute-bound: reading both 67 MB volumes takes 39 ms against ~6.7 s of metric
compute, so I/O is 0.5% of the work. **Staging a cohort to node-local `$TMPDIR` is therefore not
worth doing** — Lustre already delivers 3.4 GB/s here, well past what the metrics can consume.

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

## 7. Implementing a report-to-volume model

Two functions in
[`cli/predict_r2v.py`](../contrastive-pretraining/mrrate_r2v/cli/predict_r2v.py):

```python
def load_r2v_model(checkpoint, device):      # return an object with .generate(...)
def generate_one(model, report_text, case, seed) -> np.ndarray   # [X, Y, Z]
```

Everything else already works. Honor two things:

- **Output on the cohort's grid** (`case.shape`, `case.spacing_mm`). A different grid is allowed but
  those cases get excluded rather than resized.
- **Seed from `stable_seed(args.seed, case.case_id)`** so a rerun reproduces the same volumes.

For training, use the Dataset directly — see the
[data README](../contrastive-pretraining/mrrate_r2v/data/README.md).

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
    cli/               the seven entry points
  scripts/             the contrastive-pretraining pipeline (separate; see its own README)
  slurm/               _common.sh + five numbered job scripts
  tests/
docs/
  R2V.md               this file
  design/archive/      the audits and design records behind these decisions
  nhr_official_docs/   FAU cluster documentation
```

`scripts/data.py` is shared: the R2V Dataset imports its volume preprocessing unchanged, so both
pipelines can never drift apart on how a volume is prepared. Everything else in `scripts/` belongs
to the contrastive model and is untouched by this pipeline.

---

## 10. Storage and hygiene

Cohorts and prediction sets are large: **~67 MB per volume at 256³ float32**. A 4-sequence,
200-per-sequence cohort is ~54 GB, and each prediction set the same again.

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
