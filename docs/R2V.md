# Report-to-Volume: the complete guide

Everything about generating brain MRI volumes from radiology reports and scoring the result.
This is the only document you need; the per-module READMEs
([data](../contrastive-pretraining/mrrate_r2v/data/README.md),
[eval](../contrastive-pretraining/mrrate_r2v/eval/README.md)) go deeper on their own area.

For the *conditioning* side — which text encoder, which report format, and where modality/spacing
should come from — see **[TEXT_ENCODERS.md](TEXT_ENCODERS.md)** and the
[textenc README](../contrastive-pretraining/mrrate_r2v/textenc/README.md).

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
                                                              metrics.json
```

**Train and test are the same program up to the point where one trains and the other infers.**
That is the whole design. Nothing is frozen to disk between preprocessing and metrics: no cohort
directory, no prediction set, no `.npy` on disk. A case is preprocessed, generated, scored and
released, one at a time.

Three properties follow, and they are the reason it has this shape:

| Property | How it is guaranteed |
|---|---|
| **Test preprocesses exactly like train** | There is one `R2VDatasetConfig` and one `build_dataset`. Not a convention — there is no second code path that *could* differ. |
| **Evaluation is always the same** | One `LiveEvaluator.run`, and one metric definition in `eval/challenge_metrics.py`. Every task, every model — the task only decides how a volume is *produced*. |
| **The same cases, every time** | Case selection uses **no RNG**: ordered by `(study_uid, series_id)` within each (modality, plane) bucket, round-robined across buckets. Prefix-stable, so `--n-per-bucket` is a prefix of the full run. |
| **The same volumes, every time** | Sampler noise is `--seed + <dataset index>` — a function of the case's position in the manifest, not of iteration order. A rerun, a resume or a smaller `--n-per-bucket` all reproduce the same volume for the same case. |

What records comparability: `metrics.json` carries `dataset_config.geometry_fingerprint()` —
every preprocessing setting that affects the tensor — alongside the model identity, the task, the
split and the sample cap. Two runs whose fingerprints agree saw the same cases at the same geometry
under the same preprocessing.

> ⚠️ **There is no longer an automatic refusal on a train/test preprocessing mismatch.** An earlier
> version compared `--posterior-shift-mm` / `--normalizer` / `--geometry-mode` against what the
> adapter recorded at training time and refused to run before any GPU work; that check went with
> the declutter. **Check the fingerprint against the run's `train_summary.json` yourself** — this is
> exactly the class of divergence that voided a whole sweep once (see the box below).

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

### Then, per model — evaluate

There is no cohort or prediction step any more: one command builds the Dataset, generates and
scores.

```bash
WS=/hnvme/workspace/y100dc19-nvidia-mri-brain

# the frozen autoencoder — the ceiling any generator is measured against
python -m mrrate_r2v.cli.evaluate --task reconstruction \
    --manifest .../manifest_shards_native.csv \
    --report-index .../report_index_shards_native.csv \
    --split test --n-per-bucket 200 \
    --vae-checkpoint $WS/models/autoencoder_v1.pt \
    --out $WS/cache/r2v/results/vae_v1

# a trained report adapter
python -m mrrate_r2v.cli.evaluate --task report2volume \
    --manifest .../manifest_shards_native.csv \
    --report-index .../report_index_shards_native.csv \
    --split test --n-per-bucket 200 \
    --checkpoint $WS/runs/r2v_final_D_report2ct_style/adapter_last.pt \
    --out $WS/cache/r2v/results/r2v_D
```

Read `<out>/metrics.json` — `SSIM_mean` is the platform's primary metric. Check `n_scored_files`
against `n_total_files` every time (§5).

> **`--posterior-shift-mm` and `--normalizer` must match what the adapter trained under.** Nothing
> checks this for you any more. `slurm/evaluate.sbatch` defaults to `15` / `percentile`, which is
> what the final runs used.

### On the cluster

```bash
sbatch slurm/evaluate.sbatch reconstruction vae_baseline
sbatch slurm/evaluate.sbatch generation     nvidia_uncond
sbatch slurm/evaluate.sbatch report2volume  r2v_D  $WS/runs/r2v_final_D_report2ct_style/adapter_last.pt
```

Scale with `R2V_N_PER_BUCKET` (`8` ≈ 10 min wiring check, `200` ≈ 4 h, unset = the full 29,027-case
split ≈ 60 h). Paths and the apptainer invocation live in `slurm/_common.sh` — edit them in that
one place. For a whole sweep of arms and guidance scales, use `slurm/final_eval/run_sweep.sh`.

---

## 3. The three tasks

`--task` decides only **how a volume is produced**. Every task is then scored identically, by
`eval/challenge_metrics.py`.

| `--task` | The model was given | Paired ground truth | What the number means |
|---|---|---|---|
| `reconstruction` | a real volume, to encode and decode | yes, that exact volume | the **ceiling**: no generator can beat its own autoencoder |
| `report2volume` | the case's report | yes, the series that report describes | the result you are actually after |
| `generation` | a modality label only, report-blind | none in principle — but see below | the **floor** |

> ⚠️ **`generation` is now scored voxelwise too, and that number needs care.** An unconditional
> generator is told "make a T1w brain" and nothing about any patient, yet it is still compared
> against the real series for that case — so its SSIM/PSNR measures *how similar two unrelated
> brains are*, not generation quality. An earlier design made this structurally impossible
> (`eval/tasks.py` marked the task unpaired); the port of the official metrics removed that
> distinction, since the challenge scores whatever you submit against the paired ground truth
> regardless.
>
> Read `generation` as the **floor** — the score obtainable with no report information at all. A
> `report2volume` run that does not clearly beat it has not learned to use its report. That is a
> genuinely useful baseline; treating it as "the unconditional model's quality" is not.

**Expect `report2volume` fidelity to sit well below `reconstruction`.** A report constrains
pathology and gross anatomy, not voxel positions. A low PSNR there is not necessarily a bug —
compare against other report-to-volume runs on the same split, and against the two bounds above,
never against a reconstruction number alone.

---

## 4. Controlling the experiment

Every knob below belongs to `cli.evaluate` (and, where it affects the tensor, to `cli.train_r2v`
too) and is recorded in `metrics.json` under `dataset_config`. **Changing one changes the numbers,
and nothing stops you comparing across the change** — the automatic refusal is gone (§1), so
compare `dataset_config` between two runs before reading their metrics against each other.

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
(verified: `(240, 240, 176)` fails with "Expected size 14 but got size 15"). The `generation` task
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
Superior-Inferior) — the same order the Dataset returns, `LiveCase` records, and the NVIDIA model
uses. Internally the preprocessing works in `(D, H, W)`; the CLI converts at the boundary via
`geometry.xyz_to_dhw`. You never see `(D, H, W)` from outside the package.

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
not a reason to score a model on a grid it never saw. Both CLIs expose the knob and the training
flag's default is read off the dataclass so the two cannot drift — but **the cross-check that used
to refuse a mismatched evaluation no longer exists**, so the value in `dataset_config` has to be
compared against the run's `train_summary.json` by hand.

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

One file per run: **`<out>/metrics.json`**.

```json
{
  "metrics": {"MSE_mean": ..., "PSNR_mean": ..., "SSIM_mean": ..., "dice": ...,
              "FID_2p5D_XY": ..., "FID_2p5D_XZ": ..., "FID_2p5D_YZ": ..., "FID_2p5D_Avg": ...,
              "n_total_files": 2000, "n_scored_files": 1988,
              "n_missing_outputs": 12, "n_excluded_out_of_scope_modality": 0},
  "per_case": [{"case_id": "...", "bucket": "T1w_AXIAL", "status": "scored",
                "MSE": ..., "PSNR": ..., "SSIM": ...}, ...],
  "task": "report2volume", "split": "test", "n_per_bucket": 200, "elapsed_sec": 14203.6,
  "model": {...}, "dataset_config": {...}
}
```

The old `metrics_per_bucket.csv` / `metrics_summary.csv` / `per_case_metrics.csv` /
`summary.json` / `distribution_metrics.json` / `anatomy_metrics.json` / `excluded_cases.json` /
`run_manifest.json` set is gone, along with `slurm/check_run.py` and `SUCCESS_CRITERIA.md`.
Per-bucket aggregates are no longer precomputed — group `per_case` by `bucket` yourself.

### The metrics

All eight come from `eval/challenge_metrics.py`, a port of the official evaluation container. **This
is the same code `validation.py` calls**, so a training curve and a final score are the same
numbers at different sample sizes.

| Metric | Tells you | Direction | Watch out |
|---|---|---|---|
| `SSIM_mean` | does it look like the same image | higher, max 1.0 | the platform's **primary** metric |
| `dice` | nothing extra — a literal copy of `SSIM_mean` | higher | not a real Dice. It is what the ranking config reads; do not "fix" it |
| `MSE_mean` | mean squared error after percentile normalisation | lower | scale-invariant, so it says nothing about output range |
| `PSNR_mean` | the same error on a log scale | higher | `data_range` is fixed at 1.0 post-normalisation, not a real max |
| `FID_2p5D_{XY,XZ,YZ}` | do the real and produced *populations* match, per plane | lower | squeezenet1_1 features; never trained on medical images |
| `FID_2p5D_Avg` | the mean of the three | lower | needs a few hundred cases before it is stable |
| `n_scored_files` | how many actually entered the means | — | **always check against `n_total_files`** |
| `n_missing_outputs` | generations that failed | — | these are *excluded*, not penalised — a run can look good by failing |

**Two behaviours copied deliberately from the official code, both of which look like bugs:**

1. `dice` is `SSIM_mean`, not Dice.
2. A case whose generation failed is **dropped from the means** rather than scored as worst-case.
   So a model that crashes on its hardest cases scores *better*. `n_missing_outputs` is the only
   thing that reveals it — read it every time.

**What the metric does to your volumes.** `compute_basic_metrics` percentile-normalises (0.5/99.5)
*both* volumes, then, if the shapes differ, resamples the **generated** volume onto the **real**
one's shape with `scipy.ndimage.zoom(order=1)`. Two consequences:

- The metrics are **invariant to the decoder's output range**, which is why
  `cli/evaluate.py` passes `postprocess=False` — not to fix the scale, but to keep the volume in the
  ground truth's space.
- The **ground truth is authoritative in both shape and scale.** Whatever grid the real volume is
  on is the grid the comparison happens on.

That last point matters for the FID too, in a different way: `fid_2p5d` extracts slice features
from each volume independently (every slice resized to 224²), so it never needs the two to share a
shape — but the slice *count* follows each volume's own geometry.

### Comparing two runs

Compare `dataset_config` and `model` in the two `metrics.json` files. If the geometry fingerprint,
split and `n_per_bucket` agree, the runs saw the same cases at the same geometry under the same
preprocessing.

**Nothing enforces this any more** — the automatic train/test preprocessing check is gone (see §1).
Two runs at different `--posterior-shift-mm` will happily produce comparable-looking numbers that
are not comparable.

### Speed

Measured 6.7 s/case at 30 inference steps on one H200, plus ~1 s/case of scoring; I/O is a few
percent. Scale with **ranks**, not workers: `LiveEvaluator` shards cases `index % world_size` and
each rank accumulates its own `ChallengeAccumulator`, pooled once at the end (feature pooling for
FID, not averaged distances — so a sharded run equals a single-process one).

---

## 6. Scoring volumes that already exist

**Short answer: you cannot re-score a finished run without regenerating it, and the earlier sweeps
did not keep their volumes.**

The `cli.import_predictions` path documented here previously is gone with the cohort/prediction-set
layer. Evaluation now builds the Dataset, generates and scores in one pass, so there is no
"prediction set" for an external file tree to be imported as.

**What survives from a sweep run** is `metrics.json` (or, for pre-2026-08-18 runs, the old CSV set)
and a handful of example NIfTIs under `figures/`. `SAVE_VOLUMES=1` in
`slurm/final_eval/run_sweep.sh` was only ever set for the `headline` stage — the cheap stages
"exist to be thrown away", at ~19 GB per run — and no headline stage was run. So the generated
volumes for every cfg/format/epoch sweep point are **not on disk**.

To score an old configuration with the current metrics, **re-run it**:

```bash
CHECKPOINT_KIND=last slurm/final_eval/run_D_report2ct_style.sh    # or any arm / epoch / cfg
```

This is not merely a formality — it is the only correct way, because the old and new numbers are
not comparable anyway (different metric definitions, §5). Regeneration is deterministic
(`--seed + <dataset index>`), so the volumes are the ones the old run would have produced, given
the same checkpoint and geometry.

### `--gt-space native` — scoring against the released ground truth

```bash
python -m mrrate_r2v.cli.evaluate --task report2volume --gt-space native ...
# or, through the sweep:  GT_SPACE=native slurm/final_eval/run_sweep.sh cfg
```

| `--gt-space` | Ground truth is | Use |
|---|---|---|
| `model` (default) | the preprocessed volume on the model's bucket grid | comparing models to each other |
| `native` | the released volume, RAS-reoriented and otherwise untouched — no resample, no normalize, no crop/pad | estimating what the challenge server will score |

**Generation is unaffected either way.** The model always samples on its own bucket grid
(`case.shape`); the metric resamples that volume onto the ground truth (§5). So `native` changes
what you are compared *against*, never what the model produces.

The two are **not comparable to each other**, so `gt_space` is recorded in `dataset_config` in every
`metrics.json`.

**Before doing that, know what the released volumes look like.** Measured over a random 200-series
sample of the test split (T1w/T2w/FLAIR/SWI, seed 0, 2026-08-18):

| Property | Measured |
|---|---|
| Stored orientation | **RAS 52%, LAS 32%, PSR 9%, LSP 7%** — 48% non-RAS |
| Distinct shapes | **141 in 200 series** |
| Range | ~(176, 512, 512) to ~(462, 480, 37) |
| Slice thickness | up to **5.6 mm** |

This is why `native` **still reorients to RAS**, and why that is not a compromise. `zoom` rescales;
it never permutes or flips. Measured on real test-split series, reorientation is not cosmetic:

```
stored             stored shape        native (X, Y, Z)
RAS   flair-raw-sag  (176, 512, 512) -> (176, 512, 512)   unchanged
LAS   t2w-raw-axi    (396, 416,  45) -> (396, 416,  45)   flipped only
LSP   t2w-raw-cor    (462, 480,  37) -> (462,  37, 480)   axes PERMUTED
PSR   t2w-raw-sag    (464, 464,  32) -> ( 32, 464, 464)   axes PERMUTED
```

Pairing a generated RAS volume against an un-reoriented LSP one compares different anatomical axes
and scores like noise, on roughly half the split. Reorientation is what makes the two arrays refer
to the same anatomy.

Resampling, normalisation and crop/pad genuinely are dropped: they exist to put a volume on the
model's training grid, and the metric only needs a common grid, which it produces itself.

### Keeping the generated volumes

`--save-volumes` (or `SAVE_VOLUMES=1`, or `SAVE_VOLUMES_FOR="D:3.0 E:3.0"` for specific sweep runs)
writes `<out>/volumes/<bucket>.f16.raw` plus an `index.json` giving each stack's shape, dtype and
case order.

**One file per bucket, not one per case**: `/hnvme`'s binding limit is a file-count quota (61k soft,
81k hard) and it is already at it, so 2,000 loose files per run is what actually breaks — not space.
Read one back with:

```python
index = json.loads((out / "volumes" / "index.json").read_text())["T1w__AXIAL"]
stack = np.fromfile(out / "volumes" / "T1w__AXIAL.f16.raw", dtype=np.float16)
stack = stack.reshape([len(index["case_ids"])] + index["shape"])
volume = stack[index["case_ids"].index(case_id)]
```

---

## 7. The report-to-volume model

A frozen NV-Generate-MR-Brain denoiser plus a trained report adapter. Two extra stages:

```
cli.train_r2v     →  adapter_*.pt        (frozen base + text encoder; only the adapter learns)
cli.generate_r2v  →  one .nii.gz         (one report, to look at)
cli.evaluate      →  metrics.json        (generates and scores the split in one pass)
```

`slurm/train_r2v.sbatch`, `slurm/generate_r2v.sbatch` and `slurm/evaluate.sbatch` are the runnable
versions.
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

### Training more epochs onto a finished run

`slurm/final/run_D_continue.sh` is the worked example: two more epochs of configuration D, resumed
from its `adapter_last.pt`, writing into a **new** run directory (`..._cont`) so `--keep-last-n`
cannot prune the baseline's checkpoints or overwrite its `train_summary.json`.

**`--resume-lr-schedule restart` is mandatory when the source run finished, and the reason is a
silent one.** `PolynomialLR` reaches exactly 0 at `total_iters` and stays there. A completed run's
checkpoint therefore stores a dead schedule *and* an optimizer whose learning rate is already 0 --
`r2v_final_D_report2ct_style/adapter_last.pt` holds
`{total_iters: 4493, last_epoch: 4493, _last_lr: [0.0]}`. Plain `--resume` restores both, so the
continuation trains at LR 0 for its entire walltime: the loss curve looks like a converged model and
the checkpoint it writes is bit-for-bit its input. `fit` now refuses an exhausted schedule outright
rather than running it (`test_extending_a_finished_run_refuses_its_dead_lr_schedule`), and `restart`
discards it and builds a fresh one from `--lr` over the new run's horizon. Adam's moments are kept
either way -- they describe the loss surface, not the schedule.

Two settings that a resume makes different from a fresh run, both deliberate:

* **`--epochs` and `--max-steps` count what *this* job does**, not the model's history. The epoch
  *numbering* is still absolute, so a continuation logs and checkpoints epochs 3 and 4
  (`adapter_epoch003.pt`, `adapter_epoch004.pt`) and never replays epoch 0's shuffle seed.
* **Save intervals are anchored at the resumed step.** Anchored at 0, `4493 // 600 > 0 // 600` fires
  a validation, a full validation and a checkpoint on the *first* optimizer step after every resume.

Pick the restart LR against what the adapter is now, not against the sweep. The sweep's 1e-3 is the
right peak for a run starting from a zero-init projection; a converged adapter re-entered at full
peak spends its first epoch undoing its last. D's continuation uses **3e-4** (power 2 over 4,493
steps → mean ~1e-4).

Use `--save-every-epochs 1`. An epoch boundary is not a multiple of `--save-every-steps` in general
(2 epochs of D is 4,493 optimizer steps, so the boundary sits at 2246.5), and `adapter_epoch*.pt` is
outside `--keep-last-n`'s glob, so retention cannot delete the checkpoint the next decision rests on.

**Budget from the measured number, not the estimate in `_final_common.sh`'s header.** The four
completed arms took **22.17 h** for 2 epochs, of which only 0.47 h was validation → **10.85 h of
training per epoch**. Three epochs does not fit `h200`'s hard 24 h cap at all; two fit with ~1.8 h of
margin. Past that there is only `preempt` (48 h), which is survivable now that resume works but
still loses everything since the last checkpoint.

Smoke it first — `SMOKE=1 NODES=1 slurm/final/run_D_continue.sh` carries the resume flags (they sit
outside the SMOKE branch on purpose), so it exercises the exact code path the long job takes.

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

## 7b. Evaluating a trained adapter (the five-arm run)

Training produces `adapter_*.pt`; this is how those become numbers.

### Running it

Everything goes through `slurm/final_eval/`, which knows each arm's run directory and checkpoint:

```bash
cd contrastive-pretraining

SMOKE=1 slurm/final_eval/run_D_report2ct_style.sh    # 8/bucket, ~20 min, wiring only
slurm/final_eval/run_D_report2ct_style.sh            # the real thing

slurm/final_eval/run_sweep.sh cfg        # best guidance scale per arm      (25 jobs)
slurm/final_eval/run_sweep.sh format     # report-format robustness         (12 jobs)
slurm/final_eval/run_sweep.sh headline   # the arm-vs-arm result             (5 jobs)
slurm/final_eval/run_sweep.sh epochs     # D at 3 and 4 epochs               (4 jobs)
```

**Smoke each arm once before committing long jobs.** A SMOKE run is a wiring check and its metrics
mean nothing at 8 cases per bucket — the run says so itself.

**Checkpoint selection**: `adapter_last.pt` where it exists, `adapter_step0004200.pt` for C, whose
training job was killed by a host-RAM OOM in its final validation. C is therefore compared at 4,200
optimizer steps against 4,493 for the others — quote that caveat wherever the five-way table
appears, or set `CHECKPOINT_KIND=step4200` for the step-matched comparison. `CHECKPOINT_KIND` also
accepts `epoch003`-style names for the per-epoch checkpoints a continuation run writes; note that
`model.training.epoch` in the results is **0-indexed** while the filename is 1-based.

### `evaluate` vs `generate_r2v`

| | writes | scoreable |
|---|---|---|
| `evaluate.sbatch` (`cli.evaluate`) | `metrics.json` | **yes** — it generates and scores in one pass |
| `generate_r2v.sbatch` (`cli.generate_r2v`) | one `.nii.gz` in NVIDIA's int16 `[0, 1000]` range | no — use it to *look* at a sample |

### W&B

`slurm/final_eval/` logs to W&B by default (`R2V_WANDB=online`, project `mr-rate-r2v-eval`, one
group per stage, run named after the arm). A hand-run `evaluate.sbatch` stays offline unless you set
`R2V_WANDB`.

**Panels embed patient report text**, so they are gated behind `--wandb-log-reports`
(`R2V_WANDB_REPORTS=1`, the default in `final_eval/`) exactly as in `cli.train_r2v`. Keep that
project private, or set `R2V_WANDB_REPORTS=0` — the metrics are logged either way and only the
panels are withheld. `--wandb-panels` is per bucket and small by design; 2,000 interactive panels is
~1 GB of base64 and an unusable workspace (measured ~1.2 MB each).

> The blinded report-classifier consistency metric, the `check_run.py` acceptance gate and its
> `ER1`–`ER4` criteria were removed with the old metric suite. Scoring is now exactly the official
> challenge metric set (§5) and nothing else.

### What a pre-2026-08-10 evaluation is worth

Nothing — and there was no such evaluation, because two of these five defects made the predict path
refuse to start for three of the four arms. The three CLIs named below (`cli.predict_r2v` and
friends) no longer exist, but the failure modes are properties of *any* inference path and the
lesson is why `tests/test_r2v_inference_contracts.py` exists:

1. **The predict path wrote volumes in NVIDIA's int16 `[0, 1000]` range** against a `[0, 1]` ground
   truth. Every paired metric consumed a 1000×-offset pair and returned a plausible number.
   `cli/evaluate.py` now generates with `postprocess=False`, and the current metric normalises both
   volumes anyway (§5) — but the flag still matters, and a test asserts the call site keeps it.
2. **Configurations B and C could not be rebuilt at inference.** `rebuild_embedder` had no branch
   for `kind="tokens"`, so those adapters fell through to a path defaulting to RadBERT — and since
   CXR-BERT and RadBERT are both 768-wide, every downstream shape check passed. A CXR-BERT arm
   silently loaded RadBERT.
3. **Configuration D could not run at all**, since nothing passed it per-section text.
4. **`assert_report_format_matches` compared nothing.** It read `cohort.spec.geometry_fingerprint`,
   an attribute a parsed dict never has, so the built format read as `None` unconditionally — which
   passes silently when the adapter records no format and refuses *everything* when it records one.
   Its unit test survived because the stub exposed the same wrong attribute; mocking the wrong
   interface is what hid it.
5. **The format check was applied to configuration D**, which never reads the joined string and
   records `report_format=None` by construction. The check now takes the embedder and exempts a
   sectioned configuration.

Three of these produced complete, valid-looking output; the other two were only reachable by
running the thing.

### And what a pre-2026-08-18 evaluation is worth

The numbers are real but **not comparable to anything produced now**: they came from the removed
metric suite (`mae_fg`, `ncc_whole`, `fvd`, `medicalnet_fid`, anatomy, report-consistency), not the
official challenge metrics. The sweep runs also did not keep their volumes, so an old configuration
can only be re-scored by regenerating it (§6).

---

## 8. Testing

```bash
cd contrastive-pretraining
python -m pytest                          # everything, with coverage
python -m pytest --no-cov -q              # faster (~3.5 min, 812 tests)
python -m pytest --no-cov tests/test_eval_challenge_metrics.py tests/test_eval_live.py -v
```

No GPU, no real data, no checkpoints. `testpaths` covers `tests/` **and** `../submission`, so the
container's own tests run too — they previously did not, because `testpaths` named only `tests`.

**The R2V tests are not in version control, and CI does not run them.** `.gitignore` ignores
`/contrastive-pretraining/tests`, so only the 8 original *contrastive* test files are tracked. This
is deliberate and left as is, but it has two consequences: the suite will not survive a fresh clone,
and a change that breaks a load-bearing invariant will pass CI. Run `python -m pytest` locally
before trusting any R2V change.

Files whose subjects were deleted by the 2026-08-18 declutter are retired in place as
`tests/*.py.obsolete` — not collected, kept as a record of what went.

Three files are load-bearing invariants rather than ordinary unit tests. **Do not weaken them to
make a change pass:**

- `test_r2v_validation.py::test_every_rank_enters_the_panel_gather_not_just_rank_zero` — that a
  collective is entered by *every* rank. `gather_objects` is a no-op at world size 1, so this
  asserts the **call**, not the numbers; the bug it guards crashed every multi-GPU run with an
  `OutOfMemoryError: Tried to allocate more than 1EB memory` that named nothing useful.
- `test_eval_live.py` — that the ground truth always comes from the dataset (a generator cannot
  nominate an easier target), that an out-of-scope modality never reaches the generator, and that
  one failed case is recorded as missing rather than losing the run.
- `test_eval_challenge_metrics.py` — that pooling shards equals a single process (means combine
  trivially; a Fréchet distance does not, so features must be pooled before the distance), and that
  the official code's two quirks survive: `dice == SSIM_mean`, and a missing output is excluded from
  the means rather than penalised.

---

## 9. Where things are

```
contrastive-pretraining/
  mrrate_r2v/
    data/              manifest -> Dataset -> preprocessed volume   (README.md)
    eval/              task -> metrics                              (README.md)
    models/nvidia.py   the only place NVIDIA-authored model-loading code is used
    models/report_conditioned_unet.py
                       the pretrained diffusion UNet + report cross-attention adapters,
                       and the strict pretrained-checkpoint loader
    models/adapter.py  what is trainable (asserted, not assumed) + the adapter checkpoint format
    text.py            the replaceable text-encoder seam: RadBERT, a test mock, one registry,
                       plus encode_reports (the one dispatch seam) and rebuild_embedder
    textenc/           the encoder zoo + report formats + fusion            (README.md)
                       conditioning.py = the named configurations (textenc/README.md Part 4)
    conditioning.py    modality ids, NVIDIA's own modality dropout, report dropout, CFG
    training.py        adapter training; mirrors NV-Generate-CTMR/scripts/diff_model_train.py
    validation.py      step-based validation during training: FID + alignment proxy, DDP-safe
    sampling.py        report-to-volume sampling; mirrors scripts/diff_model_infer.py
    cli/               the entry points: build_manifest, train_r2v, evaluate, generate_r2v,
                       download_text_encoders
  scripts/             the contrastive-pretraining pipeline (separate; see its own README)
  slurm/               _common.sh + the numbered job scripts
  tests/
docs/
  R2V.md               this file
  TEXT_ENCODERS.md     report analysis, the encoder zoo, formats, metadata conditioning
  design/archive/      the audits and design records behind these decisions
  nhr_official_docs/   FAU cluster documentation
```

`mrrate_r2v` has no import dependency on `contrastive-pretraining/scripts/` (or on `mr_rate/` /
`vision_encoder/`). Volume preprocessing (`mrrate_r2v/data/_preprocess_ops.py`) was originally
forked verbatim from the contrastive pipeline's `scripts/data.py`, but the two are now
maintained independently — this is deliberate, so `mrrate_r2v` can be extracted into its own
repository without carrying any of `contrastive-pretraining/scripts/`, `mr_rate/`, or
`vision_encoder/` along with it.

---

## 10. Storage and hygiene

**Evaluation no longer writes volumes at all.** A case is preprocessed, generated, scored and
released one at a time, so a finished run is a single `metrics.json` of a few hundred KB. The old
cohort and prediction directories — ~14 MB per volume, ~28 GB per artifact set — are gone, and with
them ~365 GB per experiment set.

That also means **an evaluation cannot be re-scored later**: there is nothing kept to re-score (§6).
`slurm/final_eval/run_sweep.sh` sets `SAVE_VOLUMES=1` only for the `headline` stage, on the
principle that the cheap stages exist to be thrown away.

`/hnvme` enforces a **file-count** quota (61k soft), not just a space quota — worth remembering
before writing one file per volume anywhere.

- Write results to a workspace (`/hnvme/workspace/...`), never to git or `$HOME`.
- `metrics.json` is identifier-free and safe to copy or share: cases appear as `case_id`, a hash of
  `(study_uid, series_id)`.
- `study_uid`/`series_id` live only in the manifest CSV. Keep it in the workspace and do not paste
  identifiers into logs, issues, or papers — quote the `case_id` instead.

---

## 11. Background

Design records, audits, and the evidence behind these choices are in
[`docs/design/archive/`](design/archive/) — see its
[INDEX.md](design/archive/INDEX.md). You do not need them to use the pipeline; read them when you
want to know *why* a default is what it is.
