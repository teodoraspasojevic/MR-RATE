# `mrrate_r2v.cli` — the eight entry points

Everything you run is here. Each script does one stage and writes one directory; the next stage
reads that directory and nothing else.

Full pipeline context: [`docs/R2V.md`](../../../docs/R2V.md). This file is the command reference —
what each script does, every flag it takes, and what it leaves on disk.

All commands are run from `contrastive-pretraining/`:

```bash
cd contrastive-pretraining
python -m mrrate_r2v.cli.<script> ...
```

---

## The order they run in

```
   ┌─ stage 0 ─────────────────────────────────────────────── once per storage location
   │  build_manifest      →  manifest.csv + report_index.csv       (an index; no pixels)
   │
   ├─ stage 1 ─────────────────────────────────────────────── once per experiment set
   │  preprocess          →  COHORT dir  (cohort.json, cohort_id)  (frozen ground truth)
   │
   ├─ training ───────────────────────────────────────────────  report-to-volume only
   │  train_r2v           →  adapter_last.pt                       (reads the manifest directly,
   │                                                                NOT the cohort)
   ├─ stage 2 ─────────────────────────────────────────────── once per model
   │  predict_vae         →  PREDICTION dir     VAE reconstruction
   │  predict_generation  →  PREDICTION dir     NVIDIA unconditional generation
   │  predict_r2v         →  PREDICTION dir     report-conditioned generation  ← ours
   │  import_predictions  →  PREDICTION dir     someone else's .nii.gz files
   │  generate_r2v        →  a .nii.gz          free-form, one report, no cohort needed
   │
   └─ stage 3 ─────────────────────────────────────────────── per prediction set
      evaluate --task ... →  RESULTS dir  (metrics_per_bucket.csv + summary + figures)
```

**Two things hold this together.** A cohort directory freezes the case list, FOV, sample count,
normalizer and seed into a `cohort_id` hash. Every prediction set records the `cohort_id` it was
produced against, and `evaluate` refuses to score a mismatch. There is no `--force` and no
"close enough" — that refusal is the reason experiments are comparable.

**Which scripts need which stack:** `build_manifest --source shards_parquet` needs pyarrow but not
torch; `--source extracted_dir` needs torch but not pyarrow; `evaluate` needs neither the data stack
nor a model (it reads `.npy` only); `train_r2v` / `generate_r2v` / `predict_r2v` additionally need
`transformers` for the text encoder — on the cluster that means the `nvidia+redbert.sif` image, not
the base one.

---

## Stage 0 — `build_manifest`

One CSV row per eligible (study, series) pair: what exists and where. Independent of split,
geometry, report source and sampling, so you build it once and every cohort afterwards reads it.

```bash
python -m mrrate_r2v.cli.build_manifest --source shards_parquet \
    --shards-root <data_ws>/MR-Rate-raw \
    --out-csv <data_ws>/r2v_manifest/manifest_shards_native.csv \
    --out-report-index-csv <data_ws>/r2v_manifest/report_index_shards_native.csv \
    --verify-sample 20
```

| flag | default | meaning |
|---|---|---|
| `--source` | `shards_parquet` | `shards_parquet` \| `data_path_archive` \| `extracted_dir` |
| `--out-csv` | — | output manifest (required unless `--dry-run`) |
| **shards_parquet** | | |
| `--shards-root` | — | directory of `shard-*.tar` + `series.parquet` |
| `--out-report-index-csv` | — | also write the `(study_uid, archive_path)` index `ShardReportStore` needs. **Pass it** — training needs it. |
| **data_path_archive** | | |
| `--data-root` | — | directory of un-extracted `batchNN.tar` |
| `--batch-tar-pattern` | `{batch_id}.tar` | filename template |
| **extracted_dir** | | |
| `--data-folder` | — | an already-extracted MR-RATE tree |
| `--space` | `native_space` | `native_space` \| `coreg_space` \| `atlas_space` |
| **shared** | | |
| `--metadata-csv` | — | CSV or `.tar.gz` of per-batch CSVs |
| `--splits-csv` | — | the splits CSV |
| `--splits` | `train val test` | which splits to include |
| `--excluded-modalities` | MRA + derived/localizer set | pass with no values to exclude nothing |
| `--include-derived` / `--include-localizer` | off | defensive re-checks; the public release already filtered these |
| `--verify-sample` | `20` | resolve N random archive rows *for real* afterwards. **Always leave this on** — it is what catches a changed filename convention. |
| `--dry-run` | off | report what would be built, write nothing |

The two archive sources open no archives during the build — they construct locators from each root's
own index, so a full build takes seconds.

---

## Stage 1 — `preprocess` (build a cohort)

Selects cases, preprocesses their volumes, writes them with a `cohort.json` contract. **Run once per
experiment set.** FOV, case list, sample count and normalization are decided here and nowhere else.

```bash
python -m mrrate_r2v.cli.preprocess \
    --manifest-csv <manifest.csv> --report-index-csv <report_index.csv> \
    --split test --sequences T1w T2w FLAIR SWI \
    --n-per-bucket 200 --out <ws>/cohorts/test_v1
```

| flag | default | meaning |
|---|---|---|
| `--manifest-csv` | *required* | from `build_manifest` |
| `--report-index-csv` | — | shard report index (**preferred** report source) |
| `--report-csv` / `--report-jsonl` | — | alternative report sources |
| `--split` | `test` | `train` \| `val` \| `test` |
| `--sequences` | T1w T2w FLAIR SWI | which modalities to include |
| `--n-per-bucket` | `200` | cases per (modality, plane) bucket; `0` = every eligible case. 10 buckets exist in the real test split, so 200 → 2000 cases. Per **bucket**, not per sequence, so every bucket has equal statistical power (the real plane mix is ~81% axial). |
| `--series-selection` | `one_per_study_per_bucket` | which series of a study may be picked. **Leave it alone for cohorts** — see below. |
| `--seed` | `42` | deterministic selection; part of the `cohort_id` |
| `--geometry-mode` | `per_modality_plane` | each bucket gets NVIDIA's published FOV exactly, at a shape the UNet accepts. `fixed` forces one grid — only for a deliberate single-geometry study. |
| `--fixed-shape` / `--fixed-spacing-mm` | — | `X Y Z`; `--geometry-mode fixed` only |
| `--normalizer` | `percentile` | `percentile` (NVIDIA's own MRI transform) \| `zscore` \| `minmax` |
| `--posterior-shift-mm` | `0.0` | defacing compensation. Measured: 8 of 10 buckets align better at 0 than at 15 mm. The 15 mm value belongs to the contrastive pipeline's oversized fixed FOV, not to this model. |
| `--env-config` / `--model-config` / `--network-config` | NVIDIA's | override the vendored configs |
| `--out` | *required* | cohort directory to create |
| `--overwrite` | off | replace a non-empty output directory |
| `--archive-access-mode` | `stream` | `stream` (no disk write) \| `node_local_cache` (`$TMPDIR`) |
| `--report-sections` | `findings impression` | which sections become the conditioning text |
| `--report-format` | — | a **single** named format from `textenc.formats`. Must be one the adapter was trained on, or `predict_r2v`/`generate_r2v` refuse the cohort — a cohort freezes composed text, so the format cannot be changed afterwards. |
| `--dry-run` | off | select the cohort, print what would be written, stop |

**Writes:** `cohort.json` (the contract + `cohort_id`, no identifiers — safe to share),
`index.csv` (the *only* file with real `study_uid`/`series_id` — workspace only),
`volumes/<modality>__<plane>.npz` (one archive per bucket), `reports.json`.

---

### What is a "bucket"?

**A bucket is one (modality, plane) pair** — `T1w__SAGITTAL`, `T2w__AXIAL`, and so on. One sequence
in one view. It is the unit of everything downstream: one voxel grid, one `.npz` archive on disk, one
sampling quota, one row in `metrics_per_bucket.csv`, one FID. The real test split has 10 of them.

### What decides which cases end up in the cohort

Two flags, applied in this order:

1. **`--series-selection`** — which series of a study are *allowed* to be candidates.
2. **`--n-per-bucket`** — how many of those candidates are then drawn per bucket (default 200).

The second cannot rescue the first: if selection left only 6 T2w SAGITTAL candidates, you get 6
cases, silently.

### The five `--series-selection` modes

A patient's one scanner visit is a **study**. It produces several **series** — different sequences
(T1w, T2w, FLAIR, SWI: each makes different tissue bright) in different planes (axial, sagittal,
coronal), plus occasional repeats. Measured on the real test split: **7 series per study on average**
(max 21), with 63% of studies covering all four sequences and 70% covering all three planes. All of
them are the same brain from the same visit.

The **report is written once per study, not per series.** So every series this flag keeps gets paired
with *the same* report text, and the choice is really about **how often each report gets reused.**

One detail behind two of the rows: each study has a **center-modality series** (the reference the
others were aligned to), which on MR-RATE is essentially always the axial T1w. Any mode that has to
pick one series out of a mixed group prefers it.

| mode | what gets paired with the report | verdict |
|---|---|---|
| `all` | **Every series.** A study has 7 on average (max 21), so its one report is reused ~7 times — including 1.19 near-identical scans per bucket. | **Train, not eval.** More data is good for training; for evaluation the reuse means near-identical scans count as separate observations, so multi-series studies dominate and error bars come out too narrow. |
| **`one_per_study_per_bucket`** *(default)* | **One series per bucket** — and a bucket *is* a sequence-and-view pair. A study with axial T1w, sagittal T1w and axial FLAIR puts one case into each of those 3 buckets, so its report appears 3 times in the cohort but never twice inside one bucket. | **Use for cohorts.** Metrics are computed per bucket, and inside a bucket every case is a different patient. Reuse *across* buckets doesn't matter, since those are scored separately. |
| `one_per_study_per_sequence` | **One series per sequence**, and the pick is the axial one. The report is reused once per sequence, but always on the same view. | Only for a deliberately plane-agnostic cohort. **Sagittal and coronal starve**: a request for 200 T2w SAGITTAL cases returned 6. |
| `one_per_study_deterministic` | **One series per study** — each report used exactly once, on the center-modality series. | Maximum independence, but that series is almost always axial T1w: 4861 T1w vs 7 T2w and 0 SWI on a 4-sequence request. **Only for a single-sequence cohort.** |
| `one_per_study_random` | **One series per study, redrawn each epoch** — the report is used once per epoch, paired with a different series over time. | **Training only.** A cohort built this way could not be reproduced from its own contract. |

`cli.train_r2v` hardcodes `all`, so there is nothing to set on the training side.

---

## Training — `train_r2v`

Trains the report-conditioning adapter on a **frozen** NVIDIA denoiser. Reads the manifest directly;
it does **not** use a cohort (cohorts are an evaluation artefact).

```bash
# CPU wiring check: synthetic latents and fabricated reports, no data/VAE/GPU needed
python -m mrrate_r2v.cli.train_r2v --dry-run --max-steps 2 --out /tmp/r2v_dryrun --text-encoder mock

# real single-GPU run
python -m mrrate_r2v.cli.train_r2v \
    --manifest <data_ws>/r2v_manifest/manifest_shards_native.csv \
    --report-index <data_ws>/r2v_manifest/report_index_shards_native.csv \
    --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \
    --vae-checkpoint  <ws>/models/autoencoder_v1.pt \
    --text-encoder radbert --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \
    --out <ws>/runs/r2v_adapter_v1 --epochs 1 --batch-size 1

# multi-GPU, the official launcher shape
torchrun --nproc_per_node=4 -m mrrate_r2v.cli.train_r2v --num-gpus 4 ...  # same flags
```

**data**

| flag | default | meaning |
|---|---|---|
| `--manifest` | required¹ | manifest CSV from `build_manifest` |
| `--report-index` | required¹ | report index CSV for `ShardReportStore` |
| `--split` | `train` | which split to train on |
| `--report-sections` | `findings impression` | concatenated into the conditioning text; ignored when `--report-format` is given |
| `--report-format` | the `--conditioning` config's own (A/B: `findings_impression_meta,impression_findings_meta`) | one named format, or **several comma-separated to sample one per training sample**. Validation is pinned to the first name; the checkpoint records the whole spec. See `mrrate_r2v/README.md` §4.2.1. |
| `--geometry-mode` | `per_modality_plane` | as in `preprocess`; per-bucket FOVs |
| `--bucket-order` | `interleave` | `interleave` = each `(modality, plane)` bucket spread evenly across the epoch, so consecutive batches carry different modalities \| `shuffle` = one flat shuffle |
| `--num-workers` | `4` | DataLoader workers |
| `--dry-run` | off | synthetic latents + fabricated reports; no manifest, VAE or GPU needed |

¹ required unless `--dry-run`.

Series selection is fixed at `"all"` — every eligible series is a training sample, so a study's
report is paired with each of its ~7 series. That contrast (one report, several modalities,
distinguished only by `class_labels`/`spacing_tensor`) is what stops the report adapter from
absorbing modality; the one-per-study modes exist for cohort construction, not training. On the
real train split that is **575,536 series over 82,233 studies**, 7.0 series per study.

One batch is always one `(modality, plane)` bucket, in either geometry mode — see
[`../data/README.md`](../data/README.md#batching-geometrybucketbatchsampler). No frequency
weighting is applied in either `--bucket-order`: one epoch is one pass over every series, and a
bucket's share of it is its share of the data.

**model**

| flag | default | meaning |
|---|---|---|
| `--base-checkpoint` | required¹ | NVIDIA diffusion UNet — **frozen**, loaded strictly |
| `--vae-checkpoint` | required¹ | NVIDIA autoencoder — **frozen**, encodes volumes to latents on the fly |
| `--network-config` | NVIDIA's | `config_network_rflow.json`; the base geometry is read from it |
| `--cross-attention-dim` | `512` | the adapters' width |
| `--conditioning-levels` | base model's `attention_levels` = `0 0 1 1` | where the report enters the UNet — see below |
| `--no-condition-mid` | (mid on) | drop the bottleneck adapter |
| `--context-hidden-dim` | `4 × cross_attention_dim` | hidden width of `context_proj` |

#### What `--conditioning-levels 0 0 1 1` means

The UNet is a U: it downsamples the latent through 4 **resolution levels**, crosses the bottleneck,
then upsamples back through the mirrored 4. One flag position per *level*, not per block — a `1`
places a report cross-attention adapter on **both** sides of the U at that resolution.

```
level                 0         1         2         3      ── bottleneck ──
resolution (latent)  full      1/2       1/4       1/8
channels              64       128       256       512
--conditioning-levels  0         0         1         1
encoder              –         –       adapter   adapter
decoder              –         –       adapter   adapter        + mid adapter
                                                                  (unless --no-condition-mid)
```

So `0 0 1 1` = **5 adapters**: the two coarsest encoder blocks, the two decoder blocks that mirror
them (i.e. the first two after the bottleneck, which are also the coarsest), and the bottleneck.

Levels are ordered coarse-to-the-right: index `0` is the full-resolution, cheap-channel end;
index `3` is the deep, low-resolution end. The decoder is indexed by the level it *mirrors*, so you
never have to think about the reversed block order — position `i` always means "this resolution,
going down and coming back up".

Why the default is the two coarsest levels: it matches where the pretrained model already puts its
own attention, a report describes global findings rather than per-voxel texture, and attention cost
grows with voxel count — turning on level 0 means attending over the full-resolution latent at every
step. Start here; widening it is an experiment, not a fix.

**text encoder**

| flag | default | meaning |
|---|---|---|
| `--text-encoder` | `radbert` | `radbert` \| `mock` (deterministic hash-based stand-in for CPU tests) |
| `--text-checkpoint` | — | local RadBERT snapshot directory; **required** for `radbert` |
| `--max-report-tokens` | `512` | truncation length. Exceeding the encoder's real limit is refused up front, not as a CUDA index error at step 1. |
| `--mock-output-dim` | `32` | `--text-encoder mock` only |

**optimisation**

| flag | default | meaning |
|---|---|---|
| `--out` | *required* | output directory for checkpoints |
| `--epochs` | `1` | |
| `--max-steps` | — | stop after N steps regardless of epochs (use for smoke runs) |
| `--batch-size` | `1` | > 1 needs the bucket batch sampler, which is already wired in |
| `--lr` | `1e-5` | NVIDIA's own `diffusion_unet_train.lr` |
| `--grad-accumulation-steps` | `1` | effective batch = this × `--batch-size` |
| `--grad-clip-norm` | — | off by default (as in the official trainer) |
| `--no-amp` | (amp on) | disable mixed precision |
| `--seed` | `0` | also seeds the dedicated report-dropout generator (`seed + 12345`) |
| `--log-every` | `10` | steps between log lines |
| `--save-every-steps` | — | periodic checkpoints; otherwise only `adapter_last.pt` |
| `--validate-every-steps` | — | ⚠️ **inert today** — see "Known gaps" |
| `--save-format` | `adapter` | `adapter` (8M) \| `full` (NVIDIA's layout) \| `both` |
| `--resume` | — | adapter checkpoint to resume from (restores optimizer, scheduler, scaler, RNG, step) |
| `--scale-factor` | `auto` | `auto` = from the base checkpoint (**correct for a frozen denoiser**) \| `recompute` = `1/std(z)` of the first batch (official *training* behaviour) \| a literal float |
| `--num-gpus` | `1` | with `torchrun`; DDP uses `find_unused_parameters=True` |
| `--device` | `cuda` if available | |

**conditioning**

| flag | default | meaning |
|---|---|---|
| `--report-dropout-probability` | `0.1` | fraction of samples whose report is replaced by the learned null token. This is what makes report classifier-free guidance possible at inference. |
| `--modality-dropout-probability` | `0.1` | NVIDIA's own `augment_modality_label(prob=...)`, imported unchanged |

**Writes:** `adapter_last.pt` (+ `adapter_step*.pt` if `--save-every-steps`), `train_summary.json`
(steps, wall time, final/mean loss, trainable vs frozen parameter counts, text-encoder identity,
base-checkpoint identity).

**Logged per step:** `loss` (L1 on the rectified-flow velocity target `x0 − ε` — NVIDIA's own
objective, no auxiliary alignment term), `lr`, `n_dropped_reports`, `timestep_mean`.

---

## Stage 2 — the predict scripts

All four write the same directory layout: `predictions.json` (task, `cohort_id`, model provenance
incl. checkpoint sha256, item list, failures) + `volumes/<modality>__<plane>.npz`. **None of them
compute a metric.**

### `predict_r2v` vs `generate_r2v` — same model, different purpose

They load **the identical model** (both call `generate_r2v.build_sampler`, so they cannot diverge).
The difference is only what goes in and what comes out:

| | `generate_r2v` | `predict_r2v` |
|---|---|---|
| you give it | a report you typed, and a grid (`--dim`, `--spacing`) | a **cohort** |
| the report comes from | you | each case's own real report |
| the grid comes from | you | each case's own bucket |
| you get back | plain `.nii.gz` files you can open in a viewer | a **prediction set** — volumes plus the bookkeeping `evaluate` needs (`cohort_id`, case ids, per-case seeds) |
| can you score it? | no | yes |

**In short:** `generate_r2v` is for *looking* — type a report, see what the model draws. `predict_r2v`
is for *measuring* — run the whole cohort and get something the evaluator will accept.

(`generate_r2v --cohort` also loops a cohort, but still writes loose `.nii.gz` files plus a manifest,
not a scoreable prediction set. If you want numbers, use `predict_r2v`.)

### `predict_r2v` — report-conditioned generation (ours)

```bash
python -m mrrate_r2v.cli.predict_r2v \
    --cohort <ws>/cohorts/test_v1 \
    --checkpoint <ws>/runs/r2v_adapter_v1/adapter_last.pt \
    --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \
    --vae-checkpoint  <ws>/models/autoencoder_v1.pt \
    --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \
    --out <ws>/predictions/r2v_v1
```

| flag | default | meaning |
|---|---|---|
| `--cohort` | *required* | cases and their reports come from here |
| `--out` | *required* | prediction directory to create |
| `--checkpoint` | *required* | the **adapter** (from `train_r2v`) |
| `--base-checkpoint` | *required* | frozen NVIDIA diffusion UNet |
| `--vae-checkpoint` | *required* | frozen NVIDIA autoencoder |
| `--text-checkpoint` | whatever the adapter recorded | text encoder directory |
| `--network-config` | NVIDIA's | |
| `--model-name` | `report2volume` | recorded in the prediction set's provenance |
| `--modality` | each case's own `sequence` | force one class label for every case instead |
| `--num-inference-steps` | `30` | NVIDIA's own default |
| `--report-guidance-scale` | `4.0` | `0` disables the report term and reproduces NVIDIA's guidance exactly |
| `--modality-guidance-scale` | `10.0` | NVIDIA's `cfg_guidance_scale` for mr-brain |
| `--allow-base-mismatch` | off | load an adapter trained against a *different* base checkpoint |
| `--device` | `cuda` | |
| `--seed` | `42` | per-case seed is `stable_seed(seed, case_id)`, so a rerun reproduces every volume |
| `--limit` | — | first N cases only, for a smoke test |
| `--overwrite` | off | |

| `--allow-report-format-mismatch` | off | predict even when the cohort's `report_format` is not one the adapter was trained on |

> **`--modality` now defaults to each case's own sequence.** It used to default to `T1w` for the
> whole run, so on the default 10-bucket cohort every T2w/FLAIR/SWI case was generated with the T1w
> class label while the prediction item recorded the case's real `sequence` — a wrong conditioning
> input that generates perfectly well and scores as if the model were bad at three of four
> sequences. Pass `--modality` only to force one label deliberately.
>
> The cohort's `report_format` is also checked here now (it was only checked in `generate_r2v`),
> so the path that produces the numbers can no longer condition on differently-composed text.

### `generate_r2v` — free-form, no cohort needed

The same model assembly (`build_sampler` is shared with `predict_r2v`, so they cannot diverge), but
driven by a report string. Use it to eyeball what a checkpoint produces.

```bash
python -m mrrate_r2v.cli.generate_r2v \
    --base-checkpoint ... --vae-checkpoint ... --adapter <run>/adapter_last.pt \
    --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \
    --report "Findings: 12 mm enhancing lesion in the right frontal lobe." \
    --modality T1w --plane AXIAL \
    --report-guidance-scale 4 --out <ws>/samples/case001.nii.gz
```

Same checkpoint/sampler/guidance flags as `predict_r2v`, plus:

| flag | default | meaning |
|---|---|---|
| `--report` / `--report-file` / `--cohort` | — | **exactly one.** `--cohort` generates one volume per case from its paired report. |
| `--out` | — | output `.nii.gz` (with `--report`/`--report-file`) |
| `--out-dir` | — | output directory (with `--cohort`) |
| `--modality` | `T1w` | class label, geometry bucket, and the `[MODALITY]` marker |
| `--plane` | `AXIAL` | geometry bucket and the `[PLANE]` marker |
| `--dim` | the `(--modality, --plane)` bucket's shape | output shape, **X Y Z**. Must be divisible by the latent divisor (4). Unknown bucket → NVIDIA's `256 256 256`. |
| `--spacing` | the bucket's spacing | mm, **X Y Z**. A real conditioning input to the model, not just header metadata — and, under a `*_meta` format, the `[SPACING]` marker too. |
| `--allow-report-format-mismatch` | off | generate from a cohort composed under a format the adapter was not trained on |
| `--text-encoder` | whatever the adapter recorded | `radbert` \| `mock` |
| `--max-report-tokens` | the adapter's recorded value | |
| `--no-batched-guidance` | (batched) | run the CFG branches as separate forwards — slower, identical numbers |
| `--latent-only` | off | stop after the diffusion loop, skip the VAE decode (cheap smoke test) |
| `--seed` | `1234` | NVIDIA's own default |

Writes the volume plus a `*.manifest.json` recording seed, geometry, guidance scales, sampler
settings, all checkpoint paths + sha256, and the text-encoder identity.

The adapter checkpoint names its own text encoder, context width and adapter geometry, so none of
that has to be re-specified — a mismatch becomes impossible rather than merely unlikely.

### `predict_vae` — reconstruction baseline

Encodes and decodes every cohort volume with the frozen autoencoder. Score with
`--task reconstruction`. Padding to the encoder's divisor is applied before encoding and the *exact
same* amount cropped back after decoding (tracked by a `CropPadRecord`), so the reconstruction always
returns on the cohort's own grid.

| flag | default | meaning |
|---|---|---|
| `--cohort` / `--out` / `--checkpoint` | *required* | cohort dir / output dir / NVIDIA autoencoder |
| `--env-config` / `--model-config` / `--network-config` | NVIDIA's | |
| `--device` | `cuda` | |
| `--limit` | — | first N cases only |
| `--overwrite` | off | |

### `predict_generation` — NVIDIA unconditional baseline

Generates from a modality label alone. Score with `--task generation` (distribution metrics only).

| flag | default | meaning |
|---|---|---|
| `--cohort` / `--out` | *required* | |
| `--n-per-bucket` | the cohort's own per-bucket counts | so the real and generated populations have identical composition |
| `--checkpoint` | *required* | autoencoder (VAE) |
| `--unet-checkpoint` | the filename NVIDIA's env config names, next to `--checkpoint` | **absolute path required** |
| `--env-config` / `--model-config` / `--network-config` | NVIDIA's | |
| `--device` | `cuda` | |
| `--seed` | `42` | |
| `--overwrite` | off | |

Every bucket is generated at **its own** geometry — the cohort's — because spacing is a real
conditioning input. Bucket shapes must be a multiple of 32 (the UNet's constraint); a wrong shape is
refused before any GPU work rather than padded, since padding would change the FOV the model was
asked for.

### `import_predictions` — someone else's `.nii.gz` files

Does the identifier matching once, records the result, produces a directory `evaluate` accepts like
any other.

| flag | default | meaning |
|---|---|---|
| `--cohort` / `--predictions-csv` / `--out` | *required* | |
| `--task` | `report2volume` | `reconstruction` \| `report2volume` |
| `--model-name` | `external_checkpoint` | provenance |
| `--overwrite` | off | |

CSV schema — `study_key`/`series_key` must match the manifest's `study_uid`/`series_id` exactly,
never a filename or row position:

```
study_key,prediction_path[,series_key,modality,acquisition_plane]
```

`series_key` may be omitted only for a study with exactly one case in the cohort. Anything ambiguous,
duplicated or unmatched is rejected with a reason into `import_report.json` rather than guessed at.

---

## Stage 3 — `evaluate`

One command scores every task. `--task` decides which metrics run; nothing else does.

```bash
python -m mrrate_r2v.cli.evaluate --task report2volume \
    --gt <ws>/cohorts/test_v1 --pred <ws>/predictions/r2v_v1 \
    --out <ws>/results/r2v_v1 \
    --distribution-metrics --medicalnet-checkpoint <ws>/pretrained/medicalnet/....pth \
    --workers $SLURM_CPUS_PER_TASK
```

| flag | default | meaning |
|---|---|---|
| `--task` | *required* | `reconstruction` \| `report2volume` \| `generation` |
| `--gt` | *required* | ground-truth **cohort** directory (also the real reference population for `generation`) |
| `--pred` | *required* | prediction directory |
| `--out` | *required* | results directory to create |
| `--overwrite` | off | |
| `--distribution-metrics` | off | FID / Inception Score / precision-recall-density-coverage. Always on for `generation`. |
| `--medicalnet-checkpoint` | — | MedicalNet ResNet-10 weights for the **3D** FID; without it only the 2.5D Inception variant is computed |
| `--device` | `cuda` | for the feature extractors |
| `--fid-bootstrap` | `30` | bootstrap resamples for the FID CI |
| `--min-subgroup-n` | `10` | skip per-bucket distribution metrics below this many cases — they are not stable |
| `--diversity-k` | `5` | k for precision/recall/density/coverage |
| `--seed` | `42` | |
| `--save-figures` | `3` | orthogonal-slice montages per bucket → `<out>/figures/` (0 to disable). Paired tasks show GT / prediction / \|diff\| at evenly-spaced metric ranks, so N=3 gives worst, median and best — **the worst one is where failure modes are visible.** |
| `--save-nifti-cases` | `0` | also export gt/pred/absdiff as `.nii.gz` for the first N figured cases per bucket |
| `--skip-metric-groups` | none | drop groups the task would compute. `perceptual` is ~40% of per-case time. **Can only remove, never add** — `generation` still cannot acquire a voxelwise metric. Recorded in `summary.json`. |
| `--workers` | `1` | per-case scoring processes. Evaluation is CPU-compute-bound (I/O is 0.5% of the work), so this is the one knob that speeds it up; results are identical regardless of value. |
| `--wandb-mode` / `--wandb-entity` / `--wandb-project` / `--wandb-group` | `disabled` | optional W&B logging |

**Writes:** `metrics_per_bucket.csv` and `metrics_summary.csv` (the deliverable), `summary.json`
(machine-readable mirror), `run_manifest.json` (what actually ran), per-case rows, exclusions with
reasons, and `figures/`.

Read `summary.json` first, and run the acceptance checker before treating any number as a result:

```bash
python3 slurm/check_run.py --cohort <ws>/cohorts/test_v1 \
    [--pred-vae <dir>] [--pred-gen <dir>] [--results-recon <dir>] [--results-gen <dir>]
```

It scores the run against [`slurm/SUCCESS_CRITERIA.md`](../../slurm/SUCCESS_CRITERIA.md), names every
check with its criterion id, and reports absent inputs as **SKIP, not PASS**.

### What each task computes

| `--task` | paired? | metric groups |
|---|---|---|
| `reconstruction` | yes | fidelity, perceptual, distribution, anatomy |
| `report2volume` | yes | fidelity, perceptual, distribution, anatomy, report_alignment |
| `generation` | **no** | distribution, anatomy |

`generation` gets no voxelwise metric because no real patient corresponds to a generated volume.
That is a property of the task declared in [`eval/tasks.py`](../eval/tasks.py), not a flag anyone can
forget to pass. Details in [`eval/README.md`](../eval/README.md).

---

## Slurm

[`slurm/_common.sh`](../../slurm/_common.sh) holds every path and the apptainer invocation; the
numbered scripts source it by absolute path via `$R2V_REPO`.

| script | wraps | note |
|---|---|---|
| `01_smoke_test.sbatch` | `preprocess` → `predict_vae` → `evaluate` on **2 real cases** | a few minutes; must pass before any pilot or full run |
| `02_preprocess.sbatch` | `cli.preprocess` | CPU partition is `cpu` on Helma |
| `03_predict_vae.sbatch` | `cli.predict_vae` | |
| `04_predict_generation.sbatch` | `cli.predict_generation` | |
| `05_evaluate.sbatch` | `cli.evaluate` | |
| `06_train_r2v.sbatch` | `cli.train_r2v` | uses `$SIF_IMAGE_TEXT` (`nvidia+redbert.sif`) |
| `07_generate_r2v.sbatch` | `cli.generate_r2v` | takes the adapter path **positionally** |

```bash
sbatch slurm/06_train_r2v.sbatch                                     # 4-step smoke run
sbatch --export=ALL,R2V_MAX_STEPS=0,R2V_EPOCHS=1 --time=24:00:00 \
       slurm/06_train_r2v.sbatch                                     # full run
sbatch slurm/07_generate_r2v.sbatch <ws>/runs/<tag>/adapter_last.pt
sbatch --export=ALL,R2V_COHORT=<ws>/cohorts/test_v1 \
       slurm/07_generate_r2v.sbatch <adapter.pt>                     # whole cohort
```

> ⚠️ `#SBATCH --export=NONE` is set on every script (FAU's recommendation). A plain
> `VAR=x sbatch ...` therefore does **not** reach the job — overrides must go through
> `--export=ALL,VAR=x`. And every Helma partition silently defaults to a **10-minute** walltime if
> `--time` is omitted.

Useful `07_` variables: `R2V_REPORT_GUIDANCE=0` (modality-only, i.e. NVIDIA's original behaviour),
`R2V_REPORT_GUIDANCE=1` (plain report-conditioned prediction), `R2V_MODALITY_GUIDANCE=0` (report
guidance only), `R2V_LATENT_ONLY=1` (skip the decode), `R2V_STEPS`, `R2V_MODALITY`, `R2V_SEED`,
`R2V_DIM_{X,Y,Z}`, `R2V_SPACING_{X,Y,Z}`, `R2V_COHORT`.

---

## Common failures, and what they mean

| message | cause |
|---|---|
| `prediction set was produced against cohort <id>, ... ` | you are scoring predictions against a different cohort. Regenerate one of them; there is no bypass by design. |
| `checkpoint has N tensors with no home in the model` | the base checkpoint was not trained on this architecture — check `--network-config`. |
| `adapter was trained against base checkpoint <sha> but the loaded base is <sha>` | wrong `--base-checkpoint`. `--allow-base-mismatch` downgrades it to a warning, but the adapter's zero-point is then a different frozen model. |
| `adapter checkpoint does not match this model: missing=... unexpected=...` | `--cross-attention-dim` / `--conditioning-levels` / `--condition-mid` differ from what the checkpoint was trained with. |
| `N base-model parameters are trainable` | the freeze gate caught a leak — you would have been fine-tuning NVIDIA's denoiser. |
| `output shape (...) is not divisible by the latent divisor 4` | `--dim` must be a multiple of 4 (bucket shapes are multiples of 32 anyway). |
| `collate_fn_r2v got a batch with mismatched image shapes` | `batch_size > 1` under `per_modality_plane` without the bucket batch sampler. |
| `--text-checkpoint is required for --text-encoder radbert` | pass the local RadBERT snapshot dir, or use `--text-encoder mock` for a dry run. |
| a bare `FileNotFoundError` from inside NVIDIA's loader | a relative checkpoint path — their env config resolves against cwd. Always pass absolute. |

---

## Known gaps

- **`--validate-every-steps` does nothing today.** `MRRateAdapterTrainer.fit(train_loader, validate)`
  only calls the hook when a callable is passed, and `train_r2v.py` calls `fit(train_loader)`. The
  plumbing exists; nothing is wired into it yet.
- **`report_alignment` has no model.** `report_image_similarity_score` reports
  `available=False, reason="no validated MRI image-text model exists in this project yet"` rather
  than substituting a different model and calling the result report alignment.

---

## Testing

```bash
cd contrastive-pretraining
python -m pytest tests/test_cli_imports.py tests/test_r2v_sampling_and_cli.py \
                 tests/test_eval_tasks_and_runner.py -v --no-cov
```
