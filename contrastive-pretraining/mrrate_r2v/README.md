# `mrrate_r2v` — report-to-volume for brain MRI

Generate a 3D brain MRI volume from a radiology report, and measure how good it is.

This is the package entry point: **what each piece is, how the pieces connect, and how to run
training and inference.** For *why* the experiment is designed this way — sampling policy, FOV
choices, what makes two runs comparable — read [`docs/R2V.md`](../../docs/R2V.md). The two files are
deliberately split so they don't drift: that one is the experiment guide, this one is the code map.

Layer guides: [`data/README.md`](data/README.md) · [`models/README.md`](models/README.md) ·
[`cli/README.md`](cli/README.md) · [`eval/README.md`](eval/README.md)

No installation needed — it is importable from `contrastive-pretraining/` as a plain package.

---

## 1. What the model actually is

One sentence: **NVIDIA's `NV-Generate-MR-Brain` diffusion model, frozen, with a small trainable
cross-attention adapter that lets a radiology report steer it.**

```
   report text ──► RadBERT ─────────────────────────────► [B, L, 768]   FROZEN
                                                                │
                                                         context_proj   TRAINABLE
                                                                │
                                                          [B, L, 512]
                                                                │
  real volume ──► VAE encoder ──► latent ──► +noise ──► DIFFUSION UNET ──► predicted velocity
                     FROZEN                             ├─ NVIDIA's blocks    FROZEN
                                                        └─ 5 cross-attn adapters  TRAINABLE
                                                                │
                                                       L1 loss vs (latent − noise)
```

**Trainable: 8,080,000 of 188,580,868 parameters (4.28%).** The trainer asserts that before the
first optimizer step and refuses to start otherwise.

Why so little? Because the adapters are zero-initialised, the conditioned model is *numerically
identical* to NVIDIA's at step 0. Training starts **at** a working brain-MRI generator and only ever
adds the report on top — instead of trying to relearn brain anatomy from 88k studies.

---

## 2. The pipeline, in five stages

```
stage 0   build_manifest    →  manifest.csv (+ report_index.csv)     once per storage location
stage 1   preprocess        →  COHORT dir, cohort_id                 once per experiment set
 (train)  train_r2v         →  adapter_last.pt                       reads the manifest, not a cohort
stage 2   predict_r2v       →  PREDICTION dir                        once per model
stage 3   evaluate --task   →  RESULTS dir (CSV + summary + figures) per prediction set
```

Two on-disk **contracts** hold this together, and they are the reason the package is structured this
way at all:

- **[`cohort.py`](cohort.py)** — a cohort directory freezes the case list, FOV, sample count,
  normalizer and seed, hashed into a `cohort_id`. Quote that hash in a paper: two directories with
  the same `cohort_id` hold the same cases at the same geometry with the same preprocessing.
- **[`predictions.py`](predictions.py)** — a prediction set records the `cohort_id` it was produced
  against. `evaluate` hard-fails on a mismatch. **There is no `--force` and no "close enough"
  comparison, by design** — that refusal is what makes two experiments comparable.

Both use stdlib + numpy only, so an evaluation never needs the data stack or a model loaded.

---

## 3. Module map

### Top level

| module | what it is |
|---|---|
| [`cohort.py`](cohort.py) | the frozen ground-truth contract: `Cohort`, `CohortCase`, `cohort_id` |
| [`predictions.py`](predictions.py) | its mirror: `PredictionSet`, `PredictionItem`, `PredictionReader.assert_matches_cohort` |
| [`volumes.py`](volumes.py) | on-disk volume storage — one `.npz` archive per (modality, plane) bucket, random-access by `case_id` |
| [`text.py`](text.py) | the replaceable text-encoder seam: `TextEmbedder` protocol, `RadBertEmbedder`, `MockTextEmbedder`, one registry |
| [`conditioning.py`](conditioning.py) | modality class ids, NVIDIA's own modality dropout, report dropout, classifier-free guidance |
| [`training.py`](training.py) | `MRRateAdapterTrainer` — mirrors `NV-Generate-CTMR/scripts/diff_model_train.py` |
| [`sampling.py`](sampling.py) | `ReportToVolumeSampler` — mirrors `scripts/diff_model_infer.py` |

### Subpackages

| package | guide | what it does |
|---|---|---|
| [`data/`](data/) | [README](data/README.md) | manifest → archive reads → preprocessing → `(volume, report)` pairs |
| [`models/`](models/) | [README](models/README.md) | the frozen NVIDIA nets, the report adapter, what is trainable |
| [`cli/`](cli/) | [README](cli/README.md) | the eight entry points and every flag |
| [`eval/`](eval/) | [README](eval/README.md) | one runner, three tasks, one result layout |

### Two structural rules that look arbitrary but are not

- **`eval/__init__.py` re-exports nothing**, and `data/{storage,manifest,reports,geometry}.py` import
  no torch. A heavy dependency in one module must not make another unimportable — a pyarrow-only
  interpreter has to be able to build a shards manifest.
- **The evaluator reads `.npy` files and nothing else.** No manifest, no archive, no Dataset, no
  model. That is what makes it *impossible* for the evaluator to preprocess differently than the
  cohort did.

---

## 4. Walkthrough: from a NIfTI on disk to a number

### 4.1 One sample

[`MRReportToVolumeDataset.__getitem__`](data/dataset.py), per index:

1. resolve the (modality, plane) **bucket** and its target geometry;
2. read the NIfTI bytes straight out of the `.zip`/tar archive (no extraction in `stream` mode);
3. **RAS reorient → resample → crop/pad → normalize** (`percentile` 0–99.5 → [0,1], which is
   NVIDIA's own MRI transform) — this is `scripts/data.py`'s code, imported unchanged, so the two
   pipelines in this repo cannot drift on how a volume is prepared;
4. permute `(D,H,W)` → `(X,Y,Z)` **exactly once**;
5. fetch the report and concatenate the requested sections.

```python
sample = {
  "image":             tensor [1, X, Y, Z],   # preprocessed, model-ready, no further reshaping
  "report_text":       "Findings: ...  Impression: ...",
  "modality":          "T1w",                 # becomes the class label
  "acquisition_plane": "AXIAL",
  "target_spacing_mm": tensor([1., 1., 1.]),  # a real conditioning input, not just header metadata
  "target_shape":      tensor([256, 256, 256]),
  "study_key", "series_key":                  # identifiers — never log verbatim
}
```

**One sample = one report paired with one real volume.** A study with 6 series produces 6 samples
during training (`series_selection="all"`), all conditioned on the same report.

> ⚠️ **Axis order is the single most bug-prone thing in this package.** Internal geometry is
> `(D,H,W)=(S,R,A)`; everything crossing the package boundary is `(X,Y,Z)=(R,A,S)`. Convert only via
> `geometry.dhw_to_xyz` / `xyz_to_dhw`. A skipped conversion is **silent** for a 256³ cube at
> isotropic spacing and scrambles axes otherwise.

### 4.2 Batching

`collate_fn_r2v` stacks the tensors and keeps strings as lists. Under `per_modality_plane` geometry
each bucket has its own shape, so `GeometryBucketBatchSampler` only ever emits batches drawn from
**one** bucket — that is what makes `batch_size > 1` legal.

`drop_last=True` (what `cli.train_r2v` uses) drops a bucket's *remainder*, never the **bucket**. A
bucket smaller than `batch_size` has no full batch at all, so a plain `drop_last` deletes it from
every epoch and the model silently never sees that (modality, plane). On the real train split at
`batch_size=8` that is SWI CORONAL (4 series) and SWI SAGITTAL (2) — and neither exists in val or
test, so no metric could have revealed it. Such buckets keep one short batch and are logged as
`undersized_buckets`.

### 4.2.1 Report format

`R2VDatasetConfig.report_format` takes one name from
[`textenc/formats.py`](textenc/formats.py) — or **several, comma-separated**, in which case one is
drawn per sample, uniformly, deterministically from `(seed, epoch, index)`.

The shipped training spec is `findings_impression_meta,impression_findings_meta`
(`ORDER_AGNOSTIC_META_SPEC`). Two reasons, both about the challenge rather than about MR-RATE:

- **Section order.** The challenge's report layout is unknown and nothing at submission time can
  detect that the order flipped. Training on both orders means the model has seen every section in
  first position — which matters most under truncation, where a 512-token encoder keeps the head of
  the string (8–10% of MR-RATE studies truncate; RadBERT 9.2%).
- **`[MODALITY] … [PLANE] … [SPACING] x y z`** leads both orderings. Spacing is `(X, Y, Z)` and
  matches the sample's own `target_spacing_mm` exactly. It is *also* a numeric input via
  `spacing_tensor`, but the text encoder is the only path that sees modality, plane and spacing
  together, and it is the path a challenge request can populate with no volume attached.

Validation is pinned to the spec's **first** name — a sampled format would add format variance to a
curve whose only job is to show model improvement. The checkpoint records the whole spec, so
`cli.generate_r2v` accepts a cohort built under either ordering.

At inference (`cli.generate_r2v --report`) the prefix is prepended from `--modality`, `--plane` and
`--spacing`, and `--dim`/`--spacing` default to that bucket's own trained grid. Only pass them to
override; an unknown (modality, plane) still lands on NVIDIA's 256³ @ 1 mm.

### 4.3 One training step

```python
latents   = vae.encode_stage_2_inputs(image) * scale_factor      # frozen, no_grad
spacing   = target_spacing_mm * 1e2                              # NVIDIA's own transform
modality  = augment_modality_label(class_ids, prob=0.1)          # NVIDIA's own dropout
cond      = radbert.encode(report_text)                          # frozen -> tokens + mask
drop      = sample_report_drop_mask(B, 0.10)                     # 10% get the learned null token

noise     = randn_like(latents)
timesteps = scheduler.sample_timesteps(latents)                  # RFlow picks its own
noisy     = scheduler.add_noise(latents, noise, timesteps)

pred   = unet(x=noisy, timesteps=timesteps, spacing_tensor=spacing, class_labels=modality,
              context=cond.token_embeddings, context_mask=cond.attention_mask,
              context_drop_mask=drop)
target = latents - noise                       # rectified-flow velocity
loss   = L1Loss()(pred, target)
```

**The loss is NVIDIA's own**: `L1` on the velocity target `x0 − ε`. Not MSE, not SNR-weighted, and
**no auxiliary report/image alignment term** — the adapter has to earn its keep on the original
generative objective. `Adam(lr=1e-5)`, `PolynomialLR(power=2)`, AMP + `GradScaler`. **No EMA**,
because there is none in the official code either.

`training.py`'s module docstring carries the line-by-line mapping to
`NV-Generate-CTMR/scripts/diff_model_train.py`, including the three deliberate differences (optimizer
sees only the adapter; `scale_factor` comes from the checkpoint; latents encoded on the fly).

### 4.4 How report conditioning works

Four steps — full detail in [`models/README.md`](models/README.md):

1. **RadBERT** turns the report into `[B, L, 768]` token embeddings plus an attention mask. Frozen,
   `no_grad`, permanently in `eval()`.
2. **`context_proj`** (a small trainable MLP) projects to `[B, L, 512]`. This is the *only* place the
   text encoder's width appears, which is why swapping encoders needs no other code change.
3. **Five cross-attention adapters** sit at the input of each conditioned UNet level and at the
   bottleneck. Each is `x + proj_out(attn(norm(x)))`, with `proj_out` zero-initialised. Read it as:
   *every voxel of the feature map asks which words of this report are relevant to it, and adds the
   answer to itself.* The mask is honoured, so padding never joins the softmax.
4. **A learned `null_context` token** replaces the report for 10% of training samples. So the model
   learns both "with this report" and "with no report" — and the difference between the two is
   exactly the report's contribution, which guidance amplifies at inference. Training dropout and
   inference CFG go through the *same* code path.

### 4.5 Inference

`sampling.py` is `diff_model_infer.py` step for step: latent noise at `dim // 4`, RFlow timesteps,
guided model output, `scheduler.step`, then NVIDIA's `ReconModel` decode under a
`SlidingWindowInferer` (roi 80³, gaussian, overlap 0.4), then their MR postprocessing to int16
`[0, 1000]` and an axis-aligned affine.

Guidance is **hierarchical**, with the report as an increment on NVIDIA's modality term:

```
D_guided = D_00 + s_modality · (D_m0 − D_00) + s_report · (D_mr − D_m0)
```

| branch | modality label | report |
|---|---|---|
| `D_00` | null class | null token |
| `D_m0` | real class | null token |
| `D_mr` | real class | **the report** |

- `--report-guidance-scale 0` collapses this to `diff_model_infer.py:207` **exactly** — asserted
  numerically in `tests/test_r2v_conditioning.py`, so report guidance cannot silently change what
  NVIDIA's model does.
- `--report-guidance-scale 1 --modality-guidance-scale 1` gives plainly `D_mr`.
- All branches run as one batched UNet call (the same trick official uses for its two).

### 4.6 Evaluation

`eval/runner.py:run_evaluation` is the only evaluation path — the CLI and the tests both call it.
`--task` decides the metric set; nothing else does. Full detail in [`eval/README.md`](eval/README.md).

---

## 5. Metrics

### During training

There is **no validation metric**. Per step the trainer logs:

| key | meaning |
|---|---|
| `loss` | the L1 velocity loss — the only optimisation signal |
| `lr` | current learning rate |
| `n_dropped_reports` | how many samples got the null token this step |
| `timestep_mean` | mean sampled timestep — a sanity check on scheduler behaviour |

`train_summary.json` adds steps, wall time, final/mean loss, and trainable vs frozen counts.

### At test time (`--task report2volume`)

| group | metrics |
|---|---|
| **fidelity** | `mae_whole`, `mse_whole`, `psnr_whole`, `ncc_whole`, `ssim3d_whole`, `mae_fg`, `mse_fg`, `psnr_fg`, `ncc_fg`, `relative_intensity_error_fg` |
| **perceptual** | `edge_preservation_fg`, `laplacian_variance_ratio_fg`, `hf_energy_ratio`, `ssim2d_{sagittal,coronal,axial}_mean` |
| **distribution** | MedicalNet 3D FID (bootstrap CI), 2.5D Inception FID, Inception Score, precision / recall / density / coverage, intra-set MS-SSIM (mode-collapse probe) |
| **anatomy** | L-R symmetry NCC, intracranial fraction, tissue-contrast separation, foreground compactness, background purity — compared to the real population by two-sample KS |
| **report_alignment** | `report_image_similarity_score` — ⚠️ a **hook**, currently always `available=False` (no validated MRI image-text model exists in this project yet). It never substitutes a different model and calls the result report alignment. |

Three properties worth stating explicitly:

- **The foreground mask always comes from the ground truth**, never the prediction — otherwise a
  degenerate prediction picks its own easier evaluation region.
- **A shape-mismatched prediction is excluded with a reason, never resized.** Equal `.shape` is never
  treated as proof of the same physical space.
- **A frequency-weighted aggregate weights by population bucket counts**, never cohort counts — the
  cohort is balanced per bucket by design, so cohort counts are a sampling artefact.

`--task generation` structurally cannot get a voxelwise metric, whatever flags you pass.

---

## 6. How to run it

Paths below use `<ws>` for the workspace (`/hnvme/workspace/y100dc19-nvidia-mri-brain`) and
`<data_ws>` for the data workspace. Cohorts and prediction sets are ~17–24 MB/volume — **workspace
only, never git or `$HOME`.**

### 6.0 First: does it wire up at all? (CPU, seconds, no data)

```bash
cd contrastive-pretraining
python -m mrrate_r2v.cli.train_r2v --dry-run --max-steps 2 --out /tmp/r2v_dryrun --text-encoder mock
```

Synthetic latents, fabricated reports, a two-level UNet — the whole trainer path with no manifest,
VAE, checkpoint or GPU. If this fails, nothing else will work.

### 6.1 Build the manifest (once)

```bash
python -m mrrate_r2v.cli.build_manifest --source shards_parquet \
    --shards-root <data_ws>/MR-Rate-raw \
    --out-csv           <data_ws>/r2v_manifest/manifest_shards_native.csv \
    --out-report-index-csv <data_ws>/r2v_manifest/report_index_shards_native.csv \
    --verify-sample 20
```

`--out-report-index-csv` is not optional in practice — training needs it.

### 6.2 Train the adapter

```bash
python -m mrrate_r2v.cli.train_r2v \
    --manifest      <data_ws>/r2v_manifest/manifest_shards_native.csv \
    --report-index  <data_ws>/r2v_manifest/report_index_shards_native.csv \
    --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \
    --vae-checkpoint  <ws>/models/autoencoder_v1.pt \
    --text-encoder radbert --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \
    --max-report-tokens 512 --report-sections findings impression \
    --geometry-mode per_modality_plane --split train \
    --out <ws>/runs/r2v_adapter_v1 \
    --epochs 1 --batch-size 1 --lr 1e-5 \
    --report-dropout-probability 0.10 --modality-dropout-probability 0.10 \
    --scale-factor auto --save-format adapter --num-workers 8 --device cuda
```

On the cluster:

```bash
sbatch slurm/06_train_r2v.sbatch                                     # 4-step smoke run first
sbatch --export=ALL,R2V_MAX_STEPS=0,R2V_EPOCHS=1 --time=24:00:00 \
       slurm/06_train_r2v.sbatch                                     # then the real run
```

**The arguments that actually change the result** (everything else is in
[`cli/README.md`](cli/README.md)):

| argument | default | why you would change it |
|---|---|---|
| `--lr` | `1e-5` | NVIDIA's own value. The adapter is small and starts at identity, so this is conservative — raising it is the first experiment to try. |
| `--batch-size` / `--grad-accumulation-steps` | `1` / `1` | effective batch. Memory-bound: one 256³ volume per sample. |
| `--report-dropout-probability` | `0.10` | how much unconditional signal the model sees. Too low → guidance has nothing to amplify; too high → wasted capacity. |
| `--cross-attention-dim` | `512` | adapter capacity. Bigger = more parameters at every conditioned level. |
| `--conditioning-levels` | `attention_levels` | where the report enters. Adding low levels means high-resolution attention → expensive. |
| `--report-sections` | `findings impression` | what the model is told. Adding `clinical_information` leaks indication, not observation. |
| `--max-report-tokens` | `512` | truncation. RadBERT's usable budget is `max_position_embeddings − 2`; exceeding it is refused up front. |
| `--scale-factor` | `auto` | **leave it.** `recompute` rescales the latent space out from under a denoiser that is frozen and cannot follow. |
| `--seed` | `0` | also seeds the dedicated report-dropout generator, so a resumed run reproduces the same drops. |

**Outputs:** `<out>/adapter_last.pt` (~8M params, plus optimizer/scheduler/scaler/RNG state, the base
checkpoint's sha256, and the text-encoder identity) and `<out>/train_summary.json`. Resume with
`--resume <adapter.pt>`.

### 6.3 Look at one sample (quickest way to know if training did anything)

```bash
python -m mrrate_r2v.cli.generate_r2v \
    --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \
    --vae-checkpoint  <ws>/models/autoencoder_v1.pt \
    --adapter <ws>/runs/r2v_adapter_v1/adapter_last.pt \
    --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \
    --report "Findings: A 12 mm peripherally enhancing lesion in the right frontal lobe with
              surrounding vasogenic oedema. Impression: Solitary enhancing mass." \
    --modality T1w --dim 256 256 256 --spacing 1 1 1 \
    --report-guidance-scale 4 --seed 1234 \
    --out <ws>/samples/case001.nii.gz
```

or `sbatch slurm/07_generate_r2v.sbatch <ws>/runs/r2v_adapter_v1/adapter_last.pt`.

**The inference knobs:**

| argument | default | effect |
|---|---|---|
| `--report-guidance-scale` | `4.0` | how hard the report is pushed. `0` = NVIDIA's original behaviour exactly; `1` = the plain conditioned prediction; higher = stronger and eventually over-saturated. **The first thing to sweep.** |
| `--modality-guidance-scale` | `10.0` | NVIDIA's own `cfg_guidance_scale`. `0` leaves the modality conditioned and guides the report only. |
| `--num-inference-steps` | `30` | NVIDIA's default; quality vs wall-clock |
| `--dim` / `--spacing` | the `(--modality, --plane)` bucket's own grid | the output grid. Spacing is a **real conditioning input**, so asking for a bucket's own FOV is what makes generated and real populations comparable. `--dim` must be divisible by 4. Unknown (modality, plane) → NVIDIA's 256³ @ 1 mm. |
| `--modality` | `T1w` | the class label the frozen model is conditioned on, and the `[MODALITY]` marker under a `*_meta` format |
| `--plane` | `AXIAL` | selects the geometry bucket and the `[PLANE]` marker |
| `--seed` | `1234` | reproducibility |
| `--latent-only` | off | skip the VAE decode — a cheap check that the diffusion loop runs |

A sanity ladder that isolates where a problem is: `--report-guidance-scale 0` should reproduce
NVIDIA's unconditional output (if *that* looks wrong, the base/VAE loading is wrong, not your
adapter); then `1`, then `4`, on the same seed and report.

### 6.4 Score a checkpoint properly

```bash
# 1. a frozen ground-truth cohort (once per experiment set)
python -m mrrate_r2v.cli.preprocess \
    --manifest-csv <data_ws>/r2v_manifest/manifest_shards_native.csv \
    --report-index-csv <data_ws>/r2v_manifest/report_index_shards_native.csv \
    --split test --sequences T1w T2w FLAIR SWI --n-per-bucket 200 \
    --report-format findings_impression_meta \
    --out <ws>/cohorts/test_v1
#   ^ must be one of the formats the adapter was trained on, or step 2 refuses the cohort. A cohort
#     stores already-composed text, so the format cannot be changed after the fact.

# 2. one volume per case, from that case's own report and its own modality
python -m mrrate_r2v.cli.predict_r2v \
    --cohort <ws>/cohorts/test_v1 \
    --checkpoint      <ws>/runs/r2v_adapter_v1/adapter_last.pt \
    --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \
    --vae-checkpoint  <ws>/models/autoencoder_v1.pt \
    --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \
    --out <ws>/predictions/r2v_v1 --limit 20        # drop --limit for the real run

# 3. metrics
python -m mrrate_r2v.cli.evaluate --task report2volume \
    --gt <ws>/cohorts/test_v1 --pred <ws>/predictions/r2v_v1 \
    --out <ws>/results/r2v_v1 \
    --distribution-metrics --medicalnet-checkpoint <ws>/pretrained/medicalnet/*.pth \
    --workers $SLURM_CPUS_PER_TASK --save-figures 3
```

Then, **before** reading any number as a result:

```bash
python3 slurm/check_run.py --cohort <ws>/cohorts/test_v1 \
    --pred-vae <ws>/predictions/vae_v1 --results-recon <ws>/results/vae_v1
```

It checks a finished run against [`slurm/SUCCESS_CRITERIA.md`](../slurm/SUCCESS_CRITERIA.md) and
exits 0 only if every applicable check passes. Checks whose inputs are absent are reported as
**SKIP, not PASS** — "not run" must never read as "fine".

Useful baselines to score against the same cohort: `predict_vae` (`--task reconstruction`) is the
upper bound any latent-space method can reach; `predict_generation` (`--task generation`) is
NVIDIA's model with no report at all. If report-conditioned FID is not better than that, the report
is not contributing.

---

## 7. Things that will bite you

| | |
|---|---|
| **Axis order** | internal `(D,H,W)`, external `(X,Y,Z)`. Convert only via `geometry.dhw_to_xyz`/`xyz_to_dhw`. Silent for isotropic cubes, scrambling otherwise. |
| **Two divisors** | `4` for sampling (output ÷ latent), `16` for VAE-encode padding. Swapping them yields a valid file 4× too small on every axis. |
| **`--series-selection`** | `one_per_study_per_bucket` is the default and must stay it. `one_per_study_per_sequence` silently collapses the *planes*. |
| **`--export=NONE`** | on every sbatch script — `VAR=x sbatch ...` does not reach the job. Use `--export=ALL,VAR=x`. Every Helma partition defaults to a **10-minute** walltime without `--time`. |
| **`drop_last` and tiny buckets** | a bucket smaller than `batch_size` keeps one short batch instead of vanishing. Before the fix, `drop_last=True` removed SWI CORONAL and SWI SAGITTAL from every epoch and nothing failed — see §4.2. |
| **W&B needs the proxy** | compute nodes have no direct route off-site. `slurm/_common.sh:setup_proxy` exports `http(s)_proxy` **and** passes them into the container. Without it `wandb.init(mode="online")` fails and `WandbRun` degrades to a silent no-op, so the run looks like "W&B just isn't logging". `R2V_WANDB=online` probes `api.wandb.ai` before any GPU work; `R2V_NO_PROXY=1` disables the proxy. |
| **`report_format` at inference** | a `*_meta` format's `[MODALITY]/[PLANE]/[SPACING]` prefix is a trained token sequence. `cli.generate_r2v` prepends it for free-form text and refuses a cohort composed under a format the adapter never saw. |
| **Privacy** | `study_uid`/`series_id` appear only in a cohort's `index.csv`. Everything else on disk uses `case_id` and `cohort_id`. Don't log identifiers verbatim or write them into results. |
| **Disk** | `/hnvme` has a **file-count** quota (61k soft), not a space quota — hence one archive per bucket instead of one file per volume. |
| **Fréchet distance** | computed in `eval/distribution.py`, not via `monai.metrics.FIDMetric` (that path passes a `disp=` kwarg scipy removed in 1.17). Don't reintroduce the monai call. |

---

## 8. Testing

```bash
cd contrastive-pretraining
python -m pytest                                   # everything, with coverage
python -m pytest tests/test_r2v_training.py -v --no-cov
```

All R2V tests run on CPU with synthetic fixtures — no checkpoints, no real data, seconds.

Two files are **load-bearing invariants, not ordinary unit tests** — do not weaken them to make a
change pass:

- `tests/test_cohort_contract.py` — that `cohort_id` changes when the seed, FOV, normalizer or case
  list changes, and that a prediction set from a different cohort is refused.
- `tests/test_eval_tasks_and_runner.py` — that `--task generation` never produces a voxelwise
  metric, that all tasks share one result layout, and that a shape-mismatched prediction is excluded
  rather than resized.
