# `mrrate_r2v` — report-to-volume for brain MRI

Generate a 3D brain MRI volume from a radiology report, and measure how good it is.

This file is the map: what each piece is, where it lives, and how to run training and
inference. For *why* it's designed this way, see [`docs/R2V.md`](../../docs/R2V.md) (the
experiment guide) and [`DEVELOPER_NOTES.md`](DEVELOPER_NOTES.md) (implementation-level detail,
edge cases, and history — not needed to just use the pipeline).

No installation needed — it's importable from `contrastive-pretraining/` as a plain package.

---

## 1. What it is

**NVIDIA's `NV-Generate-MR-Brain` diffusion model, frozen, with a small trainable
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

Only **4.28%** of parameters (~8M of ~189M) are trainable. The adapters start
zero-initialized, so the conditioned model is numerically identical to NVIDIA's at step 0 —
training starts from a working brain-MRI generator and just adds the report on top.

## 2. The pipeline, in three steps

```
stage 0   build_manifest    →  manifest.csv (+ report_index.csv)     once per storage location
 train    train_r2v         →  adapter_last.pt                       --split train
 test     evaluate --task   →  metrics.json                          --split test
```

`train_r2v` and `evaluate` share one dataset-building path, so they can't preprocess
differently. Evaluation streams generate→score one case at a time — nothing intermediate is
written to disk (no cohort, no prediction set, no `.npy`).

## 3. Module map

| module | what it is |
|---|---|
| [`text.py`](text.py) | the text-encoder seam: `TextEmbedder` protocol, `RadBertEmbedder`, `MockTextEmbedder` |
| [`conditioning.py`](conditioning.py) | modality class ids, modality/report dropout, classifier-free guidance |
| [`training.py`](training.py) | `MRRateAdapterTrainer` — the training loop |
| [`sampling.py`](sampling.py) | `ReportToVolumeSampler` — the inference loop |

| package | guide | what it does |
|---|---|---|
| [`data/`](data/) | [README](data/README.md) | manifest → archive reads → preprocessing → `(volume, report)` pairs |
| [`models/`](models/) | [README](models/README.md) | the frozen NVIDIA nets, the report adapter, what is trainable |
| [`cli/`](cli/) | [README](cli/README.md) | the entry points and their flags |
| [`eval/`](eval/) | [README](eval/README.md) | one runner, three tasks, one result layout |
| [`textenc/`](textenc/) | [README](textenc/README.md) | turning report text into conditioning tensors |

## 4. One sample, at a glance

[`MRReportToVolumeDataset.__getitem__`](data/dataset.py) turns one (study, series) pair into:

```python
sample = {
  "image":             tensor [1, X, Y, Z],   # preprocessed, model-ready, no further reshaping
  "report_text":       "Findings: ...  Impression: ...",
  "modality":          "T1w",                 # becomes the class label
  "acquisition_plane": "AXIAL",
  "target_spacing_mm": tensor([1., 1., 1.]),  # a real conditioning input, not just metadata
  "target_shape":      tensor([256, 256, 256]),
  "study_key", "series_key":                  # identifiers — never log verbatim
}
```

One sample = one report paired with one real volume; a study with 6 series produces 6 samples
during training.

> ⚠️ **Axis order is the single most bug-prone thing in this package.** Internal geometry is
> `(D,H,W)`; everything crossing the package boundary is `(X,Y,Z)`. Convert only via
> `geometry.dhw_to_xyz`/`xyz_to_dhw` — see [`DEVELOPER_NOTES.md`](DEVELOPER_NOTES.md) for why a
> skipped conversion can be silent.

Batching uses `collate_fn_r2v` + (for `geometry_mode="per_modality_plane"`)
`GeometryBucketBatchSampler`, which only ever draws a batch from one (modality, plane) bucket at
a time — that's what makes `batch_size > 1` legal when bucket shapes differ.

## 5. How report conditioning works

1. A **frozen text encoder** turns the report into `[B, n, D]` token embeddings + an attention
   mask. Which encoder, and whether `n` is the token count or 1, is set by `--conditioning` —
   see [`textenc/README.md`](textenc/README.md).
2. **`context_proj`** (trainable) projects to the adapters' width. This is the only place the
   text encoder's width appears, so swapping encoders needs no other code change.
3. **Five cross-attention adapters**, one per conditioned UNet level plus the bottleneck: every
   voxel asks which words of the report are relevant to it, and adds the answer to itself.
4. A **learned "no report" token** replaces the report for 10% of training samples, so the model
   learns the difference a report makes — which classifier-free guidance amplifies at inference.

Full detail (including the guidance formula and the exact training step) is in
[`DEVELOPER_NOTES.md`](DEVELOPER_NOTES.md) and [`models/README.md`](models/README.md).

## 6. How to run it

Paths below use `<ws>` for the workspace and `<data_ws>` for the data workspace. Results are
tens of MB per volume — **workspace only, never git or `$HOME`.**

### 6.0 First: does it wire up at all? (CPU, seconds, no data)

```bash
cd contrastive-pretraining
python -m mrrate_r2v.cli.train_r2v --dry-run --max-steps 2 --out /tmp/r2v_dryrun --text-encoder mock
```

Synthetic latents, fabricated reports, a two-level UNet — the whole trainer path with no
manifest, VAE, checkpoint, or GPU. Try this first if anything seems broken.

### 6.1 Build the manifest (once per storage location)

```bash
python -m mrrate_r2v.cli.build_manifest \
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
sbatch slurm/train_r2v.sbatch                                     # 4-step smoke run first
sbatch --export=ALL,R2V_MAX_STEPS=0,R2V_EPOCHS=1 --time=24:00:00 \
       slurm/train_r2v.sbatch                                     # then the real run
```

The handful of flags most worth tuning (everything else is in
[`cli/README.md`](cli/README.md)): `--lr` (default `1e-5`, conservative — raising it is often
the first experiment to try), `--batch-size`/`--grad-accumulation-steps`, and
`--report-dropout-probability`. Leave `--scale-factor` at `auto`.

**Outputs:** `<out>/adapter_last.pt` and `<out>/train_summary.json`. Resume with
`--resume <adapter.pt>`.

### 6.3 Look at one sample

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

or `sbatch slurm/generate_r2v.sbatch <ws>/runs/r2v_adapter_v1/adapter_last.pt`.

`--report-guidance-scale` is the first thing worth sweeping (`0` reproduces NVIDIA's original,
unconditional output — a useful sanity check that base/VAE loading is correct before blaming
your adapter).

### 6.4 Score a checkpoint

One command builds the dataset, generates one volume per case from that case's own report, and
scores it — no cohort to build first, no prediction set to hand off.

```bash
python -m mrrate_r2v.cli.evaluate --task report2volume \
    --manifest     <data_ws>/r2v_manifest/manifest_shards_native.csv \
    --report-index <data_ws>/r2v_manifest/report_index_shards_native.csv \
    --split test \
    --checkpoint      <ws>/runs/r2v_adapter_v1/adapter_last.pt \
    --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \
    --vae-checkpoint  <ws>/models/autoencoder_v1.pt \
    --report-format findings_impression_meta \
    --out <ws>/results/report2volume_r2v_v1 \
    --n-per-bucket 8        # drop this for the entire test split
```

`--report-format` must be one the adapter was trained on, or the run is refused.

| Scale | Cases | Wall clock (1×H200) |
|---|---|---|
| `--n-per-bucket 8` | 80 | ~10 min — wiring check; no metric means anything yet |
| `--n-per-bucket 200` | 2,000 | ~4 h |
| unset (**default**) | 29,027 | ~60 h — the entire test split |

Two useful baselines, same command shape:

```bash
# the upper bound any latent-space method can reach
python -m mrrate_r2v.cli.evaluate --task reconstruction --vae-checkpoint ... --out .../recon
# NVIDIA's model with no report at all -- if report-conditioned FID isn't better, the report
# isn't contributing
python -m mrrate_r2v.cli.evaluate --task generation --base-checkpoint ... --vae-checkpoint ... \
    --out .../generation
```

See [`eval/README.md`](eval/README.md) for what each metric means.

## 7. Things that will bite you

| | |
|---|---|
| **Axis order** | internal `(D,H,W)`, external `(X,Y,Z)`. Convert only via `geometry.dhw_to_xyz`/`xyz_to_dhw`. |
| **`--series-selection`** | `one_per_study_per_bucket` is the default for evaluation and must stay it — other modes silently collapse planes or modalities. |
| **`--export=NONE`** | on every sbatch script — `VAR=x sbatch ...` does not reach the job. Use `--export=ALL,VAR=x`. Every Helma partition defaults to a **10-minute** walltime without `--time`. |
| **W&B needs the proxy** | compute nodes have no direct route off-site. `slurm/_common.sh:setup_proxy` handles it; without it, `wandb` silently degrades to a no-op instead of erroring. |
| **`report_format` at inference** | a `*_meta` format's prefix is a trained token sequence — `cli.generate_r2v` prepends it for you, and `cli.evaluate` refuses a format the adapter never trained on. |
| **Privacy** | `study_uid`/`series_id` are identifiers — don't log them verbatim or write them into results. |

More of these, with the full reasoning, are in [`DEVELOPER_NOTES.md`](DEVELOPER_NOTES.md).

## 8. Testing

```bash
cd contrastive-pretraining
python -m pytest                                   # everything, with coverage
```

There is currently no r2v-specific automated test suite in this repo (the module docstrings and
`docs/R2V.md` reference test files from an earlier state of the package that no longer exist) —
treat that as a gap to fill, not as a passing suite to trust.
