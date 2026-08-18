# `mrrate_r2v.cli` — the entry points

Everything you run is here.

Full pipeline context: [`../README.md`](../README.md) and
[`docs/R2V.md`](../../../docs/R2V.md). Flag-choice rationale, measured evidence, and the history
of an earlier (now-deleted) 5-stage design live in
[`../DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md).

All commands run from `contrastive-pretraining/`:

```bash
cd contrastive-pretraining
python -m mrrate_r2v.cli.<script> ...
```

---

## The five entry points

```
   ┌─ stage 0 ─────────────────────────────────────────────── once per storage location
   │  build_manifest          →  manifest.csv + report_index.csv   (an index; no pixels)
   │
   ├─ train ──────────────────────────────────────────────────  report-to-volume only
   │  train_r2v               →  adapter_last.pt
   │
   ├─ test ───────────────────────────────────────────────────  per model
   │  evaluate --task ...     →  metrics.json  (the official VLM3D challenge metrics)
   │  generate_r2v            →  a .nii.gz     free-form, one report, for eyeballing
   │
   └─ utility
      download_text_encoders  →  stages a pretrained text-encoder checkpoint
```

`train_r2v` and `evaluate` share one dataset-building path (`build_r2v_dataset`), so they can't
preprocess differently. Nothing is written to disk in between (no cohort, no prediction set) —
evaluation streams generate→score one case at a time.

**Which scripts need which stack:** `build_manifest` needs pyarrow but not torch; `train_r2v`,
`generate_r2v`, and `evaluate --task report2volume` need `transformers` for the text encoder —
on the cluster that means the `nvidia+redbert.sif` image, not the base one.

---

## `build_manifest`

One CSV row per eligible (study, series) pair — what exists and where. Build it once; every run
afterwards reads it.

```bash
python -m mrrate_r2v.cli.build_manifest \
    --shards-root <data_ws>/MR-Rate-raw \
    --out-csv <data_ws>/r2v_manifest/manifest_shards_native.csv \
    --out-report-index-csv <data_ws>/r2v_manifest/report_index_shards_native.csv \
    --verify-sample 20
```

| flag | default | meaning |
|---|---|---|
| `--out-csv` | — | output manifest (required unless `--dry-run`) |
| `--shards-root` | — | directory of `shard-*.tar` + `series.parquet` |
| `--out-report-index-csv` | — | also write the index `ShardReportStore` needs. **Pass it** — training needs it. |
| `--splits` | `train val test` | which splits to include |
| `--excluded-modalities` | MRA + derived/localizer set | pass with no values to exclude nothing |
| `--verify-sample` | `20` | resolve N random archive rows *for real* afterwards — **always leave this on** |
| `--dry-run` | off | report what would be built, write nothing |

---

## `train_r2v`

Trains the report-conditioning adapter on a frozen NVIDIA denoiser.

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

# multi-GPU, torchrun
torchrun --nproc_per_node=4 -m mrrate_r2v.cli.train_r2v --num-gpus 4 ...  # same flags
```

Series selection is fixed at `"all"` — every eligible series is a training sample (a study's
report is paired with each of its ~7 series). One batch is always one `(modality, plane)`
bucket, in either geometry mode.

**Data**

| flag | default | meaning |
|---|---|---|
| `--manifest` / `--report-index` | required¹ | from `build_manifest` |
| `--split` | `train` | |
| `--report-sections` | `findings impression` | ignored when `--report-format` is given |
| `--report-format` | the `--conditioning` configuration's own | one format, or several comma-separated to sample per training sample |
| `--geometry-mode` | `per_modality_plane` | per-bucket FOVs, or `fixed` for one shared grid |
| `--bucket-order` | `interleave` | `interleave` spaces each bucket's batches evenly across the epoch; `shuffle` is a flat shuffle |
| `--num-workers` | `4` | DataLoader workers |

¹ required unless `--dry-run`.

**Model**

| flag | default | meaning |
|---|---|---|
| `--base-checkpoint` | required¹ | NVIDIA diffusion UNet — frozen, loaded strictly |
| `--vae-checkpoint` | required¹ | NVIDIA autoencoder — frozen |
| `--cross-attention-dim` | `512` | the adapters' width |
| `--conditioning-levels` | base model's `attention_levels` | which UNet levels get a report adapter — see [`models/README.md`](../models/README.md) |
| `--no-condition-mid` | (mid on) | drop the bottleneck adapter |

**Report conditioning** — normally the only flag you set

| flag | default | meaning |
|---|---|---|
| `--conditioning` | — | one of five named configurations (encoder + pooling + format together) — see [`textenc/README.md`](../textenc/README.md) Part 4 |
| `--max-report-tokens` | `512` | truncation length |
| `--text-pooling` | the configuration's own | pooled configurations only |
| `--text-checkpoint` | — | override the staged snapshot directory |
| `--text-encoder` | `radbert` | pre-`--conditioning` path; `radbert` \| `mock` (for `--dry-run`); ignored when `--conditioning` is given |

**Optimisation**

| flag | default | meaning |
|---|---|---|
| `--out` | *required* | output directory for checkpoints |
| `--epochs` / `--max-steps` | `1` / — | `--max-steps` stops early regardless of epochs (smoke runs) |
| `--batch-size` | `1` | > 1 needs the bucket batch sampler, already wired in |
| `--lr` | `1e-5` | NVIDIA's own value — raising it is often the first experiment worth trying |
| `--grad-accumulation-steps` | `1` | effective batch = this × `--batch-size` |
| `--save-format` | `adapter` | `adapter` (~8M) \| `full` (NVIDIA's layout) \| `both` |
| `--resume` | — | adapter checkpoint to resume from |
| `--scale-factor` | `auto` | **leave it** — `recompute` rescales a latent space the frozen denoiser can't follow |
| `--num-gpus` | `1` | with `torchrun` |
| `--report-dropout-probability` | `0.1` | fraction of samples whose report is replaced by the null token — what makes CFG possible at inference |

**Writes:** `adapter_last.pt` (+ `adapter_step*.pt` if `--save-every-steps`),
`train_summary.json`. **Check `skipped_steps` before reading any result** — a nonzero count
means a non-finite-gradient batch was dropped rather than crashing the run; see
[`DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) for why that can silently zero out one bucket's
training.

---

## `evaluate`

One command builds the dataset, generates, and scores against the official VLM3D challenge
metrics — no cohort or prediction directory to build first.

```bash
python -m mrrate_r2v.cli.evaluate --task report2volume \
    --manifest <data_ws>/r2v_manifest/manifest_shards_native.csv \
    --report-index <data_ws>/r2v_manifest/report_index_shards_native.csv \
    --split test \
    --checkpoint <ws>/runs/r2v_adapter_v1/adapter_last.pt \
    --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \
    --vae-checkpoint  <ws>/models/autoencoder_v1.pt \
    --out <ws>/results/r2v_v1 --n-per-bucket 8
```

| flag | default | meaning |
|---|---|---|
| `--task` | *required* | `report2volume` (trained adapter) \| `reconstruction` (frozen VAE round-trip) \| `generation` (frozen base UNet, report-blind) |
| `--manifest` / `--report-index` / `--split` | *required* / *required* / `test` | same dataset flags as `train_r2v` |
| `--n-per-bucket` | unset (entire split) | deterministic prefix per (modality, plane) bucket |
| `--checkpoint` / `--base-checkpoint` / `--vae-checkpoint` | task-dependent | which are required depends on `--task` |
| `--seed` | `42` | seeds sampler noise only |
| `--wandb-mode` / `--wandb-project` / ... | `disabled` | one metrics table plus example ground-truth-vs-generated panels |

**Writes:** one `<out>/metrics.json`. See [`eval/README.md`](../eval/README.md) for what each key
means.

---

## `generate_r2v`

Same model assembly as `evaluate --task report2volume`, driven by a report string you type. Use
it to eyeball what a checkpoint produces.

```bash
python -m mrrate_r2v.cli.generate_r2v \
    --base-checkpoint ... --vae-checkpoint ... --adapter <run>/adapter_last.pt \
    --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \
    --report "Findings: 12 mm enhancing lesion in the right frontal lobe." \
    --modality T1w --plane AXIAL \
    --report-guidance-scale 4 --out <ws>/samples/case001.nii.gz
```

| flag | default | meaning |
|---|---|---|
| `--report` / `--report-file` | — | the conditioning text |
| `--out` | — | output `.nii.gz` |
| `--modality` / `--plane` | `T1w` / `AXIAL` | class label, geometry bucket |
| `--dim` / `--spacing` | the `(--modality, --plane)` bucket's own grid | output grid, X Y Z. `--dim` must be divisible by 4. |
| `--report-guidance-scale` | `4.0` | `0` reproduces NVIDIA's unconditional output exactly — a useful sanity check |
| `--modality-guidance-scale` | `10.0` | NVIDIA's own `cfg_guidance_scale` |
| `--latent-only` | off | skip the VAE decode (cheap check the diffusion loop runs) |
| `--seed` | `1234` | |

Writes the volume plus a `*.manifest.json` recording seed, geometry, guidance scales, and every
checkpoint's sha256. The adapter checkpoint records its own text encoder and adapter geometry,
so none of that needs re-specifying.

---

## `download_text_encoders`

```bash
python -m mrrate_r2v.cli.download_text_encoders --list   # what's staged
python -m mrrate_r2v.cli.download_text_encoders --all    # fetch everything (~4 GB)
```

See [`textenc/README.md`](../textenc/README.md) for the encoder table.

---

## Slurm

[`slurm/_common.sh`](../../slurm/_common.sh) holds every path and the apptainer invocation; the
other scripts source it by absolute path via `$R2V_REPO`.

| script | wraps |
|---|---|
| `train_r2v.sbatch` | `cli.train_r2v` |
| `evaluate.sbatch` | `cli.evaluate` |
| `generate_r2v.sbatch` | `cli.generate_r2v` (adapter path positional) |
| `train_conditioning.sbatch` | `cli.train_r2v` with one of the named `--conditioning` configurations |
| `submission_smoke.sbatch` | a local Docker-free smoke test of the `submission/` container |

```bash
sbatch slurm/train_r2v.sbatch                                     # 4-step smoke run
sbatch --export=ALL,R2V_MAX_STEPS=0,R2V_EPOCHS=1 --time=24:00:00 \
       slurm/train_r2v.sbatch                                     # full run
sbatch slurm/generate_r2v.sbatch <ws>/runs/<tag>/adapter_last.pt
```

> ⚠️ `#SBATCH --export=NONE` is set on every script. A plain `VAR=x sbatch ...` does **not**
> reach the job — overrides must go through `--export=ALL,VAR=x`. Every Helma partition defaults
> to a **10-minute** walltime if `--time` is omitted.

---

## Common failures, and what they mean

| message | cause |
|---|---|
| `adapter was trained against base checkpoint <sha> but the loaded base is <sha>` | wrong `--base-checkpoint`. `--allow-base-mismatch` downgrades it to a warning, but the adapter's zero-point is then a different frozen model. |
| `adapter checkpoint does not match this model: missing=... unexpected=...` | `--cross-attention-dim`/`--conditioning-levels`/`--condition-mid` differ from what the checkpoint was trained with. |
| `N base-model parameters are trainable` | the freeze gate caught a leak — you would have been fine-tuning NVIDIA's denoiser. |
| `output shape (...) is not divisible by the latent divisor 4` | `--dim` must be a multiple of 4. |
| `collate_fn_r2v got a batch with mismatched image shapes` | `batch_size > 1` under `per_modality_plane` without the bucket batch sampler. |
| `--text-checkpoint is required for --text-encoder radbert` | pass the local RadBERT snapshot dir, or use `--text-encoder mock` for a dry run. |
| a bare `FileNotFoundError` from inside NVIDIA's loader | a relative checkpoint path — always pass absolute. |

---

## Testing

There is currently no automated test suite for these CLIs (see the top-level
[`README.md`](../README.md) §8) — treat any test-file references you see elsewhere in the
codebase as aspirational, not passing.
