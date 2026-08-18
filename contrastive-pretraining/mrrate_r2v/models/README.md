# `mrrate_r2v.models` — the networks

A frozen NVIDIA generative model, plus the small trainable piece that makes it listen to a
radiology report.

Full pipeline context: [`../README.md`](../README.md) and
[`docs/R2V.md`](../../../docs/R2V.md). Deep dives (the silent-divisor trap, the cuDNN SDPA bug,
the strict checkpoint loader's internals) are in
[`../DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md).

---

## Three modules

| file | what it is |
|---|---|
| [`nvidia.py`](nvidia.py) | the **only** place NVIDIA-authored code is used. Loads their autoencoder and diffusion UNet, exposes their config paths. |
| [`report_conditioned_unet.py`](report_conditioned_unet.py) | that same diffusion UNet, structurally unchanged, **plus** report cross-attention adapters. |
| [`adapter.py`](adapter.py) | what is trainable (asserted, not assumed) and the adapter checkpoint format. |

## The mental model

There are **three** neural networks in play, and only a sliver of one of them is trained.

```
   report text ──► RadBERT ──────────────────────────► [B, L, 768]   FROZEN  (text.py)
                                                            │
                                                     context_proj    TRAINABLE  ~4.2M
                                                            │
                                                       [B, L, 512]
                                                            │
  real volume ──► VAE encoder ──► latent ──► + noise ──► DIFFUSION UNET
                     FROZEN                              ├─ NVIDIA's blocks   FROZEN  180M
                                                         └─ 5 cross-attn adapters  TRAINABLE  ~3.9M
                                                                    │
                                                              predicted velocity
```

Counts are logged at the start of every run and asserted before the first optimizer step:

```
trainable   8,080,000 params in N tensors (4.28% of 188,580,868); frozen 180,500,868
```

---

## 1. `nvidia.py` — the seam

Loads the pretrained autoencoder and diffusion UNet, and exposes the JSON configs
([`nvidia_configs/`](nvidia_configs/)) that describe their exact architecture, so nothing
downstream restates those numbers. The network classes themselves are `monai` classes; this
module adds the loading glue on top. A few small functions (and the config JSON) are copied,
not reimplemented, from NVIDIA's NV-Generate-CTMR release — see the banner comment near the top
of `nvidia.py`.

```python
from mrrate_r2v.models.nvidia import (
    DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_NETWORK_CONFIG,   # NVIDIA's own configs
    load_autoencoder,          # -> (autoencoder, cfg_args, required_divisor)
    load_autoencoder_and_unet, # -> (autoencoder, unet, scale_factor, cfg_args)
    define_instance, load_config, set_random_seed,
)
```

`nvidia_unet_kwargs()` in [`report_conditioned_unet.py`](report_conditioned_unet.py) parses
NVIDIA's own network config and hands the result to the constructor — nothing in this package
hardcodes the UNet's architecture numbers.

⚠️ **Two different divisors matter here** (padding for VAE-encode vs. the sampling
output/latent ratio) — mixing them up produces a *valid* volume that's silently 4× too small.
See [`DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) for which is which.

⚠️ **Pass absolute checkpoint paths.** NVIDIA's env config stores paths relative to the current
working directory; a relative path here fails with a bare `FileNotFoundError`.

---

## 2. `report_conditioned_unet.py` — how the report gets in

Cross-attention adapters are added *alongside* NVIDIA's pretrained blocks (not by turning on
MONAI's own conditioning switch, which would replace pretrained modules and break weight
loading — see [`DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) for why). Each adapter is a residual
branch, zero-initialized, so at step 0 the conditioned model is numerically identical to
NVIDIA's frozen one.

### `forward` signature

```python
unet(
    x,                    # [B, 4, X/4, Y/4, Z/4]  noisy latent
    timesteps,            # [B]
    context=...,          # [B, L, context_dim]  RAW text-encoder output (or None)
    context_mask=...,     # [B, L] bool   True = real token   (HuggingFace convention)
    context_drop_mask=..., # [B] bool     True = use the null embedding for this sample
    class_labels=...,     # [B] long      NVIDIA modality class id
    spacing_tensor=...,   # [B, 3]        spacing_mm * 1e2
)  # -> [B, 4, X/4, Y/4, Z/4]  predicted velocity
```

`context` is the **raw** encoder output — the projection to the adapters' width happens inside,
which is why swapping text encoders needs no change here.

### The strict loader

`load_pretrained_maisi_weights(model, checkpoint_path)` raises unless the checkpoint and the
model agree completely (every tensor accounted for, no shape mismatches, every shared tensor
bit-equal after loading) — deliberately not NVIDIA's own `strict=False` loader, which would
accept a silent mismatch. Returns a `PretrainedLoadReport` whose `.format()` belongs in every
run log.

---

## 3. `adapter.py` — what is trainable, and what a checkpoint holds

```python
freeze_report = freeze_to_adapter_only(unet, text_embedder)   # sets requires_grad
assert_only_adapter_trainable(unet, optimizer, text_embedder) # PROVES it — raises otherwise
```

The trainer calls both before the first step, so a wrong `requires_grad` can't silently
fine-tune the 180M base model instead of the 8M adapter.

**Adapter checkpoint** (default, `save_adapter_checkpoint`) holds only the ~8M adapter tensors,
plus which base checkpoint (by sha256) and which text encoder it was trained against —
`load_adapter_checkpoint` refuses to load onto a different base unless you say so explicitly,
and `cli.generate_r2v` doesn't need those flags re-specified. A full-UNet-format checkpoint is
also available (`--save-format full|both`) for NVIDIA's own tooling.

---

## Using it from Python

### Build a model and load the pretrained base

```python
from mrrate_r2v.models.report_conditioned_unet import (
    build_report_conditioned_unet, load_pretrained_maisi_weights)
from mrrate_r2v.text import build_text_embedder

embedder = build_text_embedder("radbert", checkpoint="<ws>/pretrained/RadBERT-RoBERTa-4m")

unet = build_report_conditioned_unet(
    context_dim=embedder.output_dim,     # 768 for RadBERT
    cross_attention_dim=512,
    condition_mid=True,
    use_flash_attention=torch.cuda.is_available(),
).to(device)

report = load_pretrained_maisi_weights(unet, "<ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt")
print(report.format())
```

### Resume a trained adapter

```python
from mrrate_r2v.models.adapter import load_adapter_checkpoint, sha256_file

load_adapter_checkpoint("<run>/adapter_last.pt", unet,
                        base_checkpoint_sha256=sha256_file(base_ckpt))   # raises on mismatch
unet.eval()
```

### Constructor arguments worth knowing

| argument | default | meaning | CLI flag |
|---|---|---|---|
| `context_dim` | *required* | width of the incoming report embedding — always `embedder.output_dim` | — (derived) |
| `cross_attention_dim` | `512` | the adapters' internal width | `--cross-attention-dim` |
| `conditioning_levels` | `attention_levels` = `[F,F,T,T]` | which UNet levels get an adapter | `--conditioning-levels 0 0 1 1` |
| `condition_mid` | `True` | also condition the bottleneck | `--no-condition-mid` |
| `network_config` | NVIDIA's `config_network_rflow.json` | where the base geometry is read from | `--network-config` |
| `use_flash_attention` | from config | base-UNet attention kernel; needs CUDA, pass `False` on CPU | — (auto from `--device`) |

---

## Traps worth knowing

- `use_flash_attention=True` requires CUDA. Adapter attention itself always uses
  `scaled_dot_product_attention` (exact, CPU-capable), so adapter logic stays testable on CPU.
- `--conditioning-levels` changes the adapter set, so a checkpoint trained with one value can't
  load into a model built with another.
- `monai` is a floating dependency (`monai>=1.5.0`); `MaskedCrossAttention` checks the MONAI
  attributes it borrows exist at construction, rather than discovering a rename as silently
  wrong numbers.

More traps (and the measured evidence behind each) are in
[`../DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md).
