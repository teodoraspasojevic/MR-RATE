# `mrrate_r2v.models` — the networks

A frozen NVIDIA generative model, plus the small trainable piece that makes it listen to a
radiology report.

Full pipeline context: [`docs/R2V.md`](../../../docs/R2V.md). This file is about the model layer —
read it when you want to know what is trainable, what a checkpoint contains, or how the report gets
into the denoiser.

---

## Three modules

| file | what it is |
|---|---|
| [`nvidia.py`](nvidia.py) | the **only** place NVIDIA-authored code is used. Loads their autoencoder and diffusion UNet, exposes their config paths. |
| [`report_conditioned_unet.py`](report_conditioned_unet.py) | that same diffusion UNet, structurally unchanged, **plus** report cross-attention adapters — and a strict pretrained-weight loader. |
| [`adapter.py`](adapter.py) | what is trainable (asserted, not assumed) and the adapter checkpoint format. |

---

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

Counts, logged at the start of every run and asserted before the first optimizer step:

```
trainable   8,080,000 params in N tensors (4.28% of 188,580,868); frozen 180,500,868
```

---

## 1. `nvidia.py` — the seam

Loads the pretrained autoencoder and diffusion UNet, and exposes the JSON configs
([`nvidia_configs/`](nvidia_configs/)) that describe their exact architecture, so nothing
downstream has to restate those numbers. The network classes themselves are `monai` classes; this
module adds the loading glue on top — config parsing, instantiating a network from that config, and
the unconditional sampling loop the report-blind evaluation baseline uses.

A few small functions here (and the config JSON) are copied, not reimplemented, from NVIDIA's
NV-Generate-CTMR release — see the banner comment near the top of `nvidia.py` for exactly which.

```python
from mrrate_r2v.models.nvidia import (
    DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_NETWORK_CONFIG,   # NVIDIA's own configs
    load_autoencoder,          # -> (autoencoder, cfg_args, required_divisor)
    load_autoencoder_and_unet, # -> (autoencoder, unet, scale_factor, cfg_args)  [NVIDIA's own loader]
    define_instance, load_config, set_random_seed,
)
```

### The architecture is read, never restated

`nvidia_unet_kwargs()` in [`report_conditioned_unet.py`](report_conditioned_unet.py) parses
`nvidia_configs/config_network_rflow.json` and hands the result to the constructor. For
NV-Generate-MR-Brain that is:

```
num_channels      [64, 128, 256, 512]     4 levels
attention_levels  [False, False, True, True]
num_head_channels [0, 0, 32, 32]
num_res_blocks    2      include_spacing_input True      num_class_embeds 128
```

Nothing in this package hardcodes those numbers. Change NVIDIA's config and the model follows.

### ⚠️ Two different divisors — the classic silent bug

| function | value (mr-brain) | what it is for |
|---|---|---|
| `nvidia.required_spatial_divisor(ae, cfg)` | **16** | padding a volume before the VAE **encodes** it (2^n_downsample × `num_splits`) |
| `sampling.official_latent_divisor(num_channels)` | **4** | the output-size ÷ latent-size ratio when **sampling** (`2 ** (len(num_channels) - 2)`) |

Using 16 where 4 belongs produces a volume 4× too small on every axis. The run succeeds, the NIfTI
is valid, and it is wrong. Each call site names which one it means.

### Absolute checkpoint paths are mandatory

NVIDIA's env config stores `model_dir="./models"` relative to the *current working directory*.
`load_autoencoder_and_unet` therefore requires absolute overrides and rewrites
`model_dir`/`model_filename`/`existing_ckpt_filepath` consistently. Passing a relative path fails
with a bare, uninformative `FileNotFoundError`.

---

## 2. `report_conditioned_unet.py` — how the report gets in

### Why not just `with_conditioning=True`?

MONAI's `DiffusionModelUNetMaisi` already has a conditioning switch — but it **replaces** each
attention level's `SpatialAttentionBlock` with a `SpatialTransformer`. That is a different module
tree, so NVIDIA's pretrained weights no longer load, and the "frozen" part would no longer be the
architecture that was actually trained. The constructor rejects it outright:

```python
if maisi_kwargs.get("with_conditioning"):
    raise ValueError("... changes the pretrained module tree ...")
```

Instead, cross-attention is **added alongside** the pretrained blocks as new top-level modules. The
pretrained tree stays byte-identical.

### The four added pieces

| module | shape | role |
|---|---|---|
| `context_proj` | `LayerNorm(768) → Linear(768→2048) → GELU → Linear(2048→512) → LayerNorm(512)` | any text-encoder width → the adapters' width. The input LayerNorm is what makes it indifferent to the encoder's scale. |
| `down_cross_attn` | `ModuleDict{"2","3"}` | one adapter before each conditioned encoder block |
| `up_cross_attn` | `ModuleDict{"0","1"}` | one adapter before each conditioned decoder block |
| `mid_cross_attn` | one adapter | the bottleneck (`condition_mid=True` by default) |
| `null_context` | `Parameter(1, 1, 512)` | the **learned** "no report" token |

Default `conditioning_levels = attention_levels = [F, F, T, T]` → **5 adapters**: the low-resolution
levels, where the pretrained model already spends its attention budget.

### One adapter, in full

[`ReportCrossAttentionAdapter`](report_conditioned_unet.py) is a residual branch over a 3D feature
map:

```
x ──┬──────────────────────────────────────────────────────────► + ──► out
    └► GroupNorm(32) ► Conv1×1×1 (C → inner) ► flatten to (B, voxels, inner)
       ► LayerNorm ► MaskedCrossAttention(Q=voxels, K/V=report tokens)
       ► reshape ► Conv1×1×1 (inner → C)   ← zero-initialised ──────┘
```

Three deliberate properties:

- **`proj_out` is zero-initialised** (`zero_module`, MAISI's own convention). At step 0 the branch
  outputs exactly zero, so the conditioned UNet is *numerically identical* to the pretrained one.
  Fine-tuning starts **at** NVIDIA's model, not near it.
- **Conditioning enters at each block's input**, so the skip connections leaving that level carry
  the report too. Conditioning the output would leave the decoder's skips report-blind.
- **No spatial self-attention, no GEGLU feed-forward.** MONAI's `SpatialTransformer` has both, but
  the pretrained `SpatialAttentionBlock` at that same level already provides them — at O(voxels²)
  cost. Only the cross-attention is new.

`MaskedCrossAttention` subclasses MONAI's `CrossAttentionBlock` and re-implements only `forward`,
around `F.scaled_dot_product_attention`, so a **key-padding mask** can be passed. MONAI's version
takes none — without the mask, padding tokens would join the softmax and a sample's conditioning
would depend on the longest report that happened to share its batch. Same parameters, same
state-dict keys; a startup check (`_BORROWED`) fails loudly if MONAI ever renames the attributes it
relies on.

**A single conditioning token makes this block a no-op.** Softmax over one key is identically 1 for
every query, so `to_q`/`to_k` cannot affect the output and get exactly zero gradient, and the
attention output no longer depends on `x` at all — the adapter degenerates into a report-dependent
per-channel bias, added uniformly at every voxel. Measured: `to_q` gradient 1.2e-12 at `n=1` versus
4.4e-07 at `n≤512`, and 33.8% of adapter parameters inert. Prefer a token-sequence conditioning
(`textenc/README.md` Part 4).

### `sdpa_backend_guard` — why `forward` runs inside a context manager

`F.scaled_dot_product_attention` is one interface over four interchangeable CUDA kernels, and torch
picks one per call. **cuDNN's returns non-finite gradients from a finite forward** at some latent
shapes — measured at 48³ (the `(T2w, CORONAL)` bucket) in bfloat16 and float16, reproducible with
random data and no adapter involved, and confirmed by forcing that backend explicitly. The guard
restricts SDPA to FLASH + EFFICIENT + MATH (everything except cuDNN) and wraps the whole of
`ReportConditionedUNetMaisi.forward`, so trainer, sampler and both CLIs are covered by one change.

MATH stays in the list only as a fallback — `MaskedCrossAttention` passes an `attn_mask` the fused
kernels may refuse, and an empty candidate set raises "No available kernel" rather than falling
back. Guarding the forward is enough: the backend is bound into the autograd node at forward time,
so the backward inherits it. Activation checkpointing would break that; this model does not use it.
Pinned by `tests/test_sdpa_backend_guard.py`.

### The null report and `prepare_context`

`prepare_context(batch_size, context, context_drop_mask, context_mask)` turns a raw report embedding
into what the adapters consume. A sample gets the learned `null_context` when **any** of:

1. `context is None` (fully unconditional call),
2. its bit in `context_drop_mask` is set (training dropout, or a CFG branch),
3. its `context_mask` row has no real tokens (an empty report — otherwise the softmax goes NaN).

A null token repeated `L` times is attention-equivalent to a single null token (identical K/V), so
per-sample dropout is a plain `torch.where` on a same-shaped tensor — and every repeat is unmasked
so that equivalence holds.

**Training dropout and inference CFG use this same code path.** They cannot drift apart.

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

`context` is the **raw** encoder output — the projection happens inside. That is why swapping
RadBERT for a 1024-wide encoder needs no change here.

### The strict loader

`load_pretrained_maisi_weights(model, checkpoint_path)` raises unless **all** of:

- every checkpoint tensor has a home in the model (`unexpected == []`),
- no shape mismatches,
- every parameter left unfilled belongs to the conditioning path,
- every shared tensor is **bit-equal to the file after loading**.

Those four conditions together are what "same architecture plus adapters" *means*. This is
deliberately not NVIDIA's own `load_state_dict(..., strict=False)`, which accepts any subset — a
silent mismatch there is a fine-tune that quietly starts from noise. It returns a
`PretrainedLoadReport` whose `.format()` goes into every run log, so a log can prove the base was
pretrained.

It also handles: EMA sub-dicts (`prefer_ema=True`, raises if absent), wrapper prefixes
(`module.`, `_orig_mod.`, …, stripped only when *every* key carries them), and NVIDIA's habit of
pickling `scale_factor` as a MONAI `MetaTensor` (allow-listed for `weights_only=True` rather than
falling back to arbitrary unpickling).

---

## 3. `adapter.py` — what is trainable, and what a checkpoint holds

### Membership is decided twice and cross-checked

`adapter_parameter_names(model)` walks the actual `context_proj` / `*_cross_attn` submodules and the
`null_context` Parameter **by object identity**, then compares that set against the name-prefix
tuple `CONDITIONING_PREFIXES`. If the two ever disagree, it raises — rather than quietly freezing an
adapter or training a base weight.

### The startup gate

```python
freeze_report = freeze_to_adapter_only(unet, text_embedder)   # sets requires_grad
assert_only_adapter_trainable(unet, optimizer, text_embedder) # PROVES it
```

`assert_only_adapter_trainable` raises unless: every adapter parameter is trainable, **no** base
parameter is, **no** text-encoder parameter is, and the optimizer's parameter set is *exactly* the
adapter set. The trainer calls it before the first step. Without it, one wrong `requires_grad` means
fine-tuning a 180M denoiser at lr 1e-5 instead of training a 8M adapter — and nothing would crash.

Note what freezing does **not** do: it does not wrap the forward pass in `no_grad`. Autograd still
traverses the frozen convolutions, because that is the only route by which a gradient can reach an
adapter sitting in the middle of the network.

### Checkpoint formats

**Adapter checkpoint** (`save_adapter_checkpoint`, format tag `mrrate_r2v_adapter_v1`) — the default,
~8M params instead of ~700 MB:

```python
{
  "format": "mrrate_r2v_adapter_v1",
  "adapter_state_dict": {...},          # ONLY the adapter tensors
  "step", "epoch", "loss",
  "scale_factor",                        # the latent scale it was trained at
  "config": {context_dim, cross_attention_dim, conditioning_levels, condition_mid, training, ...},
  "base_checkpoint": {"path": ..., "sha256": ...},   # WHICH frozen model this adapter's zero-point is
  "text_encoder": {name, checkpoint, output_dim, max_length, ...},
  "optimizer_state_dict", "lr_scheduler_state_dict", "scaler_state_dict",
  "rng_state": {"torch": ..., "dropout_generator": ...},
}
```

The base weights are deliberately **not** duplicated — they are unchanged by construction, so storing
them would make every checkpoint 700 MB of a file the workspace already has, and would let a stale
copy diverge from the real base. Instead the file *identifies* the base by sha256, and
`load_adapter_checkpoint` refuses to load onto a different one unless `allow_base_mismatch=True`.

Because the checkpoint records its own text encoder, context width and adapter geometry,
`cli.generate_r2v` re-specifies none of it on the command line — a mismatch becomes impossible
rather than merely unlikely.

**Full-UNet checkpoint** (`save_full_unet_checkpoint`, `--save-format full|both`) — NVIDIA's own
layout (`epoch`/`loss`/`num_train_timesteps`/`scale_factor`/`unet_state_dict`), so their tooling can
read the result. Larger and redundant for adapter training; provided because their inference path
takes this format.

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

### Constructor arguments

`build_report_conditioned_unet(context_dim, network_config=None, **kwargs)` routes each kwarg to the
right signature. Conditioning arguments (below) go to `ReportConditionedUNetMaisi`; anything else
must be a valid `DiffusionModelUNetMaisi` argument and overrides NVIDIA's config. An unknown name is
a `TypeError`, not a silent no-op. Names the two signatures share (`cross_attention_dim`,
`dropout_cattn`) always mean the conditioning one.

| argument | default | meaning | CLI flag |
|---|---|---|---|
| `context_dim` | *required* | width of the incoming report embedding. Always pass `embedder.output_dim`. | — (derived) |
| `cross_attention_dim` | `512` | the adapters' internal width; what the report is projected to | `--cross-attention-dim` |
| `conditioning_levels` | `attention_levels` = `[F,F,T,T]` | per-level bool: which UNet levels get an adapter | `--conditioning-levels 0 0 1 1` |
| `condition_mid` | `True` | also condition the bottleneck | `--no-condition-mid` |
| `conditioning_num_head_channels` | base model's `num_head_channels[level]` | head width for the adapters. **Must be set explicitly** to condition a level the base model has no attention at (its value there is 0). | — |
| `dropout_cattn` | `0.0` | dropout inside the adapter attention | — |
| `context_hidden_dim` | `4 × cross_attention_dim` = 2048 | hidden width of `context_proj` | `--context-hidden-dim` |
| `network_config` | NVIDIA's `config_network_rflow.json` | where the base geometry is read from | `--network-config` |
| `use_flash_attention` | from config (`True`) | base-UNet attention kernel; **needs CUDA**, pass `False` on CPU | — (auto from `--device`) |

`load_pretrained_maisi_weights(model, path, state_dict_key=None, prefer_ema=False)` — force a
specific sub-dict, or require EMA weights (raises if the checkpoint has none).

---

## Traps worth knowing

- **Adapters have no buffers today**, and `adapter_state_dict` would silently drop them if they
  gained any — so it asserts that stays true. If you add a buffer to an adapter, extend the
  checkpoint format in the same commit.
- **`use_flash_attention=True` requires CUDA.** The adapters themselves always use
  `scaled_dot_product_attention`, which is exact and CPU-capable, so adapter logic stays testable on
  CPU while the base UNet keeps NVIDIA's own setting.
- **`--conditioning-levels` changes the adapter set**, so a checkpoint trained with one value cannot
  be loaded into a model built with another — `load_adapter_checkpoint(strict=True)` will say so
  with the missing/unexpected names.
- **`monai` is a floating dependency** (`monai>=1.5.0`). `MaskedCrossAttention` re-implements one
  method and borrows the rest, so it checks the borrowed attributes exist at construction rather
  than discovering a rename as silently wrong numbers.

---

## Testing

```bash
cd contrastive-pretraining
python -m pytest tests/test_models_report_conditioned_unet.py \
                tests/test_models_pretrained_checkpoint_loading.py \
                tests/test_r2v_conditioning.py -v --no-cov
```

CPU, small synthetic UNets, seconds. No real checkpoints needed. `test_r2v_conditioning.py` is where
the guidance arithmetic is asserted numerically, including that `report_guidance_scale=0` reproduces
NVIDIA's formula exactly.
