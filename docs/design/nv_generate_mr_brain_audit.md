# NV-Generate-MR-Brain / NV-Generate-CTMR Code-Level Audit

Scope: `NV-Generate-CTMR/` as added to this repo (a local copy of NVIDIA's `nvidia-medtech/NV-Generate-CTMR`), focused on the `rflow-mr-brain` variant (backing the `NV-Generate-MR-Brain` HuggingFace weights). No network access, no weight downloads, no cluster jobs were performed. Every claim below is graded **VERIFIED** / **INFERRED** / **ASSUMED** / **UNKNOWN** and cited to `NV-Generate-CTMR/relative/path:Lstart-Lend` (code) or `path.md:section-or-lines` (docs). Line numbers refer to the file states read during this audit (2026-07-28).

## 0. Scope limitation — read this before trusting any architecture claim

**VERIFIED:** `NV-Generate-CTMR` ships **no local model source**. Every VAE, U-Net, ControlNet, and scheduler class is imported from the external `monai` PyPI package (`monai>=1.5.0`, `NV-Generate-CTMR/requirements.txt:6`), which is **not installed in this environment or vendored into the repo**. The classes are referenced only by Python import path inside JSON configs, e.g. `"_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi"` (`NV-Generate-CTMR/configs/config_network_rflow.json:40`).

Consequence: this audit can verify, with high confidence, **every configuration parameter and every call-site contract** (what tensors are constructed, what kwargs are passed to `unet(...)`/`controlnet(...)`, what the training loss target is, what gets saved/loaded) because all of that lives in this repo's `scripts/`. It **cannot** verify internal layer-by-layer wiring (e.g. exactly how `class_labels`/`spacing_tensor` are merged inside `DiffusionModelUNetMaisi.forward()`, or whether that specific subclass's constructor even accepts `cross_attention_dim`) because that code is not present locally. Those points are marked **UNKNOWN** or **INFERRED** below and should be re-verified directly against the installed `monai` source before any implementation work begins — per the task's own instruction to work from local artifacts, not general knowledge.

---

## 1. Model architecture

### 1.1 VAE (`autoencoder_def` → `monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi`)

**VERIFIED** (`NV-Generate-CTMR/configs/config_network_rflow.json:12-38`, identical in `config_network_ddpm.json:12-38` except `num_splits`):

| Param | Value | Note |
|---|---|---|
| `spatial_dims` | 3 | |
| `in_channels`/`out_channels` | 1 | single-channel image |
| `latent_channels` | 4 | |
| `num_channels` | `[64, 128, 256]` | 3 stages → 2 downsampling steps |
| `num_res_blocks` | `[2, 2, 2]` | |
| `attention_levels` | `[false, false, false]` | **no attention anywhere in the VAE** |
| `with_encoder_nonlocal_attn` / `with_decoder_nonlocal_attn` | `false` | confirms no non-local/self-attention blocks either |
| `norm_num_groups` / `norm_eps` | 32 / 1e-6 | GroupNorm |
| `norm_float16` | `true` | |
| `use_convtranspose` | `false` | upsampling is not transposed-conv |
| `num_splits` / `dim_split` | 4 (rflow) / 8 (ddpm) / 1 | tiling knob for the sliding-window/tensor-parallel decode, not an architecture choice |

The VAE is a **purely convolutional residual autoencoder with no attention**, spatially downsampling by **4× per axis** (2 halvings). Cross-checked independently against the latent-size table in `docs/training.md:130-137` / `docs/performance.md:7-20` (e.g. 512×512×512 image → latent `4×128×128×128`; 512/128 = 4). Same VAE architecture (`autoencoder_v1.pt`) is used for CT, general MRI, and MR-brain — see §6.

### 1.2 Latent representation

**VERIFIED:** latent shape = `(4, H/4, W/4, D/4)`, fp32/fp16 float tensor (not quantized), normalized at train/inference time by a scalar `scale_factor = 1/std(first_training_batch_latent)` computed once and frozen into the diffusion checkpoint (`NV-Generate-CTMR/scripts/diff_model_create_training_data... ` — actually computed in `scripts/diff_model_train.py:174-195`, `calculate_scale_factor`). Both the noisy-latent input to the U-Net and the final decode divide/multiply by this stored scalar (`scripts/utils_infer.py:83` `ReconModel.forward`: `self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)`).

### 1.3 Rectified-flow scheduler (`noise_scheduler` → `monai.networks.schedulers.rectified_flow.RFlowScheduler`)

**VERIFIED** (`NV-Generate-CTMR/configs/config_network_rflow.json:134-141`):
```json
{"num_train_timesteps": 1000, "use_discrete_timesteps": false, "use_timestep_transform": true, "sample_method": "uniform", "scale": 1.4}
```
- Training: timesteps sampled via `noise_scheduler.sample_timesteps(images)` (RFlow-specific method, not uniform integer sampling) (`scripts/diff_model_train.py:298-299`); regression target is the **rectified-flow velocity** `images - noise` (linear-path target), not a DDPM `epsilon`/`sample`/`v` target (`scripts/diff_model_train.py:327-329`).
- Inference: `set_timesteps(num_inference_steps, input_img_size_numel=...)` — the RFlow scheduler's step schedule is a **function of the latent volume size**, not fixed regardless of output size (`scripts/diff_model_infer.py:153-157`, `scripts/utils_infer.py:203-207`). `num_inference_steps=30` shipped for all rflow variants (`README.md` table; `configs/config_maisi_diff_model_rflow-mr-brain.json:32`).
- CT's `ddpm-ct` variant uses `monai.networks.schedulers.ddpm.DDPMScheduler` instead (1000 steps, `scaled_linear_beta`, `clip_sample: false`) — a different, older (MAISI-v1) recipe (`configs/config_network_ddpm.json:142-149`).

### 1.4 3D U-Net (`diffusion_unet_def` → `monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi`)

**VERIFIED**, shared by `rflow-ct`/`rflow-mr`/`rflow-mr-brain` via the single `config_network_rflow.json:39-65`:

| Param | Value |
|---|---|
| `in_channels`/`out_channels` | 4 (= `latent_channels`) |
| `num_channels` | `[64, 128, 256, 512]` — 4 resolution levels |
| `attention_levels` | `[false, false, true, true]` — attention **only at the 2 coarsest/deepest levels** (256-ch, 512-ch) |
| `num_head_channels` | `[0, 0, 32, 32]` | matches attention_levels |
| `num_res_blocks` | 2 (per level) |
| `use_flash_attention` | `true` |
| `resblock_updown` | `true` (rflow) vs. `false` (ddpm, `config_network_ddpm.json:68`) |
| `include_fc` | `true` (rflow, both `config_network_rflow.json:64` and mr-brain's own `configs/config_maisi_diff_model_rflow-mr-brain.json`) vs. `false` (ddpm) |
| `num_class_embeds` | 128 | modality/class-embedding vocabulary size |
| `include_top_region_index_input` / `include_bottom_region_index_input` | `@include_body_region` = **`false`** for the whole rflow family (`config_network_rflow.json:5,59-60`) — only `ddpm-ct` sets this `true` (`config_network_ddpm.json:5`) |
| `include_spacing_input` | `true` (both rflow and ddpm) |

`resblock_updown` and `include_fc` are the only two structural differences between the MAISI-v2 (rflow) and MAISI-v1 (ddpm) U-Net configs found locally; their exact internal effect is **UNKNOWN** (not documented locally, lives in `monai` source not present here) — treat as "MAISI-v2 architectural upgrades," not re-derived from source.

**No `with_conditioning`/`cross_attention_dim` key is set anywhere on `diffusion_unet_def`** in either `config_network_rflow.json` or `config_network_ddpm.json` — as configured, the main image-generating U-Net (the one used by `rflow-mr-brain`) has **no cross-attention conditioning input at all**. See §7 for the one place in the repo that *does* use cross-attention.

### 1.5 Attention blocks — summary

**VERIFIED:**
- VAE: none.
- Main image diffusion U-Net (`DiffusionModelUNetMaisi`): self-attention only, only at the 2 deepest of 4 levels, flash-attention implementation.
- `ControlNetMaisi` (CT only): mirrors the frozen U-Net's `attention_levels`/`num_head_channels` exactly (`config_network_rflow.json:66-90`), since it's weight-initialized by copying the pretrained U-Net's matching layers (`monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())`, `scripts/train_controlnet.py:307`, `scripts/utils_infer.py:356`).
- `mask_generation_diffusion_def` (CT-only 125-class organ-mask generator, a *different*, generic `monai.networks.nets.diffusion_model_unet.DiffusionModelUNet`, **not** the Maisi subclass): the **only** network in the repo configured with `with_conditioning: true, cross_attention_dim: 10, upcast_attention: true` (`config_network_rflow.json:119-132`) — cross-attention conditioned on a 10-dimensional anatomy-size vector (`skills/infer_mask-only.md:14-70`), not text, and not used by `rflow-mr-brain` at all (mr-brain has no mask stage).

### 1.6 Conditioning mechanisms (main image U-Net, as used by `rflow-mr-brain`)

**VERIFIED**, from every call site that builds `unet_inputs` (`scripts/diff_model_infer.py:174-207`, `scripts/utils_infer.py:250-273`, `scripts/diff_model_train.py:305-324`, `scripts/train_controlnet.py:211-232`):

1. **Timestep** — standard `timesteps` kwarg.
2. **Spacing** — `spacing_tensor`: per-axis float spacing × 100, cast to fp16, **always** passed (never gated by an `include_spacing` check in the scripts — matches `include_spacing_input: true` baked into both network configs) (`scripts/utils_infer.py:92-96`, `scripts/diff_model_train.py:120-122,291`).
3. **Modality** — `class_labels`: integer code (`modality_mapping.json`) looked up through a class-embedding table of size `num_class_embeds=128`, gated by `unet.num_class_embeds is not None` (`scripts/diff_model_infer.py:137,190-195`).
4. **Body region** (`top_region_index_tensor`/`bottom_region_index_tensor`) — architecturally present as a capability but **disabled** (`include_body_region: false`) for every rflow variant including `rflow-mr-brain`; only `ddpm-ct` uses it.
5. **No cross-attention, no text, no pixel/voxel-level conditioning image** for the base image DM (image-only path). ControlNet (spatial residual conditioning) exists **only for CT** — `"There is currently no ControlNet for MRI — MR variability is too large to train one whole-body model."` (`docs/inference.md:16,36`). The MR-Brain-specific *cross-sequence* ControlNet (T1↔T2↔FLAIR↔SWI) is explicitly listed as **"Coming soon"** (`README.md` model-variant table, "Model: ControlNet" row for `rflow-mr-brain`).

### 1.7 Classifier-free guidance (CFG)

**VERIFIED** — CFG is applied **only on the modality/class-label conditioning**, not on spacing, not on text (none exists), not on tumor/anatomy (that's a CT-mask-only concept):

- **Training-time null-conditioning**: `augment_modality_label(modality_tensor, prob=0.1)` (`scripts/diff_model_train.py:36-66`) does three things every step: (a) any modality code `≥ 9` (every brain contrast code: 9,10,11,16,20,29-33) has a 10% chance of being coarsened to the generic `mri` code 8; (b) CT sub-codes 2/3 have a 10% chance of coarsening to plain `ct` (1); (c) **10% of samples have their modality zeroed entirely** (`mask_zero`, lines 62-64) — this is what teaches the network to also produce a sensible "unconditional" output for modality-code 0 (`"unknown"`, `modality_mapping.json:2`), which is exactly what CFG's null branch relies on at inference.
- **Inference-time CFG**: the batch is duplicated (conditional copy + a copy with `class_labels` zeroed), one forward pass over the doubled batch, then `model_output = model_uncond + cfg_guidance_scale * (model_t - model_uncond)` (`scripts/diff_model_infer.py:197-207`, identical pattern for the ControlNet path at `scripts/utils_infer.py:264-279`). A `cfg_guidance_scale > 0` **roughly doubles U-Net compute/VRAM** per step (`skills/infer_image-only.md:166`).
- Shipped default for `rflow-mr-brain`: `cfg_guidance_scale = 10` (`configs/config_maisi_diff_model_rflow-mr-brain.json:34`), matching the documented "MR → 10" guidance (`docs/inference.md:197`, `skills/infer_image-only.md:172-176`). CT variants ship `0` (guidance has nothing to amplify since CT modality is fixed).
- **Discrepancy found**: the shipped `config_maisi_diff_model_rflow-mr.json:34` sets `cfg_guidance_scale: 15`, not the documented default of 10 (`docs/inference.md:197`, `skills/infer_image-only.md:175` both say rflow-mr's shipped default is 10). This affects the plain `rflow-mr` config file, **not** `rflow-mr-brain` (whose own shipped config correctly matches the docs at 10). Flagged as a minor doc/config inconsistency, not an mr-brain issue.

---

## 2. Exact inference inputs (`rflow-mr-brain`)

**VERIFIED**, from `configs/config_maisi_diff_model_rflow-mr-brain.json` + `configs/environment_maisi_diff_model_rflow-mr-brain.json` + `scripts/diff_model_infer.py`:

| Input | Shipped default | Source |
|---|---|---|
| `modality` | `9` (T1 whole-brain) | config `diffusion_unet_inference.modality`; valid values 9,10,11,16,20 (whole-brain) / 29-33 (skull-stripped) per `docs/inference.md:172-181` |
| `dim` | `[256, 256, 256]` | user-editable; see §4 for constraints |
| `spacing` | `[1, 1, 1]` mm | |
| `random_seed` | `1234` | becomes `seed + local_rank` under multi-GPU (`scripts/diff_model_infer.py:277-278`); fed to `monai.utils.set_determinism` |
| `num_inference_steps` | `30` | RFlow |
| `cfg_guidance_scale` | `10` | modality CFG, §1.7 |
| `top_region_index`/`bottom_region_index` | present as JSON keys (`[0,1,0,0]`/`[0,0,1,0]`) | **vestigial for this variant** — never consumed, since `unet.include_top_region_index_input` is `False` for the rflow family (`scripts/diff_model_infer.py:136,181-188`) |

- **Skull state** is not a separate flag — it is folded entirely into the `modality` integer (29-33 = skull-stripped counterparts of 9,10,11,20,16) (`README.md:134-142`, `docs/inference.md:177-181`).
- **No body-region / anatomical-region input is meaningful** for this variant (architecturally disabled, see §1.6).
- **No text/report input exists anywhere** in the config schema or CLI (`argparse` args in `scripts/diff_model_infer.py:346-352` are only `-e/-c/-t/-g`; confirmed by exhaustive grep, §7).
- **Acquisition plane is not an explicit input** — see §4.
- **Batch size at inference is fixed to 1 volume per process** — the noise/latent tensor is built with a hardcoded leading dim of 1 (`scripts/diff_model_infer.py:139-148`, `scripts/utils_infer.py:98-107`); multi-sample generation is achieved by running multiple DDP ranks or repeated CLI invocations with different seeds, each producing one file (`scripts/diff_model_infer.py:311-338`), not by a batch dimension.

---

## 3. Exact output

**VERIFIED**, all from `scripts/diff_model_infer.py:214-261` (image-only path; the ControlNet-conditioned path in `scripts/utils_infer.py:293-320` is CT-only and not used by `rflow-mr-brain`):

- **Tensor layout**: decoded volume is squeezed from `(1, 1, H, W, D)` → `(H, W, D)` (`data = synthetic_images.squeeze().cpu().detach().numpy()`, line 224) and saved with **no axis permutation** — NIfTI array shape exactly equals the requested `dim`.
- **Intensity range (MR, modality code ≥ 8)**: decoder output (already in `[0,1]` from the VAE decoder) is affinely rescaled `a_min,a_max,b_min,b_max = 0,1000,0,1` then **clipped only at the bottom** (`np.clip(data, a_min, None)`, line 229) — i.e. `[0, +∞)` in practice, **not** a bounded Hounsfield-like scale (CT by contrast is clipped both sides to `[-1000, 1000]`, line 233). Cross-checked against `skills/infer_image-only.md:192-195` ("MR (codes 8..32) | int16 NIfTI | `[0, +∞)`") and the mirrored logic in `scripts/utils_infer.py:309-318`. This asymmetric (unclipped-above) MR handling is consistent with the *training-time* intensity transform also being unclipped on the upper tail for MR (`ScaleIntensityRangePercentilesd(..., clip=False)`, §5.2) — an internally-consistent design, not an oversight.
- **dtype**: `np.int16` (`return np.int16(data)`, line 234) — the float decoder output is cast/rounded to int16 before saving, so the NIfTI header dtype is int16.
- **NIfTI orientation / affine construction**: **VERIFIED, notable finding** — `save_image()` builds a **diagonal-only affine**: `out_affine = np.eye(4); out_affine[i,i] = out_spacing[i]` (lines 254-256). There is **no rotation / direction-cosine information and no non-zero origin/translation** — the affine is purely a voxel→mm scaling matrix. The implicit axis convention is RAS because training data was reoriented with `Orientationd(axcodes="RAS")` before the network ever saw it (`scripts/diff_model_create_training_data.py:58`, and again in the VAE-training pipeline, `scripts/transforms.py:156`), but that convention is **not asserted in the output file's affine/qform beyond the identity-diagonal** — downstream viewers that assume a populated direction-cosine matrix will only see identity-aligned axes with the given spacing, not a genuine patient-frame orientation.
- **Shape**: exactly the requested `dim`/`output_size` (the AE-decode step operates on the final denoised latent, which is `dim // 4` per axis given the VAE's fixed 4× downsample, and produces a decoded volume of exactly `dim` again — confirmed via the divisor-computation logic discussed in §4).
- **Spacing**: exactly the requested `out_spacing`, written verbatim into the affine diagonal — the network does not "choose" its own output spacing; it is purely a conditioning input echoed back into the saved file's header.
- **Filename**: `{output_prefix}_seed{seed}_size{H}x{W}x{D}_spacing{sx}x{sy}x{sz}_{timestamp}_rank{r}_modality{m}.nii.gz` (`scripts/diff_model_infer.py:328`).

---

## 4. Geometry support

**VERIFIED — critical finding**: for the `rflow-mr-brain` (and plain `rflow-mr`) **image-only** inference entry point (`scripts/diff_model_infer.py`, the script the README/skills actually document for MR-Brain generation), **no geometry/spacing validator is ever invoked for MR modalities**:

```python
modality = args.diffusion_unet_inference["modality"]
if modality >= 1 and modality <= 7:
    check_input_ct(None, None, None, output_size, out_spacing, None)   # scripts/diff_model_infer.py:294-296
```
`check_input_mr` (defined at `scripts/sample_mask.py:284-330+`) is imported and called **only** by `scripts/inference.py:25,133,142` and `scripts/sample.py` — the paired CT mask+image pipeline. It is **never imported or called by `scripts/diff_model_infer.py`**, confirmed by a repo-wide grep of every call site of both functions. This means the claim in `skills/infer_image-only.md:142` — *"Hard constraints (validated by `check_input_ct` / `check_input_mr`)"* — **does not describe an enforced runtime check for the actual `rflow-mr-brain` code path**; the listed constraints (dim/spacing allowed-value tables) are documentation-only for MR, not code-enforced, for this specific entry point.

- **Divisibility / latent downsampling factor**: **VERIFIED** 4× per axis (VAE downsample factor, §1.1-1.2). The `dim`→latent-shape conversion in `scripts/diff_model_infer.py:300-308` computes `divisor = 2 ** (num_downsample_level - 2)` where `num_downsample_level = len(diffusion_unet_def["num_channels"])` = 4 → `divisor = 4`. **Notable code-smell / silent-mismatch risk**: this divisor is derived from the *diffusion U-Net's* channel-list length, **not** from the actual `autoencoder_def` configuration — the two configs must be kept in manual lock-step for this formula to reflect the true VAE downsample factor; a future edit to either config independently would silently miscompute the noise-latent shape rather than raise an error. `dim` values not exactly divisible by 4 are silently floor-divided (`output_size[i] // divisor`, line 143-146) — no error, just a latent that's marginally smaller than the exact requested `dim`.
- **`docs/inference.md:190`** documents a general "must be divisible by 16" rule for `output_size`/`dim`, presumably from the paired CT/mask pipeline's combined AE + mask-AE tiling constraints; this is **not the enforced constraint on the mr-brain image-only path**, where the effective enforced-by-arithmetic constraint is divisibility by 4 (and even that only silently truncates rather than erroring).
- **Minimum/maximum dims**: no coded hard minimum found beyond needing a non-degenerate (≥1-voxel) latent after ÷4; practical minimum is **UNKNOWN** (not stated locally). **Documented maximum** for `rflow-mr-brain`: `512×512×256` (`README.md` model table; `skills/infer_image-only.md` variant table) — this is a **documented/tested ceiling**, not a coded `assert`; no config preset file exists locally beyond the single default-256³ config for mr-brain (unlike CT, which ships `configs/config_infer_*_512x512x*.json` presets up to 512×512×768).
- **Whether arbitrary shapes/spacing "work"**: mechanically yes on this code path (no validator blocks it, confirmed above) — but explicitly **not** guaranteed to produce valid anatomy outside the training FOV distribution. The maintainers themselves flag this as the dominant failure mode, not an inference of this audit: `docs/inference.md:217` ("Unusable output despite valid inputs → FOV is probably out-of-distribution"), `README.md:112-117` callout, `skills/infer_image-only.md` opening warning with a worked failure example (256³ @ 0.5mm → 128 mm FOV → "noise").
- **How spacing enters the U-Net**: a dedicated numeric side-tensor (`spacing_tensor`, ×100-scaled, fp16), passed on every forward call — a direct conditioning input, not implemented via image resampling inside the network. The exact internal merge mechanism (additive/FiLM vs. concatenation) inside `DiffusionModelUNetMaisi.forward()` is **UNKNOWN** (monai source not present locally).
- **Acquisition plane**: **VERIFIED absent as an explicit input.** No `plane` key exists anywhere in `modality_mapping.json`, `config_maisi_diff_model_rflow-mr-brain.json`, or any inference script argument (exhaustive grep + direct reading of every touched config). Plane is represented **only indirectly**, through the chosen `(dim, spacing)` FOV shape matching the plane-specific training distribution: *"set `dim` so the slice-stacking axis maps to the smaller `dim[i]=128` (axial→z, sagittal→x, coronal→y)"* (`skills/infer_image-only.md:84`). This is exactly the "indirect, via geometry + training distribution" mechanism named in the task, and no alternative explicit-plane code path exists.
- **Documented FOV distributions by modality and plane**: full table at `docs/inference.md:136-162` (T1/T2/FLAIR/SWI/MRA × axial/sagittal/coronal, with per-row training-image counts). Total unique training images across skull conditions: **318,825** (`docs/inference.md:138`). Explicit low-N warnings for **MRA** (all planes: 37/98/11) and **SWI sagittal (2)** / **SWI coronal (4)** (`docs/inference.md:142`). Oblique-orientation scans exist in training data but are excluded from the reference table (`docs/inference.md:140`).
- **OOD risks**: as above — this is first-party, explicitly-documented maintainer guidance, not a risk this audit is inferring independently.

---

## 5. Training pipeline

### 5.1 Raw volume preprocessing → embedding creation (`scripts/diff_model_create_training_data.py`)

**VERIFIED** (`create_transforms`, lines 33-74; `process_file`, lines 110-194; `round_number`, lines 77-90):
`LoadImaged` → `EnsureChannelFirstd` → `Orientationd(axcodes="RAS")` → `EnsureTyped(float32)` → **[modality-conditional fixed intensity transform, §5.2]** → `Resized(spatial_size=new_dim, mode="trilinear")`, where `new_dim[i]` = original `dim[i]` **rounded to the nearest multiple of 128** (minimum 128) — a whole-volume trilinear resample to a 128-multiple voxel grid, **not** to a fixed physical spacing (spacing is preserved as metadata for the `spacing_tensor` conditioning input, not enforced as a target).
The pre-resize (`plain_transforms`) pass reads the *original* `dim`/`pixdim` purely to compute `new_dim` (lines 143-150) — this metadata read is separate from the actual resampled tensor.

### 5.2 Intensity normalization

**VERIFIED** (`scripts/transforms.py:42-71`):
- CT: `ScaleIntensityRanged(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True)` — fixed HU window, clipped both sides.
- MRI: `ScaleIntensityRangePercentilesd(lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False)` — **percentile-based**, not a fixed range, and **not clipped** on the upper tail (the top 0.5% of intensities may map above 1.0). This directly explains why inference-time MR output clipping is bottom-only (§3) — an internally consistent design choice traceable end-to-end.

### 5.3 Orientation

**VERIFIED**: `Orientationd(axcodes="RAS")` applied identically in both the VAE-training pipeline (`scripts/transforms.py:156`) and the diffusion-model embedding-creation pipeline (`scripts/diff_model_create_training_data.py:58`) — one consistent convention across the whole stack.

### 5.4 Resize/resample behavior — VAE vs. diffusion-model training differ

**VERIFIED**:
- **VAE training** operates on **random/central patches**, not whole resampled volumes: `SpatialPadd` + `RandSpatialCropd(roi_size=patch_size)` for training, `DivisiblePadd`/`ResizeWithPadOrCropd` for validation (`scripts/transforms.py:204-225`). Released model: patch size `[64,64,64]` initially, then `[128,128,128]` (`docs/training.md:27,126`). Shipped `spacing_type: "rand_zoom"` for the released VAE (`configs/config_maisi_vae_train.json:4-5`) — i.e. the released VAE does **not** resample to a fixed physical spacing; it randomly zooms the patch instead (`RandZoomd` + `RandRotated`, `scripts/transforms.py:179-202`), confirmed directly from the shipped training config.
- **Diffusion-model training** consumes **whole (128-multiple-resized) latents**, one full volume per gradient step, `batch_size=1` (§5.6) — no patch-cropping code found in `scripts/diff_model_train.py`'s `prepare_data` (lines 85-139).

### 5.5 VAE encoding for embedding creation

**VERIFIED**: `autoencoder.encode_stage_2_inputs` is invoked through a `SlidingWindowInferer(roi_size=[320,320,160], overlap=0.4)` + the repo's `dynamic_infer` helper (`scripts/diff_model_create_training_data.py:174-186`, `scripts/utils.py:787-818`) — encoding is **tiled in pixel space** at 320×320×160-voxel windows (falls back to a single whole-volume forward pass when the volume is small enough, per `dynamic_infer`'s size check).

### 5.6 Latent sidecars / Dataset output / gap found

**VERIFIED** output of embedding creation: `<image>_emb.nii.gz`, float32, latent-space affine copied from the resized image, channel-last `(X,Y,Z,C)` layout via `.transpose(1,2,3,0)` (`scripts/diff_model_create_training_data.py:189-191`).

**VERIFIED** the diffusion-model training dataloader (`scripts/diff_model_train.py:85-139`, `prepare_data`) expects, per training file: `{"image": <embedding_path>, "spacing": <embedding_path>+".json", ["top_region_index"/"bottom_region_index"/"modality": same .json]}`, loaded via `Lambdad` transforms that read specific JSON keys out of that sidecar file (lines 110-135). **Dataset item output** = dict with keys `image` (loaded latent tensor), `spacing` (float tensor ×100), optionally `top_region_index`/`bottom_region_index` (×100) and `modality` (mapped through `modality_mapping.json`, long dtype).

**UNKNOWN / genuine gap**: `scripts/diff_model_create_training_data.py`'s `process_file` (lines 110-194) writes **only** the `_emb.nii.gz` file — it does **not** write the `.json` sidecar that `diff_model_train.py` expects to find alongside it. No script in this repo copy is confirmed to produce that per-embedding JSON for the diffusion-model training path (the closest documented JSON-authoring example, `data/README.md:249-274`, is for **ControlNet** training data, a structurally similar but distinct pipeline). Whether this sidecar is hand-authored, produced by an internal NVIDIA tool not included in this release, or derived some other way is **not determinable from this repo**.

### 5.7 Batch size, loss, modality/spacing conditioning

**VERIFIED**:
- Batch size: `diffusion_unet_train.batch_size = 1` for `rflow-mr-brain` (`configs/config_maisi_diff_model_rflow-mr-brain.json:3`) — same value (1) shipped for every other diffusion-model/ControlNet/VAE training config found (`config_maisi_diff_model_rflow-mr.json:3`, `config_maisi_controlnet_train_rflow-mr.json:3`, `config_maisi_vae_train.json:9`); `docs/training.md:26` explicitly invites increasing it if GPU memory allows.
- Loss (diffusion U-Net): `torch.nn.L1Loss()` (`scripts/diff_model_train.py:476`) between `model_output` and, for the RFlow scheduler, the fixed target `images - noise` (rectified-flow velocity; **not** any of the DDPM `epsilon`/`sample`/`v_prediction` branches, which only execute for non-RFlow schedulers) (`scripts/diff_model_train.py:327-341`).
- Loss (VAE): perceptual + KL + adversarial + reconstruction(L1|L2), per the hyperparameter list in `docs/training.md:31-34` (`perceptual_weight`, `kl_weight`, `adv_weight`, `recon_loss`); the **existence** of these terms is verified from the shipped config, but the precise combination formula lives in `train_vae_tutorial.ipynb`, which this audit did not fully execute/read cell-by-cell — treat the *existence* of each term as **VERIFIED** and the *exact combination* as **INFERRED** (standard LDM VAE loss), not independently re-derived here.
- Modality conditioning at training: `class_labels=modality_tensor` via `augment_modality_label` (§1.7) — includes coarsening of **every** brain-specific code `≥9` (not just 9-12) to generic `mri`(8) with 10% probability per step (re-verified directly against the code: `mask_mri = modality_tensor >= 9` has no upper bound, `scripts/diff_model_train.py:58-60`).
- Spacing conditioning at training: `spacing_tensor` loaded from the sidecar JSON, always passed (never gated), matching `include_spacing_input: true`.

---

## 6. Pretrained weights

**VERIFIED**, from `data/README.md:22-32` (table) and `scripts/download_model_data.py:82-153`:

`rflow-mr-brain` downloads **exactly two** checkpoint files:

| File | Source repo | Trained on | Belongs to |
|---|---|---|---|
| `autoencoder_v1.pt` | `nvidia/NV-Generate-CT` | 37,243 CT + 17,887 MRI volumes (chest/abdomen/head-neck CT; brain/skull-stripped-brain/breast/prostate-region MR) — **no MR-RATE listed** (`data/README.md:37-68`) | **VAE** |
| `diff_unet_3d_rflow-mr-brain_v0.pt` | `nvidia/NV-Generate-MR-Brain` | **MR-RATE** — *"The training data of this version v0 is MR-RATE"* (`README.md`, News section, "🎆 March 2026 🎇") | **3D diffusion U-Net** |

**Key finding**: within the four variants shipped in this repo (`ddpm-ct`, `rflow-ct`, `rflow-mr`, `rflow-mr-brain`), **only the `rflow-mr-brain` diffusion U-Net was trained on MR-RATE.** The VAE it uses (`autoencoder_v1.pt`) is the same general-purpose CT+MRI foundation VAE used by every other variant, trained on none of MR-RATE's sources. The sibling `rflow-mr` variant uses a *different*, larger foundation VAE (`autoencoder_v2.pt`) and a diffusion U-Net trained on 16,291 MR volumes from 17 datasets (`data/README.md:140-163`) — also none from MR-RATE. There is **no ControlNet** shipped for `rflow-mr-brain` (`download_model_data.py`'s `rflow-mr-brain` branch fetches no `controlnet_*.pt`; `docs/inference.md:16`; `README.md` "Coming soon" row).

- **Checkpoint format**: `torch.save({"epoch":..., "loss":..., "num_train_timesteps":..., "scale_factor":..., "unet_state_dict":...}, path)` (`scripts/diff_model_train.py:369-400`, `save_checkpoint`). Loaded everywhere with `strict=False` (`scripts/diff_model_infer.py:70`, `scripts/utils_infer.py:351`) — the loader **silently tolerates state-dict key mismatches**. This is favorable for planned architecture extension (new conditioning params can be added without an immediate load-time crash) but also means an accidental shape/name mismatch elsewhere would silently drop weights rather than error — worth an explicit shape-diff check before trusting any extended checkpoint.
- **VAE checkpoint**: plain `state_dict` (or a `{"unet_state_dict": ...}`-wrapped variant, handled defensively, `scripts/diff_model_infer.py:62-65`) — no `scale_factor`/epoch bookkeeping (only the diffusion model needs `scale_factor`).
- **Code license**: Apache 2.0 for all of `NV-Generate-CTMR/` (`LICENSE`; every script's header comment; `README.md:256`).
- **Weight licenses** (`README.md:252-260` table, cross-checked against `LICENSE.weights`):
  - `NV-Generate-CT` weights (→ `autoencoder_v1.pt`): **NVIDIA Open Model License** — research **and commercial** use.
  - `NV-Generate-MR` weights (`autoencoder_v2.pt`, `diff_unet_3d_rflow-mr.pt`): **NVIDIA Non-Commercial ("OneWay") License** — `LICENSE.weights:20`: *"may be used or intended for use non-commercially... 'non-commercially' means for research or evaluation purposes only."* **Not used by `rflow-mr-brain`.**
  - `NV-Generate-MR-Brain` weights (`diff_unet_3d_rflow-mr-brain_v0.pt`): **NVIDIA Open Model License** (`README.md:259`) — i.e., despite being the MR-RATE-trained checkpoint, it carries the *more permissive* license, not the Non-Commercial one.
  - **Net conclusion for `rflow-mr-brain` specifically**: both components it actually uses (VAE from NV-Generate-CT, U-Net from NV-Generate-MR-Brain) are under the **Open Model License** — the stricter Non-Commercial license only governs the unrelated `autoencoder_v2.pt`/`diff_unet_3d_rflow-mr.pt` pair that `rflow-mr-brain` does not load. This is a non-obvious, easy-to-get-wrong conclusion, stated here with direct citations.

---

## 7. Text conditioning

**VERIFIED — exhaustive check**: a repo-wide, case-insensitive grep for `cross[_-]?attention`, `encoder_hidden_states`, `text[_ ]?encoder`, `clip`, `bert`, `T5`, `prompt`, `report` across every `.py`/`.json`/`.md` file, plus a keyword scan of every notebook cell's source text in all three tutorial notebooks, returns **zero** hits for any text encoder, BERT/CLIP/T5 reference, `encoder_hidden_states`, or free-text report conditioning anywhere in code, config, or documentation. The only matches are unrelated: `torch.clip`/intensity clipping, a GitHub bug-report-template link, an example "AI agent prompt" string in the README, and the two `cross_attention_dim: 10` occurrences (both on the CT-only 10-d anatomy-size mask generator, §1.5). **Conclusion: arbitrary report text is not supported anywhere in this codebase today** — not merely absent-by-omission, but actively absent from every conditioning code path traced in §1-§6.

- **Existing cross-attention/conditioning injection points**: exactly **one**, `mask_generation_diffusion_def` — a generic `monai.networks.nets.diffusion_model_unet.DiffusionModelUNet` (not the MAISI subclass), `with_conditioning: true, cross_attention_dim: 10`, consuming a 10-dimensional **anatomy-size** vector for the **CT-only** organ-mask generator (`config_network_rflow.json:119-140`; `skills/infer_mask-only.md:14-70`). The main image U-Net used by `rflow-mr-brain` (`DiffusionModelUNetMaisi`) has no such key set in any config it appears in.
- **Whether the current U-Net API can accept `encoder_hidden_states`**: **UNKNOWN** at the code level available locally. `DiffusionModelUNetMaisi`'s source is not present in this repo (external `monai` package, not installed, §0). What can be said: its sibling class in the *same* package family, `monai.networks.nets.diffusion_model_unet.DiffusionModelUNet`, demonstrably supports `with_conditioning`/`cross_attention_dim` (it's used exactly that way for the mask generator). It is **INFERRED, not verified**, that `DiffusionModelUNetMaisi` — being MAISI's own subclass/variant of that same family, with an otherwise near-identical constructor signature (`num_channels`, `attention_levels`, `num_head_channels`, `num_res_blocks`, `use_flash_attention`, ...) — likely exposes the same `with_conditioning`/`cross_attention_dim` knobs, simply unset in every shipped config here. **This must be checked directly against the installed `monai` source before any implementation work**, not assumed from this audit.
- **Layers requiring modification** (at the level this audit can see, i.e. config + call sites, not source lines): (a) set `with_conditioning`/`cross_attention_dim` on `diffusion_unet_def` (and `controlnet_def`, if retained); (b) extend every `unet_inputs`/`controlnet_inputs` dict-builder to pass a new `encoder_hidden_states` tensor — call sites: `scripts/diff_model_infer.py:174-179`, `scripts/utils_infer.py:230-235,250-256`, `scripts/diff_model_train.py:305-310`, `scripts/train_controlnet.py:196-201,211-218`; (c) add a text-embedding branch to the training dataloader (`scripts/diff_model_train.py:117-139`, `prepare_data`/`train_transforms_list`, currently has no text-handling code at all).
- **Implemented vs. "coming soon"**: **Implemented** today: CT image+mask ControlNet, CFG-on-modality, spacing conditioning, the CT-only 10-d anatomy-size cross-attention mask generator. **Explicitly announced, not yet shipped**: only the MR-Brain **cross-sequence** ControlNet (T1↔T2↔FLAIR↔SWI image-conditioned, "Coming soon" in `README.md`'s variant table) — an *image*-conditioning roadmap item, not a text-conditioning one. **No text/report-conditioning roadmap item is mentioned anywhere** in the News section, README tables, or docs — report-conditioning would be a wholly new capability relative to the maintainers' own stated roadmap, not an accelerated version of something already announced.

---

## 8. Report-conditioning strategy evaluation

All six strategies are evaluated against the concrete architecture facts above: attention exists only at the 2 deepest (of 4) U-Net levels; the only existing cross-attention precedent in the repo is a 10-d **non-text, non-spatial** vector into a *different* (CT-only) network; the existing ControlNet is a **spatial/image-shaped** conditioning mechanism (`controlnet_cond` must be `(B, C_cond, H, W, D)`, `NV-Generate-CTMR/scripts/utils_infer.py:147-151`); modality/spacing conditioning are flat vectors merged somewhere inside a class not visible locally; CFG-via-label-zeroing is an already-proven training recipe in this exact codebase (§1.7).

### A. Frozen clinical text encoder + cross-attention blocks in the U-Net

- **Compatibility with pretrained checkpoint**: Loadable in principle — `strict=False` loading (§6) means new cross-attention modules would simply initialize fresh while everything else warm-starts from `diff_unet_3d_rflow-mr-brain_v0.pt`. **Whether `DiffusionModelUNetMaisi.forward()` actually threads cross-attention through is UNKNOWN** (§7) — the single largest risk for this strategy.
- **Newly initialized parameters**: cross-attention Q/K/V/out projections at the 2 (of 4) levels that already have `attention_levels: true`, × 2 res-blocks/level — a real but not enormous addition relative to the full U-Net.
- **Expected GPU cost**: moderate increase (~10-30%, rough estimate from architecture proportions, not measured) — self-attention already runs at those 2 levels; adding cross-attention roughly doubles attention-block cost only there.
- **Expected data requirement**: moderate — only new layers need to learn a useful association, image backbone stays pretrained; the standard, most data-efficient of the strategies that add genuine text conditioning.
- **Spatially localized findings representable?** **Yes** — per-token spatial cross-attention is the closest of the six to letting different report phrases attend to different 3D regions (though only at the coarsest 2 U-Net levels, so localization is coarse, not voxel-precise).
- **Training stability**: generally the proven SD/DiT-style recipe (stable with a frozen text encoder + small LR on new layers); residual risk from the unverified MAISI forward-signature.
- **Implementation complexity**: **HIGH** — requires inspecting/patching the actual `monai` `DiffusionModelUNetMaisi` source (not vendored here) to confirm/insert cross-attention plumbing, plus a text encoder + tokenization pipeline.
- **Recommended priority**: **HIGH as the target end-state**, but gated on an external engineering unknown — not the first experiment to run.

### B. Text embeddings via FiLM / adaptive normalization

- **Compatibility**: **Likely the lowest-risk change** — no new `cross_attention_dim`/`with_conditioning` needed; modality-class and spacing conditioning are already flat vectors fed into (almost certainly) the same additive/embedding conditioning pathway used for the timestep embedding — adding a pooled report vector into that same pathway is architecturally analogous to something already proven to train in this exact codebase.
- **Newly initialized parameters**: one small projection MLP (pooled text embedding → the U-Net's existing conditioning-embedding width) — small.
- **Expected GPU cost**: minimal — no new attention ops; cost dominated by the frozen text encoder's forward pass (cheap vs. a 3D U-Net).
- **Expected data requirement**: lower than A — a single global vector is an easier target to fit, at the cost of expressive ceiling.
- **Spatially localized findings representable?** **No** — one pooled/global vector modulates the whole feature map per level/channel; cannot distinguish "abnormality in the left vs. right hemisphere."
- **Training stability**: **high** — smallest architectural delta, closely analogous to the already-proven modality/spacing conditioning.
- **Implementation complexity**: **low-medium** — still requires locating the actual insertion point in `monai` source, but the insertion pattern (additive to an existing conditioning vector) is far simpler than wiring cross-attention K/V.
- **Recommended priority**: **HIGH as the first practical experiment** — cheapest way to test whether report-conditioning helps at all before investing in A.

### C. Trainable ControlNet-style side network

- **Compatibility with pretrained checkpoint**: architecturally mismatched for text. The repo's own `ControlNetMaisi` requires a **spatial**, `(B, C_cond, H_out, W_out, D_out)`-shaped conditioning tensor (`conditioning_embedding_in_channels: 8`, `NV-Generate-CTMR/configs/config_network_rflow.json:66-90`) — designed for a literal organ-label mask, not free text. Reusing it for a report requires first converting text into a pseudo-spatial map (e.g. broadcasting a global embedding uniformly across the volume), which forfeits most of ControlNet's designed purpose.
- **Newly initialized parameters**: **largest of the six** — an entire second copy of the U-Net's encoder/mid path, weight-initialized by copying the frozen U-Net (`copy_model_state`, §1.5), plus new zero-init output heads.
- **Expected GPU cost**: **HIGH** — an extra near-U-Net-sized forward pass every training step *and* every one of the 30 RFlow denoising steps at inference.
- **Expected data requirement**: the repo's own ControlNet-finetuning example (Kidney Tumor on a C4KC-KiTS subset, `data/README.md:204-274`) shows sample-efficient finetuning **for a genuinely spatial new condition** — that sample-efficiency argument does not obviously transfer to a broadcast-text condition.
- **Spatially localized findings representable?** Only if a separate text→coarse-heatmap module is engineered first; a naive uniform broadcast degenerates to a costlier version of strategy B.
- **Training stability**: the repo's freeze-U-Net / copy-init-ControlNet / L1(+ optional region-contrastive) recipe is proven for spatial-mask conditioning (`scripts/train_controlnet.py`); unproven for broadcast-text.
- **Implementation complexity**: **HIGH** — needs a new "text→condition-tensor" module on top of all existing ControlNet training/inference machinery, for a modality (free text) that isn't naturally spatial.
- **Recommended priority**: **LOW** for global-report conditioning as specified; becomes the natural strategy **only if/when the report is first converted into something genuinely spatial** (e.g. a derived lesion-location heatmap or bounding box) — not for a first pass.

### D. Fixed condition embedding concatenated with modality/spacing embeddings

- **Compatibility**: same family as B (a flat-vector conditioning change), implemented via concatenation rather than an additive FiLM merge; the exact internal merge operator used by `DiffusionModelUNetMaisi` (concat vs. add vs. separate embed-and-sum) is **UNKNOWN** without monai source, but the call-site tensors (all flat per-sample vectors) strongly suggest this family of change is architecturally compatible.
- **Newly initialized parameters**: a small linear/MLP projection from pooled report embedding → fixed size, plus (if concatenating) a resized first linear layer at the merge point — comparable to or smaller than B.
- **Expected GPU cost**: minimal, same profile as B.
- **Expected data requirement**: same as B — easiest of the six to get *some* signal from with less data; same weak ceiling (global vector only).
- **Spatially localized findings representable?** **No** — identical limitation to B.
- **Training stability**: **high** — one of the two most mechanical changes of the six.
- **Implementation complexity**: **low**, potentially the single cheapest to prototype if the concatenation can happen at the `unet_inputs` call-site level rather than inside the frozen module's constructor — plausible given the pattern already used for modality/spacing, but not verified locally since the actual merge point is inside `monai`'s unseen `forward()`.
- **Recommended priority**: **tied-high with B** — pick whichever is easier to wire once `DiffusionModelUNetMaisi.forward()` is actually inspected.

### E. Latent text-to-image diffusion model built around the existing VAE

- **Compatibility**: full compatibility with the **frozen VAE only** (`autoencoder_v1.pt` reused unchanged) — the diffusion U-Net itself would likely need to be built new or heavily rewired (different attention wiring throughout, not just added layers), largely forfeiting warm-start from `diff_unet_3d_rflow-mr-brain_v0.pt`.
- **Newly initialized parameters**: **the entire diffusion U-Net** — by far the largest newly-initialized parameter count of the six strategies.
- **Expected GPU cost**: **HIGH for training** — training a full 3D conditional U-Net from scratch (even in latent space) is comparable to or exceeds the original MAISI diffusion-model training compute; inference cost need not exceed the existing profile (`docs/performance.md`) if final depth/width match.
- **Expected data requirement**: **highest of the six** — training attention/conditioning from scratch typically needs substantially more paired data than fine-tuning new layers on a pretrained backbone.
- **Spatially localized findings representable?** Yes, achievable by design (full architectural freedom, SD-style cross-attention) — but bought at much higher cost/data than strategy A, which gets the same capability while keeping the pretrained backbone.
- **Training stability**: training a full conditional 3D diffusion model from scratch is generally harder/slower to converge than fine-tuning a pretrained backbone — this is **general diffusion-modeling practice, not something demonstrated in this repo** (no from-scratch text-to-image experiment exists locally); flagged as **ASSUMED**, not verified.
- **Implementation complexity**: **HIGH** — effectively a new 3D LDM stack, decoupled from `scripts/diff_model_train.py`'s assumption of `DiffusionModelUNetMaisi`'s specific call signature throughout.
- **Recommended priority**: **LOW** — dominated by strategy A wherever reusing the pretrained mr-brain U-Net is acceptable; only worth it if A turns out to be genuinely blocked by an architectural incompatibility once `monai` internals are actually inspected.

### F. LoRA on attention/conditioning layers, freezing most pretrained weights

- **Compatibility**: **HIGH** — LoRA adds low-rank deltas on top of existing weight matrices without changing their shapes; combined with the `strict=False` loading pattern already used everywhere (§6), LoRA adapter modules introduce new, non-conflicting state-dict keys — a "wrap, don't replace" pattern, low-risk by construction.
- **Important caveat**: applied to the base model's **existing** self-attention layers alone, LoRA does **not** add any text input pathway — it can only reweight what the network already does with its existing conditioning inputs (timestep/spacing/modality). To actually inject report text, LoRA must be paired with one of A/B/D's new conditioning entry point; it is a fine-tuning-efficiency technique layered on top, **not a standalone 7th strategy**.
- **Newly initialized parameters**: smallest of any strategy that touches the frozen backbone at all.
- **Expected GPU cost**: **LOW** for the LoRA parameters themselves; overall cost dominated by whichever base mechanism (A/B/D) it's layered onto.
- **Expected data requirement**: **LOW-MEDIUM** — well suited to a modest paired (image, report) set relative to the ~319k-image pretraining set the base checkpoint saw (`docs/inference.md:138`).
- **Spatially localized findings representable?** Inherits whatever the underlying new conditioning pathway provides (A→yes, B/D→no); LoRA itself neither adds nor removes localization capability.
- **Training stability**: generally **high** — directly mitigates a real risk in this setting: MR-RATE's brain-only, presumably narrower report corpus is much smaller/more homogeneous than the ~319k-image multi-source pretraining set, so full fine-tuning of the whole U-Net risks overfitting/forgetting the base checkpoint's broader geometry/modality generalization; LoRA + frozen backbone limits that risk.
- **Implementation complexity**: **low-medium as an add-on**, but non-functional alone for text — must be paired with B/D first, then A.
- **Recommended priority**: **HIGH as the fine-tuning strategy**, paired with B/D initially and (later) A — not a competing standalone option.

---

## 9. Recommendation — reuse unchanged / freeze / fine-tune / extend

**Reuse unchanged (no modification, no retraining):**
- `autoencoder_v1.pt` (VAE) — frozen, no architectural or weight change.
- `RFlowScheduler` and its shipped hyperparameters (`num_train_timesteps=1000`, `scale=1.4`, `sample_method="uniform"`) — a proven recipe already specific to this exact `diff_unet_3d_rflow-mr-brain_v0.pt` checkpoint.
- Existing modality/spacing conditioning and CFG-on-modality mechanism — keep as-is; a new report-conditioning pathway should be **additive**, not replacing these.
- Existing data-pipeline conventions (RAS reorientation, MR percentile intensity scaling, 128-multiple resize-for-embedding) — keep so any new training data stays consistent with what the frozen VAE/U-Net already expect.

**Freeze (for the initial experiments):**
- The full pretrained `diff_unet_3d_rflow-mr-brain_v0.pt` backbone weights (self-attention, conv/res-blocks, timestep/class/spacing embedding paths) — start every experiment as fine-tuning on top of this checkpoint, never from scratch.

**Fine-tune:**
- Only the newly-added conditioning parameters: a FiLM/concat projection (strategies B/D) first, then — once a positive signal is confirmed — cross-attention Q/K/V/out projections (strategy A). Apply LoRA-style parameter-efficient adaptation (strategy F) to this fine-tuning stage from the start, at low LR, on MR-RATE's paired (image, report) set, precisely because that set is narrower than the ~319k-image pretraining distribution and full fine-tuning risks forgetting the base checkpoint's broader generalization.

**Extend (net-new code, not present anywhere in this repo today):**
- A frozen clinical text encoder (the repo has zero text-encoder code of any kind — this is a wholly new dependency).
- A new tokenization/text-embedding branch in the training dataloader (`scripts/diff_model_train.py`'s `prepare_data`/`train_transforms_list` has no text handling and needs one).
- (For strategy A specifically) the actual cross-attention wiring inside the U-Net — this requires either obtaining and patching the real `monai` `DiffusionModelUNetMaisi` source (not available in this repo copy) or vendoring a modified copy of it. **This is the single highest-leverage unknown to resolve before committing to a specific strategy** — nearly every claim in §7-8 about implementation complexity for A/D/E hinges on it.

**Sequencing recommendation**: B or D first (cheapest, lowest architectural risk, validates whether report-conditioning helps at all) → layer in LoRA (F) once a positive signal is confirmed → invest in full cross-attention (A) only after B/D show a positive-but-under-localized signal, since A has the highest payoff (genuine spatial localization) but also the highest engineering risk (gated on the unverified `monai` internals). **C (ControlNet) and E (from-scratch LDM) are not recommended as the primary path** — C is a poor architectural fit unless the report is first converted into an actual spatial map, and E forfeits most of the value of the MR-RATE-trained pretrained checkpoint for a much higher data/compute bill.

---

## Appendix: discrepancies and gaps found during this audit

1. **`cfg_guidance_scale` mismatch (rflow-mr only, not mr-brain)** — `configs/config_maisi_diff_model_rflow-mr.json:34` ships `15`; `docs/inference.md:197` and `skills/infer_image-only.md:175` both document the shipped MR default as `10`. `rflow-mr-brain`'s own config correctly ships `10`, matching the docs.
2. **`skills/infer_image-only.md:142`** claims MR geometry is "validated by `check_input_ct` / `check_input_mr`" — traced to the actual code, `check_input_mr` is never called by `scripts/diff_model_infer.py` (the MR image-only entry point); only `check_input_ct`, and only for CT modality codes. See §4.
3. **`scripts/download_model_data.py:140`**'s `ValueError` message ("has to be chosen from ['ddpm-ct', 'rflow-ct', 'rflow-mr']") omits `'rflow-mr-brain'` even though the preceding `if` branch (lines 82-94) handles it correctly — a stale error message, not a functional bug.
4. **Diffusion-model training JSON sidecar producer is missing from this repo copy** — see §5.6. `diff_model_train.py` expects a `.json` file per embedding that no local script is confirmed to write.
5. **`divisor` computation in `scripts/diff_model_infer.py:300-308`** derives the VAE's downsample factor from the *diffusion U-Net's* `num_channels` length rather than from `autoencoder_def` directly — a hidden coupling between two independently-editable configs (see §4).
