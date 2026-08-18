# `mrrate_r2v` — developer notes

Implementation-level detail, edge cases, measured evidence, and history, extracted from the
module READMEs to keep those beginner-friendly. **You don't need this to use the pipeline** —
start at [`README.md`](README.md). Read a section here when you want to know *why* something
is the way it is, or before changing a default that looks arbitrary.

Organized to mirror the module READMEs: top-level pipeline, then `data/`, `models/`, `cli/`,
`eval/`, `textenc/`.

---

## Top-level pipeline

### Why there's no cohort/prediction artifact anymore

The pipeline used to write an intermediate cohort directory (preprocessed volumes) and a
prediction set (generated volumes) to disk between training and scoring. That design had three
artifacts that could drift out of sync, and one did: training ran at `posterior_shift_mm=15`
while every cohort was built at `0`, displacing 15.8% of test cases — invisible because the
value existed on only one side of the split. `train_r2v` and `evaluate` now share one
`build_r2v_dataset` call, and evaluation streams generate→score one case at a time with nothing
written to disk in between. `cohort.py` and `predictions.py`, the on-disk contract that read and
wrote the old artifacts, are both gone along with the stages that used them.

There is deliberately no check that `--posterior-shift-mm`/`--normalizer`/`--geometry-mode`
match the run being scored: the challenge metrics (`eval/challenge_metrics.py`)
percentile-normalize each volume independently and resample on a shape mismatch, so a
preprocessing difference between training and evaluation doesn't invalidate the comparison the
way it would under a strict voxel-identity metric. What defines a run (task, split, model
identity, dataset geometry) is written straight into `metrics.json`, not hashed into an opaque
id.

### `eval/__init__.py` re-exports nothing, and several `data/` modules import no torch

A heavy dependency in one module must not make another unimportable — a pyarrow-only
interpreter has to be able to build a shards manifest. The evaluator also reads `.npy` files and
nothing else: no manifest, no archive, no Dataset, no model — which is what makes it
*impossible* for the evaluator to preprocess differently than the run it's scoring did.

### Report format: the shipped training spec, in detail

`R2VDatasetConfig.report_format` takes one name from [`textenc/formats.py`](textenc/formats.py)
— or several, comma-separated, in which case one is drawn per sample, uniformly,
deterministically from `(seed, epoch, index)`.

The shipped training spec is `findings_impression_meta,impression_findings_meta`
(`ORDER_AGNOSTIC_META_SPEC`). Two reasons, both about the challenge rather than about MR-RATE:

- **Section order.** The challenge's report layout is unknown and nothing at submission time
  can detect that the order flipped. Training on both orders means the model has seen every
  section in first position — which matters most under truncation, where a 512-token encoder
  keeps the head of the string (8–10% of MR-RATE studies truncate; RadBERT 9.2%).
- **`[MODALITY] … [PLANE] … [SPACING] x y z`** leads both orderings. Spacing is `(X, Y, Z)` and
  matches the sample's own `target_spacing_mm` exactly. It's also a numeric input via
  `spacing_tensor`, but the text encoder is the only path that sees modality, plane and spacing
  together, and it's the path a challenge request can populate with no volume attached.

Validation is pinned to the spec's **first** name — a sampled format would add format variance
to a curve whose only job is to show model improvement. The checkpoint records the whole spec,
so `cli.generate_r2v` accepts a cohort built under either ordering.

At inference (`cli.generate_r2v --report`) the prefix is prepended from `--modality`, `--plane`
and `--spacing`, and `--dim`/`--spacing` default to that bucket's own trained grid.

### One training step, in full

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

The loss is NVIDIA's own: L1 on the velocity target `x0 − ε`. Not MSE, not SNR-weighted, and no
auxiliary report/image alignment term — the adapter has to earn its keep on the original
generative objective. `Adam(lr=1e-5)`, `PolynomialLR(power=2)`, AMP + `GradScaler`. No EMA,
because there is none in the official code either.

`training.py`'s module docstring carries the line-by-line mapping to
`NV-Generate-CTMR/scripts/diff_model_train.py`, including the three deliberate differences
(optimizer sees only the adapter; `scale_factor` comes from the checkpoint; latents encoded on
the fly).

### Inference and the guidance formula

`sampling.py` mirrors `diff_model_infer.py` step for step: latent noise at `dim // 4`, RFlow
timesteps, guided model output, `scheduler.step`, then NVIDIA's `ReconModel` decode under a
`SlidingWindowInferer` (roi 80³, gaussian, overlap 0.4), then their MR postprocessing to int16
`[0, 1000]` and an axis-aligned affine.

Guidance is hierarchical, with the report as an increment on NVIDIA's modality term:

```
D_guided = D_00 + s_modality · (D_m0 − D_00) + s_report · (D_mr − D_m0)
```

| branch | modality label | report |
|---|---|---|
| `D_00` | null class | null token |
| `D_m0` | real class | null token |
| `D_mr` | real class | **the report** |

`--report-guidance-scale 0` collapses this to `diff_model_infer.py:207` exactly. All branches
run as one batched UNet call (the same trick the official code uses for its two).

### Batching and undersized buckets

`drop_last=True` (what `cli.train_r2v` uses) drops a bucket's *remainder*, never the bucket
itself. A bucket smaller than `batch_size` has no full batch at all, so a plain `drop_last`
would delete it from every epoch and the model would silently never see that (modality, plane).
On the real train split at `batch_size=8` that's SWI CORONAL (4 series) and SWI SAGITTAL (2) —
and neither exists in val or test, so no metric could have revealed it. Such buckets keep one
short batch instead and are logged as `undersized_buckets`.

### Metrics logged during training

| key | meaning |
|---|---|
| `loss` | the L1 velocity loss — the only optimisation signal |
| `lr` | current learning rate |
| `n_dropped_reports` | how many samples got the null token this step |
| `timestep_mean` | mean sampled timestep — a sanity check on scheduler behaviour |

`train_summary.json` adds steps, wall time, final/mean loss, and trainable vs frozen counts.
Every `--validate-every-steps` steps, `validation.ValidationRunner` additionally scores a small
fixed sample of `val` against the same metrics `cli.evaluate` uses, logged as `val/MSE_mean`,
`val/PSNR_mean`, `val/SSIM_mean`, `val/FID_2p5D_XY/XZ/YZ/Avg`, `val/dice`.

### The old `slurm/check_run.py` / `SUCCESS_CRITERIA.md`

Both were written against the old multi-file result layout (`summary.json`,
`per_case_metrics.csv`, ...) and were never updated for the single `metrics.json` this evaluator
now writes. Both were deleted (2026-08-18) rather than fixed; see
`docs/design/archive/11_success_criteria_pre_challenge_metrics_rewrite.md` for the preserved
content and the findings behind it if you're writing new success criteria against the current
schema.

### Other things that will bite you (full reasoning)

- **Two divisors**: `4` for sampling (output ÷ latent size), `16` for VAE-encode padding.
  Swapping them produces a valid `.nii.gz` that is 4× too small on every axis and no error.
- **`drop_last` and tiny buckets** — see above; before the fix, `drop_last=True` silently
  removed SWI CORONAL and SWI SAGITTAL from every training epoch.
- **W&B needs the proxy.** Compute nodes have no direct route off-site.
  `slurm/_common.sh:setup_proxy` exports `http(s)_proxy` and passes them into the container.
  Without it, `wandb.init(mode="online")` fails and `WandbRun` degrades to a silent no-op, so
  the run looks like "W&B just isn't logging" rather than erroring. `R2V_WANDB=online` probes
  `api.wandb.ai` before any GPU work; `R2V_NO_PROXY=1` disables the proxy.
- **Disk**: `/hnvme` has a file-count quota (61k soft), not a space quota — hence one archive
  per bucket instead of one file per volume.
- **Fréchet distance** is computed in `eval/challenge/fid_2p5d.py`
  (`scipy.linalg.sqrtm(..., disp=False)`), not via `monai.metrics.FIDMetric` (that path passes a
  `disp=` kwarg scipy removed in 1.17). Don't reintroduce the monai call.

---

## `data/`

See `data/README.md` for the beginner-facing version of everything below.

### Series selection: the measured numbers behind each mode

- `"all"` (default): right for training. For evaluation it's pseudo-replication — near-duplicate
  series from one session aren't independent, so plain means/CIs over them are misleading (this
  package's aggregation doesn't model clustering).
- `"one_per_study_per_bucket"`: the right choice for a per-bucket cohort.
  `"one_per_study_per_sequence"` looks similar but prefers the center-modality series (the axial
  T1w on MR-RATE), so it collapses **planes** within a sequence — measured on the real test
  split, it leaves T1w CORONAL with 16 cases and T2w SAGITTAL with 6, because those planes only
  survive for studies that happen to have no axial series of that modality.
- `"one_per_study_deterministic"`: collapses to near-single-modality (measured: 4861 T1w vs 25
  FLAIR, 7 T2w, 0 SWI) because the preferred series is always the center modality. Only
  meaningful for a single-sequence cohort, or when you deliberately want one representative
  volume per study regardless of modality.
- `"one_per_study_random"`: training only; same modality-collapse caveat within any one epoch.

### Axis order, in full

(X, Y, Z) = (Right, Anterior, Superior) is NV-Generate-CTMR's own array order, never permuted
further anywhere in its code. It's deliberately *not* the (D, H, W) = (S, R, A) order the
contrastive pipeline's `MRReportDataset` uses — that ordering exists because the VJEPA video
encoders hardcode "the axis after channel is the slice axis", a constraint that belongs to
those encoders, not to NV-Generate-CTMR. So preprocessing runs in (D, H, W) using
`_preprocess_ops.py`'s code, and `__getitem__` permutes to (X, Y, Z) exactly once as its final
step (`image.permute(0, 2, 3, 1)`), with every geometry field reindexed the same way via
`geometry.dhw_to_xyz`. The on-disk manifest stays (D, H, W); the reindex is an output-time
concern only.

Skipping the conversion is **silent** for a cube at isotropic spacing (256³ @ 1 mm — the NVIDIA
default, which is exactly why this went unnoticed) and scrambles axes for anything else. The
concrete failure mode: `crop_or_pad`'s posterior shift divides by `target_spacing[2]`, which is
the anterior-posterior spacing in (D, H, W) but the superior-inferior spacing in (X, Y, Z) — so
the FOV gets shifted along the wrong axis by the wrong amount.

`R2VDatasetConfig.geometry_fingerprint()` is one deliberate exception: it reports the
fixed-mode fields in (X, Y, Z) under `*_xyz` names, because everything else written into a
cohort directory (`CohortCase.shape`, the stored `.npy` volumes) is (X, Y, Z).

`R2VDatasetConfig`'s fixed-mode fallback values (`(256, 384, 384)` @ `(1.0, 0.5, 0.5)`) are
already (D, H, W), inherited from the contrastive loader. They only apply when you set
`geometry_mode="fixed"` without passing your own grid.

Both of `__getitem__`'s read paths — archive stream (the only one any current manifest builder
produces) and plain file — yield (D, H, W) by `preprocess_nii`/`preprocess_nii_from_bytes`'s own
contract, so the unconditional `permute(0, 2, 3, 1)` after them is correct either way. A third
read path must also yield (D, H, W), or the permute has to move into the branch.

### Storage: reading from tars without extracting

`ArchiveReader` seeks directly to a member and streams the bytes straight into
`preprocess_nii_from_bytes` — no disk write at all. Safe because every outer archive is a plain
uncompressed tar (`getmembers()` on a 592 GB tar takes ~1s; reading an arbitrary member 27-50 ms,
independent of file size), and per-study inner zips are STORED (uncompressed), so `zipfile`
reads their central directory off the tar member's own handle in ~2 ms with no intermediate
copy. Full evidence: `docs/design/archive/07_archive_backed_mrrate_storage.md`.

### `GeometryBucketBatchSampler`'s bucket-ordering measurements

Grouping is by the raw `(modality, plane)` pair, not `geometry.bucket_key` — always at least as
fine, so always shape-safe, and strictly finer in two places where the geometry key would let a
batch mix modalities: `geometry_mode="fixed"` (one key for everything) and the
`FALLBACK_GEOMETRY_KEY` collapse under `per_modality_plane`.

`"interleave"`'s stride scheduling was measured on the real train split (287,765 batches at
`batch_size=2`): **1–2 consecutive same-bucket batches**, i.e. 0.0003%, with every bucket
splitting exactly evenly across epoch quarters. A greedy alternative ("draw proportional to
remaining batches, never repeat the previous bucket") was tried and rejected: with two buckets
at a 3:1 ratio, the no-repeat rule forces strict alternation, draining the smaller bucket at
twice its natural rate and leaving the epoch's whole tail single-modality. Spacing by
construction beats constraining after the fact.

The bucket index is rebuilt whenever `dataset.samples_version` changes, so
`series_selection="one_per_study_random"` (which replaces `samples` on every `set_epoch`) can't
leave the sampler indexing a stale epoch's samples.

### Series selection: the measured skew each mode produces

- `"one_per_study_per_sequence"` prefers the center-modality series (axial T1w on MR-RATE), so
  it collapses **planes** — measured on the real test split, T1w CORONAL keeps 16 cases and
  T2w SAGITTAL 6, because those planes only survive for studies with no axial series of that
  modality.
- `"one_per_study_deterministic"` collapses **modalities** too (measured: 4861 T1w vs 25 FLAIR,
  7 T2w, 0 SWI on the test split), since the preferred series is always the center modality.

### `ShardReportStore`'s two design choices

**Presence is index-driven, content is not.** `__contains__` answers from a small index CSV
(built from the shards' own `has_report` column) without opening a tar, keeping the Dataset's
upfront "N dropped: no matching report" filter as cheap as every other store's. **Content is
read lazily, once per study, then cached** — eagerly reading `report.json` for ~5,000 studies
measured ~91 studies/s (~54s total), so a 90,000-study train split would cost 15+ minutes of
Dataset construction if done eagerly; lazy reads amortize that over the first epoch instead.

### Two manifest-building traps

**Don't trust the metadata CSV's `array_shape`/`array_spacing_mm` columns.** They come from a
plain `nib.load()` with no RAS reorientation — raw on-disk axis order, despite once being named
`ras_array_shape`. Native geometry is always derived independently via `read_native_geometry`,
which does reorient. `series.parquet`'s analogous columns are left unread for the same reason
(its build code isn't available to verify) and are resolved lazily instead.

**`series.parquet` uses abbreviated plane codes** (`axi`/`sag`/`cor`), not
`AXIAL`/`SAGITTAL`/`CORONAL`. `manifest.py` normalizes them; without that, every shards row would
silently fall back to the 256³ bucket regardless of its real modality and plane — no crash, so
easy to miss.

### Removed features (2026-08-18)

Three pieces of the data layer existed but were never reachable from any CLI flag, and were
removed as dead code rather than kept "just in case":

- **`node_local_cache` archive-access mode** — materialized an archive member onto `$TMPDIR`
  (bounded LRU cache, `NodeLocalCache`/`CacheBudget`/`resolve_node_local_root` in `storage.py`)
  before reading it. No CLI flag ever set `archive_access_mode`, so this never ran; every real
  read went through plain streaming. ~290 lines removed from `storage.py`.
- **`.npz` local-cache read path** (`use_preprocessed`/`preprocessed_dir`/`cache_allow_mismatch`
  on `MRReportToVolumeDataset`, plus `validate_cache_manifest`/`build_cache_manifest` in
  `_preprocess_ops.py`) — mirrored the contrastive pipeline's own `.npz` cache, but nothing ever
  built or pointed at one for r2v.
- **Two of three manifest builders** (`build_manifest_rows` for an extracted directory tree,
  `build_manifest_rows_from_data_path_zips` for DATA_PATH's `batchNN.tar` zips) — real,
  isolated, lazily-imported implementations, but nothing in this repo ever built a manifest from
  either layout; only `build_manifest_rows_from_shards_parquet` is used in practice.

`git log` has the removed implementations if a future storage location needs one of them back.

---

## `models/`

See `models/README.md` for the API-level version of everything below.

### Two different divisors — the classic silent bug

| function | value (mr-brain) | what it is for |
|---|---|---|
| `nvidia.required_spatial_divisor(ae, cfg)` | **16** | padding a volume before the VAE **encodes** it (`2^n_downsample × num_splits`) |
| `sampling.official_latent_divisor(num_channels)` | **4** | the output-size ÷ latent-size ratio when **sampling** (`2 ** (len(num_channels) - 2)`) |

Using 16 where 4 belongs produces a volume 4× too small on every axis. The run succeeds, the
NIfTI is valid, and it is wrong. Each call site names which one it means.

### Absolute checkpoint paths are mandatory

NVIDIA's env config stores `model_dir="./models"` relative to the *current working directory*.
`load_autoencoder_and_unet` therefore requires absolute overrides and rewrites
`model_dir`/`model_filename`/`existing_ckpt_filepath` consistently. A relative path fails with a
bare, uninformative `FileNotFoundError`.

### Why not just `with_conditioning=True`?

MONAI's `DiffusionModelUNetMaisi` already has a conditioning switch — but it *replaces* each
attention level's `SpatialAttentionBlock` with a `SpatialTransformer`. That's a different module
tree, so NVIDIA's pretrained weights no longer load, and the "frozen" part would no longer be
the architecture that was actually trained. The constructor rejects it outright. Instead,
cross-attention is added *alongside* the pretrained blocks as new top-level modules — the
pretrained tree stays byte-identical.

### The adapter, architecturally

```
x ──┬──────────────────────────────────────────────────────────► + ──► out
    └► GroupNorm(32) ► Conv1×1×1 (C → inner) ► flatten to (B, voxels, inner)
       ► LayerNorm ► MaskedCrossAttention(Q=voxels, K/V=report tokens)
       ► reshape ► Conv1×1×1 (inner → C)   ← zero-initialised ──────┘
```

Three deliberate properties: `proj_out` is zero-initialised (MAISI's own convention) so the
branch outputs exactly zero at step 0; conditioning enters at each block's *input* so the skip
connections leaving that level carry the report too; and there's no spatial self-attention or
GEGLU feed-forward, since the pretrained `SpatialAttentionBlock` at that level already provides
them.

`MaskedCrossAttention` subclasses MONAI's `CrossAttentionBlock` and re-implements only
`forward`, around `F.scaled_dot_product_attention`, so a key-padding mask can be passed (MONAI's
version takes none). A startup check (`_BORROWED`) fails loudly if MONAI ever renames the
attributes it relies on.

**A single conditioning token makes the block a no-op**: softmax over one key is identically 1
for every query, so `to_q`/`to_k` get zero gradient and the adapter degenerates into a
report-dependent per-channel bias applied uniformly at every voxel. Measured: `to_q` gradient
1.2e-12 at `n=1` versus 4.4e-07 at `n≤512`, and 33.8% of adapter parameters inert. Prefer a
token-sequence conditioning (see `textenc/README.md` Part 4).

### `sdpa_backend_guard` — a real, measured CUDA bug

`F.scaled_dot_product_attention` is one interface over four interchangeable CUDA kernels, and
torch picks one per call. cuDNN's kernel returns **non-finite gradients from a finite forward**
at some latent shapes — measured at 48³ (the `(T2w, CORONAL)` bucket) in bfloat16 and float16,
reproducible with random data and no adapter involved, confirmed by forcing that backend
explicitly. The guard restricts SDPA to FLASH + EFFICIENT + MATH (everything except cuDNN) and
wraps the whole of `ReportConditionedUNetMaisi.forward`, so trainer, sampler and both CLIs are
covered by one change. MATH stays in the list only as a fallback, since `MaskedCrossAttention`
passes an `attn_mask` the fused kernels may refuse. Guarding the forward is enough because the
backend is bound into the autograd node at forward time; activation checkpointing would break
that, and this model doesn't use it.

### The null report and `prepare_context`

`prepare_context(batch_size, context, context_drop_mask, context_mask)` produces the learned
`null_context` when: `context is None`, the sample's `context_drop_mask` bit is set (training
dropout or a CFG branch), or its `context_mask` row has no real tokens (an empty report would
otherwise NaN the softmax). Training dropout and inference CFG use this same code path, so they
can't drift apart.

### The strict pretrained-weight loader

`load_pretrained_maisi_weights` raises unless *all* of: every checkpoint tensor has a home in
the model, no shape mismatches, every unfilled parameter belongs to the conditioning path, and
every shared tensor is bit-equal to the file after loading. This is deliberately not NVIDIA's
own `load_state_dict(..., strict=False)`, which accepts any subset — a silent mismatch there is
a fine-tune that quietly starts from noise. It returns a `PretrainedLoadReport` whose `.format()`
goes into every run log. It also handles EMA sub-dicts, wrapper prefixes (`module.`,
`_orig_mod.`, stripped only when *every* key carries them), and NVIDIA's habit of pickling
`scale_factor` as a MONAI `MetaTensor`.

### Adapter membership: decided twice, cross-checked

`adapter_parameter_names(model)` walks the actual `context_proj`/`*_cross_attn`
submodules/`null_context` by object identity, then compares that set against the name-prefix
tuple `CONDITIONING_PREFIXES`, raising if they disagree. `assert_only_adapter_trainable` (called
before the first optimizer step) then proves every adapter parameter is trainable, no base or
text-encoder parameter is, and the optimizer's parameter set is exactly the adapter set — so one
wrong `requires_grad` can't silently fine-tune the 180M base at `lr=1e-5` instead of the 8M
adapter.

### Checkpoint format details

The adapter checkpoint (`mrrate_r2v_adapter_v1`) deliberately does **not** duplicate the ~700MB
base weights — it identifies the base by sha256 (`base_checkpoint.sha256`), and
`load_adapter_checkpoint` refuses to load onto a different base unless
`allow_base_mismatch=True`. It also records its own text encoder, context width, and adapter
geometry, so `cli.generate_r2v` doesn't re-specify any of it on the command line. A
full-UNet-format checkpoint (`--save-format full|both`) is also available for NVIDIA's own
tooling to read.

### Traps

- Adapters have no buffers today, and `adapter_state_dict` would silently drop any that appear —
  it asserts that stays true.
- `use_flash_attention=True` requires CUDA; the adapters themselves always use
  `scaled_dot_product_attention` (exact, CPU-capable), so adapter logic stays testable on CPU.
- `--conditioning-levels` changes the adapter set, so a checkpoint trained with one value can't
  load into a model built with another.

---

## `cli/`

See `cli/README.md` for the current flag reference. Historical note: an earlier 5-stage design
(`cli.preprocess` → `cli.predict_vae`/`predict_generation`/`predict_r2v` →
`cli.import_predictions`) built a frozen cohort and prediction set on disk; it was replaced
2026-08-10 by the current 3-stage design described in the top-level section above, and those CLI
modules were deleted (`cli/README.md` described them until 2026-08-18, when that section was
removed as stale). If you see a reference to any of them elsewhere (an archived design doc,
`docs/R2V.md`), it describes that earlier design, not the current one.

### What `--conditioning-levels 0 0 1 1` means

The UNet is a U: it downsamples the latent through 4 resolution levels, crosses the bottleneck,
then upsamples back through the mirrored 4. One flag position per *level*, not per block — a `1`
places a report cross-attention adapter on **both** sides of the U at that resolution:

```
level                 0         1         2         3      ── bottleneck ──
resolution (latent)  full      1/2       1/4       1/8
channels              64       128       256       512
--conditioning-levels  0         0         1         1
encoder              –         –       adapter   adapter
decoder              –         –       adapter   adapter        + mid adapter
                                                                  (unless --no-condition-mid)
```

So `0 0 1 1` = 5 adapters: the two coarsest encoder blocks, the two decoder blocks that mirror
them, and the bottleneck. Levels are ordered coarse-to-the-right (index 0 = full-resolution,
cheap-channel end); the decoder is indexed by the level it *mirrors*, so position `i` always
means "this resolution, going down and coming back up" regardless of the reversed block order.

Why the default is the two coarsest levels: it matches where the pretrained model already puts
its own attention, a report describes global findings rather than per-voxel texture, and
attention cost grows with voxel count — turning on level 0 means attending over the
full-resolution latent at every step.

### `skipped_steps`, and why it matters more than it looks

A training step whose gradient is non-finite is skipped so the weights survive, but the batch is
lost — and such failures have been geometry-specific in practice, so a nonzero count can mean
one `(modality, plane)` bucket was trained on *zero* times. Healthy is a flat 0. DDP and gradient
accumulation multiply the exposure: one bad micro-batch on any rank poisons the whole
accumulated, all-reduced gradient. Check it before reading any training result as meaningful.

### `--report-format`'s default per `--conditioning`

A/B/C (pooled or single-encoder token configurations) default to
`findings_impression_meta,impression_findings_meta`; D (sectioned fusion) encodes sections
separately and takes no `--report-format` at all — see `mrrate_r2v/README.md` for why the
training spec samples both orderings.

---

## `eval/`

The metrics (`MSE_mean`, `PSNR_mean`, `SSIM_mean`, `FID_2p5D_*`, `dice`) are a vendored port of
the official VLM3D `mr-volume-generation` challenge's own evaluation code
(`github.com/forithmus/VLM3D-Dockers/tree/main/mr_challenges/mrgen_evaluation`), reproduced
exactly including two behavioural quirks: a missing case is excluded from the means rather than
penalized with a worst-case value, and `dice` is a literal copy of `SSIM_mean` (the platform's
own primary-metric shim, not real Dice). An older, much larger metric set (~30 metrics:
fidelity/perceptual/distribution/anatomy/report_alignment/report_consistency) existed before
this and is gone — see `docs/design/archive/09_older_evaluation_implementation_audit.md`.

### Known gap: no report-to-volume semantic-alignment metric

Neither `cli.evaluate` (MSE/PSNR/SSIM/FID_2p5D) nor training-time validation measures whether a
generated volume matches what its conditioning report *says* — both ask "does this look like a
plausible/similar brain," not "does it show the described pathology." A prior version of this
project had a from-scratch stand-in (a blinded pathology classifier); it was removed as unused
rather than kept unwired, so revisiting this direction means rebuilding it, not re-enabling it.

---

## `textenc/`

See `textenc/README.md` for the usage-level version.

### Format choice: the evidence behind the defaults

`findings_impression` is the default because findings and impression carry complementary
information and neither alone is sufficient (findings: median 143 words, has localisation;
impression: median 19 words, has the radiologist's synthesis, but is missing 8.9% of the time).
This also matches what the 2025 VLM3D CT-track winner used.

`impression_findings` has exactly the same content and token count as the default — the only
difference is what survives a 512-token truncation. Since ~9% of reports truncate at 512, this
is free insurance for a short-context encoder; it changes nothing for an 8192-context one.

Fuller comparison (truncation rate at 512 tokens, RadBERT):

| format | truncated | works when a section is missing |
|---|---|---|
| `findings_impression` / `impression_findings` | 9.2% | yes — empty sections are dropped |
| `findings` | 2.2% | yes |
| `impression` | 0.01% | empty for 8.9% of studies |
| `clinical_findings_impression` | 9.6% | indication present for only 48% |
| `full_structured` | 16.5% | protocol text is train/test-inconsistent |
| `raw` | 19.8% | includes other body regions |

Every format obeys three rules: negation is never removed (no format cleans, rewrites,
summarises, or samples sentences — only selects and marks released fields); empty sections are
dropped, never emitted as a bare marker; and no format invents text (the one apparent exception,
`findings_impression_meta`, only adds values the *caller* supplies from structured metadata,
never parsed from the report). `findings_impression_meta`'s metadata prefix commits you to
having that metadata at inference time — a risk, since the challenge's input schema is still
unpublished; supplying modality as a separate categorical embedding (which the pipeline already
does) has no such problem, since a missing category is just the null class.

### Encoder behaviours worth knowing

- **Frozen by default, and it stays frozen** — a frozen encoder overrides `.train()`, so calling
  `.train()` on an enclosing module can't silently switch its dropout back on.
- **Truncation is counted, never silent**: the encoder tokenizes once without truncation first,
  purely to know what it's about to drop (`encoder.log_truncation_summary()`). At a 512-token
  budget, ~9% of MR-RATE reports lose their tail — one report in eleven.
- **`max_length` is checked against the real checkpoint**: RoBERTa spends 2 of its 514 position
  slots on an offset, so `radbert`'s true budget is 512; asking for more raises at build time,
  not a confusing CUDA index crash at training step 1.
- **No `trust_remote_code`, anywhere**: `cxr_bert`'s checkpoint declares custom model code that
  would otherwise execute on load; instead its standard BERT weights load into a stock
  `BertModel` with the unused heads dropped.

### The one rule for conditioning configurations: `n = 1` makes cross-attention a no-op

Same effect as in `models/`: softmax over a single key is identically 1 for every query, so the
report collapses to a per-channel bias applied uniformly at every voxel. Measured after 40 real
training steps, `to_q` gradient norm: configuration A (`n=1`) 1.2e-12 (dead, 33.8% of adapter
params inert) vs. B/C (`n≤512`) ~1e-6 vs. D (`n=2`) 2.7e-07. Keeping the token axis costs
nothing measurable (29s per 40 steps and 13.8 GiB peak, identical to `n=1`). Configuration A is
kept as a pooled baseline (CXR-BERT's CLS was CLIP-trained to summarise, and it's the exact form
CTFlow uses), not as a recommendation.

**E is D plus one token.** Configurations A/B/C put `[MODALITY]/[PLANE]/[SPACING]` at the head
of their joined string (the `*_meta` formats); D encodes each section on its own tokenizer and
never joins them, so it has nowhere to put that prefix. E gives it a conditioning token of its
own, appended, so a D result and an E result differ in exactly one token.

`n` varies per batch (padded to that batch's longest report, capped at `--max-report-tokens`)
and per tokenizer — nothing resamples it; `ContextProjection` maps `(B, n, D) → (B, n, C)`
regardless of `n`.

### `--max-report-tokens`, measured

Over 8,000 train reports at `findings_impression_meta`: CXR-BERT truncates 3.2% at 512 tokens
(mean 266, p99 629); RadBERT truncates 11.8% (mean 350, p99 805). Both hard-cap at 512, so p99
coverage isn't purchasable at that budget — `bioclinical_mbert` (8192 context) is the only
staged way to remove truncation entirely.

### Fusion cost

Fusion (`SectionedFusionEmbedder`, configurations D/E) costs what it says: three encoders means
three forward passes, roughly triple the time and memory, and a wider conditioning tensor. Only
worth it if it measurably helps — see `docs/TEXT_ENCODERS.md` for the measurement that justified
it here.
