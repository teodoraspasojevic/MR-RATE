# `mrrate_r2v.data` — getting a volume and its report

Turns MR-RATE's un-extracted archives into `(preprocessed volume, report text)` pairs.

Full pipeline context: [`docs/R2V.md`](../../../docs/R2V.md). This file is about the data layer
itself — use it when you are training a model, adding a new report source, or debugging why a
volume came out the shape it did.

---

## Use it in five lines

```python
from mrrate_r2v.data import (MRReportToVolumeDataset, R2VDatasetConfig,
                             ShardReportStore, read_manifest_csv)

rows   = read_manifest_csv("manifest_shards_native.csv")
config = R2VDatasetConfig(split="train")          # geometry_mode="per_modality_plane" by default
dataset = MRReportToVolumeDataset(rows, ShardReportStore("report_index.csv"), config)
sample  = dataset[0]
```

## What one sample is

```python
sample["image"]              # torch.Tensor [1, X, Y, Z] -- preprocessed, model-ready
                             # dtype = config.dtype (default torch.bfloat16)
sample["report_text"]        # str -- the conditioning text, composed by R2VDatasetConfig.report_format
sample["report_sections_text"]  # dict -- unjoined sections; used only by a sectioned-fusion
                             # conditioning (report2ct_style), which encodes each separately
sample["modality"]           # "T1w" | "T2w" | "FLAIR" | "SWI" | "unknown"
                             # ("MRA" only if you override --excluded-modalities)
sample["acquisition_plane"]  # "AXIAL" | "SAGITTAL" | "CORONAL" | "OBLIQUE" | "unknown"
sample["contrast_state"]     # "unknown" -- not derivable from the release
sample["skull_state"]        # "defaced_not_stripped"
sample["target_shape"]       # the grid it was resampled onto   int64  tensor [X, Y, Z]
sample["target_spacing_mm"]  #                                  float32 tensor [X, Y, Z]
sample["native_shape"]       # pre-resample geometry, for provenance   (also [X, Y, Z])
sample["native_spacing_mm"]
sample["native_fov_mm"]
sample["study_key"]          # identifiers -- for matching, never to log verbatim
sample["series_key"]
```

`sample["image"].shape[-3:] == tuple(sample["target_shape"].tolist())` always holds.

---

## Axis order: what `(X, Y, Z)`, `(D, H, W)` and `(R, A, S)` actually mean

Three notations, **two of them are array-index orders and one is a set of anatomical
direction names**. That is the whole confusion, so read this before anything else.

`(R, A, S)` is not a shape. It names *anatomical directions* — Right, Anterior, Superior —
and is what "RAS canonicalization" (`nibabel.as_closest_canonical`) guarantees: after it, each
array axis increases toward one of those three directions. Every volume in this package is
RAS-canonical. What differs between the two shape notations below is only **which array index
holds which anatomical axis** — never the physical orientation of the anatomy.

| Notation | index 0 | index 1 | index 2 | Where it is used |
|---|---|---|---|---|
| `(X, Y, Z)` | **R** (→ Right) | **A** (→ Anterior) | **S** (→ Superior) | everything crossing the package boundary |
| `(D, H, W)` | **S** (→ Superior) | **R** (→ Right) | **A** (→ Anterior) | everything internal to preprocessing |

So `(X, Y, Z) = (R, A, S) = (H, W, D)` and `(D, H, W) = (S, R, A)`: same volume, same anatomy, axes
permuted. `X/Y/Z` are the NIfTI axis names (a NIfTI file natively stores `dim0=R, dim1=A,
dim2=S`, which is why this is also NVIDIA's `dim`/`spacing` order). `D/H/W`
("depth/height/width") is borrowed video-tensor vocabulary and carries **no** anatomical
meaning of its own — here `D` happens to be the superior-inferior slice axis.

Convert only with the two helpers, never by hand:

```python
from mrrate_r2v.data.geometry import dhw_to_xyz, xyz_to_dhw
dhw_to_xyz(t)   # (D, H, W) -> (H, W, D)
xyz_to_dhw(t)   # (X, Y, Z) -> (Z, X, Y)
```

### Why two orders at all

`(X, Y, Z)` is NV-Generate-CTMR's own array order, which it never permutes internally — so
what this Dataset returns goes straight into the model with no further reshaping.

It is deliberately *not* the `(D, H, W)` order the contrastive `MRReportDataset` uses. That
ordering exists because the VJEPA video encoders hardcode "the axis after channel is the slice
axis" (asymmetric Conv3d kernels, tubelet embedding, depth chunking). That constraint belongs to
those encoders, not to the generative model this Dataset targets.

So: preprocessing runs in `(D, H, W)` using `scripts/data.py`'s shared code (its
`transpose(2, 0, 1)` is where `(X, Y, Z)` becomes `(D, H, W)`), and `__getitem__` converts back
to `(X, Y, Z)` exactly once as its last step (`image.permute(0, 2, 3, 1)`), reindexing every
geometry field the same way. The manifest on disk stays `(D, H, W)`.

### The one place this can bite you

`R2VDatasetConfig.fixed_target_shape` and `fixed_target_spacing_mm` are **`(D, H, W)`**, like
every other internal geometry parameter — *not* the `(X, Y, Z)` the Dataset returns. Anything
arriving from outside the package (a CLI flag, NVIDIA's `dim`/`spacing`) is `(X, Y, Z)` and must
be converted:

```python
R2VDatasetConfig(geometry_mode="fixed",
                 fixed_target_shape=xyz_to_dhw((256, 176, 240)),        # user thinks XYZ
                 fixed_target_spacing_mm=xyz_to_dhw((0.8, 0.9, 1.2)))   # config stores DHW
```

Skipping the conversion is **silent** for a cube at isotropic spacing (256³ @ 1 mm — the NVIDIA
default, which is exactly why this went unnoticed) and scrambles axes for anything else. The
concrete failure: `crop_or_pad`'s posterior shift divides by `target_spacing[2]`, which is the
anterior-posterior spacing in `(D, H, W)` but the superior-inferior spacing in `(X, Y, Z)` — so the
FOV gets shifted along the wrong axis by the wrong amount. `cli.preprocess` converts correctly;
`test_cohort_selection.py::test_preprocess_converts_cli_xyz_geometry_into_internal_dhw` guards it.

One exception, by design: `R2VDatasetConfig.geometry_fingerprint()` reports these two fields in
`(X, Y, Z)` under `*_xyz` names, because everything else written into a cohort directory
(`CohortCase.shape`, the stored `.npy` volumes) is `(X, Y, Z)`.

`R2VDatasetConfig`'s fixed-mode fallback values (`(256, 384, 384)` @ `(1.0, 0.5, 0.5)`) are
already `(D, H, W)`, inherited from the contrastive loader. They only apply when you set
`geometry_mode="fixed"` without passing your own grid — leave them alone.

### The `.npz` cache is `(D, H, W)` too

`use_preprocessed=True` reads `preprocess_volumes.py`'s cache, which stores `preprocess_nii`
output verbatim as `[N, D, H, W]`. So the unconditional `permute(0, 2, 3, 1)` in `__getitem__` is
correct for all three read paths — live NIfTI, archive stream, and cache. If you ever add a fourth
read path, it must also yield `(D, H, W)`, or move the permute into the branch.

`use_preprocessed=True` additionally **requires `geometry_mode="fixed"`** (the constructor raises
otherwise): one `.npz` directory carries a single shape/spacing, while `per_modality_plane` needs
a different shape per bucket. Per-bucket caching is not implemented — the cohort stage
(`cli.preprocess`) is what materializes per-bucket volumes.

---

## The five modules

| Module | Owns | Needs |
|---|---|---|
| `storage.py` | reading bytes out of un-extracted tars; the node-local cache | stdlib |
| `manifest.py` | which (study, series) pairs exist and where they live | stdlib |
| `reports.py` | where the conditioning text comes from | stdlib |

The Dataset only *composes* the text; it never tokenises or encodes. Which words go in is
`report_format` (`textenc/formats.py`), and which model turns them into numbers is `--conditioning`
(`textenc/README.md` Part 4). The Dataset is unaware of both — it emits strings.
| `geometry.py` | what voxel grid each series is resampled onto | stdlib |
| `dataset.py` | assembling all of it into tensors | torch |

Only `dataset.py` imports torch. That is on purpose: it lets a pyarrow-only interpreter build a
manifest, which is why there is no duplicate standalone builder script. `__init__.py`'s re-exports
are lazy (PEP 562) to keep that true.

Volume preprocessing itself — RAS reorient → resample → crop/pad → normalize — is
`scripts/data.py`'s code imported unchanged via `_preprocess_ops.py`. Both pipelines therefore
prepare a volume identically, by construction rather than by discipline.

---

## Geometry: which FOV do I get?

Two modes, set by `R2VDatasetConfig.geometry_mode`.

**`"per_modality_plane"`** (the default, and what evaluation cohorts use) — each
(modality, plane) bucket gets its own grid, built from NVIDIA's published median training FOV for
that pair (`NV_BRAIN_FOV_MM`):

- `shape` = FOV rounded to the **nearest multiple of 32** (`UNET_SPATIAL_MULTIPLE`) — the
  diffusion UNet's hard constraint: 4 levels, latent = shape/4, latent must be divisible by 8.
  Note this is stricter than the VAE's own divisor of 16; a div-16-but-not-32 shape passes the
  autoencoder and then fails the UNet's skip connections.
- `spacing` = `FOV / shape`, i.e. **derived, not fixed at 1 mm**. The physical FOV therefore
  equals NVIDIA's recommendation *exactly*, and the resulting spacing lands within ±10% of 1 mm
  for every published bucket. Spacing is a real conditioning input to the UNet
  (`spacing_tensor`), and the FOV table is the quantity NVIDIA actually validated — fixing
  spacing at 1 mm and rounding the shape up instead would over-cover the recommended FOV by up
  to 30 mm on an axis.

Tighter fit per anatomy, but:

- shapes differ between buckets, so `batch_size > 1` needs `GeometryBucketBatchSampler` as the
  DataLoader's `batch_sampler` — otherwise `collate_fn_r2v` raises (with an actionable message, not
  a raw `torch.stack` traceback);
- numbers from it are not comparable with a fixed-mode run.

### Batching: `GeometryBucketBatchSampler`

**One batch is one `(modality, plane)` bucket, in every geometry mode.** The sampler groups on the
raw pair rather than on `geometry.bucket_key`, which is always at least as fine — so it stays
shape-safe for `collate_fn_r2v` — and is strictly finer in the two places where the geometry key
would let a batch mix modalities: `geometry_mode="fixed"` (one key for everything) and the
`FALLBACK_GEOMETRY_KEY` collapse under `per_modality_plane`.

Enforcement lives in the sampler, not in `collate_fn_r2v`. The collate function only *checks*
shapes, so a plain `DataLoader(batch_size=N)` over a fixed-geometry dataset stays legal.

`bucket_order` picks the order the buckets come out in. Both settings use every series exactly
once per epoch, apply no frequency weighting or temperature, and never resample a bucket — a
bucket's share of the epoch is its share of the data, so the train split's 2-series
`(SWI, SAGITTAL)` bucket contributes 2 series:

- **`"interleave"`** (default) — stride scheduling: a bucket holding `n` of the epoch's `total`
  batches claims virtual times `(k + phase) * total / n`, and batches are emitted in virtual-time
  order. Each bucket therefore lands evenly spaced across the whole epoch at its natural rate.
  Measured on the real train split (287,765 batches at `batch_size=2`): **1–2 consecutive
  same-bucket batches**, i.e. 0.0003%, and every bucket splits exactly evenly across epoch
  quarters. Consecutive batches differing in bucket also means gradient accumulation accumulates
  *across* modalities — with bucket-pure micro-batches, drawing per optimiser step instead would
  make every update single-modality.
- **`"shuffle"`** — one flat shuffle over all batches (the pre-2026-08 behaviour). Same epoch
  contents; nothing prevents a run of same-bucket batches.

A greedy "draw proportional to remaining, never repeat the previous bucket" was tried and
rejected: with two buckets at 3:1 the no-repeat rule forces strict alternation, draining the
smaller bucket at twice its natural rate and leaving the epoch's whole tail single-modality.

The bucket index is rebuilt whenever `dataset.samples_version` changes, so
`series_selection="one_per_study_random"` — which replaces `samples` on every `set_epoch` — cannot
leave the sampler indexing the previous epoch's samples. Use `training.set_loader_epoch(loader,
epoch)` to advance the epoch; it reseeds the dataset, batch sampler and sampler together.

**`"fixed"`** — one shape and spacing for everything, from `fixed_target_shape` /
`fixed_target_spacing_mm`. Batching just works and volumes share one grid, but it puts every
bucket on the same FOV regardless of anatomy. Keep it for a deliberate single-grid study; it is
not the default anywhere.

Anything not in the FOV table — unknown modality, OBLIQUE plane, missing plane metadata — falls
back to 256³ at 1 mm (`DEFAULT_FALLBACK_FOV_MM`, NVIDIA's own shipped default inference
geometry), reachable as `FALLBACK_GEOMETRY_KEY`.

```python
from mrrate_r2v.data import build_geometry_table
build_geometry_table()[("T1w", "SAGITTAL")]
# GeometrySpec(target_shape=(256, 192, 256),                       # (D, H, W), all div-32
#              target_spacing=(0.9766, 0.9167, 0.9766))            # = FOV (250, 176, 250) / shape
```

The table is keyed on the 15 published (modality, plane) pairs — including MRA, even though
`manifest.py` excludes MRA series by default — plus the fallback entry.

---

## Sampling: how many times does a study appear?

`R2VDatasetConfig.series_selection` — five modes, and the choice is not cosmetic:

| Value | Samples per study | Use / trade-off |
|---|---|---|
| `"all"` (config default) | one per eligible series | training; full granularity. For *evaluation* it is pseudo-replication — near-duplicate series from one session are not independent, so means overweight multi-series studies and CIs come out falsely narrow |
| `"one_per_study_per_bucket"` | one per (study, modality, plane) | **the right choice for a per-bucket cohort** — and `cli.preprocess`'s default |
| `"one_per_study_per_sequence"` | one per (study, modality) | right for a per-sequence cohort, wrong for a per-bucket one: it prefers the center-modality series, which on MR-RATE is the *axial* T1w, so it collapses **planes** — measured on the real test split, T1w CORONAL keeps 16 cases and T2w SAGITTAL 6 |
| `"one_per_study_deterministic"` | 1, preferring the center-modality series | collapses **modalities** too (measured: 4861 T1w vs 25 FLAIR, 7 T2w, 0 SWI). Only for a single-sequence cohort, or one representative volume per study |
| `"one_per_study_random"` | 1, redrawn each epoch | training only; full coverage over many epochs, same modality-collapse caveat within any single epoch |

For `"one_per_study_random"`, call `dataset.set_epoch(epoch)` each epoch. All of this Dataset's
randomness lives in that one call — nothing is drawn from the global RNG inside `__getitem__`, so a
run is reproducible from `(seed, epoch)`.

To balance modalities rather than studies:

```python
from mrrate_r2v.data import get_modality_balanced_sampler
loader = DataLoader(dataset, batch_size=2, sampler=get_modality_balanced_sampler(dataset),
                    collate_fn=collate_fn_r2v)
```

---

## Report sources

Three, all interchangeable — implement `study_uid in store` and `store[study_uid] -> ReportRecord`
and yours works too.

| Class | Reads | Use when |
|---|---|---|
| `ShardReportStore` | per-study `report.json` in the shard tars | training from SHARDS_PATH — self-contained, **preferred** |
| `StructuredReportStore` | MR-RATE's `reports.csv` / `reports.tar.gz` | you have the DATA_PATH release CSVs |
| `SentenceJSONLReportStore` | `findings_sentences.jsonl` | fallback only — section boundaries are lost |

`ShardReportStore` answers "does this study have a report?" from a small index CSV without opening a
tar, then reads content lazily and caches it. That matters: eager reads measured ~91 studies/s, so a
90,000-study split would cost 15+ minutes of Dataset construction.

Pick sections with `report_sections=(...)`, from `"raw"`, `"clinical_information"`, `"technique"`,
`"findings"`, `"impression"` (default: `("findings", "impression")`). Each selected section is
prefixed with its own name; the full selected text is used every call — never truncated, never
randomly subsampled.

---

## Storage: reading from tars without extracting

Nothing is ever extracted. `ArchiveReader` seeks directly to a member.

```python
R2VDatasetConfig(archive_access_mode="stream")            # default: no disk write at all
R2VDatasetConfig(archive_access_mode="node_local_cache")  # materialize onto $TMPDIR, LRU-bounded
```

Use `node_local_cache` when you want OS page-cache reuse across epochs. It resolves `$TMPDIR` at
first real use and **fails loudly** if no valid node-local root exists, rather than quietly filling a
shared workspace. Budget defaults to 200 GB / 20,000 files; override with
`cache_max_bytes`/`cache_max_files`. Both settings apply only to `backend="archive"` manifest rows.

Why seeking is safe: every outer archive is a plain uncompressed tar (`getmembers()` on a 592 GB tar
takes ~1s; reading an arbitrary member 27-50 ms, independent of file size), and per-study inner zips
are STORED, so `zipfile` reads their directory off the tar member's own handle in ~2 ms.

---

## Building a manifest

Usually via the CLI:

```bash
python -m mrrate_r2v.cli.build_manifest --source shards_parquet \
    --shards-root <root> --out-csv <manifest.csv> \
    --out-report-index-csv <report_index.csv> --verify-sample 20
```

Three sources: `shards_parquet` (needs pyarrow, no torch), `data_path_archive` (needs a metadata +
splits CSV), `extracted_dir` (needs torch + nibabel). The two archive sources open no archives at
all — they build locators from each root's own index, so a full build takes seconds.

**Keep `--verify-sample` on** (default 20). It resolves N random rows for real and confirms the
filename convention still holds; the build aborts if any fail.

A manifest records only *where* a series is and *what* it is (modality, plane, split) plus its
native geometry — never a geometry policy, report source, or sampling mode. Changing
`geometry_mode`, `series_selection`, or the report store therefore costs nothing; only a change of
underlying storage needs a rebuild.

### Two traps worth knowing

**Don't trust the metadata CSV's `array_shape`/`array_spacing_mm` columns.** They come from a plain
`nib.load()` with no RAS reorientation — raw on-disk axis order, despite once being named
`ras_array_shape`. Native geometry is always derived independently via `read_native_geometry`, which
does reorient and returns `(D, H, W)`. Series.parquet's analogous columns are left unread for the
same reason (its build code isn't available to verify), and resolved lazily instead.

**`series.parquet` uses abbreviated plane codes** (`axi`/`sag`/`cor`), not `AXIAL`/`SAGITTAL`/
`CORONAL`. `manifest.py` normalizes them. Without that, every shards row silently falls back to the
256³ bucket regardless of its real modality and plane — no crash, so easy to miss.

### Eligibility

MRA is excluded by default: NV-Generate-MR-Brain does not support it, and it is ~0.02% of series.
Derived and localizer series are also excluded, though those are defensive re-checks — the public
release already filtered them. Override with `--excluded-modalities`, `--include-derived`,
`--include-localizer`.

---

## Testing

```bash
python -m pytest tests/test_data_dataset.py tests/test_data_storage.py -v --no-cov
```

CPU, synthetic fixtures, seconds. No real data or checkpoints needed.
