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
config = R2VDatasetConfig(split="train", geometry_mode="fixed",
                          fixed_target_shape=(256, 256, 256),
                          fixed_target_spacing_mm=(1.0, 1.0, 1.0))
dataset = MRReportToVolumeDataset(rows, ShardReportStore("report_index.csv"), config)
sample  = dataset[0]
```

## What one sample is

```python
sample["image"]              # torch.Tensor [1, X, Y, Z] -- preprocessed, model-ready
sample["report_text"]        # str -- the conditioning text
sample["modality"]           # "T1w" | "T2w" | "FLAIR" | "SWI"
sample["acquisition_plane"]  # "AXIAL" | "SAGITTAL" | "CORONAL" | ...
sample["contrast_state"]     # "unknown" -- not derivable from the release
sample["skull_state"]        # "defaced_not_stripped"
sample["target_shape"]       # the grid it was resampled onto      [X, Y, Z]
sample["target_spacing_mm"]  #                                     [X, Y, Z]
sample["native_shape"]       # pre-resample geometry, for provenance
sample["native_spacing_mm"]
sample["native_fov_mm"]
sample["study_key"]          # identifiers -- for matching, never to log verbatim
sample["series_key"]
```

`sample["image"].shape[-3:] == tuple(sample["target_shape"].tolist())` always holds.

### The axis order, and why it is what it is

**X = Right-Left, Y = Anterior-Posterior, Z = Superior-Inferior**, after RAS canonicalization.

That is NV-Generate-CTMR's own array order, which it never permutes internally — so what this
Dataset returns goes straight into the model with no further reshaping.

It is deliberately *not* the `(D, H, W) = (S, R, A)` order the contrastive `MRReportDataset` uses.
That ordering exists because the VJEPA video encoders hardcode "the axis after channel is the slice
axis" (asymmetric Conv3d kernels, tubelet embedding, depth chunking). That constraint belongs to
those encoders, not to the generative model this Dataset targets.

So: preprocessing runs in `(D, H, W)` using `scripts/data.py`'s shared code, and `__getitem__`
converts to `(X, Y, Z)` exactly once as its last step (`image.permute(0, 2, 3, 1)`), reindexing every
geometry field the same way. The manifest on disk stays `(D, H, W)`.

### The one place this can bite you

`R2VDatasetConfig.fixed_target_shape` and `fixed_target_spacing_mm` are **`(D, H, W)`**, like every
other internal geometry parameter — *not* the `(X, Y, Z)` the Dataset returns. Anything arriving from
outside the package (a CLI flag, NVIDIA's `dim`/`spacing`) is `(X, Y, Z)` and must be converted:

```python
from mrrate_r2v.data.geometry import xyz_to_dhw
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

The Dataset's *own* defaults (`(256, 384, 384)` @ `(1.0, 0.5, 0.5)`) are already `(D, H, W)`,
inherited from the contrastive loader — leave them alone.

### The `.npz` cache is `(D, H, W)` too

`use_preprocessed=True` reads `preprocess_volumes.py`'s cache, which stores `preprocess_nii` output
verbatim as `[N, D, H, W]`. So the unconditional `permute(0, 2, 3, 1)` in `__getitem__` is correct
for all three read paths — live NIfTI, archive stream, and cache. If you ever add a fourth read
path, it must also yield `(D, H, W)`, or move the permute into the branch.

---

## The five modules

| Module | Owns | Needs |
|---|---|---|
| `storage.py` | reading bytes out of un-extracted tars; the node-local cache | stdlib |
| `manifest.py` | which (study, series) pairs exist and where they live | stdlib |
| `reports.py` | where the conditioning text comes from | stdlib |
| `geometry.py` | what voxel grid each series is resampled onto | stdlib |
| `dataset.py` | assembling all of it into tensors | torch |

Only `dataset.py` imports torch. That is on purpose: it lets a pyarrow-only interpreter build a
manifest, which is why there is no duplicate standalone builder script.

Volume preprocessing itself — RAS reorient → resample → crop/pad → normalize — is
`scripts/data.py`'s code imported unchanged via `_preprocess_ops.py`. Both pipelines therefore
prepare a volume identically, by construction rather than by discipline.

---

## Geometry: which FOV do I get?

Two modes, set by `R2VDatasetConfig.geometry_mode`.

**`"fixed"`** — one shape and spacing for everything. Batching just works, and volumes are
comparable across models. This is what evaluation cohorts use.

**`"per_modality_plane"`** (the Dataset's own default) — each (modality, plane) gets a shape sized
from NVIDIA's published median training FOVs, rounded *up* to a multiple of 16 so anatomy is never
truncated below the distribution it targets. Tighter fit per anatomy, but:

- shapes differ between buckets, so `batch_size > 1` needs `GeometryBucketBatchSampler` as the
  DataLoader's `batch_sampler` — otherwise `collate_fn_r2v` raises (with an actionable message, not
  a raw `torch.stack` traceback);
- numbers from it are not comparable with a fixed-mode run.

Anything not in the table — unknown modality, OBLIQUE plane, missing plane metadata — falls back to
256³ at 1 mm.

```python
from mrrate_r2v.data import build_geometry_table
build_geometry_table()[("T1w", "SAGITTAL")]   # GeometrySpec(shape=(256,176,256), spacing=(1,1,1))
```

---

## Sampling: how many times does a study appear?

`R2VDatasetConfig.series_selection`:

| Value | Samples per study | Trade-off |
|---|---|---|
| `"all"` (default) | one per eligible series | full granularity; big studies count more |
| `"one_per_study_deterministic"` | 1, preferring the center-modality series | no overrepresentation; non-preferred series never seen |
| `"one_per_study_random"` | 1, redrawn each epoch | no overrepresentation, full coverage over many epochs |

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

Pick sections with `report_sections=("findings", "impression")`. The full selected text is used every
call — never truncated, never randomly subsampled.

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
`cache_max_bytes`/`cache_max_files`.

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

**Always pass `--verify-sample`.** It resolves N random rows for real and confirms the filename
convention still holds; the build aborts if any fail.

### Two traps worth knowing

**Don't trust the metadata CSV's `array_shape`/`array_spacing_mm` columns.** They come from a plain
`nib.load()` with no RAS reorientation — raw on-disk axis order, despite once being named
`ras_array_shape`. Native geometry is always derived independently via `read_native_geometry`, which
does reorient. Series.parquet's analogous columns are left unread for the same reason (its build code
isn't available to verify), and resolved lazily instead.

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
