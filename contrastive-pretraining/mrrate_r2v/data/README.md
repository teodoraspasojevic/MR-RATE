# `mrrate_r2v.data` — getting a volume and its report

Turns MR-RATE's un-extracted archives into `(preprocessed volume, report text)` pairs.

Full pipeline context: [`../README.md`](../README.md) and [`docs/R2V.md`](../../../docs/R2V.md).
Deep dives (axis-order forensics, measured series-selection statistics, why the geometry table
is built the way it is) live in [`../DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) — this file is
the "how do I use it" version.

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
                             # conditioning (report2ct_style{,_meta}), which encodes each
                             # separately
sample["modality"]           # "T1w" | "T2w" | "FLAIR" | "SWI" | "unknown"
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

## Axis order: the one thing to get right

Every volume here is RAS-canonical (`nibabel.as_closest_canonical`): each array axis increases
toward one anatomical direction (Right, Anterior, or Superior). What differs between the two
notations below is only *which array index holds which anatomical axis*:

| Notation | index 0 | index 1 | index 2 | Where it is used |
|---|---|---|---|---|
| `(X, Y, Z)` | **R** (→ Right) | **A** (→ Anterior) | **S** (→ Superior) | everything crossing the package boundary |
| `(D, H, W)` | **S** (→ Superior) | **R** (→ Right) | **A** (→ Anterior) | everything internal to preprocessing |

Convert only with the two helpers, never by hand:

```python
from mrrate_r2v.data.geometry import dhw_to_xyz, xyz_to_dhw
dhw_to_xyz(t)   # (D, H, W) -> (H, W, D)
xyz_to_dhw(t)   # (X, Y, Z) -> (Z, X, Y)
```

`R2VDatasetConfig.fixed_target_shape`/`fixed_target_spacing_mm` are `(D, H, W)`, like every
other internal geometry parameter — **not** `(X, Y, Z)`. If your value came from outside the
package (a CLI flag, NVIDIA's `dim`/`spacing`), convert it first:

```python
R2VDatasetConfig(geometry_mode="fixed",
                 fixed_target_shape=xyz_to_dhw((256, 176, 240)),        # user thinks XYZ
                 fixed_target_spacing_mm=xyz_to_dhw((0.8, 0.9, 1.2)))   # config stores DHW
```

Getting this wrong is **silent** for a cube at isotropic spacing and scrambles axes for
anything else — see [`DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) for exactly how.

---

## The five modules

| Module | Owns | Needs |
|---|---|---|
| `storage.py` | reading bytes out of un-extracted tars | stdlib |
| `manifest.py` | which (study, series) pairs exist and where they live | stdlib |
| `reports.py` | where the conditioning text comes from | stdlib |
| `geometry.py` | what voxel grid each series is resampled onto | stdlib |
| `dataset.py` | assembling all of it into tensors | torch |

Only `dataset.py` imports torch — a pyarrow-only interpreter can still build a manifest.
Volume preprocessing itself (RAS reorient → resample → crop/pad → normalize) lives in
`_preprocess_ops.py`.

---

## Geometry: which FOV do I get?

Two modes, set by `R2VDatasetConfig.geometry_mode`:

- **`"per_modality_plane"`** (default, and what evaluation cohorts use) — each
  (modality, plane) bucket gets its own grid, derived from NVIDIA's published median training
  FOV for that pair. Tighter fit per anatomy, but shapes differ between buckets, so
  `batch_size > 1` needs `GeometryBucketBatchSampler` as the DataLoader's `batch_sampler`
  (otherwise `collate_fn_r2v` raises with an actionable message), and numbers from it aren't
  comparable with a fixed-mode run.
- **`"fixed"`** — one shape/spacing for everything. Batching just works, but every bucket gets
  the same FOV regardless of anatomy. Use it only for a deliberate single-grid study.

Anything not in the FOV table (unknown modality, OBLIQUE plane, missing plane metadata) falls
back to 256³ @ 1 mm.

```python
from mrrate_r2v.data import build_geometry_table
build_geometry_table()[("T1w", "SAGITTAL")]
# GeometrySpec(target_shape=(256, 192, 256), target_spacing=(0.9766, 0.9167, 0.9766))
```

### Batching: `GeometryBucketBatchSampler`

One batch is one `(modality, plane)` bucket, in every geometry mode. `bucket_order="interleave"`
(default) spaces each bucket's batches evenly across the epoch, so consecutive batches carry
different modalities; `"shuffle"` is a plain flat shuffle. Use
`training.set_loader_epoch(loader, epoch)` to advance the epoch — it reseeds the dataset, batch
sampler, and sampler together in the right order.

---

## Sampling: how many times does a study appear?

`R2VDatasetConfig.series_selection` — the choice is not cosmetic:

| Value | Samples per study | Use it for |
|---|---|---|
| `"all"` (dataset default) | one per eligible series | training |
| `"one_per_study_per_bucket"` | one per (study, modality, plane) | **evaluation / a per-bucket cohort** |
| `"one_per_study_per_sequence"` | one per (study, modality) | a per-sequence cohort (collapses planes for a per-bucket one) |
| `"one_per_study_deterministic"` | one per study | a single-sequence cohort, or "one representative volume per study" |
| `"one_per_study_random"` | one per study, redrawn each epoch | training only |

For `"one_per_study_random"`, call `dataset.set_epoch(epoch)` each epoch — all of this Dataset's
randomness lives in that one call, so a run is reproducible from `(seed, epoch)`. See
[`DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) for the measured skew each mode produces.

---

## Report sources

Three, all interchangeable — implement `study_uid in store` and `store[study_uid] -> ReportRecord`
and yours works too.

| Class | Reads | Use when |
|---|---|---|
| `ShardReportStore` | per-study `report.json` in the shard tars | training from SHARDS_PATH — self-contained, **preferred** |
| `StructuredReportStore` | MR-RATE's `reports.csv` / `reports.tar.gz` | you have the DATA_PATH release CSVs |
| `SentenceJSONLReportStore` | `findings_sentences.jsonl` | fallback only — section boundaries are lost |

Pick sections with `report_sections=(...)`, from `"raw"`, `"clinical_information"`, `"technique"`,
`"findings"`, `"impression"` (default: `("findings", "impression")`).

---

## Building a manifest

Usually via the CLI:

```bash
python -m mrrate_r2v.cli.build_manifest \
    --shards-root <root> --out-csv <manifest.csv> \
    --out-report-index-csv <report_index.csv> --verify-sample 20
```

Builds from SHARDS_PATH's `shard-*.tar` + `series.parquet` layout (needs pyarrow, not torch).
Opens no archives — locators come purely from `series.parquet`'s own index, so a full build
takes seconds. **Keep `--verify-sample` on** (default 20): it resolves N random rows for real
and confirms the filename convention still holds; the build aborts if any fail.

A manifest records only *where* a series is and *what* it is (modality, plane, split) plus its
native geometry — never a geometry policy, report source, or sampling mode. Changing
`geometry_mode`, `series_selection`, or the report store therefore costs nothing; only a change
of underlying storage needs a rebuild.

MRA is excluded by default (NV-Generate-MR-Brain doesn't support it); override with
`--excluded-modalities`, `--include-derived`, `--include-localizer`.

---

## Testing

There is currently no automated test suite for this module (see the top-level
[`README.md`](../README.md) §8) — treat any test-file references you see elsewhere in the
codebase as aspirational, not passing.
