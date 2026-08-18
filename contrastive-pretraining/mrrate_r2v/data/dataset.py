"""The Dataset: one item = one (report text, single target volume) pair.

    sample["image"]        torch.Tensor [1, X, Y, Z]   preprocessed, model-ready
    sample["report_text"]  str                         the conditioning text
    sample["modality"], ["acquisition_plane"], ["contrast_state"], ["skull_state"]
                           str                         conditioning attributes
    sample["target_shape"], ["target_spacing_mm"]      the grid it was resampled onto
    sample["native_*"]                                 pre-resample geometry, for provenance
    sample["study_key"], ["series_key"]                identifiers -- never log verbatim

**Axis order.** (X, Y, Z) = (Right, Anterior, Superior) after RAS canonicalization --
NV-Generate-CTMR's own array order. Internally, preprocessing works in (D, H, W) (see
`_preprocess_ops.py`); `__getitem__` converts to (X, Y, Z) exactly once, at the end, via
`image.permute(0, 2, 3, 1)` and `geometry.dhw_to_xyz`. Never convert by hand elsewhere.

Volume preprocessing and the splits loader live in `_preprocess_ops.py` (vendored from the
contrastive pipeline, now maintained independently). This module adds report pairing, the
per-modality geometry policy, archive-backed reads, and batching.
"""
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ._preprocess_ops import (
    NORMALIZERS,
    preprocess_nii,
    preprocess_nii_from_bytes,
    read_native_geometry,
    read_native_geometry_from_bytes,
)
from .geometry import (
    GEOMETRY_MODES,
    UNET_SPATIAL_MULTIPLE,
    GeometryPolicy,
    GeometrySpec,
    build_geometry_table,
    dhw_to_xyz,
)
from .manifest import read_manifest_csv
from .reports import REPORT_SECTION_NAMES
from .storage import ArchiveReader

SERIES_SELECTION_MODES = ("all", "one_per_study_per_bucket", "one_per_study_per_sequence",
                          "one_per_study_deterministic", "one_per_study_random")


@dataclass
class R2VDatasetConfig:
    """Everything the Dataset needs beyond a manifest and a report source.

    The knobs you will actually turn:
      `geometry_mode`      "fixed" for comparable experiments, "per_modality_plane" for
                           per-anatomy FOVs (see `geometry.py`)
      `fixed_target_*`     the grid used when geometry_mode="fixed"
      `series_selection`   "all" for training, "one_per_study_per_sequence" for evaluation
                           (see `_select_series` -- the differences matter)
      `report_sections`    which report sections become the conditioning text
    """

    split: str = "train"
    report_sections: tuple = ("findings", "impression")
    # A named format from `textenc.formats`. None (default) keeps `report_sections` joined by
    # `ReportRecord.compose`. A comma-separated spec ("a,b") samples one format per sample,
    # deterministically from (seed, epoch, index) -- see `textenc.formats.choose_format`. Train
    # split only; use a single fixed format for validation.
    report_format: Optional[str] = None
    # Sections returned *unjoined* as `report_sections_text`, for configs that encode each
    # section separately (one cross-attention token per section). Purely additive -- doesn't
    # affect `report_text` or `geometry_fingerprint`. "acquisition" is a synthesized
    # [MODALITY]/[PLANE]/[SPACING] string, not a report field.
    conditioning_sections: tuple = ("findings", "impression", "acquisition")
    geometry_mode: str = "per_modality_plane"
    # geometry_mode="fixed" only. Both (D, H, W)-ordered like every internal geometry value --
    # convert with `geometry.xyz_to_dhw` if your value came from outside the package.
    fixed_target_shape: tuple = (256, 384, 384)
    fixed_target_spacing_mm: tuple = (1.0, 0.5, 0.5)
    posterior_shift_mm: float = 15.0
    normalizer: str = "percentile"
    normalizer_kwargs: dict = field(default_factory=lambda: dict(
        lower_percentile=0.0, upper_percentile=99.5,
        lower_limit=0.0, upper_limit=1.0, clip=False,
    ))
    series_selection: str = "all"
    dtype: torch.dtype = torch.bfloat16
    seed: int = 0

    #: Derived in `__post_init__` from `report_format`, never passed in: the parsed spec.
    report_format_names: tuple = field(init=False, repr=False, default=())

    def __post_init__(self):
        if self.series_selection not in SERIES_SELECTION_MODES:
            raise ValueError(f"Unknown series_selection '{self.series_selection}'. "
                             f"Choose from: {SERIES_SELECTION_MODES}")
        if self.geometry_mode not in GEOMETRY_MODES:
            raise ValueError(f"Unknown geometry_mode '{self.geometry_mode}'. "
                             f"Choose from: {GEOMETRY_MODES}")
        if self.normalizer not in NORMALIZERS:
            raise ValueError(f"Unknown normalizer '{self.normalizer}'. "
                             f"Choose from: {list(NORMALIZERS.keys())}")
        from ..textenc.formats import ACQUISITION_SECTION

        for name in tuple(self.report_sections):
            if name not in REPORT_SECTION_NAMES:
                raise ValueError(f"Unknown report section '{name}'. Choose from: {REPORT_SECTION_NAMES}")
        # `conditioning_sections` also accepts "acquisition"; `report_sections` does not (it only
        # feeds the joined `report_text`).
        for name in tuple(self.conditioning_sections):
            if name not in REPORT_SECTION_NAMES + (ACQUISITION_SECTION,):
                raise ValueError(f"Unknown conditioning section '{name}'. Choose from: "
                                 f"{REPORT_SECTION_NAMES + (ACQUISITION_SECTION,)}")
        if self.report_format is not None:
            from ..textenc.formats import parse_format_spec

            # Parsed once here, not every __getitem__ call.
            self.report_format_names = parse_format_spec(self.report_format)

    def geometry_fingerprint(self):
        """Preprocessing settings that change the returned tensor, recorded in a cohort's
        `cohort.json` (X, Y, Z, matching everything else stored there)."""
        # report_format is only emitted when set, so unset-format cohorts keep old cohort_ids.
        extra = {"report_format": self.report_format} if self.report_format is not None else {}
        return {
            **extra,
            "geometry_mode": self.geometry_mode,
            "unet_spatial_multiple": UNET_SPATIAL_MULTIPLE,
            "fixed_target_shape_xyz": list(dhw_to_xyz(self.fixed_target_shape)) if self.geometry_mode == "fixed" else None,
            "fixed_target_spacing_mm_xyz": list(dhw_to_xyz(self.fixed_target_spacing_mm)) if self.geometry_mode == "fixed" else None,
            "posterior_shift_mm": self.posterior_shift_mm,
            "normalizer": self.normalizer,
            "normalizer_kwargs": dict(self.normalizer_kwargs),
            "report_sections": list(self.report_sections),
        }


class MRReportToVolumeDataset(Dataset):
    """See the module docstring for the sample schema and axis contract.

    `manifest`: a path to a manifest CSV, or an in-memory list of `ManifestRow`.
    `report_store`: any object supporting `study_uid in store` and `store[study_uid] ->
    ReportRecord` (the three in `reports.py`, or your own adapter).
    """

    def __init__(self, manifest, report_store, config=None, verbose=True):
        self.config = config or R2VDatasetConfig()
        self.report_store = report_store

        if self.config.geometry_mode == "fixed":
            self.geometry = GeometryPolicy(
                mode="fixed",
                single_spec=GeometrySpec(self.config.fixed_target_shape,
                                         self.config.fixed_target_spacing_mm),
            )
        else:
            # Spacing is derived from NVIDIA's published FOV; shape rounds to the UNet's divisor.
            self.geometry = GeometryPolicy(mode="per_modality_plane",
                                           table=build_geometry_table())

        normalizer_cls = NORMALIZERS[self.config.normalizer]
        self.normalizer_obj = normalizer_cls(**self.config.normalizer_kwargs)

        # No open handles held, so cheap to build eagerly.
        self._archive_reader = ArchiveReader()

        rows = manifest if isinstance(manifest, list) else read_manifest_csv(manifest)
        rows = [r for r in rows if r.split == self.config.split]
        n_before_report_filter = len(rows)
        self.rows = [r for r in rows if r.study_uid in self.report_store]
        n_dropped = n_before_report_filter - len(self.rows)

        self._epoch = 0
        self._samples_version = 0
        self.samples = self._select_series(self.rows)

        if verbose:
            print(f"[MRReportToVolumeDataset] split={self.config.split}: "
                  f"{len(self.samples)} samples from {len(self.rows)} eligible "
                  f"(study, series) pairs ({n_dropped} dropped: no matching report; "
                  f"series_selection={self.config.series_selection})")

    @property
    def samples_version(self):
        """Bumps whenever `samples` is replaced -- only under
        series_selection="one_per_study_random" -- so `GeometryBucketBatchSampler` can detect a
        stale bucket index."""
        return self._samples_version

    def set_epoch(self, epoch):
        """Reseed series_selection="one_per_study_random" for this epoch, deterministically from
        (config.seed, epoch). Call before iterating the DataLoader, not during -- use
        `training.set_loader_epoch`, which also reseeds the batch sampler in the right order."""
        self._epoch = epoch
        if self.config.series_selection == "one_per_study_random":
            self.samples = self._select_series(self.rows)
            self._samples_version += 1

    def _select_series(self, rows):
        """How many samples a study contributes, and how they're chosen:

        "all" (default) -- one sample per eligible series. Right for training. Wrong for
          evaluation: near-duplicate series from one session aren't independent observations,
          so plain means/CIs over them are misleading (pseudo-replication).

        "one_per_study_per_bucket" -- one sample per (study, modality, plane). **Use this for a
          per-bucket cohort.** ("one_per_study_per_sequence" would collapse planes instead.)

        "one_per_study_per_sequence" -- one sample per (study, modality), preferring the
          center-modality series. Right for a per-sequence cohort, wrong for a per-bucket one.

        "one_per_study_deterministic" -- one sample per study, preferring the center-modality
          series (T1w on MR-RATE) -- collapses to near-single-modality. Only for a
          single-sequence cohort, or when you deliberately want one volume per study.

        "one_per_study_random" -- one sample per study, redrawn each epoch from
          (seed, epoch, study_uid) via `set_epoch`. Training only; same modality-collapse
          caveat as above, within any single epoch.
        """
        mode = self.config.series_selection
        if mode == "all":
            return list(rows)

        # Group key: what "one per" means for this mode.
        keys = {
            "one_per_study_per_bucket": lambda r: (r.study_uid, r.modality, r.plane),
            "one_per_study_per_sequence": lambda r: (r.study_uid, r.modality),
        }
        key_of = keys.get(mode, lambda r: r.study_uid)
        groups = {}
        for r in rows:
            groups.setdefault(key_of(r), []).append(r)

        selected = []
        for key, group in groups.items():
            group = sorted(group, key=lambda r: r.series_id)
            if mode == "one_per_study_random":
                rng = random.Random(f"{self.config.seed}:{self._epoch}:{key}")
                chosen = rng.choice(group)
            else:  # one_per_study_deterministic | one_per_study_per_sequence
                chosen = next((r for r in group if r.is_center_modality), group[0])
            selected.append(chosen)
        return selected

    def __len__(self):
        return len(self.samples)

    def _read_archive_bytes(self, row):
        return self._archive_reader.read_bytes(row.locator())

    def __getitem__(self, index):
        row = self.samples[index]
        spec = self.geometry.resolve(row.modality, row.plane)
        posterior_shift_voxels = int(round(self.config.posterior_shift_mm / spec.target_spacing[2]))

        native_shape, native_spacing = row.native_shape, row.native_spacing_mm

        if row.backend == "archive":
            raw_bytes = self._read_archive_bytes(row)
            if native_shape is None:
                native_shape, native_spacing = read_native_geometry_from_bytes(raw_bytes)
            arr = preprocess_nii_from_bytes(
                raw_bytes, spec.target_spacing, spec.target_shape,
                posterior_shift_voxels, self.normalizer_obj,
            )
        else:
            arr = preprocess_nii(
                row.image_path, spec.target_spacing, spec.target_shape,
                posterior_shift_voxels, self.normalizer_obj,
            )
        image = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0).to(self.config.dtype)
        # The one and only axis conversion: (D,H,W) above this line, (X,Y,Z) below. Both read
        # paths above yield (D,H,W); a third read path must too, or move the permute into it.
        image = image.permute(0, 2, 3, 1).contiguous()  # [1,D,H,W] -> [1,H,W,D] = [1,X,Y,Z]

        report = self.report_store[row.study_uid]
        target_spacing_xyz = dhw_to_xyz(spec.target_spacing)
        if self.config.report_format is None:
            report_text = report.compose(self.config.report_sections)
        else:
            from ..textenc.formats import choose_format, format_report

            # One name -> that name; several -> a uniform draw keyed on (seed, epoch, index), so
            # it's reproducible under any --num-workers and after a resume.
            name = choose_format(self.config.report_format_names,
                                 (self.config.seed, self._epoch, index))
            # modality/plane/spacing come from the manifest row and resolved geometry, never
            # parsed out of the report text.
            report_text = format_report(report, name, modality=row.modality, plane=row.plane,
                                        spacing_mm_xyz=target_spacing_xyz)

        # The same `ReportRecord`, section-separated and unjoined. An absent section is "" and is
        # masked out by `SectionedFusionEmbedder`. "acquisition" is synthesized from this row's
        # modality/plane/spacing (`meta_prefix_for`), not a report field, and is never empty.
        from ..textenc.formats import ACQUISITION_SECTION, meta_prefix_for

        report_sections_text = {
            name: (meta_prefix_for(row.modality, row.plane, target_spacing_xyz)
                   if name == ACQUISITION_SECTION
                   else (getattr(report, name, None) or "").strip())
            for name in self.config.conditioning_sections
        }

        native_shape = native_shape or (0, 0, 0)
        native_spacing = native_spacing or (0.0, 0.0, 0.0)
        native_fov = tuple(s * p for s, p in zip(native_shape, native_spacing))

        return {
            "image": image,                          # [1, X, Y, Z], NV-Generate-CTMR-ready
            "report_text": report_text,
            "report_sections_text": report_sections_text,
            "modality": row.modality or "unknown",
            "acquisition_plane": row.plane or "unknown",
            "contrast_state": "unknown",             # not derivable from the release
            "skull_state": "defaced_not_stripped",   # constant for native_space
            "target_spacing_mm": torch.tensor(target_spacing_xyz, dtype=torch.float32),
            "target_shape": torch.tensor(dhw_to_xyz(spec.target_shape), dtype=torch.int64),
            "native_shape": torch.tensor(dhw_to_xyz(native_shape), dtype=torch.int64),
            "native_spacing_mm": torch.tensor(dhw_to_xyz(native_spacing), dtype=torch.float32),
            "native_fov_mm": torch.tensor(dhw_to_xyz(native_fov), dtype=torch.float32),
            "study_key": row.study_uid,              # traceability -- do not log verbatim
            "series_key": row.series_id,             # traceability -- do not log verbatim
        }


def build_r2v_dataset(manifest, report_index, *, split, report_format, geometry_mode,
                      series_selection, posterior_shift_mm, normalizer, seed,
                      report_sections=("findings", "impression")) -> "MRReportToVolumeDataset":
    """The one place `cli.train_r2v` and `cli.evaluate` build their dataset, so the two can never
    preprocess differently. `series_selection` is the one deliberate difference: "all" for
    training, "one_per_study_per_bucket" for evaluation. `dataset.config` holds the config used.
    """
    from .reports import ShardReportStore

    config = R2VDatasetConfig(
        split=split, report_sections=tuple(report_sections), report_format=report_format,
        geometry_mode=geometry_mode, series_selection=series_selection,
        posterior_shift_mm=posterior_shift_mm, normalizer=normalizer,
        dtype=torch.float32, seed=seed,
    )
    return MRReportToVolumeDataset(
        str(manifest), ShardReportStore(str(report_index)), config=config
    )


# --------------------------------------------------------------------------- batching


_STACK_TENSOR_KEYS = (
    "image", "target_spacing_mm", "target_shape",
    "native_shape", "native_spacing_mm", "native_fov_mm",
)
_LIST_KEYS = (
    "report_text", "report_sections_text", "modality", "acquisition_plane",
    "contrast_state", "skull_state", "study_key", "series_key",
)


def collate_fn_r2v(batch):
    """Collates a batch of any size >= 1 (the contrastive pipeline's collate only reads
    `batch[0]`). Every image must share the same (X, Y, Z) -- always true under
    geometry_mode="fixed", only within one bucket under "per_modality_plane" (use
    `GeometryBucketBatchSampler` there). Raises an actionable error on a shape mismatch.
    """
    if not batch:
        raise ValueError("collate_fn_r2v received an empty batch.")
    shapes = {tuple(b["image"].shape) for b in batch}
    if len(shapes) > 1:
        raise ValueError(
            f"collate_fn_r2v got a batch with mismatched image shapes {sorted(shapes)}. This "
            f"happens when geometry_mode='per_modality_plane' and the batch mixes (modality, "
            f"plane) buckets. Use GeometryBucketBatchSampler as the DataLoader's "
            f"batch_sampler, or geometry_mode='fixed', for batch_size > 1."
        )
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in _STACK_TENSOR_KEYS}
    for k in _LIST_KEYS:
        out[k] = [b[k] for b in batch]
    return out


BUCKET_ORDERS = ("interleave", "shuffle")


class GeometryBucketBatchSampler(Sampler):
    """Yields batches drawn from a single (modality, plane) bucket at a time.

    Pass as a DataLoader's `batch_sampler` (not `sampler` -- it yields lists of indices).
    Grouping key is the raw `(modality, plane)` pair, always at least as fine as
    `geometry.bucket_key`, so every batch is shape-safe for `collate_fn_r2v` under both
    geometry modes.

    `bucket_order`:
    - `"interleave"` (default) -- each bucket's batches are spaced evenly across the epoch at
      its natural rate (see `_interleave`), so consecutive batches carry different modalities
      and gradient accumulation isn't single-modality.
    - `"shuffle"` -- one flat shuffle over all batches (pre-2026-08 behaviour); can produce runs
      of same-bucket batches.

    No frequency weighting or temperature: a bucket's epoch share is its share of the data.

    **`drop_last` drops remainders, never a whole bucket.** A bucket smaller than `batch_size`
    has no full batch, so it keeps one short batch rather than vanishing from every epoch --
    see `undersized_buckets`.
    """

    def __init__(self, dataset, batch_size, drop_last=False, seed=0, bucket_order="interleave"):
        if bucket_order not in BUCKET_ORDERS:
            raise ValueError(f"Unknown bucket_order '{bucket_order}'. Choose from: {BUCKET_ORDERS}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_order = bucket_order
        self.epoch = 0
        self._buckets = None
        self._buckets_version = None

    @property
    def undersized_buckets(self):
        """{(modality, plane): n} for buckets smaller than `batch_size` -- these contribute one
        short batch per epoch instead of vanishing (see class docstring). Worth logging: a
        bucket this small is memorized, not learned."""
        return {key: len(indices) for key, indices in self.buckets.items()
                if len(indices) < self.batch_size}

    @property
    def buckets(self):
        """{(modality, plane): [dataset index, ...]}, rebuilt whenever `dataset.samples` is
        replaced. Built lazily (not in `__init__`) because
        series_selection="one_per_study_random" redraws `samples` every `set_epoch`, and a
        snapshot taken at construction would go stale."""
        version = getattr(self.dataset, "samples_version", None)
        if self._buckets is None or version is None or version != self._buckets_version:
            buckets = {}
            for i, sample in enumerate(self.dataset.samples):
                buckets.setdefault((sample.modality, sample.plane), []).append(i)
            self._buckets, self._buckets_version = buckets, version
        return self._buckets

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches_per_bucket(self, rng):
        """{bucket: [batch, ...]} -- every index used exactly once, `drop_last` aside."""
        per_bucket = {}
        for key, indices in self.buckets.items():
            indices = list(indices)
            rng.shuffle(indices)
            batches = [indices[s:s + self.batch_size] for s in range(0, len(indices), self.batch_size)]
            # A bucket with a single short batch is the undersized case and keeps it -- only
            # pop the remainder when there's more than one batch.
            if self.drop_last and len(batches) > 1 and len(batches[-1]) < self.batch_size:
                batches.pop()
            if batches:
                per_bucket[key] = batches
        return per_bucket

    @staticmethod
    def _interleave(per_bucket, rng):
        """Stride-schedules batches so each bucket's `n` batches land evenly across the epoch
        (virtual time `(k + phase) * total / n`) instead of clumping. The random per-bucket
        `phase` is what stops every epoch from replaying an identical bucket sequence."""
        total = sum(len(batches) for batches in per_bucket.values())
        schedule = []
        for batches in per_bucket.values():
            stride = total / len(batches)
            phase = rng.random()
            schedule.extend(((k + phase) * stride, batch) for k, batch in enumerate(batches))
        schedule.sort(key=lambda item: item[0])
        return [batch for _virtual_time, batch in schedule]

    def __iter__(self):
        rng = random.Random(f"{self.seed}:{self.epoch}")
        per_bucket = self._batches_per_bucket(rng)
        if self.bucket_order == "shuffle":
            batches = [b for bucket_batches in per_bucket.values() for b in bucket_batches]
            rng.shuffle(batches)
            return iter(batches)
        return iter(self._interleave(per_bucket, rng))

    def __len__(self):
        n = 0
        for indices in self.buckets.values():
            full, rem = divmod(len(indices), self.batch_size)
            if rem == 0:
                n += full
            elif self.drop_last and full >= 1:
                n += full            # remainder dropped, the bucket still has full batches
            else:
                n += full + 1        # keep the short batch (drop_last off, or nothing else to keep)
        return n
