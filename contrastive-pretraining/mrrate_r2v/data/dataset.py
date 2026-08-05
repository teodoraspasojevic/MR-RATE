"""The Dataset: one item = one (report text, single target volume) pair.

    sample["image"]        torch.Tensor [1, X, Y, Z]   float, preprocessed, ready for
                                                      NV-Generate-CTMR with no permute
    sample["report_text"]  str                         the conditioning text
    sample["modality"], ["acquisition_plane"], ["contrast_state"], ["skull_state"]
                           str                         conditioning attributes
    sample["target_shape"], ["target_spacing_mm"]      the grid it was resampled onto
    sample["native_*"]                                 pre-resample geometry, for provenance
    sample["study_key"], ["series_key"]                identifiers -- never log verbatim

**Axis order.** X=Right-Left, Y=Anterior-Posterior, Z=Superior-Inferior after RAS
canonicalization. That is NV-Generate-CTMR's own array order, never permuted further
anywhere in its code, so every consumer of this Dataset gets model-ready tensors.

It is deliberately *not* the (D, H, W) = (S, R, A) order `MRReportDataset` uses. That
ordering exists because the VJEPA video encoders hardcode "the axis after channel is the
slice axis"; that constraint belongs to those encoders, not to NV-Generate-CTMR. So
preprocessing runs in (D, H, W) using `scripts/data.py`'s shared code, and `__getitem__`
permutes to (X, Y, Z) exactly once as its final step -- `image.permute(0, 2, 3, 1)`, with
every geometry field reindexed the same way via `geometry.dhw_to_xyz`. The on-disk manifest
stays (D, H, W); the reindex is an output-time concern only.

**What is reused, not reimplemented.** Volume preprocessing (RAS reorient -> resample ->
crop/pad -> normalize), the `.npz` cache format and its manifest validation, and the splits
loader are all `scripts/data.py`'s code imported unchanged (via `_preprocess_ops`). This
module adds the report pairing, the per-modality geometry policy, archive-backed reads, and
a collate function that actually keeps every item in the batch.
"""
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from ._preprocess_ops import (
    NORMALIZERS,
    preprocess_nii,
    preprocess_nii_from_bytes,
    read_native_geometry,
    read_native_geometry_from_bytes,
    validate_cache_manifest,
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
from .storage import (
    ArchiveReader,
    CacheBudget,
    NodeLocalCache,
    NodeLocalRootError,
    resolve_node_local_root,
)

SERIES_SELECTION_MODES = ("all", "one_per_study_per_bucket", "one_per_study_per_sequence",
                          "one_per_study_deterministic", "one_per_study_random")
ARCHIVE_ACCESS_MODES = ("stream", "node_local_cache")

# Bounded default budget for the node-local materialization cache. 200 GB / 20,000 files fits
# Helma's 15 TB/node $TMPDIR with room for checkpoints and other temp files. $TMPDIR's inode
# limit is undocumented, so the file count is a conservative guess, not a verified ceiling.
DEFAULT_CACHE_MAX_BYTES = 200 * (1024 ** 3)
DEFAULT_CACHE_MAX_FILES = 20_000


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
      `archive_access_mode` "stream" (no disk write) or "node_local_cache"
    """

    split: str = "train"
    report_sections: tuple = ("findings", "impression")
    # A named format from `textenc.formats` (e.g. "impression_findings",
    # "findings_impression_meta"). None -- the default -- keeps the historical behaviour exactly:
    # `report_sections` joined by `ReportRecord.compose`. Set one and it takes over, and the name
    # is recorded in `geometry_fingerprint` so a cohort built with it is a different cohort.
    report_format: Optional[str] = None
    # Which released sections are additionally returned *unjoined*, as `report_sections_text`, for
    # conditioning configurations that encode each section separately (Report2CT-style fusion ->
    # one cross-attention token per section). Purely additive: `report_text` is unaffected, which
    # is why this is deliberately NOT in `geometry_fingerprint` -- it changes no existing field and
    # would otherwise invalidate every cohort built before it existed.
    conditioning_sections: tuple = ("findings", "impression")
    geometry_mode: str = "per_modality_plane"
    # geometry_mode="fixed" only. **Both are (D, H, W)-ordered**, like every other internal
    # geometry parameter -- NOT the (X, Y, Z) order the Dataset *returns*. If your value came from
    # outside this package (a CLI flag, NVIDIA's dim/spacing config), convert it with
    # `geometry.xyz_to_dhw` first; `cli/preprocess.py` does exactly that.
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

    # Archive-backed rows only (backend="archive"); backend="file" rows never consult these.
    archive_access_mode: str = "stream"
    cache_root: Optional[str] = None           # None -> auto-resolve $TMPDIR at first use
    cache_env_vars: tuple = ("TMPDIR", "TMP")
    cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES
    cache_max_files: int = DEFAULT_CACHE_MAX_FILES

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
        if self.archive_access_mode not in ARCHIVE_ACCESS_MODES:
            raise ValueError(f"Unknown archive_access_mode '{self.archive_access_mode}'. "
                             f"Choose from: {ARCHIVE_ACCESS_MODES}")
        for name in tuple(self.report_sections) + tuple(self.conditioning_sections):
            if name not in REPORT_SECTION_NAMES:
                raise ValueError(f"Unknown report section '{name}'. Choose from: {REPORT_SECTION_NAMES}")
        if self.report_format is not None:
            from ..textenc.formats import REPORT_FORMATS

            if self.report_format not in REPORT_FORMATS:
                raise ValueError(f"Unknown report_format '{self.report_format}'. "
                                 f"Choose from: {sorted(REPORT_FORMATS)}")

    def geometry_fingerprint(self):
        """The preprocessing settings that change the returned tensor. Recorded in a
        cohort's `cohort.json` so a later run can prove it used identical preprocessing.

        The fixed-target fields are reported in **(X, Y, Z)** and named accordingly: everything
        else in a cohort directory is (X, Y, Z) (`CohortCase.shape`, the stored `.npy` volumes), so
        emitting the internal (D, H, W) here would make `cohort.json` the one file in the
        directory with a different convention.
        """
        # `report_format` is emitted only when it is set. A cohort built before formats existed
        # must keep its cohort_id, and an unset format changes no text -- so adding an always-present
        # `"report_format": null` key would invalidate every existing cohort for no reason.
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

    def __init__(self, manifest, report_store, config=None,
                 preprocessed_dir=None, use_preprocessed=False,
                 cache_allow_mismatch=False, verbose=True):
        self.config = config or R2VDatasetConfig()
        self.report_store = report_store

        if self.config.geometry_mode == "fixed":
            self.geometry = GeometryPolicy(
                mode="fixed",
                single_spec=GeometrySpec(self.config.fixed_target_shape,
                                         self.config.fixed_target_spacing_mm),
            )
        else:
            # No spacing/divisor arguments: the table now derives spacing from NVIDIA's published
            # FOV and fixes the shape at a multiple of the UNet's constraint. See build_geometry_table.
            self.geometry = GeometryPolicy(mode="per_modality_plane",
                                           table=build_geometry_table())

        self.preprocessed_dir = preprocessed_dir
        self.use_preprocessed = bool(use_preprocessed)
        self.cache_allow_mismatch = cache_allow_mismatch
        if self.use_preprocessed and self.config.geometry_mode != "fixed":
            raise ValueError(
                "use_preprocessed=True requires geometry_mode='fixed': the .npz cache has one "
                "shape/spacing for the whole directory, whereas 'per_modality_plane' needs a "
                "different shape per bucket. Per-bucket caching is not implemented."
            )

        normalizer_cls = NORMALIZERS[self.config.normalizer]
        self.normalizer_obj = normalizer_cls(**self.config.normalizer_kwargs)

        # An ArchiveReader holds no open handles, so it is cheap to build eagerly. The
        # node-local cache is built lazily on first real use, so a dataset configured for it
        # but never indexed (dry run, or a split with no archive rows) never needs $TMPDIR.
        self._archive_reader = ArchiveReader()
        self._node_local_cache = None

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

        if self.use_preprocessed:
            self._check_cache_manifest()

    @property
    def samples_version(self):
        """Bumped every time `samples` is replaced, so anything holding a derived index into
        it (`GeometryBucketBatchSampler`'s bucket lists) can tell that it went stale.

        Only `series_selection="one_per_study_random"` ever replaces `samples`; for every
        other mode this stays 0 for the dataset's whole life.
        """
        return self._samples_version

    def set_epoch(self, epoch):
        """Reseed `series_selection="one_per_study_random"` for this epoch.

        All of this Dataset's randomness lives here, so calling this fixes the whole epoch's
        selection deterministically from (config.seed, epoch). Nothing is drawn from the
        process-global RNG inside `__getitem__`.

        **Call this before iterating the DataLoader, not during.** It rebinds `samples`, so an
        in-flight iteration would mix two epochs' selections. `training.set_loader_epoch` is the
        supported way to call it -- it also reseeds the batch sampler, in the right order.
        """
        self._epoch = epoch
        if self.config.series_selection == "one_per_study_random":
            self.samples = self._select_series(self.rows)
            self._samples_version += 1

    def _select_series(self, rows):
        """How many samples a study contributes, and how they are chosen:

        "all" (default) -- one sample per eligible series. Full manifest granularity; a study
          with N series contributes N samples. Right for training; for *evaluation* it is
          pseudo-replication -- near-duplicate series from one session are not independent
          observations, so plain means over them overweight multi-series studies and plain
          std/CIs come out falsely narrow (this package's aggregation does not model clustering).

        "one_per_study_per_bucket" -- one sample per (study, sequence, plane). **Use this for a
          per-bucket cohort.** Beware the subtler trap: "one_per_study_per_sequence" prefers the
          center-modality series, which on MR-RATE is the *axial* T1w, so it collapses PLANES
          within a sequence -- measured on the real test split it leaves T1w CORONAL with 16 cases
          and T2w SAGITTAL with 6, because those planes only survive for studies that happen to
          have no axial series of that modality.

        "one_per_study_per_sequence" -- one sample per (study, sequence), preferring the
          center-modality series, else the series_id-sorted first. Right for a per-sequence
          cohort, wrong for a per-bucket one (see above).

        "one_per_study_deterministic" -- one sample per *study*, across all requested sequences.
          Beware: the preferred series is the center modality, which on MR-RATE is the T1w
          series, so a multi-sequence request collapses to almost entirely T1w (measured on the
          real test split: 4861 T1w vs 25 FLAIR, 7 T2w, 0 SWI). Only meaningful for a
          single-sequence cohort, or when you deliberately want one representative volume per
          study regardless of modality.

        "one_per_study_random" -- one sample per study, redrawn each epoch from
          (seed, epoch, study_uid) via `set_epoch`. Training only; carries the same
          modality-collapse caveat as above within any single epoch.
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

    def _cache_space_dir(self):
        return os.path.join(self.preprocessed_dir, "native_space")

    def _check_cache_manifest(self):
        validate_cache_manifest(
            self._cache_space_dir(), "native_space",
            self.geometry.single_spec.target_spacing, self.geometry.single_spec.target_shape,
            self.config.posterior_shift_mm, self.config.normalizer, self.config.normalizer_kwargs,
            allow_mismatch=self.cache_allow_mismatch, tag="MRReportToVolumeDataset",
        )

    def __len__(self):
        return len(self.samples)

    def _node_local_cache_for(self, locator):
        """Build the bounded node-local cache on first genuine need. Raises
        `NodeLocalRootError` rather than silently writing to a persistent workspace."""
        if self._node_local_cache is not None:
            return self._node_local_cache
        root = self.config.cache_root
        if root is None:
            root, _diagnostics = resolve_node_local_root(env_vars=self.config.cache_env_vars)
        elif not os.path.isdir(root):
            raise NodeLocalRootError(
                f"Configured cache_root does not exist or is not a directory (length={len(root)})."
            )
        budget = CacheBudget(max_bytes=self.config.cache_max_bytes, max_files=self.config.cache_max_files)
        self._node_local_cache = NodeLocalCache(root, budget)
        return self._node_local_cache

    def _read_archive_bytes(self, row):
        return self._archive_reader.read_bytes(row.locator())

    def __getitem__(self, index):
        row = self.samples[index]
        spec = self.geometry.resolve(row.modality, row.plane)
        posterior_shift_voxels = int(round(self.config.posterior_shift_mm / spec.target_spacing[2]))

        native_shape, native_spacing = row.native_shape, row.native_spacing_mm

        if row.backend == "archive":
            if self.config.archive_access_mode == "node_local_cache":
                cache = self._node_local_cache_for(row.locator())
                cached_path = cache.get_or_materialize(
                    row.locator(), lambda: self._read_archive_bytes(row),
                    hint=f"{row.modality or ''}_{row.plane or ''}",
                )
                if native_shape is None:
                    native_shape, native_spacing = read_native_geometry(cached_path)
                arr = preprocess_nii(
                    cached_path, spec.target_spacing, spec.target_shape,
                    posterior_shift_voxels, self.normalizer_obj,
                )
            else:  # "stream": no disk write at all
                raw_bytes = self._read_archive_bytes(row)
                if native_shape is None:
                    native_shape, native_spacing = read_native_geometry_from_bytes(raw_bytes)
                arr = preprocess_nii_from_bytes(
                    raw_bytes, spec.target_spacing, spec.target_shape,
                    posterior_shift_voxels, self.normalizer_obj,
                )
        elif self.use_preprocessed:
            cached = np.load(os.path.join(self._cache_space_dir(), f"{row.study_uid}.npz"))
            arr = np.ascontiguousarray(cached["volumes"][row.cache_index])  # [D, H, W]
        else:
            arr = preprocess_nii(
                row.image_path, spec.target_spacing, spec.target_shape,
                posterior_shift_voxels, self.normalizer_obj,
            )
        image = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0).to(self.config.dtype)
        # The one and only axis conversion: (D,H,W)=(S,R,A) above this line, (X,Y,Z)=(R,A,S) below.
        # Unconditional because ALL THREE read paths above yield (D,H,W): the two preprocess_nii
        # calls by its own contract, and the .npz cache because preprocess_volumes.py stores
        # preprocess_nii output verbatim as [N,D,H,W]. A fourth read path must match, or this
        # permute has to move into the branches. See the module docstring.
        image = image.permute(0, 2, 3, 1).contiguous()  # [1,D,H,W] -> [1,H,W,D] = [1,X,Y,Z]

        report = self.report_store[row.study_uid]
        if self.config.report_format is None:
            report_text = report.compose(self.config.report_sections)
        else:
            from ..textenc.formats import format_report

            # modality/plane come from the manifest row, never parsed out of the report -- only
            # the `*_meta` formats read them, and only because they were supplied here.
            report_text = format_report(report, self.config.report_format,
                                        modality=row.modality, plane=row.plane)

        # The same `ReportRecord`, section-separated and unjoined. Deterministic (a plain field
        # read plus strip -- no parsing, no sampling), so training, validation and sampling see
        # byte-identical text for a given study. An absent section is "" and is masked out by
        # `SectionedFusionEmbedder`, never emitted as a real attention key.
        report_sections_text = {
            name: (getattr(report, name, None) or "").strip()
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
            "target_spacing_mm": torch.tensor(dhw_to_xyz(spec.target_spacing), dtype=torch.float32),
            "target_shape": torch.tensor(dhw_to_xyz(spec.target_shape), dtype=torch.int64),
            "native_shape": torch.tensor(dhw_to_xyz(native_shape), dtype=torch.int64),
            "native_spacing_mm": torch.tensor(dhw_to_xyz(native_spacing), dtype=torch.float32),
            "native_fov_mm": torch.tensor(dhw_to_xyz(native_fov), dtype=torch.float32),
            "study_key": row.study_uid,              # traceability -- do not log verbatim
            "series_key": row.series_id,             # traceability -- do not log verbatim
        }


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
    """Collates every item in the batch, for any batch_size >= 1. (`scripts/data.py`'s
    collate functions only ever read `batch[0]` and discard the rest.)

    Every image must share the same (X, Y, Z). Always true with geometry_mode="fixed"; with
    "per_modality_plane" it only holds within one bucket, so use `GeometryBucketBatchSampler`
    for batch_size > 1 there. A mixed batch raises this function's own actionable error
    rather than a raw torch.stack traceback.
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

    **Grouping key is the raw `(modality, plane)` pair, not `geometry.bucket_key`.** Shape is a
    pure function of that pair, so this is always at least as fine as the geometry bucket and
    therefore always shape-safe for `collate_fn_r2v`. It is *strictly* finer in two places, both
    deliberate: under geometry_mode="fixed" the geometry key is a single constant, so grouping by
    it would let a batch mix modalities; and under "per_modality_plane" every pair missing from
    the FOV table collapses onto `FALLBACK_GEOMETRY_KEY`, which would put e.g. an OBLIQUE T1w and
    a plane-less SWI in one batch. One batch is now one modality in every configuration.

    **Ordering** (`bucket_order`), both of which use every sample exactly once per epoch and
    never resample a bucket -- the per-epoch count of each bucket is exactly its natural size:

    - `"interleave"` (default) -- each bucket's batches are spaced evenly across the epoch at its
      natural rate (see `_interleave`). Consecutive batches therefore carry different modalities,
      no bucket clumps, and no epoch ends with a single-modality tail. That also makes gradient
      accumulation accumulate *across* modalities: with bucket-pure micro-batches, drawing the
      bucket per optimiser step instead would make every update single-modality.
    - `"shuffle"` -- one flat shuffle over all batches. The pre-2026-08 behaviour; same epoch
      contents, but nothing stops a run of same-bucket batches.

    Neither mode applies frequency weighting or temperature: a bucket's share of the epoch is
    its share of the data, and the 2-series SWI SAGITTAL bucket contributes 2 series.
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
    def buckets(self):
        """{(modality, plane): [dataset index, ...]}, rebuilt whenever `dataset.samples` is
        replaced. Built lazily rather than in `__init__` because
        `series_selection="one_per_study_random"` redraws `samples` on every `set_epoch`: a
        snapshot taken at construction goes stale on the first epoch boundary and then hands out
        batches spanning two buckets, which `collate_fn_r2v` rejects on shape.
        """
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
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches.pop()
            if batches:
                per_bucket[key] = batches
        return per_bucket

    @staticmethod
    def _interleave(per_bucket, rng):
        """Stride-schedule the buckets: a bucket holding `n` of the epoch's `total` batches claims
        virtual times `(k + phase) * total / n`, and batches are emitted in virtual-time order.

        So a bucket's batches land evenly spaced across the whole epoch, at its natural rate --
        one batch every `total / n`. Two consecutive batches share a bucket only where the
        pigeonhole forces it (a bucket holding more than about half the epoch); on MR-RATE's
        train split the largest bucket is 16% of batches, so consecutive batches essentially
        always differ in modality. The random per-bucket `phase` is the only stochastic part, and
        it is what stops every epoch from replaying an identical bucket sequence.

        The rejected alternative was a greedy draw weighted by *remaining* batches that simply
        banned repeating the previous bucket. Measured: with two buckets at 3:1 it forces strict
        alternation, draining the smaller bucket at twice its natural rate and leaving the epoch's
        whole tail single-modality. Spacing by construction beats constraining after the fact.
        """
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
            n += full + (0 if self.drop_last or rem == 0 else 1)
        return n


def compute_modality_balance_weights(samples, key="modality", base_weight=1.0, eps=1e-6):
    """Inverse-frequency sample weights over one categorical field (default: modality).
    Same idea as the contrastive pipeline's pathology rebalancing, applied to a single field.
    """
    counts = {}
    for s in samples:
        v = getattr(s, key)
        counts[v] = counts.get(v, 0) + 1
    total = max(len(samples), 1)
    inv_freq = {k: total / max(v, eps) for k, v in counts.items()}
    weights = [base_weight * inv_freq[getattr(s, key)] for s in samples]
    return torch.tensor(weights, dtype=torch.float32)


def get_modality_balanced_sampler(dataset, key="modality", base_weight=1.0, generator=None):
    weights = compute_modality_balance_weights(dataset.samples, key=key, base_weight=base_weight)
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset.samples),
                                 replacement=True, generator=generator)
