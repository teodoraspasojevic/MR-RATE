"""What voxel grid each series is resampled onto. Stdlib only, no torch.

Two policies, both selected via `R2VDatasetConfig.geometry_mode`:

- `"fixed"` -- one shape/spacing for every series. Use this for anything you want to
  compare across models: all volumes share a grid, batching just works.
- `"per_modality_plane"` -- a per-(modality, plane) shape derived from
  NV-Generate-MR-Brain's published median training FOVs. Tighter fit per anatomy, but
  shapes differ between buckets, so batch_size > 1 needs `GeometryBucketBatchSampler`
  and the numbers are not comparable with a fixed-mode run.

Axis order here is (D, H, W) = (Superior, Right, Anterior), matching `scripts/data.py`'s
internal convention. `dhw_to_xyz` converts to the (X, Y, Z) = (R, A, S) order the Dataset
actually returns -- see `dataset.py`'s docstring for why the conversion happens exactly
once, at the end.
"""
import math
from dataclasses import dataclass

# Median training FOV per (modality, plane) for nvidia/NV-Generate-MR-Brain
# (rflow-mr-brain), transcribed from NV-Generate-CTMR/docs/inference.md's "Recommended
# FOV" table and re-expressed in (D, H, W) = (S, R, A) order.
#
# Axis assignment follows that doc's own rule -- "set dim so the slice-stacking axis maps
# to the smaller dim[i]": axial -> D, sagittal -> H, coronal -> W. All 15 published rows
# have two equal larger values plus one smaller value, and the pattern was checked to hold
# for every row, not inferred from one example.
NV_BRAIN_FOV_MM = {
    ("T1w", "AXIAL"):      (174.0, 240.0, 240.0),
    ("T1w", "SAGITTAL"):   (250.0, 176.0, 250.0),
    ("T1w", "CORONAL"):    (240.0, 240.0, 200.0),
    ("T2w", "AXIAL"):      (158.0, 240.0, 240.0),
    ("T2w", "SAGITTAL"):   (240.0, 162.0, 240.0),
    ("T2w", "CORONAL"):    (200.0, 200.0, 180.0),
    ("FLAIR", "AXIAL"):    (175.0, 250.0, 250.0),
    ("FLAIR", "SAGITTAL"): (250.0, 176.0, 250.0),
    ("FLAIR", "CORONAL"):  (250.0, 250.0, 200.0),
    ("SWI", "AXIAL"):      (145.0, 230.0, 230.0),
    ("SWI", "SAGITTAL"):   (230.0, 140.0, 230.0),
    ("SWI", "CORONAL"):    (230.0, 230.0, 155.0),
    ("MRA", "AXIAL"):      (158.0, 220.0, 220.0),
    ("MRA", "SAGITTAL"):   (250.0, 158.0, 250.0),
    ("MRA", "CORONAL"):    (240.0, 240.0, 179.0),
}

# Used for any (modality, plane) missing from the table: unknown modality, OBLIQUE plane
# (present in NVIDIA's training data but excluded from its published table), or missing
# plane metadata. Value is NV-Generate-MR-Brain's own shipped default inference geometry.
DEFAULT_FALLBACK_FOV_MM = (256.0, 256.0, 256.0)

# NV-Generate-MR-Brain's default inference spacing -- deliberately not the contrastive
# loader's (1.0, 0.5, 0.5), which was tuned for a discriminative encoder's fixed
# 256x384x384 grid rather than this model's FOV distribution.
DEFAULT_GEOMETRY_SPACING_MM = (1.0, 1.0, 1.0)

GEOMETRY_MODES = ("per_modality_plane", "fixed")

FIXED_GEOMETRY_KEY = ("__fixed__", "__fixed__")
FALLBACK_GEOMETRY_KEY = ("__fallback__", "__fallback__")


def dhw_to_xyz(t):
    """Reindex a (D, H, W) 3-tuple to (X, Y, Z) = (H, W, D).

    (D, H, W) = (Superior, Right, Anterior) internally; (X, Y, Z) = (Right, Anterior,
    Superior) on output, which is NV-Generate-CTMR's own array order. The `image` tensor's
    equivalent op is `tensor.permute(0, 2, 3, 1)`.
    """
    return (t[1], t[2], t[0])


def xyz_to_dhw(t):
    """Inverse of `dhw_to_xyz`: reindex an (X, Y, Z) 3-tuple to (D, H, W) = (Z, X, Y).

    Anything arriving from outside this package -- a CLI flag, NVIDIA's `dim`/`spacing` config --
    is in (X, Y, Z). `GeometrySpec`, `crop_or_pad`, and every other internal geometry parameter is
    in (D, H, W). Convert at the boundary with this, never by hand: passing (X, Y, Z) into a
    (D, H, W) slot is silent for a cube at isotropic spacing and scrambles axes for anything else.
    """
    return (t[2], t[0], t[1])


def _ceil_to_multiple(value_mm, spacing_mm, divisible_by):
    """Smallest voxel count >= value_mm/spacing_mm that is a multiple of divisible_by.

    Rounds up, never to nearest, so the grid's physical FOV is always at least the median
    FOV it derives from -- a geometry baseline should not truncate anatomy.
    """
    voxels = math.ceil(value_mm / spacing_mm)
    return int(math.ceil(voxels / divisible_by) * divisible_by)


@dataclass(frozen=True)
class GeometrySpec:
    """One resample/crop target, both fields (D, H, W)-ordered."""

    target_shape: tuple
    target_spacing: tuple

    @property
    def physical_fov_mm(self):
        return tuple(s * p for s, p in zip(self.target_shape, self.target_spacing))


def build_geometry_table(spacing_mm=DEFAULT_GEOMETRY_SPACING_MM, divisible_by=16,
                         fov_table=None, fallback_fov_mm=DEFAULT_FALLBACK_FOV_MM):
    """{(modality, plane): GeometrySpec} from a median-FOV table, plus a fallback entry."""
    fov_table = fov_table if fov_table is not None else NV_BRAIN_FOV_MM
    table = {}
    for key, fov in fov_table.items():
        shape = tuple(_ceil_to_multiple(fov[i], spacing_mm[i], divisible_by) for i in range(3))
        table[key] = GeometrySpec(target_shape=shape, target_spacing=tuple(spacing_mm))
    fallback_shape = tuple(_ceil_to_multiple(fallback_fov_mm[i], spacing_mm[i], divisible_by) for i in range(3))
    table[FALLBACK_GEOMETRY_KEY] = GeometrySpec(target_shape=fallback_shape, target_spacing=tuple(spacing_mm))
    return table


class GeometryPolicy:
    """Resolves a (modality, plane) pair to a GeometrySpec. See the module docstring for
    which mode to pick."""

    def __init__(self, mode="per_modality_plane", table=None, single_spec=None):
        if mode not in GEOMETRY_MODES:
            raise ValueError(f"Unknown geometry mode '{mode}'. Choose from: {GEOMETRY_MODES}")
        self.mode = mode
        if mode == "fixed":
            self.single_spec = single_spec or GeometrySpec(
                target_shape=(256, 384, 384), target_spacing=(1.0, 0.5, 0.5),
            )
            self.table = None
        else:
            self.table = table if table is not None else build_geometry_table()
            self.single_spec = None

    def bucket_key(self, modality, plane):
        if self.mode == "fixed":
            return FIXED_GEOMETRY_KEY
        key = (modality, plane)
        return key if key in self.table else FALLBACK_GEOMETRY_KEY

    def resolve(self, modality, plane):
        if self.mode == "fixed":
            return self.single_spec
        return self.table[self.bucket_key(modality, plane)]
