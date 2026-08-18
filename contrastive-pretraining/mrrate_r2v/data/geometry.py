"""What voxel grid each series is resampled onto. Stdlib only, no torch.

Two policies, both selected via `R2VDatasetConfig.geometry_mode`:

- `"fixed"` -- one shape/spacing for every series. Use this for anything you want to
  compare across models: all volumes share a grid, batching just works.
- `"per_modality_plane"` -- a per-(modality, plane) shape derived from
  NV-Generate-MR-Brain's published median training FOVs. Tighter fit per anatomy, but
  shapes differ between buckets, so batch_size > 1 needs `GeometryBucketBatchSampler`
  and the numbers are not comparable with a fixed-mode run.

Axis order here is (D, H, W) = (Superior, Right, Anterior), matching `_preprocess_ops.py`'s
internal convention. `dhw_to_xyz` converts to the (X, Y, Z) = (R, A, S) order the Dataset
actually returns -- see `dataset.py`'s docstring for why the conversion happens exactly
once, at the end.
"""
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

# Every axis of a volume handed to the diffusion UNet must be a multiple of this.
#
# Verified empirically (job 662720): every div-32 shape generates, every div-16-but-not-32 shape
# raises `RuntimeError: Sizes of tensors must match ... Expected size 14 but got size 15` from the
# UNet's skip connections. The arithmetic: the UNet has 4 levels (3 downsamples) so its latent must
# be divisible by 8, and the latent is `output_size // 4`, hence 32.
#
# Note this is STRICTER than the VAE's own requirement of 16 (`required_spatial_divisor`). Checking
# only 16 is necessary but not sufficient -- it passes the autoencoder and fails the UNet.
UNET_SPATIAL_MULTIPLE = 32

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


def _round_to_multiple(value, multiple):
    """Nearest positive multiple of `multiple`."""
    return max(multiple, int(round(value / multiple)) * multiple)


@dataclass(frozen=True)
class GeometrySpec:
    """One resample/crop target, both fields (D, H, W)-ordered."""

    target_shape: tuple
    target_spacing: tuple

    @property
    def physical_fov_mm(self):
        return tuple(s * p for s, p in zip(self.target_shape, self.target_spacing))


def build_geometry_table(fov_table=None, unet_multiple=UNET_SPATIAL_MULTIPLE,
                         fallback_fov_mm=DEFAULT_FALLBACK_FOV_MM):
    """{(modality, plane): GeometrySpec} realising NVIDIA's published FOV table exactly.

    For each bucket: pick the shape as the nearest multiple of `unet_multiple` (the diffusion
    UNet's hard constraint), then derive `spacing = FOV / shape`. The physical field of view
    therefore equals NVIDIA's recommendation **exactly**, and the resulting spacing -- which lands
    within about +/-10% of 1 mm for every published bucket -- is what the model is conditioned on.

    Why derive spacing instead of fixing it at 1 mm: `spacing` is a real conditioning input to the
    UNet (it reaches the network as `spacing_tensor`), and the FOV table is the quantity NVIDIA
    actually validated. Fixing spacing at 1 mm and rounding the shape up instead would over-cover
    the recommended FOV by up to 30 mm on an axis. Non-1 mm spacing conditioning was verified to
    run (job 662720).
    """
    fov_table = fov_table if fov_table is not None else NV_BRAIN_FOV_MM
    table = {}
    for key, fov in list(fov_table.items()) + [(FALLBACK_GEOMETRY_KEY, fallback_fov_mm)]:
        shape = tuple(_round_to_multiple(f, unet_multiple) for f in fov)
        spacing = tuple(f / s for f, s in zip(fov, shape))
        table[key] = GeometrySpec(target_shape=shape, target_spacing=spacing)
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
