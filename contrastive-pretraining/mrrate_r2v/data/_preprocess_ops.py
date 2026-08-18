"""Per-volume preprocessing: RAS reorient -> resample -> normalize -> crop/pad.

Vendored from the contrastive pipeline's `scripts/data.py` so this package has no
import dependency on anything outside `mrrate_r2v` (no `sys.path` reach into
`contrastive-pretraining/scripts`). This is a deliberate, one-time fork: R2V and
the contrastive pipeline may now diverge here, since the whole point of this
module is for `mrrate_r2v` to be extractable into its own repository.

Importing this pulls in torch and nibabel. Keep it out of module-level imports in
anything that must stay lightweight (`manifest.py` imports it lazily, inside
functions, for exactly that reason -- see its docstring).
"""
import os
import csv
import gzip

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib


def resize_array(array, current_spacing, target_spacing):
    """Resize array to match target spacing using trilinear interpolation."""
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


class ZScoreNormalizer:
    """Z-score on nonzero voxels, clip to [-5,5], rescale to [-1,1]."""

    def normalize(self, data):
        mask = data != 0
        if mask.sum() > 0:
            mean = data[mask].mean()
            std = data[mask].std()
            data = (data - mean) / (std + 1e-8)
        data = np.clip(data, -5.0, 5.0)
        data = data / 5.0
        return data


class PercentileNormalizer:
    """Rescale [lower, upper] percentile to [lower_limit, upper_limit].

    clip=True (default, unchanged behavior) additionally clamps the input to
    the [low, high] percentile bounds before rescaling, so the output is
    guaranteed to stay within [lower_limit, upper_limit]. clip=False rescales
    using the same bounds but does not clamp, so values beyond the upper
    percentile can map above upper_limit -- this matches NV-Generate-CTMR's
    MRI intensity transform (ScaleIntensityRangePercentilesd(..., clip=False))
    for consumers that want to match that pipeline exactly.
    """

    def __init__(self, lower_percentile=0.5, upper_percentile=99.5,
                 lower_limit=-1.0, upper_limit=1.0, clip=True):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.clip = clip

    def normalize(self, data):
        mask = data != 0
        if mask.sum() > 0:
            low = np.percentile(data[mask], self.lower_percentile)
            high = np.percentile(data[mask], self.upper_percentile)
        else:
            low, high = data.min(), data.max()
        if self.clip:
            data = np.clip(data, low, high)
        if high - low > 1e-8:
            data = (data - low) / (high - low)
            data = data * (self.upper_limit - self.lower_limit) + self.lower_limit
        else:
            data = np.zeros_like(data)
        return data


class MinMaxNormalizer:
    """Simple min-max rescale to [lower_limit, upper_limit]."""

    def __init__(self, lower_limit=-1.0, upper_limit=1.0):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def normalize(self, data):
        dmin = data.min()
        dmax = data.max()
        if dmax - dmin > 1e-8:
            data = (data - dmin) / (dmax - dmin)
            data = data * (self.upper_limit - self.lower_limit) + self.lower_limit
        else:
            data = np.zeros_like(data)
        return data


NORMALIZERS = {
    'zscore': ZScoreNormalizer,
    'percentile': PercentileNormalizer,
    'minmax': MinMaxNormalizer,
}


# Mapping from logical space name to the image subdirectory used in the
# raw HuggingFace download layout (layout 2). The native repo stores volumes
# in `img/`, while derivative repos (coreg, atlas) use prefixed names.
SPACE_TO_IMG_SUBDIR = {
    'native_space': 'img',
    'coreg_space': 'coreg_img',
    'atlas_space': 'atlas_img',
}

def list_nii_files(img_dir):
    """Sorted list of absolute *.nii.gz paths in img_dir (empty if it's missing)."""
    if not os.path.isdir(img_dir):
        return []
    return sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith('.nii.gz')
    )


def discover_subjects(data_folder, space):
    """Discover subjects and their NIfTI volumes under data_folder for a space.

    Report/split agnostic: returns every subject directory that has >=1 NIfTI in
    its image subdir.

    Supports two on-disk layouts:
      1) data_folder/<study_uid>/<space>/img/*.nii.gz
      2) data_folder/batchXX/<study_uid>/<img_subdir>/*.nii.gz

    Returns a list of {'subject_id': str, 'image_paths': [str, ...]}.
    """
    first_level_dirs = sorted([
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    ])
    if not first_level_dirs:
        return []

    # Auto-detect layout: layout 1 if the first entry has a <space> subfolder.
    first_dir = os.path.join(data_folder, first_level_dirs[0])
    use_space_layout = os.path.isdir(os.path.join(first_dir, space))

    found = []
    if use_space_layout:
        for study_uid in first_level_dirs:
            img_dir = os.path.join(data_folder, study_uid, space, 'img')
            nii = list_nii_files(img_dir)
            if nii:
                found.append({'subject_id': study_uid, 'image_paths': nii})
    else:
        img_subdir = SPACE_TO_IMG_SUBDIR.get(space, 'img')
        for batch_dir in first_level_dirs:
            batch_path = os.path.join(data_folder, batch_dir)
            for study_uid in sorted(os.listdir(batch_path)):
                img_dir = os.path.join(batch_path, study_uid, img_subdir)
                nii = list_nii_files(img_dir)
                if nii:
                    found.append({'subject_id': study_uid, 'image_paths': nii})
    return found


def _canonicalize(nii_img):
    """Reorient an already-loaded nibabel image to canonical RAS.

    Shared by every path- and bytes-based loader/geometry-reader in this
    module so "how we canonicalize" is defined exactly once.
    """
    return nib.as_closest_canonical(nii_img)


def _resample_canonical(nii_img, target_spacing):
    """Core of load_and_resample_nii, operating on an already-canonicalized
    nibabel image (i.e. the part that requires get_fdata()). Split out so
    load_and_resample_nii (path-based) and load_and_resample_nii_from_bytes
    (in-memory-bytes-based, used by the archive-backed storage backend in
    storage.py) share one implementation and cannot drift.
    """
    img_data = nii_img.get_fdata().astype(np.float32)
    np.nan_to_num(img_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    voxel_sizes = nii_img.header.get_zooms()
    if len(voxel_sizes) >= 3:
        # RAS: dim0=R(X), dim1=A(Y), dim2=S(Z). Reorder spacing to (Z, X, Y).
        current_spacing = (float(voxel_sizes[2]), float(voxel_sizes[0]), float(voxel_sizes[1]))
    else:
        current_spacing = (1.0, 1.0, 1.0)

    img_data = img_data.transpose(2, 0, 1)  # (X, Y, Z) -> (Z, X, Y)
    tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)
    resampled = resize_array(tensor, current_spacing, target_spacing)[0, 0]
    return resampled


def _geometry_of_canonical(nii_img):
    """Core of read_native_geometry, operating on an already-canonicalized
    nibabel image, without calling get_fdata(). Split out for the same
    drift-avoidance reason as _resample_canonical.
    """
    x, y, z = nii_img.shape[:3]
    zx, zy, zz = nii_img.header.get_zooms()[:3]
    shape = (int(z), int(x), int(y))
    spacing = (float(zz), float(zx), float(zy))
    return shape, spacing


def _looks_gzipped(data):
    return len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B


def load_and_resample_nii(path, target_spacing):
    """Load NIfTI, reorient to canonical RAS, resample to target spacing.

    Returns a float32 numpy array of shape (D, H, W) = (Z, X, Y) in RAS.
    """
    nii_img = _canonicalize(nib.load(str(path)))
    return _resample_canonical(nii_img, target_spacing)


def load_and_resample_nii_from_bytes(raw_bytes, target_spacing):
    """Same as load_and_resample_nii, but from already-read NIfTI bytes
    (a `.nii.gz` file's exact on-disk bytes, gzip-compressed or not) instead
    of a filesystem path -- no temporary file is written anywhere. Used by
    the archive-backed storage backend (storage.py) so a series read
    directly out of an un-extracted archive goes through the identical
    canonicalize/resample logic as one read from an extracted directory.
    """
    payload = gzip.decompress(raw_bytes) if _looks_gzipped(raw_bytes) else raw_bytes
    nii_img = _canonicalize(nib.Nifti1Image.from_bytes(payload))
    return _resample_canonical(nii_img, target_spacing)


def read_native_geometry(path):
    """Header-only (D, H, W)-ordered native shape/spacing, no voxel decode.

    Matches load_and_resample_nii's axis convention exactly: (D, H, W) =
    (S, R, A) after RAS canonicalization, i.e. what target_shape/target_spacing
    are indexed against everywhere else in this module. Used by generation
    tooling that must preserve pre-resample geometry alongside the
    resampled/cropped tensor; load_and_resample_nii discards this once it
    resamples, and calling it just for shape/spacing would force a full
    get_fdata() decode this function deliberately avoids.
    """
    nii_img = _canonicalize(nib.load(str(path)))
    return _geometry_of_canonical(nii_img)


def read_native_geometry_from_bytes(raw_bytes):
    """Same as read_native_geometry, but from already-read NIfTI bytes.

    Unlike the path-based version, this does need a full gzip decompress of
    the payload (nibabel's from_bytes has no lazy/partial-decompress path
    the way a real file's header can be read without decoding pixel data) --
    acceptable for MR-RATE's observed per-series compressed sizes (single-
    digit to low-hundreds of MB), and only ever called lazily at first actual
    training-time access to a given series (never during archive-backed
    manifest/index building -- see manifest.py's build_manifest_rows_from_*
    functions), so it never adds an extra decompression beyond the one
    __getitem__ already pays for.
    """
    payload = gzip.decompress(raw_bytes) if _looks_gzipped(raw_bytes) else raw_bytes
    nii_img = _canonicalize(nib.Nifti1Image.from_bytes(payload))
    return _geometry_of_canonical(nii_img)


def load_all_splits(splits_csv):
    """study_uid -> split label, for every row in a splits CSV.

    Keeps every row's split label rather than filtering to one split's
    allow-set, so a manifest can record each row's split once and be
    filtered to any split afterward without re-reading the CSV.
    """
    mapping = {}
    with open(splits_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['study_uid']] = row['split']
    return mapping


def crop_or_pad(data, target_shape, posterior_shift_voxels):
    """Center crop/pad a (D, H, W) array to target_shape, with posterior W shift.

    The W axis (Y in RAS = anterior-posterior) is shifted posteriorly by
    posterior_shift_voxels to compensate for defacing; if the shift pushes past
    the posterior edge the crop starts at index 0. Pad value is 0 (background).

    Returns a float32 numpy array of shape target_shape.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))

    td, th, tw = target_shape
    d, h, w = tensor.shape

    # Center crop start indices
    d_start = max((d - td) // 2, 0)
    h_start = max((h - th) // 2, 0)

    # W axis (Y/AP): shift center posteriorly (toward lower index in RAS)
    w_center = w // 2 - posterior_shift_voxels
    w_start = w_center - tw // 2
    w_start = max(w_start, 0)
    w_start = min(w_start, max(w - tw, 0))

    tensor = tensor[d_start:d_start + td, h_start:h_start + th, w_start:w_start + tw]

    pad_d_before = (td - tensor.size(0)) // 2
    pad_d_after = td - tensor.size(0) - pad_d_before
    pad_h_before = (th - tensor.size(1)) // 2
    pad_h_after = th - tensor.size(1) - pad_h_before
    pad_w_before = (tw - tensor.size(2)) // 2
    pad_w_after = tw - tensor.size(2) - pad_w_before

    tensor = F.pad(
        tensor,
        (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after),
        value=0,
    )
    return tensor.numpy()


def preprocess_nii(path, target_spacing, target_shape, posterior_shift_voxels, normalizer_obj):
    """Full single-volume transform: load+RAS+resample -> normalize -> crop/pad.

    Returns a float32 numpy array of shape target_shape (D, H, W). Single source
    of truth for per-volume preprocessing.
    """
    resampled = load_and_resample_nii(path, target_spacing)
    normalized = normalizer_obj.normalize(resampled)
    return crop_or_pad(normalized, target_shape, posterior_shift_voxels)


def preprocess_nii_from_bytes(raw_bytes, target_spacing, target_shape, posterior_shift_voxels, normalizer_obj):
    """Same as preprocess_nii, but from already-read NIfTI bytes instead of a
    path -- mirrors preprocess_nii exactly (load step swapped for
    load_and_resample_nii_from_bytes), so an archive-backed series and an
    extracted-file series go through identical normalize/crop_or_pad code.
    Used by the archive-backed storage path (storage.py).
    """
    resampled = load_and_resample_nii_from_bytes(raw_bytes, target_spacing)
    normalized = normalizer_obj.normalize(resampled)
    return crop_or_pad(normalized, target_shape, posterior_shift_voxels)


__all__ = [
    "NORMALIZERS", "crop_or_pad", "discover_subjects", "load_all_splits",
    "load_and_resample_nii", "load_and_resample_nii_from_bytes",
    "preprocess_nii", "preprocess_nii_from_bytes",
    "read_native_geometry", "read_native_geometry_from_bytes",
]
