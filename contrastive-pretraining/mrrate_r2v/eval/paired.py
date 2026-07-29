"""Metrics on one (ground truth, prediction) pair. Plain float32 numpy arrays in, floats out.

Two families, matching `tasks.py`'s metric groups:

- **fidelity** -- `mae`, `mse`, `psnr`, `ncc`, `ssim_3d`, `relative_intensity_error`: how close
  is this volume to the target, voxel by voxel.
- **perceptual** -- `edge_preservation_ratio`, `laplacian_variance_ratio`,
  `high_frequency_energy_ratio`, `ssim_2d_mean`: is fine detail surviving or has it blurred.

Two things to know before reading a number from here:

1. **Intensities are not clipped to [0, 1].** The default percentile normalizer leaves values
   above the 99.5th percentile above 1.0. `data_range=1.0` is a fixed reference scale for
   cross-sample comparability, not a real maximum -- so treat PSNR as relative, not absolute.
2. **Geometry is not checked here.** These functions only ever see arrays. The caller must have
   gotten STRICT_MATCH from `geometry_contract.compare_geometry` first; `runner.py` does.

`skimage` is imported lazily inside the SSIM functions, so the rest of this module works
without it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

EPS = 1e-8


def _sk_ssim():
    """scikit-image's SSIM, imported on first use so a missing scikit-image only breaks SSIM."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "SSIM requires scikit-image (`pip install scikit-image`). Every other metric in "
            "mrrate_r2v.eval.paired works without it."
        ) from e
    return structural_similarity


def foreground_mask_from_intensity(volume: np.ndarray, percentile: float = 1.0) -> np.ndarray:
    """Heuristic foreground mask (NOT an anatomical segmentation -- this repository's R2V Dataset
    has no per-sample brain-mask field to use instead, unlike the old implementation's real
    HD-BET masks). Voxels above the given percentile of the volume's own intensity distribution.
    """
    if volume.max() <= 0:
        return np.zeros_like(volume, dtype=bool)
    thresh = np.percentile(volume, percentile)
    return volume > max(thresh, EPS)


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(a - b)
    if mask is not None:
        if not mask.any():
            return float("nan")
        return float(diff[mask].mean())
    return float(diff.mean())


def mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a - b) ** 2
    if mask is not None:
        if not mask.any():
            return float("nan")
        return float(diff[mask].mean())
    return float(diff.mean())


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0, mask: np.ndarray | None = None) -> float:
    m = mse(a, b, mask)
    if not np.isfinite(m) or m <= 0:
        return float("inf") if m == 0 else float("nan")
    return float(10.0 * np.log10((data_range**2) / m))


def relative_intensity_error(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """mean(|a-b|) / mean(|a|) over the mask -- a scale-normalized error, unlike absolute MAE.
    Can blow up (large std) when the mask's mean intensity is near zero; not hidden, reported as-is.
    """
    if mask is not None:
        a_m, b_m = a[mask], b[mask]
    else:
        a_m, b_m = a.ravel(), b.ravel()
    denom = np.abs(a_m).mean()
    if denom < EPS:
        return float("nan")
    return float(np.abs(a_m - b_m).mean() / denom)


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Normalized cross-correlation, [-1, 1], higher better."""
    if mask is not None:
        a_m, b_m = a[mask], b[mask]
    else:
        a_m, b_m = a.ravel(), b.ravel()
    a_c, b_c = a_m - a_m.mean(), b_m - b_m.mean()
    denom = np.sqrt((a_c**2).sum() * (b_c**2).sum())
    if denom < EPS:
        return float("nan")
    return float((a_c * b_c).sum() / denom)


def ssim_3d(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    return float(_sk_ssim()(a, b, data_range=data_range))


def ssim_2d_mean(a: np.ndarray, b: np.ndarray, axis: int, data_range: float = 1.0, min_fg_frac: float = 0.01) -> dict:
    """Mean 2D SSIM over slices along `axis`, excluding background-only slices (foreground
    fraction below `min_fg_frac` of the slice, by the same intensity heuristic as
    `foreground_mask_from_intensity`) -- an empty/background slice's SSIM is not informative and
    would bias the mean toward whatever a near-constant comparison happens to score.
    """
    ssim = _sk_ssim()
    n = a.shape[axis]
    scores = []
    for i in range(n):
        sa = np.take(a, i, axis=axis)
        if (sa > 1e-3).mean() < min_fg_frac:
            continue
        sb = np.take(b, i, axis=axis)
        scores.append(ssim(sa, sb, data_range=data_range))
    return {"mean": float(np.mean(scores)) if scores else None, "n_slices_used": len(scores), "n_slices_total": n}


def gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    gx, gy, gz = np.gradient(volume)
    return np.sqrt(gx**2 + gy**2 + gz**2)


def edge_preservation_ratio(orig: np.ndarray, recon: np.ndarray, mask: np.ndarray | None = None) -> float:
    """mean(|grad(recon)|) / mean(|grad(orig)|) over the mask -- 1.0 means edges preserved,
    <1 means smoothing/blur (the typical VAE failure mode), >1 would mean spurious high-frequency
    content added.
    """
    g_o, g_r = gradient_magnitude(orig), gradient_magnitude(recon)
    if mask is not None:
        g_o, g_r = g_o[mask], g_r[mask]
    denom = g_o.mean()
    if denom < EPS:
        return float("nan")
    return float(g_r.mean() / denom)


def laplacian_variance_ratio(orig: np.ndarray, recon: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance-of-Laplacian is a classic no-reference sharpness proxy; the ratio recon/orig
    behaves like edge_preservation_ratio but via a different (2nd-derivative) operator.
    """
    lap_o = ndi.laplace(orig)
    lap_r = ndi.laplace(recon)
    if mask is not None:
        lap_o, lap_r = lap_o[mask], lap_r[mask]
    var_o = lap_o.var()
    if var_o < EPS:
        return float("nan")
    return float(lap_r.var() / var_o)


_HF_MASK_CACHE: dict = {}


def _high_frequency_mask(shape: tuple, cutoff_frac: float) -> np.ndarray:
    """Boolean mask of fftshift-ed frequencies above `cutoff_frac` of Nyquist.

    Depends only on (shape, cutoff), so it is built once and cached rather than rebuilt for
    every volume -- the meshgrid+sqrt allocates three 134 MB float64 arrays at 256^3 and
    measured 239 ms, which was ~25% of this metric's cost and was being paid twice per case.
    """
    key = (tuple(shape), float(cutoff_frac))
    mask = _HF_MASK_CACHE.get(key)
    if mask is None:
        shape_arr = np.array(shape)
        center = shape_arr // 2
        grids = np.meshgrid(*[np.arange(s) - c for s, c in zip(shape_arr, center)], indexing="ij")
        radius = np.sqrt(sum(g**2 for g in grids)) / (np.linalg.norm(shape_arr / 2) + EPS)
        mask = radius > cutoff_frac
        _HF_MASK_CACHE[key] = mask
    return mask


def high_frequency_energy_ratio(orig: np.ndarray, recon: np.ndarray, cutoff_frac: float = 0.25) -> float:
    """Fraction of 3D FFT energy above `cutoff_frac` of the Nyquist frequency, recon/orig ratio.
    Whole-volume only (no mask) since FFT is not meaningfully maskable.
    """
    mask = _high_frequency_mask(orig.shape, cutoff_frac)

    def hf_energy(vol):
        power = np.abs(np.fft.fftshift(np.fft.fftn(vol))) ** 2
        return power[mask].sum()

    denom = hf_energy(orig)
    if denom < EPS:
        return float("nan")
    return float(hf_energy(recon) / denom)


def through_plane_consistency(volume: np.ndarray, axis: int) -> np.ndarray:
    """Slice-to-adjacent-slice NCC along `axis` -- a single volume's own internal consistency
    (no reference needed), useful for spotting generation artifacts like banding/tearing between
    slices. Returns one value per adjacent-slice pair (length shape[axis]-1).
    """
    n = volume.shape[axis]
    scores = []
    for i in range(n - 1):
        s0, s1 = np.take(volume, i, axis=axis), np.take(volume, i + 1, axis=axis)
        scores.append(ncc(s0, s1))
    return np.array(scores, dtype=np.float64)


def intensity_percentiles(volume: np.ndarray, mask: np.ndarray | None = None, ps=(0.5, 1, 5, 25, 50, 75, 95, 99, 99.5)) -> dict:
    vals = volume[mask] if mask is not None else volume.ravel()
    if vals.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": float(np.percentile(vals, p)) for p in ps}


def latent_statistics(z_mu: np.ndarray, z_sigma: np.ndarray) -> dict:
    return {
        "z_mu_mean": float(z_mu.mean()), "z_mu_std": float(z_mu.std()),
        "z_sigma_mean": float(z_sigma.mean()), "z_sigma_std": float(z_sigma.std()),
    }


def compression_ratio(input_shape, latent_shape, input_bytes_per_voxel: int = 4, latent_bytes_per_voxel: int = 4) -> dict:
    input_voxels = int(np.prod(input_shape))
    latent_voxels = int(np.prod(latent_shape))
    return {
        "input_voxels": input_voxels, "latent_voxels": latent_voxels,
        "compression_ratio": (input_voxels * input_bytes_per_voxel) / max(latent_voxels * latent_bytes_per_voxel, 1),
    }
