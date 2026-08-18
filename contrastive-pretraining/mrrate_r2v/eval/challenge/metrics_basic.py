"""MSE / PSNR / SSIM for one 3D volume pair. Ported from the official `metrics_basic.py`."""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _normalize01(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    if hi - lo < 1e-6:
        return np.zeros_like(vol)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)


def compute_basic_metrics(real: np.ndarray, fake: np.ndarray) -> dict:
    """MSE/PSNR/SSIM for one (real, fake) pair. If shapes differ, `fake` is resampled onto
    `real`'s shape with nearest-neighbor-order-1 zoom -- the official code's fallback, not ours."""
    if real.shape != fake.shape:
        from scipy.ndimage import zoom

        factors = [r / f for r, f in zip(real.shape, fake.shape)]
        fake = zoom(fake, factors, order=1)

    real_n = _normalize01(real)
    fake_n = _normalize01(fake)

    mse = float(np.mean((real_n - fake_n) ** 2))
    psnr = float(peak_signal_noise_ratio(real_n, fake_n, data_range=1.0))
    ssim = float(structural_similarity(real_n, fake_n, data_range=1.0))

    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}
