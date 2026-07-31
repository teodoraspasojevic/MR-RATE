"""Anatomical plausibility: does a volume look like a *brain*, not just like the right texture?

FID and PSNR can both look acceptable for a volume that is anatomically wrong -- asymmetric,
the wrong size, or with grey and white matter at indistinguishable intensities. These metrics
check properties that hold for essentially every real head MRI, so a generator violating them is
implausible regardless of what the distribution metrics say.

**Deliberately named "anatomical plausibility", not "clinical".** Every measure here is computed
from voxel intensities with no segmentation model and no radiologist input. They cannot tell you
whether a volume shows the pathology its report describes -- that needs the image-text model this
project does not have, or a validated segmentation network. Calling these clinical would oversell
them.

| Measure | Real brains | Reads as |
|---|---|---|
| `lr_symmetry_ncc` | ~0.85-0.95 | correlation between the volume and its left-right mirror. Brains are grossly symmetric; a low value means implausible asymmetry |
| `intracranial_fraction` | stable per (modality, plane) | foreground voxels / total. Detects heads that are too large or too small for the FOV |
| `tissue_contrast_separation` | > ~1.0 | distance between the two dominant foreground intensity modes, in units of their pooled width. Low means grey/white matter are not distinguishable |
| `foreground_compactness` | ~0.5-0.8 | foreground volume / its bounding-box volume. A head is a solid blob; noise or fragments lower this |
| `background_purity` | ~1.0 | fraction of the outside-the-head region that is genuinely near-zero. Catches haze in the air |

Distributions are compared real-vs-produced (per sequence and overall) with a two-sample
Kolmogorov-Smirnov test, so the output is "do these populations differ", not a paired score --
that makes the group valid for unconditional generation as well as for paired tasks.

numpy + scipy only; no torch, no model weights.
"""
from __future__ import annotations

import numpy as np

ANATOMY_MEASURES = (
    "lr_symmetry_ncc",
    "intracranial_fraction",
    "tissue_contrast_separation",
    "foreground_compactness",
    "background_purity",
)


def _foreground(volume: np.ndarray) -> np.ndarray:
    """Head mask, robust to a non-zero background floor.

    Thresholds relative to the volume's own background level rather than at a fixed value: the VAE
    reconstruction's background sits at ~0.058 while ground truth is bit-exact 0, and a fixed
    threshold would silently select the whole volume for one and the head for the other.
    """
    bg = float(np.median(volume))
    hi = float(np.percentile(volume, 99.0))
    if hi <= bg:
        return np.zeros(volume.shape, dtype=bool)
    return volume > bg + 0.15 * (hi - bg)


def lr_symmetry_ncc(volume: np.ndarray, lr_axis: int = 0) -> float:
    """Normalized cross-correlation between the volume and its mirror across `lr_axis`.

    `lr_axis=0` is correct for this package's (X, Y, Z) = (R-L, A-P, S-I) output order.

    Evaluated over the **union** of the foreground and its mirror, not the intersection. The
    intersection would be blind to the failure this metric exists to catch: if a generator omits a
    whole hemisphere, the intersection excludes exactly the missing region and the remaining
    symmetric part still correlates near 1.0. The union counts tissue-present-on-one-side-only as
    the mismatch it is. Padding outside both is excluded either way, so it cannot inflate the score.
    """
    fg = _foreground(volume)
    if fg.sum() < 1000:
        return float("nan")
    mirrored = np.flip(volume, axis=lr_axis)
    both = fg | np.flip(fg, axis=lr_axis)
    if both.sum() < 1000:
        return float("nan")
    a, b = volume[both].astype(np.float64), mirrored[both].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def intracranial_fraction(volume: np.ndarray) -> float:
    """Foreground voxels as a fraction of the whole array."""
    return float(_foreground(volume).mean())


def tissue_contrast_separation(volume: np.ndarray, n_modes: int = 2) -> float:
    """Separation of the two dominant foreground intensity modes, in pooled-standard-deviation units.

    A 1-D two-component Gaussian fit by expectation-maximization over foreground intensities --
    cheap, deterministic, and enough to answer "are there two distinguishable tissue populations".
    Returns |mu1 - mu2| / sqrt((var1 + var2)/2). Values near 0 mean a single blurred intensity
    population, i.e. no usable grey/white contrast.
    """
    fg = _foreground(volume)
    x = volume[fg].astype(np.float64)
    if x.size < 1000:
        return float("nan")
    if x.size > 200_000:                      # deterministic subsample, this is a 1-D fit
        x = x[:: max(1, x.size // 200_000)]
    lo, hi = np.percentile(x, [10, 90])
    mu = np.array([lo, hi], dtype=np.float64)
    var = np.full(n_modes, max(x.var(), 1e-12))
    w = np.full(n_modes, 1.0 / n_modes)
    for _ in range(40):
        d = x[:, None] - mu[None, :]
        logp = -0.5 * (d * d) / var[None, :] - 0.5 * np.log(2 * np.pi * var)[None, :] + np.log(w)[None, :]
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        r /= np.clip(r.sum(axis=1, keepdims=True), 1e-300, None)
        nk = np.clip(r.sum(axis=0), 1e-12, None)
        mu = (r * x[:, None]).sum(axis=0) / nk
        var = np.clip((r * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk, 1e-12, None)
        w = nk / x.size
    pooled = np.sqrt(var.mean())
    return float(abs(mu[1] - mu[0]) / pooled) if pooled > 0 else float("nan")


def foreground_compactness(volume: np.ndarray) -> float:
    """Foreground volume divided by the volume of its axis-aligned bounding box.

    A head fills a good fraction of its own bounding box. Scattered noise or disconnected
    fragments -- a classic generative failure -- push this down.
    """
    fg = _foreground(volume)
    if not fg.any():
        return float("nan")
    box = 1
    for ax in range(3):
        proj = fg.any(axis=tuple(i for i in range(3) if i != ax))
        idx = np.where(proj)[0]
        box *= int(idx[-1] - idx[0] + 1)
    return float(fg.sum() / box) if box else float("nan")


def background_purity(volume: np.ndarray) -> float:
    """Fraction of the outside-the-head region that is genuinely near-zero.

    Uses the head's bounding box to define "outside", so it does not depend on the same threshold
    the foreground mask uses. Catches the haze a decoder leaves in the air -- the VAE here fills
    the background with ~0.058 instead of 0, which this reports as low purity.
    """
    fg = _foreground(volume)
    if not fg.any():
        return float("nan")
    outside = np.ones(volume.shape, dtype=bool)
    sl = []
    for ax in range(3):
        proj = fg.any(axis=tuple(i for i in range(3) if i != ax))
        idx = np.where(proj)[0]
        sl.append(slice(int(idx[0]), int(idx[-1]) + 1))
    outside[tuple(sl)] = False
    if not outside.any():
        return float("nan")
    vals = np.abs(volume[outside])
    hi = float(np.percentile(volume, 99.0))
    return float(np.mean(vals < 0.02 * max(hi, 1e-12)))


def measure(volume: np.ndarray) -> dict:
    """Every anatomical measure for one volume."""
    return {
        "lr_symmetry_ncc": lr_symmetry_ncc(volume),
        "intracranial_fraction": intracranial_fraction(volume),
        "tissue_contrast_separation": tissue_contrast_separation(volume),
        "foreground_compactness": foreground_compactness(volume),
        "background_purity": background_purity(volume),
    }


def _ks_two_sample(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy import stats

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return {"ks_statistic": None, "p_value": None, "skipped": "fewer than 3 finite values"}
    res = stats.ks_2samp(a, b)
    return {"ks_statistic": float(res.statistic), "p_value": float(res.pvalue)}


def compare_populations(real_measures: list, produced_measures: list) -> dict:
    """Real vs produced, per measure: both distributions summarized, plus a two-sample KS test.

    Unpaired by design, so this works for unconditional generation. A large KS statistic with a
    small p-value means the produced population is anatomically unlike the real one on that measure.
    """
    out = {"n_real": len(real_measures), "n_produced": len(produced_measures)}
    for name in ANATOMY_MEASURES:
        a = np.array([m.get(name, np.nan) for m in real_measures], dtype=np.float64)
        b = np.array([m.get(name, np.nan) for m in produced_measures], dtype=np.float64)
        fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
        out[name] = {
            "real_mean": float(fa.mean()) if fa.size else None,
            "real_std": float(fa.std()) if fa.size else None,
            "produced_mean": float(fb.mean()) if fb.size else None,
            "produced_std": float(fb.std()) if fb.size else None,
            "n_real_valid": int(fa.size),
            "n_produced_valid": int(fb.size),
            **_ks_two_sample(a, b),
        }
    return out
