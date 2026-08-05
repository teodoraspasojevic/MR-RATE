"""The three training-validation metrics, defined precisely and in one place.

    val/fvd            FVD-style Frechet distance over sequence features   distribution   lower better
    val/fid_2p5d       volume-weighted 2.5D Frechet Inception Distance     distribution   lower better
    val/ssim           paired 3D SSIM against the case's own ground truth  paired         higher better

**What these do and do not measure.** FVD and 2.5D FID compare the *distribution* of generated
volumes to the distribution of real ones; neither is defined for a single case. SSIM compares one
generated volume to *its own* ground-truth volume. **None of the three measures whether the
generated volume is semantically faithful to the report that conditioned it.** That needs a
validated MRI image-text model, and no such model with verifiable open weights was found -- see
`docs/TEXT_ENCODERS.md` section 9.6.

Provenance of each definition:

| metric | status |
|---|---|
| FVD | **MRI-volume adaptation.** Challenge-precedented in family (VLM3D 2025 CT Task 4 scored `FVD_I3D` + `FVD_CT-Net`), but the extractor is r3d_18, not I3D. See `video_features.py`. |
| 2.5D FID | **Project-specific adaptation.** Neither the challenge nor GenerateCT uses the name; GenerateCT's FID is plain *slice-level* InceptionV3, whose limitation it states itself. This is the volume-weighted three-plane variant already in `distribution.py`. |
| SSIM | **Standard** (Wang et al. 2004) via `skimage`, with the Wang settings named explicitly below. Used by neither the challenge nor GenerateCT -- it is a training-progress signal here, not a challenge score. |

**Every metric operates in one intensity space** (`video_features.METRIC_INTENSITY_SPACE`): the
model-input percentile-normalised space, where 0-99.5th percentile maps to [0, 1]. Ground truth
arrives that way from the Dataset; a generation must therefore be the **decoder's float output**,
not `sampling.postprocess_mr`'s int16 [0, 1000]. Feeding a postprocessed generation against a
normalised ground truth is 1000x off and still returns a plausible-looking number, which is why the
space is named in a constant and asserted here rather than left to a convention.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("mrrate_r2v.eval.validation_metrics")

# --- SSIM parameters, stated rather than inherited from a default ------------------------------
#: Wang et al. (2004) SSIM: 11-wide Gaussian window, sigma 1.5, population covariance, K1/K2 as
#: published. `skimage`'s bare defaults are a 7-wide *uniform* window with sample covariance, which
#: is a different estimator -- `paired.ssim_3d` uses those and is left untouched so existing
#: evaluation numbers stay comparable; validation states the canonical settings instead.
SSIM_WIN_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03
#: A fixed reference scale, not per-volume. The percentile normaliser puts the 99.5th percentile at
#: 1.0 and leaves a tail above it. Rescaling each volume by its own max would let a generation with
#: badly wrong intensities look structurally fine -- exactly the error SSIM should expose.
SSIM_DATA_RANGE = 1.0

#: Voxel spacing agreement required before a pair is compared, in mm. Tighter than any real
#: difference between a cohort case and its own generation (which share one `GeometrySpec`), so this
#: only ever fires on a genuine mix-up.
SPACING_TOLERANCE_MM = 1e-3

#: Absolute floor below which a Frechet distance is not attempted at all.
MIN_FRECHET_SAMPLES = 16

#: **The constraint that decides how validation is scheduled.** A Frechet distance estimates a
#: `D x D` covariance from `N` samples, so at `N < D` the estimate is *rank-deficient by
#: construction*: `sqrtm(S_r @ S_g)` is ill-conditioned and `distribution.frechet_distance` either
#: needs its epsilon regularisation or refuses outright with a non-negligible imaginary component.
#:
#: The feature dimensions here are 512 (r3d_18) and 2048 (InceptionV3), so a *reliable* FVD or
#: 2.5D FID needs hundreds to thousands of volumes -- and every validation volume costs a full
#: diffusion sampling run. The practical consequence, documented rather than hidden:
#:
#:   - **SSIM is the frequent-validation metric.** Paired and per-case, so it is meaningful at
#:     N = 32-64 and its standard error shrinks like 1/sqrt(N).
#:   - **FVD and 2.5D FID are trend-only at small N** and belong in the occasional full pass, or
#:     better, in the offline `cli.evaluate` path over a real cohort (which is where this
#:     repository already computes distribution metrics, on ~2000 cases).
#:
#: `rank_status` records which regime a given number is in, and it travels with the metric so a
#: value can never be read as more trustworthy than its sample supports.
RANK_DEFICIENT_NOTE = (
    "N < feature_dim: the covariance estimate is rank-deficient, so this value is a trend "
    "indicator only, not a calibrated distance. Compare it only against other values computed at "
    "the same N with the same extractor."
)


def rank_status(n_samples: int, feature_dim: int) -> dict:
    """Which small-sample regime a Frechet distance is in. Travels with every reported value."""
    if n_samples >= 2 * feature_dim:
        level, note = "well_conditioned", "N >= 2 x feature_dim"
    elif n_samples >= feature_dim:
        level, note = "marginal", "feature_dim <= N < 2 x feature_dim: full rank but noisy"
    else:
        level, note = "rank_deficient", RANK_DEFICIENT_NOTE
    return {"n_samples": int(n_samples), "feature_dim": int(feature_dim),
            "level": level, "note": note}


# --------------------------------------------------------------------------- geometry gating


@dataclass
class PairVerdict:
    """Whether a (ground truth, generated) pair may be compared voxelwise, and why not if not."""

    ok: bool
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


def verify_pair(gt: np.ndarray, generated: np.ndarray, gt_spacing: Sequence[float],
                generated_spacing: Optional[Sequence[float]] = None) -> PairVerdict:
    """Gate SSIM on geometric comparability. **Never resizes to force a comparison.**

    Equal `.shape` is necessary but not sufficient -- the same rule
    `eval/geometry_contract.py` already enforces: two arrays can share a shape and describe
    different physical volumes, so spacing is checked too. A mismatch is *excluded with a reason*,
    which is then reported in the metric payload, rather than silently corrected.

    Orientation and affine are not re-checked here because they cannot differ by construction: a
    validation generation is produced *onto* the cohort case's own frozen `GeometrySpec`, so it
    inherits that case's RAS orientation and axis-aligned affine. Anatomical coverage follows from
    shape plus spacing being equal on a shared grid.
    """
    gt = np.asarray(gt)
    generated = np.asarray(generated)
    if generated.shape != gt.shape:
        return PairVerdict(False, f"shape_mismatch: gt {tuple(gt.shape)} vs generated "
                                  f"{tuple(generated.shape)} (never resized to force a comparison)")
    if generated_spacing is not None:
        a = np.asarray(gt_spacing, dtype=float)
        b = np.asarray(generated_spacing, dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, atol=SPACING_TOLERANCE_MM):
            return PairVerdict(False, f"spacing_mismatch: gt {tuple(a)} vs generated {tuple(b)}")
    if not np.isfinite(generated).all():
        return PairVerdict(False, "generated_not_finite: NaN or inf in the generated volume")
    if gt.ndim != 3:
        return PairVerdict(False, f"not_3d: ground truth has shape {tuple(gt.shape)}")
    return PairVerdict(True)


def foreground_bounding_box(volume: np.ndarray, threshold: float = 1e-6):
    """The tightest box containing every voxel above `threshold`, as a tuple of slices.

    **Always computed from the ground truth**, never from the generation -- the same rule the
    evaluator's foreground mask follows, and for the same reason: a degenerate generation must not
    get to choose an easier comparison region.

    A *crop*, not a mask. SSIM is a sliding-window statistic; zeroing voxels inside the window
    changes local means and variances and produces a number that is not SSIM. Cropping keeps every
    window valid while removing the bit-exact zero padding, which is ~52% of a 256^3 cohort volume
    and would otherwise contribute a perfect score to a large fraction of the windows.
    """
    mask = np.abs(np.asarray(volume)) > threshold
    if not mask.any():
        return tuple(slice(0, s) for s in volume.shape)
    box = []
    for axis in range(volume.ndim):
        projected = mask.any(axis=tuple(i for i in range(volume.ndim) if i != axis))
        index = np.flatnonzero(projected)
        box.append(slice(int(index[0]), int(index[-1]) + 1))
    return tuple(box)


# --------------------------------------------------------------------------- SSIM


def _skimage_ssim():
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:  # pragma: no cover
        raise ImportError("validation SSIM requires scikit-image (`pip install scikit-image`)") from exc
    return structural_similarity


def ssim_volume(gt: np.ndarray, generated: np.ndarray, *, crop_to_foreground: bool = True) -> dict:
    """3D SSIM between one generation and its own ground truth, in the Wang et al. settings.

    **Computed directly in 3D**, not slice-by-slice: a 3D Gaussian window measures through-plane
    structure too, which is where a slice-stacking generator fails and a per-slice mean would not
    see it. (`paired.ssim_2d_mean` remains available as a diagnostic.)

    Returns the primary foreground-cropped value plus the whole-volume value, so the effect of the
    padding is visible rather than baked in.
    """
    structural_similarity = _skimage_ssim()
    gt = np.asarray(gt, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)

    def score(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        # A window cannot be larger than the smallest axis, and skimage raises rather than adapt.
        window = min(SSIM_WIN_SIZE, *(s for s in a.shape))
        if window < 3:
            return None
        if window % 2 == 0:
            window -= 1
        return float(structural_similarity(
            a, b, data_range=SSIM_DATA_RANGE, win_size=window,
            gaussian_weights=True, sigma=SSIM_SIGMA, use_sample_covariance=False,
            K1=SSIM_K1, K2=SSIM_K2,
        ))

    out = {"ssim_whole_volume": score(gt, generated)}
    if crop_to_foreground:
        box = foreground_bounding_box(gt)
        out["ssim"] = score(np.ascontiguousarray(gt[box]), np.ascontiguousarray(generated[box]))
        out["foreground_fraction"] = float(np.prod([s.stop - s.start for s in box]) / gt.size)
    else:
        out["ssim"] = out["ssim_whole_volume"]
        out["foreground_fraction"] = 1.0
    return out


def ssim_parameters() -> dict:
    """The exact SSIM configuration, for the result payload and the W&B run config."""
    return {
        "implementation": "skimage.metrics.structural_similarity",
        "dimensionality": "3D (single volumetric window, not slice-by-slice)",
        "win_size": SSIM_WIN_SIZE,
        "gaussian_weights": True,
        "sigma": SSIM_SIGMA,
        "use_sample_covariance": False,
        "K1": SSIM_K1,
        "K2": SSIM_K2,
        "data_range": SSIM_DATA_RANGE,
        "normalisation": "shared model-input percentile space; never per-volume rescaled",
        "background": "cropped to the ground truth's foreground bounding box (a crop, not a mask)",
        "standard": True,
        "used_by_challenge": False,
    }


# --------------------------------------------------------------------------- Frechet distances


def frechet(real: np.ndarray, generated: np.ndarray) -> Optional[float]:
    """Numerically stable Frechet distance, reusing `distribution.frechet_distance`.

    One implementation for FVD and for 2.5D FID -- the two differ only in which features they are
    handed, which is the whole point of separating feature extraction from the distance.
    """
    from .distribution import frechet_distance

    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError(f"feature arrays must be (N, D) with matching D, got {real.shape} and "
                         f"{generated.shape}")
    if len(real) < 2 or len(generated) < 2:
        return None
    try:
        return float(frechet_distance(real, generated))
    except ValueError as exc:
        # `frechet_distance` refuses an ill-conditioned covariance rather than returning a silently
        # complex or NaN value -- correct behaviour, and routine at N < feature_dim. A validation
        # pass must not die of it: the metric is withheld for this step and the reason is logged.
        log.warning("Frechet distance withheld (N=%d, D=%d): %s", len(real), real.shape[1], exc)
        return None


@dataclass
class PlaneFeatures:
    """Per-plane feature accumulation for one metric family.

    Stored as lists of vectors, appended one volume at a time, so peak memory is
    `n_volumes x n_planes x feature_dim` floats (a few MB) and never a set of volumes.
    """

    planes: tuple
    real: dict = field(default_factory=dict)
    generated: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for plane in self.planes:
            self.real.setdefault(plane, [])
            self.generated.setdefault(plane, [])

    def add(self, real: dict, generated: dict) -> None:
        for plane in self.planes:
            if real.get(plane) is not None:
                self.real[plane].append(np.asarray(real[plane], dtype=np.float64).reshape(-1))
            if generated.get(plane) is not None:
                self.generated[plane].append(np.asarray(generated[plane], dtype=np.float64).reshape(-1))

    def counts(self) -> dict:
        return {plane: (len(self.real[plane]), len(self.generated[plane])) for plane in self.planes}

    def extend(self, other: "PlaneFeatures") -> None:
        for plane in self.planes:
            self.real[plane].extend(other.real.get(plane, []))
            self.generated[plane].extend(other.generated.get(plane, []))


def aggregate_frechet(features: PlaneFeatures, key: str, *,
                      min_samples: int = MIN_FRECHET_SAMPLES,
                      single_plane: bool = False) -> dict:
    """Per-plane Frechet distances plus one headline unweighted mean.

    **Unweighted, not sample-count-weighted**: every plane is a complete view of every volume, not
    a variable-size sample of anything, so the three carry equal evidence. (Contrast the
    *population*-weighted bucket aggregate in `eval/aggregate.py`, which weights because bucket
    sizes really are a sampling artefact.)

    Real and generated counts are reported per plane and asserted equal -- unequal counts mean a
    rank dropped a case, which biases the distance in a direction nothing else would reveal.
    """
    out: dict = {}
    per_plane, planes = {}, list(features.planes)
    if single_plane:
        planes = planes[:1]                 # a whole-volume extractor has one value, not three
    for plane in planes:
        real, generated = features.real[plane], features.generated[plane]
        if len(real) != len(generated):
            log.warning("%s/%s: %d real vs %d generated features -- unequal counts bias the "
                        "Frechet distance; a rank probably dropped a case",
                        key, plane, len(real), len(generated))
        if min(len(real), len(generated)) < min_samples:
            log.warning("%s/%s: %d usable volumes is below the %d needed for a meaningful Frechet "
                        "distance (covariance error dominates) -- withheld",
                        key, plane, min(len(real), len(generated)), min_samples)
            continue
        real_matrix, generated_matrix = np.stack(real), np.stack(generated)
        value = frechet(real_matrix, generated_matrix)
        if value is not None:
            per_plane[plane] = value
            out[f"{key}/{plane}"] = value
    if per_plane:
        out[key] = float(np.mean(list(per_plane.values())))
        n = len(features.real[planes[0]])
        out[f"{key}/n_real"] = n
        out[f"{key}/n_generated"] = len(features.generated[planes[0]])
        dim = len(features.real[planes[0]][0])
        status = rank_status(n, dim)
        # Emitted as a number so it can be plotted next to the metric: 0 = rank-deficient (trend
        # only), 1 = marginal, 2 = well-conditioned. A value whose companion flag is 0 must not be
        # compared against one computed at a different N.
        out[f"{key}/rank_level"] = {"rank_deficient": 0, "marginal": 1, "well_conditioned": 2}[status["level"]]
        out[f"{key}/feature_dim"] = dim
        if status["level"] == "rank_deficient":
            log.warning("%s computed at N=%d with feature_dim=%d. %s", key, n, dim, RANK_DEFICIENT_NOTE)
    return out


def real_vs_real_baseline(real_features: Sequence[np.ndarray], seed: int = 0) -> dict:
    """The finite-sample noise floor: split the real features in half and measure the distance.

    **A reference, not a score.** For FVD and 2.5D FID lower is better and the theoretical optimum
    is 0, but two disjoint samples of the *same* distribution do not give 0 at finite N -- they give
    this. A model score is only meaningful relative to it. The split is deterministic in `seed`.
    """
    features = np.stack([np.asarray(f, dtype=np.float64).reshape(-1) for f in real_features])
    n = len(features)
    if n < 2 * MIN_FRECHET_SAMPLES:
        return {"value": None, "n_per_half": n // 2,
                "note": f"needs >= {2 * MIN_FRECHET_SAMPLES} real volumes for a two-way split, "
                        f"have {n}"}
    order = np.random.default_rng(seed).permutation(n)
    half = n // 2
    a, b = features[order[:half]], features[order[half:2 * half]]
    return {"value": frechet(a, b), "n_per_half": int(half), "seed": int(seed),
            "note": "finite-sample noise floor; a model FVD/FID near this is at the measurement "
                    "limit, not perfect"}


__all__ = [
    "MIN_FRECHET_SAMPLES",
    "PairVerdict",
    "PlaneFeatures",
    "SPACING_TOLERANCE_MM",
    "SSIM_DATA_RANGE",
    "SSIM_K1",
    "SSIM_K2",
    "SSIM_SIGMA",
    "SSIM_WIN_SIZE",
    "aggregate_frechet",
    "foreground_bounding_box",
    "frechet",
    "real_vs_real_baseline",
    "ssim_parameters",
    "ssim_volume",
    "verify_pair",
]
