"""Distribution-level metrics: 3D Frechet distance, a properly-defined 2.5D FID (all three
anatomical planes, volume-weighted, documented slice policy), Inception Score, and
manifold-based diversity/mode-collapse diagnostics (precision/recall/density/coverage).

Ported and corrected from the older evaluation implementation
(`~/NV-Generate-CTMR/evaluation/distribution_metrics.py`). Per
docs/design/archive/09_older_evaluation_implementation_audit.md §9/§15: the Frechet-distance math and
feature extractors are "reusable after adaptation" (float32 casting for this repo's bfloat16
Dataset tensors); the old 2.5D FID was axial-only and is reimplemented properly here (all three
planes, explicit volume-weighted aggregation); precision/recall/density/coverage is ported
unchanged (pure numpy, no dataset-specific assumptions).

Feature-based metrics may resize/resample inputs to whatever the encoder needs (a bilinear resize
to 299x299 for Inception, an adaptive pool inside MedicalNet's own forward pass) -- this is METRIC
PREPROCESSING, not proof that two volumes are spatially paired (see geometry.py's module
docstring). `real` and `generated` populations always receive IDENTICAL preprocessing; the encoder,
feature layer, slice policy, and resize policy are fixed and recorded in every cache/result via
`feature_configuration()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("distribution_metrics")

DISTRIBUTION_METRICS_VERSION = "1.0"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_INPUT_SIZE = 299
MEDICALNET_FEATURE_DIM = 512
INCEPTION_FEATURE_DIM = 2048

# 2.5D FID plane policy, explicit and tested (see test_evaluation_distribution_metrics.py):
# axis indices match this repository's Dataset axis order (geometry.DATASET_AXIS_ORDER =
# (X,Y,Z) = sagittal,coronal,axial) -- NOT re-derived here, imported from geometry.py so the two
# modules can never silently disagree about which axis is which.
PLANE_AXES = (("sagittal", 0), ("coronal", 1), ("axial", 2))
MAX_SLICES_PER_VOLUME_PER_PLANE = 32
MIN_FOREGROUND_FRACTION = 0.01


def feature_configuration(encoder_name: str, checkpoint_sha256: str | None, feature_dim: int) -> dict:
    """The explicit, versioned description of what produced a set of features -- required in
    every cache and result file per the "name and version the feature extractor" / "record the
    exact feature layer and dimension" requirements.
    """
    return {
        "encoder_name": encoder_name, "checkpoint_sha256": checkpoint_sha256, "feature_dim": feature_dim,
        "distribution_metrics_version": DISTRIBUTION_METRICS_VERSION,
        "plane_policy": {"planes": [p for p, _ in PLANE_AXES], "max_slices_per_volume_per_plane": MAX_SLICES_PER_VOLUME_PER_PLANE, "min_foreground_fraction": MIN_FOREGROUND_FRACTION, "aggregation": "volume_weighted_mean_pool_then_frechet"},
    }


# --------------------------------------------------------------------------- slice selection


def non_empty_slices(volume: np.ndarray, axis: int, min_fg_frac: float = MIN_FOREGROUND_FRACTION, max_slices: int = MAX_SLICES_PER_VOLUME_PER_PLANE) -> np.ndarray:
    """Indices of slices along `axis` with foreground content above `min_fg_frac`, evenly
    subsampled to `max_slices` if more are found -- deterministic (no randomness), so repeated
    calls on the same volume return the same indices. Falls back to the mid-slice if every slice
    is background-only (should not happen on real data; must not crash if it does).
    """
    n = volume.shape[axis]
    fg_frac = np.array([(np.take(volume, i, axis=axis) > 1e-3).mean() for i in range(n)])
    candidates = np.where(fg_frac >= min_fg_frac)[0]
    if len(candidates) == 0:
        return np.array([n // 2])
    if len(candidates) <= max_slices:
        return candidates
    pick_idx = np.linspace(0, len(candidates) - 1, max_slices).round().astype(int)
    return candidates[pick_idx]


def mid_slice(volume: np.ndarray, axis: int) -> np.ndarray:
    return np.take(volume, volume.shape[axis] // 2, axis=axis)


def to_inception_input(slices_2d: np.ndarray, device: str) -> torch.Tensor:
    """slices_2d: (N,H,W) float32 in this repo's model-input intensity space. Direct bilinear
    resize to 299x299, NO center crop (preserves the full anatomical field of view -- a center
    crop is safe for a centered photograph, not safe for a brain slice where a crop could cut
    into cortex at the periphery; same reasoning as the ported implementation's own choice).
    """
    x = torch.from_numpy(slices_2d).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device)


# --------------------------------------------------------------------------- feature extractors


class MedicalNetFeatureExtractor:
    """3D MedicalNet (Med3D) ResNet-10 -- a medically-motivated but NOT brain-MRI-specific 3D
    feature extractor (Med3D's "23-dataset" is a multi-anatomy CT/MR compilation). Category B
    (conditionally interpretable), same classification as the older implementation used --
    inherited, not re-derived, since nothing about this repository changes that classification.
    """

    # Class-level defaults so the attributes always exist even for an instance built without
    # __init__ (tests do this to inject a fake network rather than load 200 MB of weights).
    normalize = True
    crop_to_foreground = True

    def __init__(self, checkpoint_path: Path, device: str, normalize: bool = True,
                 crop_to_foreground: bool = True):
        from monai.networks.nets import resnet10

        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"MedicalNet checkpoint not found at {checkpoint_path}")
        self.model = resnet10(pretrained=str(checkpoint_path), spatial_dims=3, n_input_channels=1, feed_forward=False, shortcut_type="B", bias_downsample=False)
        self.model.eval().to(device)
        self.device = device
        self.normalize = normalize
        self.crop_to_foreground = crop_to_foreground

    @staticmethod
    def preprocess(volume: np.ndarray, normalize: bool = True, crop_to_foreground: bool = True) -> np.ndarray:
        """Put a volume into the intensity domain Med3D was trained on.

        **This is not optional.** Med3D's own pipeline z-scores each volume using its
        strictly-positive voxels (Tencent/MedicalNet `__itensity_normalize_one_volume__`). Feeding
        this repo's percentile-normalized [0, ~2] volumes raw leaves the features almost
        information-free: measured on the real cohort, FID(T1w real, T2w real) came out 12x
        *smaller* than FID(T1w real, its own VAE reconstruction) -- i.e. the features could not
        tell two grossly different contrasts apart. See `mrrate_r2v/eval/README.md`.

        `crop_to_foreground` additionally trims the bit-exact zero padding (~52% of a 256^3 cohort
        volume). Global average pooling over a region every subject shares dilutes between-subject
        variation; cropping recovers a further ~4x in feature spread.
        """
        v = np.asarray(volume, dtype=np.float32)
        if crop_to_foreground:
            m = np.abs(v) > 1e-6
            if m.any():
                sl = []
                for ax in range(3):
                    proj = m.any(axis=tuple(i for i in range(3) if i != ax))
                    idx = np.where(proj)[0]
                    sl.append(slice(int(idx[0]), int(idx[-1]) + 1))
                v = np.ascontiguousarray(v[tuple(sl)])
        if normalize:
            pos = v[v > 0]
            if pos.size:
                v = ((v - pos.mean()) / max(float(pos.std()), 1e-8)).astype(np.float32)
        return v

    @torch.no_grad()
    def extract(self, volume: np.ndarray) -> np.ndarray:
        v = self.preprocess(volume, self.normalize, self.crop_to_foreground)
        x = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return self.model(x).squeeze(0).float().cpu().numpy()


class InceptionFeatureExtractor:
    """torchvision Inception-v3 (ImageNet-1k) -- an explicit natural-image proxy, included only
    because it is the canonical backbone the conventional FID/Inception-Score definitions are
    specified against. Not a validated medical-imaging metric.
    """

    def __init__(self, device: str):
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.model.eval().to(device)
        self.device = device
        self._captured = None
        self.model.fc.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._captured = inputs[0].detach()

    @torch.no_grad()
    def extract_batch(self, slices_2d: np.ndarray, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        feats, probs = [], []
        for start in range(0, len(slices_2d), batch_size):
            x = to_inception_input(slices_2d[start : start + batch_size], self.device)
            logits = self.model(x)
            feats.append(self._captured.cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(feats, axis=0), np.concatenate(probs, axis=0)


# --------------------------------------------------------------------------- Frechet distance


def frechet_distance(features_real: np.ndarray, features_gen: np.ndarray, epsilon: float = 1e-6) -> float:
    """Frechet distance between two feature populations, treated as multivariate Gaussians:

        ||mu_r - mu_g||^2 + tr(S_r + S_g - 2 (S_r S_g)^(1/2))

    Computed in float64 throughout. If the covariance product is near-singular -- routine at
    this project's cohort sizes, where n can be smaller than the feature dimension -- both
    covariances get `epsilon` added to the diagonal and the square root is retried, rather than
    returning a NaN or a silently complex value. Any residual imaginary part is dropped only
    after checking it is numerically negligible.

    Implemented here rather than delegated to `monai.metrics.FIDMetric`: that call path passes a
    `disp=` argument to `scipy.linalg.sqrtm` which scipy removed in 1.17, so it raises on any
    current scipy. Keeping the ~15 lines in-package also means an evaluation run needs no monai.
    """
    from scipy import linalg

    r = np.asarray(features_real, dtype=np.float64)
    g = np.asarray(features_gen, dtype=np.float64)
    mu_r, mu_g = r.mean(axis=0), g.mean(axis=0)
    sigma_r, sigma_g = np.cov(r, rowvar=False), np.cov(g, rowvar=False)
    sigma_r = np.atleast_2d(sigma_r)
    sigma_g = np.atleast_2d(sigma_g)

    diff = mu_r - mu_g

    def _sqrtm_product(a, b):
        out = linalg.sqrtm(a @ b)
        return np.asarray(out[0] if isinstance(out, tuple) else out)

    covmean = _sqrtm_product(sigma_r, sigma_g)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_r.shape[0]) * epsilon
        covmean = _sqrtm_product(sigma_r + offset, sigma_g + offset)

    if np.iscomplexobj(covmean):
        imag_scale = np.max(np.abs(covmean.imag))
        if imag_scale > 1e-3:
            raise ValueError(
                f"Frechet distance: matrix square root has a non-negligible imaginary component "
                f"({imag_scale:.3e}); the covariance estimate is too ill-conditioned to trust"
            )
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * np.trace(covmean))


def frechet_distance_with_diagnostics(features_real: np.ndarray, features_gen: np.ndarray, n_bootstrap: int = 30, seed: int = 42) -> dict:
    """`frechet_distance` plus the small-sample diagnostics this project's cohort sizes need:
    an explicit rank-deficiency flag and a bootstrap CI, so a single FID number is never read as
    more precise than the sample supports.
    """
    n_real, n_gen = len(features_real), len(features_gen)
    feature_dim = features_real.shape[1]

    def _fd(r, g):
        return frechet_distance(r, g)

    if n_real < 2 or n_gen < 2:
        return {"fid": None, "n_real": n_real, "n_gen": n_gen, "feature_dim": feature_dim, "skipped": "fewer than 2 samples"}

    point_estimate = _fd(features_real, features_gen)
    rng = np.random.RandomState(seed)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx_r, idx_g = rng.randint(0, n_real, n_real), rng.randint(0, n_gen, n_gen)
        try:
            val = _fd(features_real[idx_r], features_gen[idx_g])
            if np.isfinite(val):
                boot_vals.append(val)
        except Exception as e:  # noqa: BLE001
            log.warning("bootstrap resample failed: %s", e)

    covariance_rank_deficient = n_real < feature_dim or n_gen < feature_dim
    return {
        "fid": point_estimate, "n_real": n_real, "n_gen": n_gen, "feature_dim": feature_dim,
        "covariance_rank_deficient": covariance_rank_deficient,
        "bootstrap_n_requested": n_bootstrap, "bootstrap_n_successful": len(boot_vals),
        "bootstrap_mean": float(np.mean(boot_vals)) if boot_vals else None,
        "bootstrap_ci95": [float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))] if boot_vals else None,
    }


def inception_score(probs: np.ndarray, n_splits: int = 10, seed: int = 42) -> dict:
    n = probs.shape[0]
    effective_splits = max(1, min(n_splits, n // 2)) if n >= 2 else 1
    eps = 1e-12
    split_scores = []
    for i in range(effective_splits):
        part = probs[i * n // effective_splits : (i + 1) * n // effective_splits]
        if len(part) == 0:
            continue
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        split_scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return {"mean": float(np.mean(split_scores)) if split_scores else None, "std": float(np.std(split_scores)) if split_scores else None, "n_samples": n, "n_splits_effective": effective_splits, "splits_reduced": effective_splits < n_splits}


# --------------------------------------------------------------------------- diversity / mode collapse (unchanged from the ported version -- pure numpy, no dataset assumptions)


def _pairwise_sqeuclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    return np.clip(a2 + b2 - 2.0 * (a @ b.T), 0.0, None)


def _kth_nn_radius(features: np.ndarray, k: int) -> np.ndarray:
    d2 = _pairwise_sqeuclidean(features, features)
    np.fill_diagonal(d2, np.inf)
    k = min(k, d2.shape[0] - 1)
    return np.partition(d2, kth=k - 1, axis=1)[:, k - 1]


def precision_recall_density_coverage(features_real: np.ndarray, features_gen: np.ndarray, k: int = 5, chunk_size: int = 2048) -> dict:
    """Manifold-based diversity/mode-collapse diagnostics (Kynkaanniemi et al. 2019 precision/
    recall + Naeem et al. 2020 density/coverage), independent of and complementary to Frechet
    distance -- FID can look good under first/second-moment agreement even when a generator has
    collapsed onto a subset of the real manifold; this catches that.
    """
    n_real, n_gen = len(features_real), len(features_gen)
    if n_real < 2 or n_gen < 2:
        return {"n_real": n_real, "n_gen": n_gen, "skipped": "fewer than 2 samples in real and/or generated set"}

    k_eff = max(1, min(k, n_real - 1, n_gen - 1))
    real_radii = _kth_nn_radius(features_real.astype(np.float32), k_eff)
    gen_radii = _kth_nn_radius(features_gen.astype(np.float32), k_eff)

    gen_covered = np.zeros(n_gen, dtype=bool)
    real_covered_by_gen = np.zeros(n_real, dtype=bool)
    real_covered_by_real = np.zeros(n_real, dtype=bool)
    density_hits = np.zeros(n_gen, dtype=np.int64)

    for start in range(0, n_gen, chunk_size):
        chunk = features_gen[start : start + chunk_size].astype(np.float32)
        d2 = _pairwise_sqeuclidean(features_real.astype(np.float32), chunk)
        inside_real_ball = d2 <= real_radii[:, None]
        gen_covered[start : start + chunk_size] = inside_real_ball.any(axis=0)
        density_hits[start : start + chunk_size] = inside_real_ball.sum(axis=0)
        inside_gen_ball = d2 <= gen_radii[None, start : start + chunk_size]
        real_covered_by_gen |= inside_gen_ball.any(axis=1)
        real_covered_by_real |= inside_real_ball.any(axis=1)

    return {
        "n_real": n_real, "n_gen": n_gen, "k_requested": k, "k_effective": k_eff,
        "precision": float(gen_covered.mean()), "recall": float(real_covered_by_gen.mean()),
        "density": float(density_hits.mean() / k_eff), "coverage": float(real_covered_by_real.mean()),
        "interpretation": "precision=realism, recall/coverage=diversity. Low recall+coverage with high precision = mode collapse; low precision AND recall = a realism/domain gap, not collapse.",
        "unstable_small_sample": n_real < 50 or n_gen < 50,
    }


# --------------------------------------------------------------------------- orchestration


@dataclass
class CaseFeatures:
    case_id: str
    sequence: str
    bucket: str = ""      # "<modality>__<plane>" -- the primary grouping key (see compute_distribution_metrics)
    medicalnet_real: np.ndarray | None = None
    medicalnet_gen: np.ndarray | None = None
    inception_2p5d_real: dict | None = None  # {"sagittal": vec, "coronal": vec, "axial": vec} -- volume-weighted mean-pooled
    inception_2p5d_gen: dict | None = None
    inception_mid_probs_real: np.ndarray | None = None
    inception_mid_probs_gen: np.ndarray | None = None
    # FVD-family sequence features, {plane: vec}. Same shape as inception_2p5d_*, so the two
    # aggregate through the same code. Populated only when a sequence extractor is supplied.
    fvd_real: dict | None = None
    fvd_gen: dict | None = None
    # Mid-axial slices, kept from the same volume load the features came from, so the intra-set
    # diversity probe costs no extra I/O. Not written to the feature cache (they are pixels, not
    # features, and would inflate it ~100x).
    mid_slice_real: np.ndarray | None = None
    mid_slice_gen: np.ndarray | None = None


_PLANE_NAMES = tuple(name for name, _ in PLANE_AXES)


def case_features_to_arrays(features: list) -> dict:
    """Flattens a list[CaseFeatures] into one dict of stacked numpy arrays -- what
    `feature_cache.FeatureCache.save` writes to disk. `case_id`/`sequence` become parallel
    string arrays; any feature field that's `None` for every case is omitted entirely (so a
    partial extraction, e.g. MedicalNet-only, produces a cache without dangling all-None arrays).
    """
    out = {"case_id": np.array([f.case_id for f in features]), "sequence": np.array([f.sequence for f in features])}
    if any(f.medicalnet_real is not None for f in features):
        out["medicalnet_real"] = np.stack([f.medicalnet_real for f in features])
        out["medicalnet_gen"] = np.stack([f.medicalnet_gen for f in features])
    if any(f.inception_2p5d_real is not None for f in features):
        for name in _PLANE_NAMES:
            out[f"inception_2p5d_real_{name}"] = np.stack([f.inception_2p5d_real[name] for f in features])
            out[f"inception_2p5d_gen_{name}"] = np.stack([f.inception_2p5d_gen[name] for f in features])
    if any(f.inception_mid_probs_real is not None for f in features):
        out["inception_mid_probs_real"] = np.stack([f.inception_mid_probs_real for f in features])
        out["inception_mid_probs_gen"] = np.stack([f.inception_mid_probs_gen for f in features])
    return out


def arrays_to_case_features(arrays: dict) -> list:
    """Inverse of `case_features_to_arrays` -- reconstructs `list[CaseFeatures]` from a loaded
    cache dict (e.g. `FeatureCache.load()`'s return value).
    """
    n = len(arrays["case_id"])
    out = []
    for i in range(n):
        cf = CaseFeatures(case_id=str(arrays["case_id"][i]), sequence=str(arrays["sequence"][i]))
        if "medicalnet_real" in arrays:
            cf.medicalnet_real, cf.medicalnet_gen = arrays["medicalnet_real"][i], arrays["medicalnet_gen"][i]
        if f"inception_2p5d_real_{_PLANE_NAMES[0]}" in arrays:
            cf.inception_2p5d_real = {name: arrays[f"inception_2p5d_real_{name}"][i] for name in _PLANE_NAMES}
            cf.inception_2p5d_gen = {name: arrays[f"inception_2p5d_gen_{name}"][i] for name in _PLANE_NAMES}
        if "inception_mid_probs_real" in arrays:
            cf.inception_mid_probs_real, cf.inception_mid_probs_gen = arrays["inception_mid_probs_real"][i], arrays["inception_mid_probs_gen"][i]
        out.append(cf)
    return out


def extract_2p5d_inception_features(volume: np.ndarray, extractor: "InceptionFeatureExtractor") -> dict:
    """Per plane: mean-pool Inception features over the plane's own non-empty slices -> ONE
    feature vector per volume per plane. This is what makes the resulting FID volume-weighted
    (every volume contributes exactly one vector per plane to the population, however many
    slices it had) rather than slice-weighted (which would silently overweight volumes with more
    non-empty slices).
    """
    out = {}
    for name, axis in PLANE_AXES:
        idx = non_empty_slices(volume, axis=axis)
        slices = np.stack([np.take(volume, i, axis=axis) for i in idx])
        feats, _ = extractor.extract_batch(slices)
        out[name] = feats.mean(axis=0)
    return out


def compute_per_plane_frechet(all_features: list, real_attr: str, gen_attr: str,
                              n_bootstrap: int = 30, seed: int = 42) -> dict:
    """Per-plane Frechet distance plus an unweighted mean across the 3 planes.

    Unweighted, not sample-count-weighted, because every plane is a full, equally-valid view of
    every volume (not a variable-count sample of anything). **This per-plane-then-average shape is
    what the VLM3D challenge itself reports** -- its `ranking_config` exposes `FID_2p5D_XY`,
    `FID_2p5D_XZ`, `FID_2p5D_YZ` and the headline `FID_2p5D_Avg` -- so it is matched here
    deliberately rather than invented.

    Shared by the 2.5D Inception FID and the FVD-family sequence features: both are
    `{plane: vector}` per volume, so both aggregate identically.
    """
    out = {}
    for name, _axis in PLANE_AXES:
        real = np.stack([getattr(f, real_attr)[name] for f in all_features if getattr(f, real_attr)])
        gen = np.stack([getattr(f, gen_attr)[name] for f in all_features if getattr(f, gen_attr)])
        out[name] = frechet_distance_with_diagnostics(real, gen, n_bootstrap, seed)
    finite = [out[name]["fid"] for name, _ in PLANE_AXES if out[name].get("fid") is not None]
    out["combined_unweighted_mean"] = float(np.mean(finite)) if finite else None
    out["combination_method"] = "unweighted mean across sagittal/coronal/axial plane-level distance"
    return out


def compute_2p5d_fid(all_features: list, n_bootstrap: int = 30, seed: int = 42) -> dict:
    """The 2.5D Inception FID: per-plane, then the unweighted mean. See `compute_per_plane_frechet`."""
    return compute_per_plane_frechet(all_features, "inception_2p5d_real", "inception_2p5d_gen",
                                     n_bootstrap, seed)


def compute_distribution_metrics(all_features: list, sequences: list, min_subgroup_n: int = 10, n_bootstrap: int = 30, seed: int = 42, k_diversity: int = 5, buckets=None) -> dict:
    """Per-bucket + per-sequence + overall aggregation, grouping the same way for every metric
    family so results line up in one report.

    `buckets` is the primary grouping: a (modality, plane) bucket is the only level at which every
    volume shares a geometry, so it is the only level at which a Frechet distance compares
    like with like. Per-sequence and overall groups mix geometries and are kept as a coarse
    summary only.
    """
    groups = {"overall": all_features}
    for b in (buckets or []):
        groups[b] = [f for f in all_features if getattr(f, "bucket", None) == b]
    for seq in sequences:
        groups[seq] = [f for f in all_features if f.sequence == seq]

    out = {}
    for group_name, feats in groups.items():
        if len(feats) < 2:
            out[group_name] = {"n": len(feats), "skipped": "fewer than 2 samples"}
            continue
        unstable = len(feats) < min_subgroup_n
        entry = {"n": len(feats), "unstable_small_sample": unstable}

        mn_real = [f.medicalnet_real for f in feats if f.medicalnet_real is not None]
        mn_gen = [f.medicalnet_gen for f in feats if f.medicalnet_gen is not None]
        if mn_real and mn_gen:
            entry["medicalnet_fid_3d"] = frechet_distance_with_diagnostics(np.stack(mn_real), np.stack(mn_gen), n_bootstrap, seed)
            entry["diversity_precision_recall_density_coverage"] = precision_recall_density_coverage(np.stack(mn_real), np.stack(mn_gen), k=k_diversity)

        if any(f.inception_2p5d_real for f in feats) and any(f.inception_2p5d_gen for f in feats):
            entry["inception_2p5d_fid"] = compute_2p5d_fid(feats, n_bootstrap, seed)

        # FVD, on the same per-plane-then-average shape. Offline FVD did not exist before
        # 2026-08-10: it was computed only in the training-time validation loop, which meant the
        # challenge's own headline metric family had no offline counterpart.
        if any(f.fvd_real for f in feats) and any(f.fvd_gen for f in feats):
            entry["fvd"] = compute_per_plane_frechet(feats, "fvd_real", "fvd_gen",
                                                     n_bootstrap, seed)

        sl_real = [f.mid_slice_real for f in feats if f.mid_slice_real is not None]
        sl_gen = [f.mid_slice_gen for f in feats if f.mid_slice_gen is not None]
        if sl_real and sl_gen:
            from .paired import intra_set_ms_ssim_slices
            entry["intra_set_ms_ssim_real"] = intra_set_ms_ssim_slices(sl_real, seed=seed)
            entry["intra_set_ms_ssim_produced"] = intra_set_ms_ssim_slices(sl_gen, seed=seed)
            entry["intra_set_ms_ssim_note"] = ("mean pairwise SSIM within each population. Compare "
                                               "produced against real: clearly higher = less variety "
                                               "than the data, i.e. mode collapse.")

        probs_real = [f.inception_mid_probs_real for f in feats if f.inception_mid_probs_real is not None]
        probs_gen = [f.inception_mid_probs_gen for f in feats if f.inception_mid_probs_gen is not None]
        if probs_gen:
            entry["inception_score_generated_or_reconstructed"] = inception_score(np.stack(probs_gen), seed=seed)
        if probs_real:
            entry["inception_score_real_reference"] = inception_score(np.stack(probs_real), seed=seed)

        out[group_name] = entry
    return out
