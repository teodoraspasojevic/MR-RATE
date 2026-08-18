"""FID_2p5D -- streaming 2.5D Frechet Inception Distance. Ported from the official `fid_2p5d.py`.

Method taken from the VLM3D CT track's own code (`compute_fid_2-5d_ct.py`): slice-based feature
extraction with squeezenet1_1 over three orthogonal planes (XY/XZ/YZ), then Frechet distance per
plane. Streaming: `FIDAccumulator` takes one (real, fake) volume pair at a time, extracts that
pair's slice features immediately, and never holds a whole volume longer than it takes to slice it --
only the small per-slice feature vectors accumulate.

`raw_features`/`finalize_pooled` below are additions, not part of the official file: they let
multiple worker ranks each accumulate their own shard's features and pool them into one global
Frechet distance before `finalize()`'s math runs -- the same computation a single process would do,
just fed a feature set assembled across ranks instead of within one.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from scipy import linalg

_FEATURE_DIM = 512
_INPUT_SIZE = 224

#: Planes: axis=2 (XY, Z fixed), axis=1 (XZ, Y fixed), axis=0 (YZ, X fixed).
_AXIS_FOR_PLANE = {"XY": 2, "XZ": 1, "YZ": 0}


class SqueezeNetFeatureExtractor(nn.Module):
    """squeezenet1_1 with its classifier dropped, returning the global-average-pooled 512-d
    feature vector."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        weights = tv_models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        base = tv_models.squeezenet1_1(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.eval()
        self.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


def _normalize_slice(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(np.float32)
    lo, hi = np.percentile(sl, 0.5), np.percentile(sl, 99.5)
    if hi - lo < 1e-6:
        return np.zeros_like(sl)
    return np.clip((sl - lo) / (hi - lo), 0.0, 1.0)


def _slice_to_tensor(sl: np.ndarray) -> torch.Tensor:
    sl = _normalize_slice(sl)
    t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False
    )
    t = t.repeat(1, 3, 1, 1)
    return t.squeeze(0)


def _iter_slices(volume: np.ndarray, axis: int, stride: int = 4):
    n = volume.shape[axis]
    for idx in range(0, n, stride):
        yield np.take(volume, idx, axis=axis)


class RunningMoments:
    """Accumulates slice features across many volumes without ever holding all of them at once,
    then computes mean/covariance in a single pass at the end.

    Feature vectors are kept (float32, 512-d, a few hundred slices per volume -- about 1000x
    smaller than the volume itself), never the volumes, so this stays cheap even over a large
    dataset.
    """

    def __init__(self):
        self._chunks: list[np.ndarray] = []

    def add(self, feats: np.ndarray) -> None:
        if feats.shape[0] > 0:
            self._chunks.append(feats)

    def array(self) -> np.ndarray:
        """The concatenated feature matrix accumulated so far, `(N, 512)`."""
        if not self._chunks:
            return np.zeros((0, _FEATURE_DIM), dtype=np.float32)
        return np.concatenate(self._chunks, axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, int]:
        all_feats = self.array()
        if all_feats.shape[0] == 0:
            return np.zeros(_FEATURE_DIM), np.eye(_FEATURE_DIM), 0
        mu = all_feats.mean(axis=0)
        sigma = np.cov(all_feats, rowvar=False)
        return mu, sigma, all_feats.shape[0]


class FIDAccumulator:
    """Takes volume pairs ONE AT A TIME (streaming), extracting and accumulating slice features
    immediately. The volumes themselves are never held in bulk -- the caller may release each pair
    right after `add_pair` returns."""

    def __init__(self, device: str = "auto", stride: int = 4, batch_size: int = 32):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = SqueezeNetFeatureExtractor(device=device)
        self.device = device
        self.stride = stride
        self.batch_size = batch_size
        self.real_moments = {plane: RunningMoments() for plane in _AXIS_FOR_PLANE}
        self.fake_moments = {plane: RunningMoments() for plane in _AXIS_FOR_PLANE}

    @torch.no_grad()
    def _extract_volume_features(self, volume: np.ndarray, axis: int) -> np.ndarray:
        feats_list = []
        batch = []
        for sl in _iter_slices(volume, axis=axis, stride=self.stride):
            batch.append(_slice_to_tensor(sl))
            if len(batch) == self.batch_size:
                x = torch.stack(batch).to(self.device)
                feats_list.append(self.extractor(x).cpu().numpy())
                batch = []
        if batch:
            x = torch.stack(batch).to(self.device)
            feats_list.append(self.extractor(x).cpu().numpy())
        if not feats_list:
            return np.zeros((0, _FEATURE_DIM), dtype=np.float32)
        return np.concatenate(feats_list, axis=0)

    def add_pair(self, real_vol: np.ndarray, fake_vol: np.ndarray) -> None:
        """Process one (real, fake) volume pair. `real_vol`/`fake_vol` may be released by the
        caller as soon as this returns -- nothing here keeps a reference to either."""
        for plane, axis in _AXIS_FOR_PLANE.items():
            self.real_moments[plane].add(self._extract_volume_features(real_vol, axis))
            self.fake_moments[plane].add(self._extract_volume_features(fake_vol, axis))

    def finalize(self) -> dict:
        results = {}
        for plane in _AXIS_FOR_PLANE:
            mu_r, sigma_r, n_r = self.real_moments[plane].finalize()
            mu_f, sigma_f, n_f = self.fake_moments[plane].finalize()
            if n_r < 2 or n_f < 2:
                results[f"FID_2p5D_{plane}"] = float("nan")
                continue
            results[f"FID_2p5D_{plane}"] = frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

        valid_vals = [v for v in results.values() if not np.isnan(v)]
        results["FID_2p5D_Avg"] = float(np.mean(valid_vals)) if valid_vals else float("nan")
        return results

    def raw_features(self) -> dict:
        """`{plane: {"real": array, "fake": array}}` -- for pooling this rank's accumulated
        features with other ranks' before a single, global `finalize_pooled` call. Not part of the
        official file: single-process runs never need this, only our DDP-sharded evaluation does."""
        return {plane: {"real": self.real_moments[plane].array(),
                        "fake": self.fake_moments[plane].array()}
                for plane in _AXIS_FOR_PLANE}


def finalize_pooled(raw_features_per_rank: list) -> dict:
    """The same computation `FIDAccumulator.finalize()` does, fed a feature set assembled from
    multiple ranks' `raw_features()` instead of one process's own accumulation. Concatenating
    feature chunks before taking mean/covariance is associative, so this is bit-identical to a
    single process having seen every pair itself."""
    results = {}
    for plane in _AXIS_FOR_PLANE:
        real = np.concatenate([r[plane]["real"] for r in raw_features_per_rank if r[plane]["real"].shape[0]]
                              or [np.zeros((0, _FEATURE_DIM), dtype=np.float32)], axis=0)
        fake = np.concatenate([r[plane]["fake"] for r in raw_features_per_rank if r[plane]["fake"].shape[0]]
                              or [np.zeros((0, _FEATURE_DIM), dtype=np.float32)], axis=0)
        if real.shape[0] < 2 or fake.shape[0] < 2:
            results[f"FID_2p5D_{plane}"] = float("nan")
            continue
        mu_r, sigma_r = real.mean(axis=0), np.cov(real, rowvar=False)
        mu_f, sigma_f = fake.mean(axis=0), np.cov(fake, rowvar=False)
        results[f"FID_2p5D_{plane}"] = frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

    valid_vals = [v for v in results.values() if not np.isnan(v)]
    results["FID_2p5D_Avg"] = float(np.mean(valid_vals)) if valid_vals else float("nan")
    return results


def _matrix_sqrt(a: np.ndarray) -> np.ndarray:
    """`scipy.linalg.sqrtm`, without the `disp=` kwarg the official code passes -- scipy >= 1.17
    removed it (this project's `requirements.txt` already flags the same incompatibility for the
    reason `frechet_distance` used to avoid `monai.metrics.FIDMetric`). Pre-1.17 `sqrtm` returns
    `(result, info)` when `disp=False` is given and just `result` otherwise; without the kwarg it
    always returns just `result`, on every scipy version -- so this is a version-compatibility fix,
    not a change to the Frechet-distance math itself.
    """
    result = linalg.sqrtm(a)
    return result[0] if isinstance(result, tuple) else result


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean = _matrix_sqrt(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _matrix_sqrt((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)
