"""Volume -> sequence features, for the FVD-style Frechet distance.

**This is an MRI-volume adaptation of FVD, not standard FVD. It is named so everywhere.**

What standard FVD is, from the reference implementation
(`google-research/frechet_video_distance/frechet_video_distance.py`, VERIFIED by reading it):

- extractor: I3D RGB from TF-Hub `deepmind/i3d-kinetics-400/1`, pretrained on Kinetics-400;
- feature: the tensor `RGB/inception_i3d/Mean:0` -- the spatiotemporal global-average pool,
  **1024-d**. (The source comment says "logits layer"; the tensor it actually takes is the
  pre-logits pooled activation. The code is the definition, not the comment.)
- input: `[N, T, 224, 224, 3]`, bilinear resize, scaled to **[-1, 1]** by `2x/255 - 1`;
- batch size hardcoded to 16, so N must be divisible by 16;
- Frechet distance via `tfgan.eval.frechet_classifier_distance_from_activations`;
- anchor worth knowing: the reference example reports FVD ~= **131 for empty frames**, not 0.

Why this module is not that, and what it is instead:

1. **No TensorFlow in either container and no I3D checkpoint anywhere** -- I3D is not in
   torchvision (only r3d_18 / mc3_18 / r2plus1d_18 / s3d / mvit / swin3d). So the primary
   extractor here is torchvision's **r3d_18, KINETICS400_V1** (official weights, 33.4M params,
   Kinetics-400 acc@1 63.2, feature dim **512** from the pre-`fc` global pool). Same *lineage* as
   FVD -- a Kinetics-400-pretrained 3D CNN -- and a different network. Hence
   `fvd_r3d18_kinetics400`, never "FVD".
2. **A brain's slice axis is not a temporal axis.** r3d_18's 3D convolutions were trained on
   motion; here they see through-plane anatomical structure. That is a domain transfer, not a
   principled temporal model. It is also exactly what the VLM3D CT track does: GenerateCT
   (arXiv:2305.16037) computes `FVD_I3D` on CT volumes-as-videos *and* `FVD_CT-Net` using CT-Net,
   which is a 3D classifier and not a video model at all. So a non-temporal domain extractor is
   established practice for this metric family, and `MedicalNetSequenceExtractor` is this
   pipeline's `FVD_CT-Net` analogue.
3. **No anatomical axis is privileged.** Rather than declare one axis "time", every volume is
   encoded once per plane (sequence axis = that plane's through-plane axis) and the three
   plane-level Frechet distances are averaged unweighted into one headline number -- the same
   convention `distribution.compute_2p5d_fid` already uses, so the two metrics aggregate alike.
   Per-plane values are diagnostics.

The slice policy, fixed and deterministic:

    T = SEQUENCE_LENGTH (16, r3d_18's own training clip length)
    frames = evenly spaced indices over the volume's *non-empty* extent along that axis
    ordering = increasing anatomical index (never shuffled, never reversed)

Fixed T is what stops a 256-slice volume from contributing more evidence than a 128-slice one:
every volume yields exactly one feature vector per plane, from exactly T frames.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

log = logging.getLogger("mrrate_r2v.eval.video_features")

#: Frames per sequence. 16 is r3d_18's training clip length; it is also short enough that the
#: temporal strides do not collapse the axis.
SEQUENCE_LENGTH = 16

#: Spatial size each frame is resized to. r3d_18's declared `crop_size`; a direct bilinear resize
#: is used rather than resize-then-crop, because a crop can remove brain (the same reason
#: `distribution.to_inception_input` resizes rather than centre-crops).
FRAME_SIZE = 112

#: torchvision's own declared Kinetics-400 normalisation for r3d_18
#: (`R3D_18_Weights.KINETICS400_V1.transforms()`), read from the weights rather than copied from a
#: paper, so it cannot drift from the checkpoint.
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)

R3D18_FEATURE_DIM = 512

#: The intensity space every validation metric operates in. Volumes reach here as the model-input
#: percentile-normalised space (0-99.5th percentile mapped to [0, 1], tail above 1 not clipped) --
#: NOT `sampling.postprocess_mr`'s int16 [0, 1000]. Mixing the two is the trap this constant
#: exists to name: the ground truth comes from the Dataset in [0, ~2] and a postprocessed
#: generation is 1000x larger, which would make every metric meaningless while still returning a
#: number. `validation.py` therefore uses the decoder's float output directly.
METRIC_INTENSITY_SPACE = "model_input_percentile_0_995_to_0_1"

#: (name, axis) for this repo's (X, Y, Z) = (sagittal, coronal, axial) Dataset order. Imported
#: shape from `distribution.PLANE_AXES` rather than re-declared, so the two cannot disagree.
try:
    from .distribution import MIN_FOREGROUND_FRACTION, PLANE_AXES
except Exception:  # noqa: BLE001 -- keeps this module importable without the heavy eval deps
    PLANE_AXES = (("sagittal", 0), ("coronal", 1), ("axial", 2))
    MIN_FOREGROUND_FRACTION = 0.01


# --------------------------------------------------------------------------- slice policy


def sequence_indices(volume: np.ndarray, axis: int, n_frames: int = SEQUENCE_LENGTH,
                     min_fg_frac: float = MIN_FOREGROUND_FRACTION) -> np.ndarray:
    """`n_frames` evenly spaced slice indices along `axis`, in increasing anatomical order.

    Spans the volume's **non-empty extent** (first to last slice with foreground above
    `min_fg_frac`) rather than the whole array, because ~52% of a padded 256^3 cohort volume is
    exactly zero and a sequence dominated by black frames carries no anatomy. Falls back to the
    full axis when nothing passes the threshold, so a degenerate generation still produces a
    feature vector instead of an exception -- a collapsed generation must score badly, not crash
    the validation pass.

    Deterministic: no randomness, and the same volume always gives the same indices.
    """
    size = int(volume.shape[axis])
    if size == 0:
        raise ValueError(f"volume has zero extent along axis {axis}: {volume.shape}")
    moved = np.moveaxis(np.asarray(volume), axis, 0)
    per_slice = np.abs(moved).reshape(size, -1)
    fraction = (per_slice > 1e-6).mean(axis=1)
    present = np.flatnonzero(fraction > min_fg_frac)
    lo, hi = (int(present[0]), int(present[-1])) if present.size else (0, size - 1)
    if hi <= lo:
        lo, hi = 0, size - 1
    # `linspace` over an inclusive range, rounded: for n_frames > extent this repeats slices rather
    # than padding with black, which keeps the sequence anatomically continuous.
    return np.linspace(lo, hi, int(n_frames)).round().astype(int)


def volume_to_frames(volume: np.ndarray, axis: int, n_frames: int = SEQUENCE_LENGTH) -> np.ndarray:
    """`(T, H, W)` float32 frames along `axis`, display-agnostic (no flips -- a feature extractor
    does not care which way is up, only that real and generated are treated identically)."""
    indices = sequence_indices(volume, axis, n_frames)
    return np.stack([np.take(volume, int(i), axis=axis) for i in indices]).astype(np.float32)


def frames_to_clip(frames: np.ndarray, device, size: int = FRAME_SIZE,
                   mean: Sequence[float] = KINETICS_MEAN,
                   std: Sequence[float] = KINETICS_STD) -> torch.Tensor:
    """`(T, H, W)` -> `(1, 3, T, size, size)`, normalised the way the checkpoint expects.

    Three steps, in this order and identically for real and generated:

    1. **clip to [0, 1]**. The percentile normaliser leaves a tail above 1 (0-99.5th pct -> [0, 1]),
       and a Kinetics-normalised input is only meaningful on [0, 1]. Clipping rather than
       rescaling per volume is deliberate: a per-volume rescale would hide the intensity errors
       these metrics exist to catch.
    2. **bilinear resize** each frame to `size x size`. Aspect ratio is not preserved; it is not
       preserved for real volumes either, and per-bucket shapes are fixed, so within a plane every
       volume is distorted identically.
    3. **replicate to 3 channels, then normalise** with the checkpoint's own Kinetics mean/std.
       Replication is the standard grayscale->RGB route for an RGB-pretrained network (the same one
       `distribution.to_inception_input` uses).
    """
    x = torch.from_numpy(np.ascontiguousarray(frames)).float().clamp_(0.0, 1.0)
    x = x.unsqueeze(1)                                                    # (T, 1, H, W)
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)                                              # (T, 3, s, s)
    mean_t = torch.tensor(list(mean)).view(1, 3, 1, 1)
    std_t = torch.tensor(list(std)).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t
    return x.permute(1, 0, 2, 3).unsqueeze(0).to(device)                  # (1, 3, T, s, s)


# --------------------------------------------------------------------------- extractors


class R3D18SequenceExtractor:
    """torchvision `r3d_18` KINETICS400_V1, `fc` replaced by identity -> 512-d pooled features.

    The primary FVD-adaptation extractor. Weights are the official torchvision release, resolved
    offline from `TORCH_HOME` once staged, so no run depends on a third-party mirror.
    """

    name = "r3d18_kinetics400"
    feature_dim = R3D18_FEATURE_DIM

    def __init__(self, device: str = "cpu", torch_home: Optional[str] = None,
                 weights_path: Optional[str] = None) -> None:
        import os

        if torch_home:
            os.environ.setdefault("TORCH_HOME", str(torch_home))
        from torchvision.models.video import R3D_18_Weights, r3d_18

        weights = R3D_18_Weights.KINETICS400_V1
        if weights_path:
            model = r3d_18(weights=None)
            state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        else:
            # Downloads once into TORCH_HOME, then resolves from cache. `--offline` runs must have
            # been staged beforehand; the error from torchvision is explicit if not.
            model = r3d_18(weights=weights)
        transforms = weights.transforms()
        # Read from the checkpoint's own declared transform, never hardcoded here, so a torchvision
        # update that changes the preprocessing cannot silently desynchronise the features.
        self.mean = tuple(float(v) for v in getattr(transforms, "mean", KINETICS_MEAN))
        self.std = tuple(float(v) for v in getattr(transforms, "std", KINETICS_STD))
        crop = getattr(transforms, "crop_size", [FRAME_SIZE, FRAME_SIZE])
        self.frame_size = int(crop[0])
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.weights_url = weights.url

    def configuration(self) -> dict:
        return {
            "extractor": self.name,
            "weights_url": self.weights_url,
            "feature_dim": self.feature_dim,
            "feature_layer": "global average pool before fc (fc replaced by Identity)",
            "sequence_length": SEQUENCE_LENGTH,
            "frame_size": self.frame_size,
            "normalisation": {"mean": self.mean, "std": self.std},
            "intensity_space": METRIC_INTENSITY_SPACE,
            "planes": [name for name, _ in PLANE_AXES],
            "aggregation": "unweighted mean of per-plane Frechet distances",
            "standard_fvd": False,
            "adaptation_note": "MRI-volume adaptation of FVD: Kinetics-400 3D CNN (r3d_18), not "
                               "I3D, and the sequence axis is anatomical, not temporal.",
        }

    @torch.no_grad()
    def extract_plane(self, volume: np.ndarray, axis: int) -> np.ndarray:
        frames = volume_to_frames(volume, axis, SEQUENCE_LENGTH)
        clip = frames_to_clip(frames, self.device, self.frame_size, self.mean, self.std)
        return self.model(clip).squeeze(0).float().cpu().numpy()

    def extract(self, volume: np.ndarray) -> dict:
        """One feature vector per plane. `{plane_name: (512,)}`."""
        return {name: self.extract_plane(volume, axis) for name, axis in PLANE_AXES}


class MedicalNetSequenceExtractor:
    """The staged 3D MedicalNet ResNet-10, wrapped in the same interface -- this pipeline's
    analogue of GenerateCT's `FVD_CT-Net`: a domain 3D classifier rather than a video network.

    It consumes the **whole volume**, not a plane sequence, so `extract` returns the same 512-d
    vector under every plane key. That is not a bug and it is not hidden: it means the "per-plane"
    breakdown is meaningless for this extractor and only its single value is reported. Kept behind
    the same interface so the two can be swapped without touching the runner.
    """

    name = "medicalnet_resnet10"
    feature_dim = 512

    def __init__(self, checkpoint_path, device: str = "cpu") -> None:
        from .distribution import MedicalNetFeatureExtractor

        self.inner = MedicalNetFeatureExtractor(Path(checkpoint_path), device=device)
        self.checkpoint_path = str(checkpoint_path)

    def configuration(self) -> dict:
        return {
            "extractor": self.name,
            "checkpoint": self.checkpoint_path,
            "feature_dim": self.feature_dim,
            "feature_layer": "global average pool (monai resnet10, feed_forward=False)",
            "preprocessing": "Med3D intensity z-score over positive voxels + foreground crop "
                             "(MedicalNetFeatureExtractor.preprocess)",
            "intensity_space": METRIC_INTENSITY_SPACE,
            "planes": "n/a -- whole-volume 3D extractor, one value not three",
            "standard_fvd": False,
            "adaptation_note": "GenerateCT's FVD_CT-Net analogue: a domain 3D classifier, not a "
                               "video model. No sequence axis is involved.",
        }

    def extract(self, volume: np.ndarray) -> dict:
        feature = np.asarray(self.inner.extract(volume), dtype=np.float64).reshape(-1)
        return {name: feature for name, _ in PLANE_AXES}


def build_sequence_extractor(name: str, device: str = "cpu", **kwargs):
    """The one place a sequence extractor is constructed."""
    if name in ("r3d18", "r3d18_kinetics400"):
        return R3D18SequenceExtractor(device=device, torch_home=kwargs.get("torch_home"),
                                      weights_path=kwargs.get("weights_path"))
    if name in ("medicalnet", "medicalnet_resnet10"):
        checkpoint = kwargs.get("checkpoint_path")
        if not checkpoint:
            raise ValueError("medicalnet sequence extractor needs checkpoint_path=")
        return MedicalNetSequenceExtractor(checkpoint, device=device)
    raise ValueError(f"unknown sequence extractor '{name}'. Choose from: r3d18, medicalnet")


__all__ = [
    "FRAME_SIZE",
    "KINETICS_MEAN",
    "KINETICS_STD",
    "METRIC_INTENSITY_SPACE",
    "MedicalNetSequenceExtractor",
    "R3D18_FEATURE_DIM",
    "R3D18SequenceExtractor",
    "SEQUENCE_LENGTH",
    "build_sequence_extractor",
    "frames_to_clip",
    "sequence_indices",
    "volume_to_frames",
]
