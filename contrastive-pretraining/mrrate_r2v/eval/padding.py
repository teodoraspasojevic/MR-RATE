"""End-only zero-padding to a divisor, and its exact inverse.

NVIDIA's VAE needs every spatial axis divisible by a model-derived divisor. `pad_to_divisible`
computes the smallest such padding; `crop_using_record` undoes exactly that padding afterward, so
`cli/evaluate.py:reconstruct` and `training.py`'s latent encoding always return on the input's own
grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropPadRecord:
    """Exact, deterministic provenance for end-only padding: `per_axis[i]` is
    `{"op": "pad"|"none", "before": 0, "after": N}` for axis i."""

    per_axis: tuple[dict, ...]


def pad_to_divisible(shape: tuple, divisor: int) -> tuple[tuple, CropPadRecord | None]:
    """Smallest end-only zero-padding that makes every axis of `shape` a multiple of `divisor`.
    Returns `(shape, None)` if already divisible everywhere -- no `CropPadRecord` fabricated for
    a no-op."""
    per_axis = []
    new_shape = []
    any_pad = False
    for size in shape:
        pad_after = (-size) % divisor
        new_shape.append(size + pad_after)
        per_axis.append({"op": "pad" if pad_after else "none", "before": 0, "after": int(pad_after)})
        any_pad = any_pad or pad_after > 0
    if not any_pad:
        return tuple(shape), None
    return tuple(new_shape), CropPadRecord(per_axis=tuple(per_axis))


def crop_using_record(array: np.ndarray, crop_pad: CropPadRecord) -> np.ndarray:
    """Exactly inverts the padding `pad_to_divisible` describes -- crops off precisely the
    recorded `after` amount per axis, nothing else."""
    slices = tuple(slice(a["before"], array.shape[i] - a["after"]) for i, a in enumerate(crop_pad.per_axis))
    return array[slices]
