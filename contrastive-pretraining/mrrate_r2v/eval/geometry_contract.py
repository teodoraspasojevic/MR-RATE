"""Can these two volumes legitimately be compared voxel by voxel?

One `GeometryRecord` per volume says exactly what physical grid it occupies.
`compare_geometry` returns one of four verdicts, never a silent guess:

    STRICT_MATCH                  same grid -- compute paired metrics
    DECODER_BOUNDARY_CORRECTABLE  shape differs but everything else agrees; fixable only if
                                  the caller can *prove* the padding it applied
    WORLD_ALIGNED_ELIGIBLE        different grid, same anatomy; resamplable on explicit opt-in,
                                  and reported separately from strict results
    INCOMPATIBLE                  excluded, with a recorded reason

**The one rule this module exists to enforce: equal `.shape` is never treated as proof that
two volumes occupy the same physical space.** Shape, spacing, orientation, affine rotation,
and origin/FOV overlap are each checked separately. Two crops of different brain regions can
share a shape; one scan resampled at two spacings has two shapes and is still the same scan.
The implementation this replaced resized whenever shapes merely differed, which produces
precise-looking numbers that prove nothing.

Rationale and citations: docs/design/archive/10_evaluation_geometry_contract_and_shape_mismatch_policy.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import nibabel as nib
import numpy as np

GEOMETRY_CONTRACT_VERSION = "1.0"

# The axis order `data/dataset.py` guarantees: axis0=Right-Left, axis1=Anterior-Posterior,
# axis2=Superior-Inferior, after RAS canonicalization.
DATASET_AXIS_ORDER = ("X", "Y", "Z")
DATASET_ANATOMICAL_AXIS_MEANING = ("R-L", "A-P", "S-I")
DATASET_ORIENTATION = "RAS"

_AXCODE_TO_ANATOMICAL = {
    "R": "R-L", "L": "R-L", "A": "A-P", "P": "A-P", "S": "S-I", "I": "S-I",
}


# --------------------------------------------------------------------------- crop/pad provenance


@dataclass(frozen=True)
class CropPadRecord:
    """Exact, deterministic provenance for a shape change this evaluation package itself
    performed (never a change of unknown origin -- see `pad_to_divisible`/`crop_using_record`).
    `per_axis[i]` describes what happened to axis i, in the SAME axis order as the GeometryRecord
    it is attached to.
    """

    per_axis: tuple[dict, ...]
    reason: str
    method: str  # "end" (only pad/crop at the end of each axis) -- the only method implemented,
    #              because it is the only one with a trivially exact, unambiguous inverse.

    def as_dict(self) -> dict:
        return {"per_axis": list(self.per_axis), "reason": self.reason, "method": self.method}


def pad_to_divisible(shape: tuple, divisor: int) -> tuple[tuple, CropPadRecord | None]:
    """Smallest end-only zero-padding that makes every axis of `shape` a multiple of `divisor`.
    Returns (new_shape, None) if `shape` is already divisible everywhere (no padding performed,
    no CropPadRecord fabricated for a no-op). Pure arithmetic -- does not touch any array; pass
    the returned `CropPadRecord` to `pad_array`/`crop_using_record` to actually do so.
    """
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
    return tuple(new_shape), CropPadRecord(per_axis=tuple(per_axis), reason="divisibility_padding", method="end")


def pad_array(array: np.ndarray, crop_pad: CropPadRecord) -> np.ndarray:
    pad_width = [(a["before"], a["after"]) for a in crop_pad.per_axis]
    return np.pad(array, pad_width, mode="constant")


def crop_using_record(array: np.ndarray, crop_pad: CropPadRecord) -> np.ndarray:
    """Exactly inverts `pad_array(array_before_pad, crop_pad)` -- crops off precisely the
    `before`/`after` amounts recorded, nothing else. This is the ONLY shape-correction primitive
    in this module that does not require a resampling decision, because the record proves the
    correspondence rather than assuming it from matching sizes.
    """
    # Explicit per-axis stop index: array.shape - after (avoids relying on a fragile "-0" slice).
    slices = tuple(slice(a["before"], array.shape[i] - a["after"]) for i, a in enumerate(crop_pad.per_axis))
    return array[slices]


# --------------------------------------------------------------------------- GeometryRecord


@dataclass(frozen=True)
class GeometryRecord:
    """One volume's physical grid. Always build it via a classmethod, never by hand, so every
    field is traceable to a real source:

        from_cohort_case        a frozen ground-truth case (the usual path at evaluation time)
        from_dataset_sample     a live `MRReportToVolumeDataset` sample (preprocessing time)
        from_nifti              an externally-saved file -- the only source with a real affine
        from_generation_condition  an unconditional generation, which has no patient provenance

    `with_crop_pad_applied` derives a record for a volume this package itself padded/cropped.
    """

    shape: tuple
    axis_order: tuple
    anatomical_axis_meaning: tuple
    spacing_mm: tuple
    orientation: str | None
    affine: np.ndarray | None
    modality: str | None
    acquisition_plane: str | None
    crop_pad: CropPadRecord | None
    valid_bounds: tuple | None  # per-axis (start, stop) of the non-padded region, this record's own shape
    preprocessing_version: str
    source: str
    study_key: str | None = None
    series_key: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.shape)
        for name, value in (("axis_order", self.axis_order), ("anatomical_axis_meaning", self.anatomical_axis_meaning), ("spacing_mm", self.spacing_mm)):
            if len(value) != n:
                raise ValueError(f"GeometryRecord: len({name})={len(value)} does not match len(shape)={n}")
        if self.affine is not None and self.affine.shape != (4, 4):
            raise ValueError(f"GeometryRecord: affine must be a 4x4 matrix, got shape {self.affine.shape}")

    @property
    def fov_mm(self) -> tuple:
        return tuple(s * p for s, p in zip(self.shape, self.spacing_mm))

    def fingerprint(self) -> dict:
        """A compact, JSON-safe dict identifying this record's geometry+preprocessing --
        intended for feature-cache keys (see distribution_metrics.py) and result provenance.
        Deliberately excludes the affine matrix (not stable/meaningful to hash exactly) and
        study/series identifiers (not part of "what preprocessing produced this").
        """
        return {
            "shape": list(self.shape), "axis_order": list(self.axis_order),
            "spacing_mm": [round(float(x), 6) for x in self.spacing_mm],
            "orientation": self.orientation, "modality": self.modality,
            "acquisition_plane": self.acquisition_plane,
            "preprocessing_version": self.preprocessing_version,
            "geometry_contract_version": GEOMETRY_CONTRACT_VERSION,
        }

    def with_crop_pad_applied(self, new_shape: tuple, crop_pad: CropPadRecord, source: str) -> "GeometryRecord":
        """A new record describing the SAME physical content after `pad_array`/`crop_using_record`
        was applied -- spacing/orientation/modality/plane/study/series are unchanged (padding
        with zeros does not move the physical grid's origin under the "method=end" convention:
        the original voxels keep their original indices), only `shape`/`crop_pad` change.
        """
        return replace(self, shape=tuple(new_shape), crop_pad=crop_pad, source=source)

    @classmethod
    def from_cohort_case(cls, case, *, preprocessing_version: str, source: str = "cohort_case",
                         shape: tuple | None = None) -> "GeometryRecord":
        """A `cohort.CohortCase` -- a frozen ground-truth volume on disk.

        `shape` overrides the case's recorded shape for the rare caller that padded the array
        before this point; leave it None and the cohort's own recorded shape is used, which is
        what almost every caller wants.

        No affine: the cohort's volumes are RAS by construction (the Dataset canonicalizes and
        never bypasses it) and are stored as bare arrays. That is recorded honestly as
        `affine=None` rather than synthesizing one that `compare_geometry` would then treat as
        evidence.
        """
        return cls(
            shape=tuple(int(x) for x in (shape if shape is not None else case.shape)),
            axis_order=DATASET_AXIS_ORDER, anatomical_axis_meaning=DATASET_ANATOMICAL_AXIS_MEANING,
            spacing_mm=tuple(float(x) for x in case.spacing_mm), orientation=DATASET_ORIENTATION,
            affine=None, modality=case.sequence, acquisition_plane=case.acquisition_plane,
            crop_pad=None, valid_bounds=None, preprocessing_version=preprocessing_version,
            source=source, study_key=case.study_key, series_key=case.series_key,
        )

    @classmethod
    def from_dataset_sample(cls, sample: dict, *, which: str = "target", preprocessing_version: str, index: int | None = None) -> "GeometryRecord":
        """One item from `MRReportToVolumeDataset` (or one row of a collated batch, via `index`).

        `which="target"`: the already-resampled grid -- what a model actually sees.
        `which="native"`: the pre-resample geometry, provenance only (no native-resolution
        tensor is ever materialized, so it is never used for a voxelwise comparison).

        No affine is available from a Dataset sample; RAS orientation is a guarantee of the
        Dataset's construction, recorded as such rather than re-derived here.
        """
        def _get(key):
            v = sample[key]
            return v[index] if index is not None else v

        def _tuple3(t):
            return tuple(int(x) if isinstance(x, (int, np.integer)) or float(x).is_integer() else float(x) for x in (t.tolist() if hasattr(t, "tolist") else t))

        if which == "target":
            shape = _tuple3(_get("target_shape"))
            spacing = tuple(float(x) for x in (_get("target_spacing_mm").tolist() if hasattr(_get("target_spacing_mm"), "tolist") else _get("target_spacing_mm")))
        elif which == "native":
            shape = _tuple3(_get("native_shape"))
            spacing = tuple(float(x) for x in (_get("native_spacing_mm").tolist() if hasattr(_get("native_spacing_mm"), "tolist") else _get("native_spacing_mm")))
        else:
            raise ValueError(f"which must be 'target' or 'native', got {which!r}")

        return cls(
            shape=shape, axis_order=DATASET_AXIS_ORDER, anatomical_axis_meaning=DATASET_ANATOMICAL_AXIS_MEANING,
            spacing_mm=spacing, orientation=DATASET_ORIENTATION, affine=None,
            modality=_get("modality"), acquisition_plane=_get("acquisition_plane"),
            crop_pad=None, valid_bounds=None, preprocessing_version=preprocessing_version,
            source=f"dataset_sample:{which}", study_key=_get("study_key"), series_key=_get("series_key"),
        )

    @classmethod
    def from_nifti(cls, path, *, modality: str | None = None, acquisition_plane: str | None = None,
                    preprocessing_version: str = "external", study_key: str | None = None, series_key: str | None = None) -> "GeometryRecord":
        """An externally-saved `.nii.gz` (an R2V prediction, or any saved volume) -- the ONE
        record type in this module with a real, file-derived affine. Orientation is read, not
        assumed: `nib.aff2axcodes(affine)` is compared against RAS and recorded honestly (a
        non-RAS file is a valid GeometryRecord, just with `orientation != "RAS"`, letting
        `compare_geometry` correctly reject it against a Dataset sample's RAS-guaranteed record).
        """
        img = nib.load(str(path))
        axcodes = nib.aff2axcodes(img.affine)
        orientation = "".join(axcodes)
        anatomical = tuple(_AXCODE_TO_ANATOMICAL[c] for c in axcodes)
        return cls(
            shape=tuple(int(x) for x in img.shape[:3]), axis_order=("axis0", "axis1", "axis2"),
            anatomical_axis_meaning=anatomical, spacing_mm=tuple(float(x) for x in img.header.get_zooms()[:3]),
            orientation=orientation, affine=np.asarray(img.affine, dtype=np.float64),
            modality=modality, acquisition_plane=acquisition_plane, crop_pad=None, valid_bounds=None,
            preprocessing_version=preprocessing_version, source=f"nifti_file:{path}",
            study_key=study_key, series_key=series_key,
        )

    @classmethod
    def from_generation_condition(cls, *, shape: tuple, spacing_mm: tuple, modality: str | None,
                                   acquisition_plane: str | None, condition: dict, preprocessing_version: str) -> "GeometryRecord":
        """An NVIDIA unconditional-generation output. There is no real-patient study/series to
        attach -- `study_key`/`series_key` stay `None` and the conditioning actually used
        (modality class code, spacing tensor, seed) is recorded in `extra["condition"]` instead,
        so a generated volume is never mistaken for having real provenance downstream.
        """
        return cls(
            shape=tuple(int(x) for x in shape), axis_order=DATASET_AXIS_ORDER,
            anatomical_axis_meaning=DATASET_ANATOMICAL_AXIS_MEANING, spacing_mm=tuple(float(x) for x in spacing_mm),
            orientation=None, affine=synthesize_diagonal_affine(spacing_mm), modality=modality,
            acquisition_plane=acquisition_plane, crop_pad=None, valid_bounds=None,
            preprocessing_version=preprocessing_version, source="nvidia_unconditional_generation",
            study_key=None, series_key=None, extra={"condition": condition, "synthetic": True},
        )


def synthesize_diagonal_affine(spacing_mm: tuple, origin_mm: tuple | None = None) -> np.ndarray:
    """A diagonal, RAS-consistent affine built ONLY from spacing (+ optional origin) -- for
    volumes with no real file-derived affine (a Dataset sample, or a from-scratch generation).
    NEVER a substitute for a real affine when one is required (`compare_geometry` never treats
    two synthesized affines as proof of `rotation_match`/`origin_match` -- see below); this exists
    only so downstream code that needs *some* affine to save a NIfTI has one, explicitly labeled
    as synthesized via the record's own `source`/`extra` fields, not silently indistinguishable
    from a real one.
    """
    aff = np.eye(4, dtype=np.float64)
    for i in range(3):
        aff[i, i] = spacing_mm[i]
    if origin_mm is not None:
        aff[:3, 3] = origin_mm
    return aff


# --------------------------------------------------------------------------- comparison policy


class GeometryDecision(str, Enum):
    STRICT_MATCH = "strict_match"
    DECODER_BOUNDARY_CORRECTABLE = "decoder_boundary_correctable"
    WORLD_ALIGNED_ELIGIBLE = "world_aligned_eligible"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class GeometryComparison:
    decision: GeometryDecision
    reasons: tuple  # non-empty iff decision != STRICT_MATCH
    shape_match: bool
    spacing_match: bool
    orientation_match: bool
    rotation_match: bool | None  # None if either side has no affine
    origin_match: bool | None  # None if either side has no affine
    fov_overlap_fraction: float | None  # None if not computable (no affines)
    modality_match: bool
    plane_match: bool

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["decision"] = self.decision.value
        d["reasons"] = list(self.reasons)
        return d


def _rotation_matches(affine_a: np.ndarray, spacing_a, affine_b: np.ndarray, spacing_b, tol_deg: float) -> bool:
    """Compares only the direction-cosine (rotation) part of two affines, independent of spacing
    and origin -- the decomposition the old implementation's single blanket affine tolerance
    (`~/NV-Generate-CTMR/evaluation/evaluate_r2v.py:132`) did not do.
    """
    dir_a = affine_a[:3, :3] / np.asarray(spacing_a)
    dir_b = affine_b[:3, :3] / np.asarray(spacing_b)
    # angle between corresponding basis vectors, worst axis
    max_angle = 0.0
    for i in range(3):
        va, vb = dir_a[:, i], dir_b[:, i]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-9 or nb < 1e-9:
            return False
        cos_angle = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
        max_angle = max(max_angle, math.degrees(math.acos(cos_angle)))
    return max_angle <= tol_deg


def _fov_overlap_fraction(affine_a, shape_a, affine_b, shape_b) -> float:
    """Fraction of volume A's world-space bounding box that overlaps volume B's, using each
    corner of each grid transformed to world space via its own affine. A coarse but
    orientation-and-origin-aware overlap estimate -- exact enough to gate "is resampling even
    physically meaningful" without a full voxel-mask intersection.
    """
    def corners_world(affine, shape):
        idx = np.array([[x, y, z, 1.0] for x in (0, shape[0]) for y in (0, shape[1]) for z in (0, shape[2])])
        return (affine @ idx.T).T[:, :3]

    wa = corners_world(affine_a, shape_a)
    wb = corners_world(affine_b, shape_b)
    lo = np.maximum(wa.min(axis=0), wb.min(axis=0))
    hi = np.minimum(wa.max(axis=0), wb.max(axis=0))
    overlap_dims = np.clip(hi - lo, 0, None)
    overlap_vol = float(np.prod(overlap_dims))
    vol_a = float(np.prod(wa.max(axis=0) - wa.min(axis=0)))
    if vol_a <= 0:
        return 0.0
    return min(overlap_vol / vol_a, 1.0)


def compare_geometry(
    a: GeometryRecord, b: GeometryRecord, *,
    spacing_tol_mm: float = 1e-3, rotation_tol_deg: float = 1.0, origin_tol_mm: float = 1.0,
    min_fov_overlap_fraction: float = 0.98, require_modality_match: bool = True, require_plane_match: bool = True,
) -> GeometryComparison:
    """The single entry point every metric computation must call before comparing two volumes
    voxelwise. Never inspects raw array data -- geometry only. `a` is conventionally the target
    (ground truth), `b` the prediction/reconstruction.
    """
    reasons = []

    modality_match = (not require_modality_match) or (a.modality == b.modality)
    if not modality_match:
        reasons.append(f"modality mismatch: {a.modality!r} vs {b.modality!r}")
    plane_match = (not require_plane_match) or (a.acquisition_plane == b.acquisition_plane)
    if not plane_match:
        reasons.append(f"acquisition_plane mismatch: {a.acquisition_plane!r} vs {b.acquisition_plane!r}")

    orientation_match = (a.anatomical_axis_meaning == b.anatomical_axis_meaning) and (a.orientation is None or b.orientation is None or a.orientation == b.orientation)
    if not orientation_match:
        reasons.append(f"orientation/axis mismatch: {a.orientation}/{a.anatomical_axis_meaning} vs {b.orientation}/{b.anatomical_axis_meaning}")

    shape_match = tuple(a.shape) == tuple(b.shape)
    spacing_match = len(a.spacing_mm) == len(b.spacing_mm) and all(abs(x - y) <= spacing_tol_mm for x, y in zip(a.spacing_mm, b.spacing_mm))
    if not spacing_match:
        reasons.append(f"spacing mismatch beyond {spacing_tol_mm}mm: {a.spacing_mm} vs {b.spacing_mm}")

    rotation_match = origin_match = None
    fov_overlap = None
    if a.affine is not None and b.affine is not None:
        rotation_match = _rotation_matches(a.affine, a.spacing_mm, b.affine, b.spacing_mm, rotation_tol_deg)
        if not rotation_match:
            reasons.append(f"affine rotation mismatch beyond {rotation_tol_deg} degrees")
        origin_diff = np.linalg.norm(a.affine[:3, 3] - b.affine[:3, 3])
        origin_match = bool(origin_diff <= origin_tol_mm)
        if not origin_match:
            reasons.append(f"affine origin mismatch: {origin_diff:.3f}mm > {origin_tol_mm}mm tolerance")
        fov_overlap = _fov_overlap_fraction(a.affine, a.shape, b.affine, b.shape)

    if not (modality_match and plane_match and orientation_match):
        return GeometryComparison(GeometryDecision.INCOMPATIBLE, tuple(reasons), shape_match, spacing_match, orientation_match, rotation_match, origin_match, fov_overlap, modality_match, plane_match)

    if shape_match and spacing_match and (rotation_match is not False) and (origin_match is not False):
        return GeometryComparison(GeometryDecision.STRICT_MATCH, (), True, True, True, rotation_match, origin_match, fov_overlap if fov_overlap is not None else 1.0, True, True)

    if (not shape_match) and spacing_match and (rotation_match is not False) and (origin_match is not False):
        reasons.append("shape differs but spacing/orientation/rotation/origin agree -- eligible for a KNOWN, provenance-based correction (see pad_to_divisible/crop_using_record) if the caller can supply one; otherwise reject rather than blind-crop")
        return GeometryComparison(GeometryDecision.DECODER_BOUNDARY_CORRECTABLE, tuple(reasons), shape_match, spacing_match, orientation_match, rotation_match, origin_match, fov_overlap, modality_match, plane_match)

    if a.affine is not None and b.affine is not None and fov_overlap is not None and fov_overlap >= min_fov_overlap_fraction:
        reasons.append(f"different voxel grids, same anatomy/FOV (overlap={fov_overlap:.3f} >= {min_fov_overlap_fraction}) -- eligible for world_aligned resampling")
        return GeometryComparison(GeometryDecision.WORLD_ALIGNED_ELIGIBLE, tuple(reasons), shape_match, spacing_match, orientation_match, rotation_match, origin_match, fov_overlap, modality_match, plane_match)

    if fov_overlap is not None:
        reasons.append(f"insufficient FOV overlap ({fov_overlap:.3f} < {min_fov_overlap_fraction}) -- not the same physical region")
    else:
        reasons.append("missing affine on at least one side -- cannot prove or resample physical correspondence")
    return GeometryComparison(GeometryDecision.INCOMPATIBLE, tuple(reasons), shape_match, spacing_match, orientation_match, rotation_match, origin_match, fov_overlap, modality_match, plane_match)


def resample_world_aligned(source_array: np.ndarray, source_geom: GeometryRecord, target_geom: GeometryRecord, *, order: int = 1) -> tuple[np.ndarray, GeometryRecord, dict]:
    """Resamples `source_array` (on `source_geom`'s grid) onto `target_geom`'s grid using both
    real affines -- only called after `compare_geometry` returns WORLD_ALIGNED_ELIGIBLE. `order=1`
    (trilinear) is the appropriate interpolation for continuous MRI intensity (order=0/nearest
    would be for label maps, never used here). Returns (resampled_array, a GeometryRecord equal to
    `target_geom` except `source`/`extra`, and an overlap-info dict for the result record).
    """
    if source_geom.affine is None or target_geom.affine is None:
        raise ValueError("resample_world_aligned requires both records to have a real affine")
    from scipy.ndimage import affine_transform

    voxel_to_voxel = np.linalg.inv(source_geom.affine) @ target_geom.affine
    resampled = affine_transform(
        source_array, matrix=voxel_to_voxel[:3, :3], offset=voxel_to_voxel[:3, 3],
        output_shape=target_geom.shape, order=order, mode="constant", cval=0.0,
    )
    overlap_fraction = _fov_overlap_fraction(source_geom.affine, source_geom.shape, target_geom.affine, target_geom.shape)
    result_geom = replace(
        target_geom, source=f"world_aligned_resample_of:{source_geom.source}",
        extra={**target_geom.extra, "world_aligned": True, "resample_order": order, "source_fov_overlap_fraction": overlap_fraction},
    )
    return resampled, result_geom, {"interpolation": "trilinear" if order == 1 else f"order={order}", "source_geometry": source_geom.fingerprint(), "target_geometry": target_geom.fingerprint(), "fov_overlap_fraction": overlap_fraction}
