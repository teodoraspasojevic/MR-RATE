"""Bridge to `scripts/data.py` -- the FORA-derived per-volume preprocessing we reuse as-is.

Everything the R2V pipeline does to a single volume (RAS reorient -> resample -> crop/pad
-> normalize) is `scripts/data.py`'s code, unmodified, so R2V and the contrastive pipeline
can never drift apart on preprocessing. This module only makes those functions importable
from inside the package; it adds no logic of its own.

Importing this pulls in torch and nibabel. Keep it out of module-level imports in anything
that must stay lightweight (`manifest.py` imports it lazily, inside functions, for exactly
that reason -- see its docstring).
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from data import (  # noqa: E402
    NORMALIZERS,
    crop_or_pad,
    discover_subjects,
    load_all_splits,
    load_and_resample_nii_from_bytes,
    preprocess_nii,
    preprocess_nii_from_bytes,
    read_native_geometry,
    read_native_geometry_from_bytes,
    validate_cache_manifest,
)

__all__ = [
    "NORMALIZERS", "crop_or_pad", "discover_subjects", "load_all_splits",
    "load_and_resample_nii_from_bytes", "preprocess_nii", "preprocess_nii_from_bytes",
    "read_native_geometry", "read_native_geometry_from_bytes", "validate_cache_manifest",
]
