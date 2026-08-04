"""Bridge to `scripts/data.py` -- the FORA-derived per-volume preprocessing we reuse as-is.

Everything the R2V pipeline does to a single volume (RAS reorient -> resample -> crop/pad
-> normalize) is `scripts/data.py`'s code, unmodified, so R2V and the contrastive pipeline
can never drift apart on preprocessing. This module only makes those functions importable
from inside the package; it adds no logic of its own.

Importing this pulls in torch and nibabel. Keep it out of module-level imports in anything
that must stay lightweight (`manifest.py` imports it lazily, inside functions, for exactly
that reason -- see its docstring).
"""
import os
import sys
from pathlib import Path

# `data` is a common enough top-level name that another package can win the import. cv2 appends
# its own directory to sys.path when imported (monai imports it), and cv2/data/ is a package --
# so anything that imports monai before this module could otherwise get cv2's `data` here. Take
# the front of sys.path unconditionally rather than only when scripts/ is absent: being present
# somewhere in sys.path is not the same as being ahead of the shadowing entry.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
while _SCRIPTS_DIR in sys.path:
    sys.path.remove(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)

_shadowing = sys.modules.get("data")
if _shadowing is not None and os.path.dirname(getattr(_shadowing, "__file__", "") or "") != _SCRIPTS_DIR:
    # Only ever drops a *wrongly* bound top-level `data`; a package's own `pkg.data` is registered
    # under its dotted name and is untouched.
    del sys.modules["data"]

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
