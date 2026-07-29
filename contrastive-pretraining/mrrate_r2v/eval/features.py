"""Feature caching with mandatory fingerprint validation. A cache entry is only ever reused if
its stored fingerprint matches the requested one EXACTLY (split, manifest identity, filters,
geometry, preprocessing, encoder version, geometry-contract version) -- any mismatch is a cache
miss, never a silent reuse. Caches from the older (pre-geometry-contract) evaluation
implementation are unreadable here by construction: `evaluation_package` is a required fingerprint
field this repo's caches always set and the old ones never did, so `_is_valid` rejects them
without needing special-case detection logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("feature_cache")

CACHE_SCHEMA_VERSION = "1.0"


def compute_fingerprint(*, split: str, manifest_sha256: str, filters: dict, geometry_fingerprint: dict, encoder_config: dict) -> dict:
    return {
        "evaluation_package": "mr_rate_contrastive_pretraining_evaluation",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "split": split, "manifest_sha256": manifest_sha256, "filters": filters,
        "geometry_fingerprint": geometry_fingerprint, "encoder_config": encoder_config,
    }


def fingerprint_key(fingerprint: dict) -> str:
    canonical = json.dumps(fingerprint, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class FeatureCache:
    """One cache = one directory. `arrays`: {name: np.ndarray}, pooled across whatever cohort
    produced them (e.g. all real MedicalNet features for a given split+geometry bucket).
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def _paths(self, fingerprint: dict):
        key = fingerprint_key(fingerprint)
        base = self.cache_dir / key
        return base.with_suffix(".npz"), base.with_suffix(".fingerprint.json")

    def load(self, fingerprint: dict) -> dict | None:
        """Returns the cached {name: array} dict, or None on any miss/mismatch/corruption --
        never raises, since a cache miss must always be recoverable by recomputing.
        """
        npz_path, fp_path = self._paths(fingerprint)
        if not npz_path.is_file() or not fp_path.is_file():
            return None
        try:
            stored_fp = json.loads(fp_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            log.warning("feature cache fingerprint unreadable, treating as miss: %s", e)
            return None
        if stored_fp != fingerprint:
            log.info("feature cache fingerprint mismatch (stale/incompatible cache) -- recomputing, not reusing: %s", npz_path)
            return None
        try:
            with np.load(npz_path) as data:
                return {k: data[k] for k in data.files}
        except (OSError, ValueError) as e:
            log.warning("feature cache file unreadable, treating as miss: %s", e)
            return None

    def save(self, fingerprint: dict, arrays: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        npz_path, fp_path = self._paths(fingerprint)
        # `np.savez_compressed` silently APPENDS ".npz" to any path that doesn't already end in
        # it -- naming the staging file "<key>.npz.tmp" would actually create
        # "<key>.npz.tmp.npz" and leave the intended tmp path missing, breaking the rename below.
        # Naming it "<key>.tmp.npz" (ends in .npz already) avoids that silent rename.
        tmp_npz = npz_path.with_name(npz_path.stem + ".tmp.npz")
        np.savez_compressed(tmp_npz, **arrays)
        tmp_npz.replace(npz_path)  # atomic on the same filesystem -- never a partially-written cache file
        fp_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
