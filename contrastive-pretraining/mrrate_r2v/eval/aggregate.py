"""Shared per-sequence/overall aggregation for paired-metric rows. Dataset-agnostic; used
identically by every evaluate_*.py entry point so aggregation logic is defined exactly once.
"""

from __future__ import annotations

import numpy as np


def aggregate_metric_rows(rows: list, group_key_fn, metric_names: list) -> dict:
    """Groups `rows` (list of dict) by `group_key_fn(row)` plus an implicit `"overall"` group
    (every row), computes mean/std per metric in `metric_names`, dropping non-finite/missing
    values from that one metric's statistics only (never dropping the whole row).
    """
    groups: dict = {"overall": list(rows)}
    for r in rows:
        groups.setdefault(group_key_fn(r), []).append(r)

    out = {}
    for group_name, group_rows in groups.items():
        entry = {"n": len(group_rows)}
        for m in metric_names:
            vals = [r[m] for r in group_rows if isinstance(r.get(m), (int, float, np.floating)) and np.isfinite(r[m])]
            entry[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_valid": len(vals)} if vals else {"mean": None, "std": None, "n_valid": 0}
        out[group_name] = entry
    return out
