"""The one place that computes the VLM3D challenge metrics. Used by both `cli.evaluate` (final
scoring) and `validation.py` (the periodic during-training curve), so the two can never define
"SSIM" or "FID" differently from each other or from the real leaderboard.

Wraps the vendored `eval/challenge/` package (a port of the official evaluation container) and
reproduces its `score.py`'s aggregation exactly: modality scope, per-case MSE/PSNR/SSIM, streaming
2.5D FID, and the final metrics dict -- including its two real quirks: a case whose generation is
missing is excluded from the MSE/PSNR/SSIM means (not penalised with a worst-case value, despite
what `score.py`'s per-case record suggests), and `dice` is a literal copy of `SSIM_mean`, not real
Dice -- that is what the platform's ranking config actually reads today.
"""
from __future__ import annotations

import numpy as np

from .challenge.fid_2p5d import FIDAccumulator, finalize_pooled
from .challenge.metrics_basic import compute_basic_metrics
from .challenge.modality_filter import ALLOWED_MODALITIES

METRIC_KEYS = (
    "dice", "MSE_mean", "PSNR_mean", "SSIM_mean",
    "FID_2p5D_XY", "FID_2p5D_XZ", "FID_2p5D_YZ", "FID_2p5D_Avg",
    "n_total_files", "n_scored_files", "n_missing_outputs", "n_excluded_out_of_scope_modality",
)


class ChallengeAccumulator:
    """Accumulate one run's (real, produced) pairs; `finalize()` reduces them to the official
    metrics dict plus a per-case breakdown. One instance per process; under multiple ranks, gather
    each rank's `state()` and call the module-level `combine()` once."""

    def __init__(self, device: str = "auto") -> None:
        self._fid = FIDAccumulator(device=device)
        self._per_case: list[dict] = []
        self.n_total = 0
        self.n_excluded = 0
        self.n_missing = 0

    @staticmethod
    def is_scored(sequence: str) -> bool:
        return (sequence or "").lower() in ALLOWED_MODALITIES

    def add(self, case_id: str, bucket: str, sequence: str, real: np.ndarray,
            produced: np.ndarray) -> None:
        """A successfully generated pair. Out-of-scope modalities are still recorded, just never
        scored -- matching the official code's own bookkeeping."""
        self.n_total += 1
        if not self.is_scored(sequence):
            self.n_excluded += 1
            self._per_case.append({"case_id": case_id, "bucket": bucket, "status": "excluded"})
            return
        metrics = compute_basic_metrics(real, produced)
        self._fid.add_pair(real, produced)
        self._per_case.append({"case_id": case_id, "bucket": bucket, "status": "scored", **metrics})

    def add_missing(self, case_id: str, bucket: str, sequence: str) -> None:
        """A case that was never generated -- excluded by modality if out of scope, else missing."""
        self.n_total += 1
        if not self.is_scored(sequence):
            self.n_excluded += 1
            self._per_case.append({"case_id": case_id, "bucket": bucket, "status": "excluded"})
        else:
            self.n_missing += 1
            self._per_case.append({"case_id": case_id, "bucket": bucket, "status": "missing"})

    def state(self) -> dict:
        """A plain, picklable snapshot for `combine()` across ranks."""
        return {"per_case": list(self._per_case), "n_total": self.n_total,
                "n_excluded": self.n_excluded, "n_missing": self.n_missing,
                "fid_raw": self._fid.raw_features()}

    def finalize(self) -> dict:
        """This process's own `{"metrics": {...}, "per_case": [...]}`. Equivalent to
        `combine([self.state()])`."""
        return combine([self.state()])


def combine(states: list) -> dict:
    """The official `score.py`'s aggregation, fed one or more ranks' `ChallengeAccumulator.state()`.
    A single-element list reproduces a plain single-process run exactly."""
    per_case = [r for s in states for r in s["per_case"]]
    scored_rows = [r for r in per_case if r["status"] == "scored"]
    n_total = sum(s["n_total"] for s in states)
    n_excluded = sum(s["n_excluded"] for s in states)
    n_missing = sum(s["n_missing"] for s in states)

    def mean(key: str) -> float:
        vals = [r[key] for r in scored_rows]
        return float(np.mean(vals)) if vals else float("nan")

    metrics = {
        "MSE_mean": mean("MSE"), "PSNR_mean": mean("PSNR"), "SSIM_mean": mean("SSIM"),
        **finalize_pooled([s["fid_raw"] for s in states]),
        "n_total_files": n_total,
        "n_scored_files": n_total - n_excluded,
        "n_excluded_out_of_scope_modality": n_excluded,
        "n_missing_outputs": n_missing,
    }
    # The platform's own primary-metric shim: a copy of SSIM_mean, not real Dice.
    metrics["dice"] = metrics["SSIM_mean"]
    return {"metrics": {k: metrics[k] for k in METRIC_KEYS}, "per_case": per_case}
