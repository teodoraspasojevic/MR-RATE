"""Step-based validation for report-adapter training.

Every `--validate-every-steps` optimizer steps, generate a fixed sample of the `val` split and
score it against the same VLM3D challenge metrics `cli.evaluate` uses
(`eval/challenge_metrics.py`), logged as:

    val/MSE_mean, val/PSNR_mean, val/SSIM_mean, val/FID_2p5D_{XY,XZ,YZ,Avg}, val/dice

So a training curve and a final `cli.evaluate` score are the same numbers, computed the same way,
just at different sample sizes -- there is exactly one metric definition in this package.

**Distributed behaviour.** Cases are sharded `index % world_size`, so no case is generated twice;
each rank accumulates its own `ChallengeAccumulator` and the small per-plane feature arrays are
gathered with `all_gather_object` before one global metric computation, identical to a single
process within float tolerance.
"""
from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .eval.challenge_metrics import ChallengeAccumulator, combine

log = logging.getLogger("mrrate_r2v.validation")


@dataclass
class ValidationConfig:
    """How many cases to validate on, and how they are generated."""

    n_samples: int = 64
    seed: int = 0
    num_inference_steps: int = 30
    #: Cases rendered into the interactive W&B panel each validation step.
    n_visualize: int = 4

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        if self.n_visualize > self.n_samples:
            raise ValueError("n_visualize must be <= n_samples: visualised cases come from the "
                             "validation sample")


@dataclass
class ValidationCase:
    """One fixed validation sample. `case_id` is a hash -- `study_key`/`series_key` never leave
    this object and are never logged or written into results."""

    index: int
    case_id: str
    report_text: str
    report_sections: dict
    modality: str
    plane: str
    shape_xyz: tuple
    spacing_xyz: tuple
    target: Optional[np.ndarray] = None

    @property
    def bucket(self) -> str:
        return f"{self.modality}_{self.plane}"


def case_id_for(study_key: str, series_key: str) -> str:
    return hashlib.sha256(f"{study_key}|{series_key}".encode()).hexdigest()[:16]


def select_validation_cases(dataset, config: ValidationConfig) -> list[int]:
    """A fixed, seeded, bucket-stratified list of dataset indices -- stratified by (modality,
    plane) so a small sample is not accidentally all one bucket, which would make every metric
    track one anatomy and call it the model. Deterministic in `config.seed` alone, so the curve
    compares like with like across runs and resumes.
    """
    by_bucket: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        key = f"{sample.modality or 'unknown'}_{sample.plane or 'unknown'}"
        by_bucket.setdefault(key, []).append(index)

    rng = random.Random(config.seed)
    for indices in by_bucket.values():
        rng.shuffle(indices)

    ordered: list[int] = []
    buckets = [by_bucket[key] for key in sorted(by_bucket)]
    position = 0
    while len(ordered) < sum(len(b) for b in buckets):
        progressed = False
        for bucket in buckets:
            if position < len(bucket):
                ordered.append(bucket[position])
                progressed = True
        if not progressed:
            break
        position += 1
    if len(ordered) < config.n_samples:
        log.warning("validation asked for %d cases but the split only has %d",
                    config.n_samples, len(ordered))
    return ordered[:config.n_samples]


# --------------------------------------------------------------------------- distributed helpers

def _dist():
    import torch.distributed as dist

    return dist if dist.is_available() and dist.is_initialized() else None


def world_size() -> int:
    d = _dist()
    return d.get_world_size() if d else 1


def rank() -> int:
    d = _dist()
    return d.get_rank() if d else 0


def gather_objects(payload: list) -> list:
    """Union of every rank's list, on every rank. A no-op single-GPU."""
    d = _dist()
    if d is None:
        return payload
    buckets = [None] * d.get_world_size()
    d.all_gather_object(buckets, payload)
    return [item for bucket in buckets if bucket for item in bucket]


# --------------------------------------------------------------------------- runner

class ValidationRunner:
    """Generates the fixed validation sample and scores it. Constructed once and reused, so the
    case list is built once. Holds no reference to the trainer: `run(trainer, step)` takes it.
    """

    def __init__(
        self,
        dataset,
        sampler_factory,
        config: Optional[ValidationConfig] = None,
        wandb_run=None,
        output_dir: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        self.dataset = dataset
        self.sampler_factory = sampler_factory
        self.config = config or ValidationConfig()
        self.wandb_run = wandb_run
        self.output_dir = Path(output_dir) if output_dir else None
        self.device = device
        self._indices = select_validation_cases(dataset, self.config)
        self._visualize = set(self._indices[: self.config.n_visualize])
        self._cases: dict[int, ValidationCase] = {}
        self._validation_index = 0
        log.info("validation: %d cases over %d buckets (seed %d)", len(self._indices),
                 len({self._bucket_of(i) for i in self._indices}), self.config.seed)

    def _bucket_of(self, index: int) -> str:
        sample = self.dataset.samples[index]
        return f"{sample.modality or 'unknown'}_{sample.plane or 'unknown'}"

    def case(self, index: int) -> ValidationCase:
        if index in self._cases:
            return self._cases[index]
        item = self.dataset[index]
        case = ValidationCase(
            index=index,
            case_id=case_id_for(item["study_key"], item["series_key"]),
            report_text=item["report_text"],
            report_sections=dict(item.get("report_sections_text") or {}),
            modality=item["modality"],
            plane=item["acquisition_plane"],
            shape_xyz=tuple(int(v) for v in item["target_shape"].tolist()),
            spacing_xyz=tuple(float(v) for v in item["target_spacing_mm"].tolist()),
            # `.float()` before `.numpy()`: R2VDatasetConfig.dtype defaults to bfloat16, which numpy
            # cannot represent. Every metric is float32 internally regardless.
            target=item["image"].squeeze(0).float().numpy().astype(np.float32),
        )
        self._cases[index] = case
        return case

    def run(self, trainer, step: int) -> dict:
        """Generate, score, restore. Returns a flat dict of `val/*` scalars ready for W&B."""
        self._validation_index += 1
        epoch = int(getattr(trainer, "epoch", 0)) + 1

        mine = [i for n, i in enumerate(self._indices) if n % world_size() == rank()]
        started = time.time()

        unet = getattr(trainer.unet, "module", trainer.unet)
        was_training = unet.training
        generate = self.sampler_factory(trainer)

        accumulator = ChallengeAccumulator(device=self.device)
        panels = []
        try:
            unet.eval()
            with torch.inference_mode():
                for index in mine:
                    case = self.case(index)
                    generated = generate(case)
                    accumulator.add(case.case_id, case.bucket, case.modality, case.target, generated)
                    if index in self._visualize:
                        panels.append({
                            "case_id": case.case_id,
                            "panel_html": self._render_panel(case, generated, step, epoch=epoch,
                                                             validation_index=self._validation_index),
                        })
                    del generated
        finally:
            unet.train(was_training)

        result = combine(gather_objects([accumulator.state()]))
        metrics = {f"val/{key}": value for key, value in result["metrics"].items()}
        # **`gather_objects` is a COLLECTIVE and must be entered by every rank.** Calling it inside
        # `if rank() == 0` leaves the other ranks never reaching `all_gather_object`, so rank 0
        # waits on a payload-size exchange that never completes and then resizes its input tensor to
        # whatever uninitialised bytes it read -- surfacing as `OutOfMemoryError: tried to allocate
        # more than 1EB`, not as a hang or a clear collective error. Gather on all ranks; only the
        # *logging* below is rank 0's job. Sorted so the panel order does not depend on which rank
        # happened to answer first.
        gathered_panels = sorted(gather_objects(panels), key=lambda r: r["case_id"])
        if rank() == 0:
            metrics["val/n_panels"] = self._log_panels(gathered_panels, step)
        metrics["val/n_cases"] = result["metrics"]["n_scored_files"]
        metrics["val/seconds"] = time.time() - started
        metrics["val/validation_index"] = self._validation_index
        metrics["val/epoch"] = epoch
        if rank() == 0:
            headline = {k: round(v, 5) for k, v in metrics.items()
                        if k in ("val/SSIM_mean", "val/PSNR_mean", "val/FID_2p5D_Avg")
                        and isinstance(v, float)}
            log.info("validation @ optimizer step %d (%d cases, %.1fs): %s", step,
                     result["metrics"]["n_scored_files"], metrics["val/seconds"], headline)
        return metrics

    def _render_panel(self, case: ValidationCase, generated: np.ndarray, step: int,
                      epoch: int = 0, validation_index: int = 0):
        """Render the interactive panel on the rank that generated the case. Returns HTML or None."""
        try:
            from .eval.figures import validation_panel_html

            return validation_panel_html(case, generated, step, epoch=epoch,
                                         validation_index=validation_index, full=False)
        except Exception as exc:  # noqa: BLE001 -- a plot must never end a training run
            log.warning("validation panel render failed for %s: %s", case.case_id, exc)
            return None

    def _log_panels(self, records: list[dict], step: int) -> int:
        """Rank 0 logs every gathered panel, whichever rank rendered it.

        **The key stays `validation/<case_id>` -- stable across validation steps on purpose.** W&B
        keeps one media panel per key with its own step slider, so a stable key is what lets you
        drag through training and watch *the same case* evolve.
        """
        if self.wandb_run is None or not getattr(self.wandb_run, "enabled", False):
            return 0
        logged = 0
        for record in records:
            html = record.get("panel_html")
            if not html:
                continue
            try:
                self.wandb_run.log_html(f"validation/{record['case_id']}", html, step=step)
                logged += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("validation panel log failed for %s: %s", record["case_id"], exc)
        return logged


__all__ = [
    "ValidationCase",
    "ValidationConfig",
    "ValidationRunner",
    "case_id_for",
    "gather_objects",
    "rank",
    "select_validation_cases",
    "world_size",
]
