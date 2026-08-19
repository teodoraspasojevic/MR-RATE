"""Streaming evaluation: the same Dataset training uses, generation, and the VLM3D challenge
metrics -- in one pass, one case at a time, nothing written to disk in between.

Builds the identical `MRReportToVolumeDataset` that `cli.train_r2v` builds, from the same manifest
and the same `R2VDatasetConfig`, and diverges only where training would call `loss.backward()` --
here it calls the sampler instead, then scores against `eval/challenge_metrics.py`.

    cli.train_r2v      build_dataset -> DataLoader -> encode -> UNet -> backward
    cli.evaluate       build_dataset -> DataLoader -> encode -> UNet -> sample -> challenge metrics
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .challenge_metrics import ChallengeAccumulator, combine

log = logging.getLogger("mrrate_r2v.eval.live")


# --------------------------------------------------------------------------- cases

def case_id_for(study_key: str, series_key: str) -> str:
    """A stable hash, so results never carry a raw study/series identifier."""
    return hashlib.sha256(f"{study_key}|{series_key}".encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class LiveCase:
    """One evaluation case. `shape`/`spacing_mm` are (X, Y, Z), the package-boundary axis order."""

    index: int                  # dataset index -- what `dataset[index]` returns this case
    case_id: str
    sequence: str                # modality: T1w / T2w / FLAIR / SWI / ...
    acquisition_plane: str
    shape: tuple
    spacing_mm: tuple

    @property
    def bucket(self) -> str:
        return f"{self.sequence}__{self.acquisition_plane}"


def select_eval_cases(dataset, n_per_bucket: int | None = None) -> list:
    """Dataset indices to evaluate, grouped by (modality, plane) bucket, each bucket sorted by
    `(study_uid, series_id)` -- a property of the data rather than of manifest row order, so the
    result is deterministic with no RNG. `n_per_bucket` caps each bucket at that many cases (the
    first N of the sorted order); `None` (the default) evaluates the entire split.
    """
    by_bucket: dict = {}
    for index, sample in enumerate(dataset.samples):
        key = (sample.modality or "unknown", sample.plane or "unknown")
        by_bucket.setdefault(key, []).append(index)

    ordered: list = []
    for key in sorted(by_bucket):
        indices = sorted(by_bucket[key],
                         key=lambda i: (str(dataset.samples[i].study_uid),
                                       str(dataset.samples[i].series_id)))
        ordered.extend(indices[:n_per_bucket] if n_per_bucket is not None else indices)
    return ordered


def build_cases(dataset, indices) -> list:
    """`LiveCase` per index, read off `dataset.samples` and the dataset's own geometry resolution
    -- the same call `__getitem__` makes, so a case's recorded shape/spacing cannot drift from the
    tensor the dataset will actually hand the model."""
    from ..data.geometry import dhw_to_xyz

    cases = []
    for index in indices:
        sample = dataset.samples[index]
        spec = dataset.geometry.resolve(sample.modality, sample.plane)
        cases.append(LiveCase(
            index=index,
            case_id=case_id_for(sample.study_uid, sample.series_id),
            sequence=sample.modality or "unknown",
            acquisition_plane=sample.plane or "unknown",
            shape=tuple(int(v) for v in dhw_to_xyz(spec.target_shape)),
            spacing_mm=tuple(float(v) for v in dhw_to_xyz(spec.target_spacing)),
        ))
    return cases


# --------------------------------------------------------------------------- the harness

@dataclass
class LiveEvalConfig:
    """Everything that decides what a live evaluation runs on."""

    task_name: str
    output_dir: Path
    split: str = "test"
    n_per_bucket: int | None = None       # None = the entire split
    device: str = "cpu"
    #: Interactive ground-truth-vs-produced panels per bucket, gated by `wandb_log_reports` since a
    #: panel embeds report text.
    wandb_panels: int = 0
    wandb_log_reports: bool = False
    log_every: int = 25
    #: Keep every generated volume under `<out>/volumes/`. Off by default: a 2,000-case run is
    #: ~19 GB, and the sweep stages exist to be thrown away.
    save_volumes: bool = False
    extra_run_metadata: dict = field(default_factory=dict)


class LiveEvaluator:
    """Generate and score one case at a time; aggregate and write once at the end.

    `generate(case, sample) -> np.ndarray` is the only thing that differs between tasks
    (report2volume / reconstruction / generation -- see `cli/evaluate.py`). Everything after that
    call is identical, which is what makes the three numbers comparable.
    """

    def __init__(self, dataset, cases, config: LiveEvalConfig) -> None:
        self.dataset = dataset
        self.cases = list(cases)
        self.config = config
        self._panel_count: dict = {}
        self._volume_files: dict = {}
        self._volume_index: dict = {}
        #: [{case_id, bucket, html}] -- filled by `run`, read by the CLI after it returns.
        self.panels: list = []

    def _save_volume(self, case: LiveCase, produced: np.ndarray) -> None:
        """Append one generated volume to its bucket's raw float16 stack.

        **One file per bucket, not one per case.** `/hnvme` enforces a file-COUNT quota (61k soft,
        81k hard) and is already at it, so 2,000 loose files per run is the constraint that
        actually bites -- space is not. Every case in a bucket shares that bucket's grid by
        construction, so a stack is well-defined; `float16` halves the bytes at a precision far
        finer than any metric here resolves.

        Written incrementally with an open file handle per bucket rather than buffered: a bucket
        holds ~200 volumes and buffering one would cost several GB of RAM.
        """
        from ..validation import rank, world_size

        directory = Path(self.config.output_dir) / "volumes"
        directory.mkdir(parents=True, exist_ok=True)
        handle = self._volume_files.get(case.bucket)
        if handle is None:
            # Rank-scoped: cases are sharded `index % world_size`, so every rank holds a disjoint
            # slice of each bucket and they would otherwise all open the same path and interleave
            # their writes into one corrupt file. The suffix is omitted at world size 1 so the
            # common case keeps a clean name.
            suffix = "" if world_size() == 1 else f".rank{rank()}"
            handle = open(directory / f"{case.bucket}{suffix}.f16.raw", "wb")
            self._volume_files[case.bucket] = handle
            self._volume_index[case.bucket] = {"shape": list(produced.shape), "dtype": "float16",
                                               "rank": rank(), "case_ids": []}
        handle.write(np.ascontiguousarray(produced, dtype=np.float16).tobytes())
        self._volume_index[case.bucket]["case_ids"].append(case.case_id)

    def _close_volumes(self) -> None:
        for handle in self._volume_files.values():
            handle.close()
        if self._volume_index:
            from ..validation import rank, world_size

            directory = Path(self.config.output_dir) / "volumes"
            name = "index.json" if world_size() == 1 else f"index.rank{rank()}.json"
            # The index is what makes the raw stacks readable: shape, dtype and case order per
            # bucket. Without it a `.raw` file is an anonymous byte blob.
            (directory / name).write_text(json.dumps(self._volume_index, indent=2))

    def _render_panel(self, case: LiveCase, sample, real, produced):
        """Panel HTML for one case, or None. Rendered inline because the volume is released as
        soon as the case is scored -- there is no later pass that could go back and read it."""
        wanted = self.config.wandb_panels
        if wanted <= 0 or not self.config.wandb_log_reports:
            return None
        n = self._panel_count.get(case.bucket, 0)
        if n >= wanted:
            return None
        self._panel_count[case.bucket] = n + 1
        try:
            from .figures import validation_panel_html
            from .wandb_evaluation import _PanelCase

            return validation_panel_html(
                _PanelCase(case, real, sample.get("report_text", ""),
                          dict(sample.get("report_sections_text") or {})),
                generated=produced, step=0, epoch=0, validation_index=0, full=True,
            )
        except Exception as exc:  # noqa: BLE001 -- a panel is never worth an evaluation
            log.warning("could not render panel for %s: %s", case.case_id, exc)
            return None

    def run(self, generate) -> dict:
        """Stream every case through `generate`, then aggregate. Returns `{"metrics", "per_case"}`
        (plus run provenance), the same dict written to `<out>/metrics.json`."""
        from ..validation import gather_objects, rank, world_size

        config = self.config
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        accumulator = ChallengeAccumulator(device=config.device)
        mine = [(n, c) for n, c in enumerate(self.cases) if n % world_size() == rank()]
        started = time.time()
        panels = []

        for position, (_n, case) in enumerate(mine, start=1):
            if not ChallengeAccumulator.is_scored(case.sequence):
                accumulator.add_missing(case.case_id, case.bucket, case.sequence)
                continue

            sample = self.dataset[case.index]
            # `gt_native` is present only under `R2VDatasetConfig.gt_space="native"`: the released
            # volume RAS-reoriented and otherwise untouched, in (X, Y, Z). Scoring against it is
            # what the challenge server does -- `compute_basic_metrics` resamples the *generated*
            # volume onto the real one's shape, so the ground truth's geometry is authoritative and
            # the model still generates on its own bucket grid (`case.shape`), unchanged.
            native = sample.get("gt_native")
            if native is not None:
                real = np.asarray(native, dtype=np.float32)
            else:
                real = sample["image"].squeeze(0).float().numpy().astype(np.float32)
            try:
                produced = generate(case, sample)
            except Exception as exc:  # noqa: BLE001 -- one bad case must not lose the whole run
                log.warning("case %s failed to generate: %s: %s", case.case_id,
                           type(exc).__name__, exc)
                accumulator.add_missing(case.case_id, case.bucket, case.sequence)
                continue

            accumulator.add(case.case_id, case.bucket, case.sequence, real, produced)

            if config.save_volumes:
                self._save_volume(case, produced)

            panel = self._render_panel(case, sample, real, produced)
            if panel:
                panels.append({"case_id": case.case_id, "bucket": case.bucket, "html": panel})
            del produced, sample, real

            if position % config.log_every == 0 or position == len(mine):
                log.info("[%d/%d] %.1fs elapsed", position, len(mine), time.time() - started)

        self._close_volumes()
        self.panels = gather_objects(panels)
        result = combine(gather_objects([accumulator.state()]))
        result.update({
            "task": config.task_name, "split": config.split, "n_per_bucket": config.n_per_bucket,
            "elapsed_sec": round(time.time() - started, 1), **config.extra_run_metadata,
        })

        (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
        log.info("done: %d/%d scored, %.1fs -> %s", result["metrics"]["n_scored_files"],
                 result["metrics"]["n_total_files"], result["elapsed_sec"], out)
        return result


__all__ = [
    "LiveCase", "LiveEvalConfig", "LiveEvaluator", "build_cases", "case_id_for",
    "select_eval_cases",
]
