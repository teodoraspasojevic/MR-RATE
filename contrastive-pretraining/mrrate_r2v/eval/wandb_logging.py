"""Optional Weights & Biases logging wrapper -- dataset-agnostic, ported from the older
evaluation implementation (`~/NV-Generate-CTMR/evaluation/wandb_logging.py`, classified "reusable
unchanged" in docs/design/archive/09_older_evaluation_implementation_audit.md §15). `wandb` import is deferred into `WandbRun.__init__` so a
missing package/credentials/network failure degrades to a no-op with a log message, never a crash.
Credentials read only from `WANDB_API_KEY` or an existing `~/.netrc` -- never hardcoded.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("wandb_logging")

WANDB_MODES = ("online", "offline", "disabled")


class WandbRun:
    def __init__(self, mode: str, entity: str | None, project: str | None, run_name: str, group: str | None = None, tags: list | None = None, config: dict | None = None):
        self.mode = mode
        self.enabled = False
        self.run = None
        self.offline_sync_hint = None
        if mode == "disabled":
            log.info("W&B logging disabled (--wandb-mode disabled)")
            return
        try:
            import wandb

            self.run = wandb.init(mode=mode, entity=entity, project=project, name=run_name, group=group, tags=tags, config=config or {})
            self.enabled = True
        except Exception as e:  # noqa: BLE001 - a W&B failure must never crash an evaluation
            log.warning("W&B init failed (%s) -- continuing without W&B logging", e)

    def log(self, data: dict, step: int | None = None) -> None:
        if self.enabled:
            try:
                self.run.log(data, step=step)
            except Exception as e:  # noqa: BLE001
                log.warning("W&B log() failed: %s", e)

    def set_summary(self, data: dict) -> None:
        """Write values into the run *summary* rather than as a time series.

        For anything measured once -- reference constants, a single full-validation pass -- a curve
        is the wrong shape: W&B draws one point, or a flat line implying it was tracked over time.
        The summary is a table on the run's overview page, which is what a constant is.
        """
        if not self.enabled:
            return
        try:
            for key, value in data.items():
                self.run.summary[key] = value
        except Exception as e:  # noqa: BLE001 - never crash a run over logging
            log.warning("W&B set_summary() failed: %s", e)

    def log_table(self, key: str, columns: list, rows: list, step: int | None = None) -> None:
        """Log a real `wandb.Table` panel.

        `set_summary` puts a value on the run's Overview tab as a key/value entry; it does **not**
        create anything in the workspace. For a set of constants meant to be *read together* -- the
        reference floors and ceilings a curve is judged against -- a Table panel is what actually
        shows up next to the charts.
        """
        if not self.enabled:
            return
        try:
            import wandb

            self.run.log({key: wandb.Table(columns=list(columns), data=[list(r) for r in rows])},
                         step=step)
        except Exception as e:  # noqa: BLE001 - never crash a run over logging
            log.warning("W&B log_table() failed: %s", e)

    def log_artifact(self, path: Path, name: str, artifact_type: str) -> None:
        if not self.enabled:
            return
        try:
            import wandb

            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(str(path))
            self.run.log_artifact(artifact)
        except Exception as e:  # noqa: BLE001
            log.warning("W&B log_artifact() failed: %s", e)

    def log_html(self, key: str, html: str, step: int | None = None) -> None:
        """Log a self-contained HTML panel (`wandb.Html`).

        `inject=False` matters: W&B otherwise wraps the payload in its own stylesheet, which
        overrides the panel's dark-background rules and can leave the slider unstyled and the
        images mis-sized.

        The key is stable across validation steps for a given case, so W&B's own step selector
        becomes the "which validation step" control and the panel's slider is the "which slice"
        control -- the two axes the panel needs, without a custom W&B plugin.
        """
        if not self.enabled:
            return
        try:
            import wandb

            self.run.log({key: wandb.Html(html, inject=False)}, step=step)
        except Exception as e:  # noqa: BLE001
            log.warning("W&B log_html() failed: %s", e)

    def finish(self) -> dict:
        summary = {"wandb_mode_requested": self.mode, "enabled": self.enabled, "run_id": None, "run_url": None, "offline_sync_hint": self.offline_sync_hint}
        if self.enabled:
            try:
                summary["run_id"] = self.run.id
                summary["run_url"] = self.run.url
                if self.mode == "offline":
                    summary["offline_sync_hint"] = f"wandb sync {Path(self.run.dir).parent}"
                self.run.finish()
            except Exception as e:  # noqa: BLE001
                log.warning("W&B finish() failed: %s", e)
        return summary
