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
