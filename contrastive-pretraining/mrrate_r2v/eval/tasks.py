"""Which metrics are valid for which task. One table, consulted by one runner.

You pass `--task` to `cli/evaluate.py`; this decides what gets computed. Nothing else in the
package chooses metrics, so a task's metric set cannot differ between two runs.

    task              paired?  metric groups
    ----------------  -------  -----------------------------------------------
    reconstruction    yes      fidelity, perceptual, distribution
    report2volume     yes      fidelity, perceptual, distribution, report_alignment
    generation        no       distribution

**Why `generation` gets no fidelity metrics.** An unconditional generator is told "make a T1w
brain" and nothing about any patient, so no specific real volume is the answer. Computing MAE
or SSIM against an arbitrary real scan would measure "how different are two random brains,"
not generation quality. The runner never computes them for that task -- this is a property of
the task, declared here, not a flag someone can forget to pass.

To add a task: add a `TaskSpec` here and, if it needs a new metric family, a group in
`METRIC_GROUPS`. The runner picks it up with no changes.
"""
from __future__ import annotations

from dataclasses import dataclass

# Metric group -> what it measures and what it needs. `needs_pair` groups are skipped
# automatically for unpaired tasks.
METRIC_GROUPS = {
    "fidelity": {
        "needs_pair": True,
        "what": "voxelwise agreement with the ground-truth volume (MAE/MSE/PSNR/NCC/SSIM)",
    },
    "perceptual": {
        "needs_pair": True,
        "what": "is fine detail preserved or blurred (edge preservation, Laplacian variance, "
                "high-frequency energy, per-plane SSIM)",
    },
    "distribution": {
        "needs_pair": False,
        "what": "do the real and produced populations look alike (MedicalNet FID, 2.5D "
                "Inception FID, Inception Score, precision/recall/density/coverage)",
    },
    "report_alignment": {
        "needs_pair": True,
        "what": "does the volume match what the report says",
    },
}


@dataclass(frozen=True)
class TaskSpec:
    """One evaluation task."""

    name: str
    paired: bool
    metric_groups: tuple
    summary: str
    unpaired_reason: str | None = None

    def groups_to_run(self, distribution_enabled: bool = True, skip=()) -> tuple:
        """The metric groups this run will actually compute.

        `skip` drops groups the task declares -- e.g. `skip=("perceptual",)` to save the ~40% of
        per-case time the detail-preservation metrics cost. It can only ever *remove*: a group the
        task does not declare cannot be added, so `generation` still cannot acquire a voxelwise
        metric by any combination of flags.

        Whatever this returns is recorded as `metric_groups_computed` in `summary.json`, so a run
        with a group skipped is never mistaken for a full one.
        """
        skip = set(skip or ())
        groups = [g for g in self.metric_groups
                  if not (METRIC_GROUPS[g]["needs_pair"] and not self.paired)]
        if not distribution_enabled:
            groups = [g for g in groups if g != "distribution"]
        return tuple(g for g in groups if g not in skip)


TASKS = {
    "reconstruction": TaskSpec(
        name="reconstruction",
        paired=True,
        metric_groups=("fidelity", "perceptual", "distribution"),
        summary="A model encoded a real volume and decoded it back. The ground truth is that "
                "exact input, so every metric applies.",
    ),
    "report2volume": TaskSpec(
        name="report2volume",
        paired=True,
        metric_groups=("fidelity", "perceptual", "distribution", "report_alignment"),
        summary="A model generated a volume from a report. The ground truth is the real series "
                "that report describes, so paired metrics apply -- but expect much weaker "
                "voxelwise agreement than reconstruction, since nothing constrains anatomy "
                "beyond the text.",
    ),
    "generation": TaskSpec(
        name="generation",
        paired=False,
        metric_groups=("distribution",),
        summary="A model generated volumes from a modality label alone. Only population-level "
                "metrics are meaningful.",
        unpaired_reason="unconditional generation -- no real patient corresponds to any "
                        "generated volume",
    ),
}

TASK_NAMES = tuple(TASKS)


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise SystemExit(f"unknown --task {name!r}; choose from {list(TASK_NAMES)}")
    return TASKS[name]


# The paired metric names the aggregator summarizes, grouped as above. `runner.py` computes
# exactly these -- keep the two in step, and a test asserts they match.
FIDELITY_METRICS = (
    "mae_whole", "mse_whole", "psnr_whole", "ncc_whole", "ssim3d_whole",
    "mae_fg", "mse_fg", "psnr_fg", "ncc_fg", "relative_intensity_error_fg",
)
PERCEPTUAL_METRICS = (
    "edge_preservation_fg", "laplacian_variance_ratio_fg", "hf_energy_ratio",
    "ssim2d_sagittal_mean", "ssim2d_coronal_mean", "ssim2d_axial_mean",
)
REPORT_ALIGNMENT_METRICS = ("report_image_similarity_score",)

GROUP_METRIC_NAMES = {
    "fidelity": FIDELITY_METRICS,
    "perceptual": PERCEPTUAL_METRICS,
    "report_alignment": REPORT_ALIGNMENT_METRICS,
}


def paired_metric_names(groups) -> list:
    """Flat list of paired metric names for the given groups, in a stable order."""
    names = []
    for g in groups:
        names.extend(GROUP_METRIC_NAMES.get(g, ()))
    return names
