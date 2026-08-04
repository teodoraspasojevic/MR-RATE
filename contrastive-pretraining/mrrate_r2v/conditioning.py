"""Conditioning bookkeeping shared by the trainer and the sampler: MR-RATE's modality strings ->
NVIDIA's class ids, report dropout, and classifier-free guidance.

Every number here is NVIDIA's, read from their files:

- the class ids come from `NV-Generate-CTMR/configs/modality_mapping.json`, not from a second copy;
- modality dropout is `scripts/diff_model_train.py:augment_modality_label` imported unchanged, so
  its three-way augmentation (coarsen CT subtype -> 1, coarsen MR subtype -> 8, drop to class 0 with
  probability `prob`) is preserved exactly;
- guidance reduces to `scripts/diff_model_infer.py:206-207`'s formula when the report term is off,
  which `tests/test_r2v_guidance.py` asserts numerically.

Report guidance is the *incremental* effect of the report on top of the modality-conditioned
prediction, so `report_guidance_scale=0` leaves NVIDIA's own behaviour untouched and
`report_guidance_scale=1` is the plain report-conditioned prediction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

# MR-RATE's `modality` column -> the key in NVIDIA's modality_mapping.json. MR-RATE native_space is
# defaced but NOT skull-stripped (`MRReportToVolumeDataset.__getitem__` reports
# skull_state="defaced_not_stripped"), so the `*_skull_stripped` ids are deliberately not used; and
# contrast state is not derivable from the release, so `mri_t1c` is never selected either.
MRRATE_MODALITY_TO_NVIDIA_KEY = {
    "T1w": "mri_t1",
    "T2w": "mri_t2",
    "FLAIR": "mri_flair",
    "SWI": "mri_swi",
}
UNCONDITIONAL_MODALITY_KEY = "unknown"


def load_modality_mapping(path=None) -> dict:
    """NVIDIA's own modality name -> class id table."""
    if path is None:
        from .models.nvidia import DEFAULT_CONFIGS_DIR

        path = DEFAULT_CONFIGS_DIR / "modality_mapping.json"
    return json.loads(Path(path).read_text())


class ModalityEncoder:
    """MR-RATE modality strings -> the `class_labels` tensor the frozen UNet expects."""

    def __init__(self, mapping: Optional[dict] = None) -> None:
        self.mapping = mapping if mapping is not None else load_modality_mapping()
        if UNCONDITIONAL_MODALITY_KEY not in self.mapping:
            raise ValueError(
                f"modality mapping has no '{UNCONDITIONAL_MODALITY_KEY}' entry, so there is no "
                f"unconditional class to use for classifier-free guidance"
            )
        self.null_id = int(self.mapping[UNCONDITIONAL_MODALITY_KEY])

    def id_for(self, modality: str) -> int:
        """Unknown or unmapped modalities fall back to the unconditional class rather than guessing,
        because a wrong class id conditions the frozen model on the wrong sequence."""
        key = MRRATE_MODALITY_TO_NVIDIA_KEY.get(modality)
        if key is None:
            return self.null_id
        return int(self.mapping[key])

    def encode(self, modalities: Sequence[str], device=None) -> torch.Tensor:
        return torch.tensor([self.id_for(m) for m in modalities], dtype=torch.long, device=device)


def augment_modality_label(modality_tensor: torch.Tensor, prob: float = 0.1) -> torch.Tensor:
    """NVIDIA's own modality conditioning-dropout, imported from the vendored trainer.

    Kept as a thin re-export rather than a copy so it cannot drift: see
    `NV-Generate-CTMR/scripts/diff_model_train.py:34-66`.
    """
    from .models import nvidia as _nvidia  # noqa: F401  -- puts the vendored root on sys.path
    from scripts.diff_model_train import augment_modality_label as official

    return official(modality_tensor, prob=prob)


# --------------------------------------------------------------------------- report dropout


@dataclass
class ConditioningConfig:
    """Dropout and guidance knobs. Defaults reproduce NVIDIA's behaviour for the modality path.

    modality_dropout_probability: NVIDIA's own default (`augment_modality_label(prob=0.1)`).
    report_dropout_probability:   0.10 -- the challenge and NVIDIA specify none, so this follows the
        modality path's own rate and the usual 10% classifier-free-guidance convention. Configurable.
    """

    modality_dropout_probability: float = 0.1
    report_dropout_probability: float = 0.1
    report_guidance_scale: float = 4.0
    modality_guidance_scale: float = 10.0  # NVIDIA's `cfg_guidance_scale` for mr-brain
    independent_dropout: bool = True

    def __post_init__(self) -> None:
        for name in ("modality_dropout_probability", "report_dropout_probability"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


def sample_report_drop_mask(
    batch_size: int, probability: float, device=None, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """`(B,)` bool, True = replace this sample's report with the learned null embedding.

    The null representation is `ReportConditionedUNetMaisi.null_context`, reached through
    `forward(context_drop_mask=...)`, so training and inference drop the report through *the same*
    code path -- and an all-padding row is never produced, so the attention softmax cannot go NaN.
    """
    if probability <= 0.0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if probability >= 1.0:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    noise = torch.rand(batch_size, generator=generator, device=None if generator is None else generator.device)
    return (noise < probability).to(device=device)


# --------------------------------------------------------------------------- guidance


def combine_guidance(
    prediction_null_null: Optional[torch.Tensor],
    prediction_modality_null: torch.Tensor,
    prediction_modality_report: Optional[torch.Tensor],
    modality_guidance_scale: float,
    report_guidance_scale: float,
) -> torch.Tensor:
    """The guided model output, in the same space and at the same point in the loop as NVIDIA's.

    Hierarchical form, with the report as an increment on top of the modality-conditioned branch:

        D_guided = D_00 + s_mod * (D_m0 - D_00) + s_rep * (D_mr - D_m0)

    - `report_guidance_scale=0` collapses to `D_00 + s_mod * (D_m0 - D_00)`, which *is*
      `diff_model_infer.py:207`'s `model_uncond + scale * (model_t - model_uncond)` with the report
      always null. NVIDIA's behaviour is recovered exactly, not approximately.
    - `report_guidance_scale=1` with `modality_guidance_scale=1` gives plainly `D_mr`.
    - `prediction_null_null=None` means modality guidance is off: the modality stays conditioned and
      only the report is guided, `D_m0 + s_rep * (D_mr - D_m0)`.
    """
    if prediction_null_null is None:
        guided = prediction_modality_null
    else:
        guided = prediction_null_null + modality_guidance_scale * (prediction_modality_null - prediction_null_null)
    if prediction_modality_report is not None and report_guidance_scale != 0.0:
        guided = guided + report_guidance_scale * (prediction_modality_report - prediction_modality_null)
    return guided


@dataclass
class GuidanceBranches:
    """Which forward passes a given guidance setting needs, in a fixed order so the batched call and
    the explicit calls cannot disagree."""

    use_null_null: bool
    use_modality_report: bool

    @property
    def n_branches(self) -> int:
        return 1 + int(self.use_null_null) + int(self.use_modality_report)

    @staticmethod
    def resolve(modality_guidance_scale: float, report_guidance_scale: float) -> "GuidanceBranches":
        return GuidanceBranches(
            use_null_null=modality_guidance_scale != 0.0,
            use_modality_report=report_guidance_scale != 0.0,
        )


def guided_model_output(
    unet,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor,
    spacing_tensor: torch.Tensor,
    context: Optional[torch.Tensor],
    context_mask: Optional[torch.Tensor],
    config: ConditioningConfig,
    modality_null_id: int,
    batched: bool = True,
    extra_unet_inputs: Optional[dict] = None,
) -> torch.Tensor:
    """One guided prediction. `batched=True` concatenates the branches into a single UNet call, the
    way `diff_model_infer.py:200-206` concatenates its two; `batched=False` runs them separately and
    must give the same numbers (asserted in the tests).

    Branch order is always: (modality, null_report), then (null_modality, null_report) if modality
    guidance is on, then (modality, report) if report guidance is on.
    """
    branches = GuidanceBranches.resolve(config.modality_guidance_scale, config.report_guidance_scale)
    batch = x.shape[0]
    extra = dict(extra_unet_inputs or {})

    labels = [class_labels]
    drops = [torch.ones(batch, dtype=torch.bool, device=x.device)]
    if branches.use_null_null:
        labels.append(torch.full_like(class_labels, modality_null_id))
        drops.append(torch.ones(batch, dtype=torch.bool, device=x.device))
    if branches.use_modality_report:
        labels.append(class_labels)
        drops.append(torch.zeros(batch, dtype=torch.bool, device=x.device))

    n = len(labels)

    def tiled(tensor, times):
        return tensor if times == 1 else torch.cat([tensor] * times)

    def shared_inputs(times):
        inputs = dict(x=tiled(x, times), timesteps=tiled(timesteps, times),
                      spacing_tensor=tiled(spacing_tensor, times))
        inputs.update({key: tiled(value, times) for key, value in extra.items()})
        inputs["context"] = None if context is None else tiled(context, times)
        inputs["context_mask"] = None if context_mask is None else tiled(context_mask, times)
        return inputs

    if batched and n > 1:
        inputs = shared_inputs(n)
        inputs.update(class_labels=torch.cat(labels), context_drop_mask=torch.cat(drops))
        predictions = list(unet(**inputs).chunk(n))
    else:
        predictions = []
        for label, drop in zip(labels, drops):
            inputs = shared_inputs(1)
            inputs.update(class_labels=label, context_drop_mask=drop)
            predictions.append(unet(**inputs))

    prediction_modality_null = predictions[0]
    index = 1
    prediction_null_null = None
    if branches.use_null_null:
        prediction_null_null = predictions[index]
        index += 1
    prediction_modality_report = predictions[index] if branches.use_modality_report else None
    return combine_guidance(
        prediction_null_null, prediction_modality_null, prediction_modality_report,
        config.modality_guidance_scale, config.report_guidance_scale,
    )


__all__ = [
    "ConditioningConfig",
    "GuidanceBranches",
    "MRRATE_MODALITY_TO_NVIDIA_KEY",
    "ModalityEncoder",
    "UNCONDITIONAL_MODALITY_KEY",
    "augment_modality_label",
    "combine_guidance",
    "guided_model_output",
    "load_modality_mapping",
    "sample_report_drop_mask",
]
