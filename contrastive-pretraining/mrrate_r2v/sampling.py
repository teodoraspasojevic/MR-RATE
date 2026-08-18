"""Report-to-volume sampling. `NV-Generate-CTMR/scripts/diff_model_infer.py:run_inference` with the
report added, step for step:

| official (`scripts/diff_model_infer.py`)                | here                                |
|---------------------------------------------------------|-------------------------------------|
| latent noise shape `dim // divisor` (:139-148)          | `ReportToVolumeSampler.sample`      |
| `RFlowScheduler.set_timesteps(steps, input_img_size_numel)` (:151-158) | same                  |
| `all_next_timesteps = cat(timesteps[1:], [0])` (:165)   | same                                |
| `spacing_tensor` `* 1e2`, `.half()` (:92-96)            | `spacing_tensor_for`                |
| `autocast("cuda", enabled=True)` around the loop (:171) | same                                |
| batch-concatenated CFG then `.chunk` (:196-207)         | `conditioning.guided_model_output`  |
| `scheduler.step(model_output, t, image, next_t)` (:212) | same                                |
| `ReconModel` + `SlidingWindowInferer(roi 80^3, gaussian, overlap 0.4)` (:213-224) | same      |
| MR postprocessing to int16 `[0, 1000]` (:225-234)       | `postprocess_mr`                    |
| affine = diag(spacing) NIfTI write (:250-259)           | `save_volume`                       |

The only substantive change is which branches CFG evaluates: official varies the modality class
label alone, this varies the report as well. With `report_guidance_scale=0` the arithmetic is
identical to official (asserted in `tests/test_r2v_guidance.py`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from monai.networks.schedulers.rectified_flow import RFlowScheduler
from torch.amp import autocast

from .conditioning import ConditioningConfig, ModalityEncoder, guided_model_output
from .text import encode_reports

log = logging.getLogger("mrrate_r2v.sampling")

MR_OUTPUT_RANGE = (0, 1000)  # diff_model_infer.py:227 -- MR branch of the official postprocessing


def official_latent_divisor(num_channels) -> int:
    """`diff_model_infer.py:300-308`: the output-size to latent-size ratio, derived from the *UNet's*
    depth -- `2 ** (len(num_channels) - 2)`, i.e. 4 for the mr-brain model.

    This is NOT `models.nvidia.required_spatial_divisor` (compression x num_splits = 16), which is the
    padding an *encode* needs. Confusing the two silently produces a volume 4x too small in every
    axis: the run succeeds, the file is valid, and it is the wrong size.
    """
    return 2 ** (max(1, len(num_channels)) - 2)


def spacing_tensor_for(spacing_mm: Sequence[float], batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
    """`diff_model_infer.py:92,96`: spacing in mm times 1e2. Official casts to `.half()` because its
    whole loop runs under autocast; the dtype is left to the caller here so a CPU test can run in
    float32 without a separate code path."""
    tensor = torch.tensor(list(spacing_mm), dtype=torch.float32, device=device) * 1e2
    return tensor.unsqueeze(0).repeat(batch_size, 1).to(dtype)


def postprocess_mr(data: np.ndarray) -> np.ndarray:
    """`diff_model_infer.py:225-234`, MR branch: the decoder's [0, 1] output back to the [0, 1000]
    range MR-RATE's `percentile` normalizer mapped away, clipped below at 0, as int16."""
    a_min, a_max, b_min, b_max = MR_OUTPUT_RANGE[0], MR_OUTPUT_RANGE[1], 0, 1
    data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
    return np.int16(np.clip(data, a_min, None))


def save_volume(data: np.ndarray, spacing_mm: Sequence[float], output_path) -> None:
    """`diff_model_infer.py:250-259`: an axis-aligned affine whose diagonal is the spacing."""
    import os

    import nibabel as nib

    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = float(spacing_mm[i])
    os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine=affine), str(output_path))


@dataclass
class SamplerConfig:
    """Sampling knobs. Defaults are NVIDIA's own from
    `configs/config_maisi_diff_model_rflow-mr-brain.json` (`diffusion_unet_inference`)."""

    num_inference_steps: int = 30
    latent_channels: int = 4
    random_seed: int = 1234
    autocast: bool = True
    recon_roi_size: tuple = (80, 80, 80)
    recon_overlap: float = 0.4
    batched_guidance: bool = True


class ReportToVolumeSampler:
    """Frozen autoencoder + report-conditioned UNet -> one synthetic volume per report."""

    def __init__(
        self,
        unet,
        autoencoder,
        text_embedder,
        noise_scheduler,
        scale_factor: float,
        divisor: int,
        device,
        conditioning: Optional[ConditioningConfig] = None,
        sampler_config: Optional[SamplerConfig] = None,
        modality_encoder: Optional[ModalityEncoder] = None,
    ) -> None:
        self.unet = unet.eval()
        self.autoencoder = autoencoder.eval() if autoencoder is not None else None
        self.text_embedder = text_embedder
        self.noise_scheduler = noise_scheduler
        self.scale_factor = float(scale_factor)
        self.divisor = int(divisor)
        self.device = torch.device(device)
        self.conditioning = conditioning or ConditioningConfig()
        self.config = sampler_config or SamplerConfig()
        self.modality_encoder = modality_encoder or ModalityEncoder()

    # -- the loop ------------------------------------------------------------------------

    @torch.no_grad()
    def sample_latent(
        self,
        report: str,
        modality: str,
        shape_xyz: Sequence[int],
        spacing_mm: Sequence[float],
        seed: Optional[int] = None,
        report_sections: Optional[dict] = None,
    ) -> torch.Tensor:
        for axis, size in enumerate(shape_xyz):
            if size % self.divisor:
                raise ValueError(
                    f"output shape {tuple(shape_xyz)} is not divisible by the latent divisor "
                    f"{self.divisor} on axis {axis}; NVIDIA's own inference has the same constraint "
                    "(diff_model_infer.py:139-148)"
                )
        if seed is not None:
            from .models.nvidia import set_random_seed

            set_random_seed(seed)

        latent_shape = (1, self.config.latent_channels, *(int(s) // self.divisor for s in shape_xyz))
        latent = torch.randn(latent_shape, device=self.device)

        if isinstance(self.noise_scheduler, RFlowScheduler):
            self.noise_scheduler.set_timesteps(
                num_inference_steps=self.config.num_inference_steps,
                input_img_size_numel=int(torch.prod(torch.tensor(latent.shape[2:]))),
            )
        else:
            self.noise_scheduler.set_timesteps(num_inference_steps=self.config.num_inference_steps)

        timesteps = self.noise_scheduler.timesteps
        next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))
        # The same seam the trainer uses, so a configuration that encodes findings and impression
        # separately gets per-section text here too. Passing only `report_text` to a sectioned
        # embedder would silently condition on one token instead of two.
        conditioning = encode_reports(
            self.text_embedder,
            {"report_text": [report],
             "report_sections_text": None if report_sections is None else [report_sections]},
            self.device,
        )
        class_labels = self.modality_encoder.encode([modality], device=self.device)
        spacing = spacing_tensor_for(spacing_mm, 1, self.device)

        amp = self.config.autocast and self.device.type == "cuda"
        with autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=amp):
            for t, next_t in zip(timesteps, next_timesteps):
                model_output = guided_model_output(
                    self.unet,
                    x=latent,
                    timesteps=torch.tensor((t,), device=self.device).float(),
                    class_labels=class_labels,
                    spacing_tensor=spacing,
                    context=conditioning.token_embeddings,
                    context_mask=conditioning.attention_mask,
                    config=self.conditioning,
                    modality_null_id=self.modality_encoder.null_id,
                    batched=self.config.batched_guidance,
                )
                if isinstance(self.noise_scheduler, RFlowScheduler):
                    latent, _ = self.noise_scheduler.step(model_output, t, latent, next_t)
                else:
                    latent, _ = self.noise_scheduler.step(model_output, t, latent)
        # Under autocast the loop leaves a half-precision latent. Official keeps the decode inside the
        # same autocast block (diff_model_infer.py:171-224); `decode` re-enters it, so the latent is
        # returned in float32 to give this function one predictable dtype.
        return latent.float()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> np.ndarray:
        """`diff_model_infer.py:213-224`: NVIDIA's `ReconModel` (which divides by `scale_factor`
        before decoding) driven by their sliding-window inferer at the same settings."""
        from .models.nvidia import ReconModel, dynamic_infer

        recon = ReconModel(autoencoder=self.autoencoder, scale_factor=self.scale_factor).to(self.device)
        inferer = SlidingWindowInferer(
            roi_size=list(self.config.recon_roi_size),
            sw_batch_size=1,
            progress=False,
            mode="gaussian",
            overlap=self.config.recon_overlap,
            sw_device=self.device,
            device=self.device,
        )
        amp = self.config.autocast and self.device.type == "cuda"
        with autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=amp):
            volume = dynamic_infer(inferer, recon, latent)
        return volume.squeeze().float().cpu().numpy()

    def generate(
        self,
        report_text: str,
        shape,
        spacing_mm,
        seed: Optional[int] = None,
        modality: str = "T1w",
        report_sections: Optional[dict] = None,
        postprocess: bool = True,
    ) -> np.ndarray:
        """Report -> `np.ndarray[X, Y, Z]` on the given grid.

        `report_sections` is required by a sectioned-fusion configuration and ignored by every
        other one; `encode_reports` raises rather than guessing if it is missing when needed.

        **`postprocess` decides which intensity space comes out, and the two are 1000x apart.**

        - `True` (the default) applies `postprocess_mr`: NVIDIA's own int16 `[0, 1000]` MR range,
          which is what a `.nii.gz` written by `cli.generate_r2v` must carry to match what their
          `diff_model_infer.py` produces.
        - `False` returns the decoder's native float `~[0, 1]`, which is the *cohort's* space --
          the percentile-normalised model input every ground-truth volume is stored in.

        Anything that will be compared against a cohort volume must pass `False`.
        `cli.evaluate` does, `validation.py` bypasses this method for the same reason
        (`_check_intensity_space` exists because the mistake is invisible: every metric consumes a
        1000x-offset pair happily and returns a plausible number).
        """
        latent = self.sample_latent(report_text, modality, shape, spacing_mm, seed=seed,
                                    report_sections=report_sections)
        if self.autoencoder is None:
            raise RuntimeError("no autoencoder was provided, so a latent cannot be decoded to a volume")
        volume = self.decode(latent)
        return postprocess_mr(volume) if postprocess else volume.astype(np.float32, copy=False)


__all__ = [
    "MR_OUTPUT_RANGE",
    "ReportToVolumeSampler",
    "SamplerConfig",
    "official_latent_divisor",
    "postprocess_mr",
    "save_volume",
    "spacing_tensor_for",
]
