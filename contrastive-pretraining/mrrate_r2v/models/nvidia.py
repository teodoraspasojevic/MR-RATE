"""The only module in this package that touches NVIDIA's model-loading code.

NV-Generate-CTMR (github.com/nvidia-medtech/NV-Generate-CTMR, Apache-2.0) ships as a plain checkout
with no installable package -- there is no `pip install`/PyPI name to depend on. Almost none of the
actual model code comes from it anyway: the diffusion UNet, autoencoder, and scheduler are all
`monai` classes (an ordinary pip dependency, see `requirements.txt`). What NV-Generate-CTMR
contributes on top of that is a handful of small functions -- a config loader, a `monai.bundle`
instantiation wrapper, and NVIDIA's own unconditional sampling loop -- copied verbatim below (marked
`# -- verbatim from NV-Generate-CTMR ... --`) rather than reimplemented, specifically so this module
has no runtime dependency on a local checkout of that repo. `report_conditioned_unet.py` reads the
pretrained architecture's geometry from `nvidia_configs/config_network_rflow.json` (also copied
verbatim) rather than restating those numbers, for the same reason `nvidia_unet_kwargs()` never
hardcodes them.

See `models/README.md` for why this stays a single seam.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import MaisiDownsample
from monai.bundle import ConfigParser
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism

DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent / "nvidia_configs"
DEFAULT_ENV_CONFIG = DEFAULT_CONFIGS_DIR / "environment_maisi_diff_model_rflow-mr-brain.json"
DEFAULT_MODEL_CONFIG = DEFAULT_CONFIGS_DIR / "config_maisi_diff_model_rflow-mr-brain.json"
DEFAULT_NETWORK_CONFIG = DEFAULT_CONFIGS_DIR / "config_network_rflow.json"


# =============================================================================================
# Verbatim from NV-Generate-CTMR (github.com/nvidia-medtech/NV-Generate-CTMR), Apache License 2.0,
# Copyright (c) MONAI Consortium -- full license text at NV-Generate-CTMR/LICENSE in this repo, or
# http://www.apache.org/licenses/LICENSE-2.0. Only mechanical changes: inlined into one module so
# nothing here depends on that repo being checked out, and each function's own relative imports
# (e.g. `from .utils import define_instance`) resolved to the copies below. No logic changed.
#
#   set_random_seed, load_models, prepare_tensors, run_inference   scripts/diff_model_infer.py
#   load_config                                                    scripts/diff_model_setting.py
#   define_instance, dynamic_infer                                 scripts/utils.py
#   ReconModel                                                      scripts/utils_infer.py
#   augment_modality_label                                          scripts/diff_model_train.py
# =============================================================================================


def set_random_seed(seed: int) -> int:
    """Set random seed for reproducibility."""
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def load_config(env_config_path: str, model_config_path: str, model_def_path: str) -> argparse.Namespace:
    """Load configuration from JSON files."""
    args = argparse.Namespace()

    with open(env_config_path) as f:
        env_config = json.load(f)
    for k, v in env_config.items():
        setattr(args, k, v)

    with open(model_config_path) as f:
        model_config = json.load(f)
    for k, v in model_config.items():
        setattr(args, k, v)

    with open(model_def_path) as f:
        model_def = json.load(f)
    for k, v in model_def.items():
        setattr(args, k, v)

    return args


def define_instance(args: argparse.Namespace, instance_def_key: str):
    """Define and instantiate an object based on the provided arguments and instance definition key."""
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def dynamic_infer(inferer, model, images):
    """Perform dynamic inference using a model and an inferer, typically a monai SlidingWindowInferer.

    Determines whether to use the model directly or the provided inferer, based on input size.
    """
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)
    else:
        spatial_dims = images.shape[2:]
        orig_roi = inferer.roi_size
        if len(orig_roi) != len(spatial_dims):
            raise ValueError(f"ROI length ({len(orig_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")
        adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
        inferer.roi_size = adjusted_roi
        output = inferer(network=model, inputs=images)
        inferer.roi_size = orig_roi
        return output


class ReconModel(torch.nn.Module):
    """Reconstructs images from latent representations via the (frozen) autoencoder."""

    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        return self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)


def load_models(args: argparse.Namespace, device, logger: logging.Logger) -> tuple:
    """Load the autoencoder and UNet models."""
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path)
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    logger.info(f"checkpoints {args.trained_autoencoder_path} loaded.")

    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(f"{args.model_dir}/{args.model_filename}", map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
    logger.info(f"checkpoints {args.model_dir}/{args.model_filename} loaded.")

    scale_factor = checkpoint["scale_factor"]
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, scale_factor


def prepare_tensors(args: argparse.Namespace, device) -> tuple:
    """Prepare necessary tensors for inference."""
    top_region_index_tensor = np.array(args.diffusion_unet_inference["top_region_index"]).astype(float) * 1e2
    bottom_region_index_tensor = np.array(args.diffusion_unet_inference["bottom_region_index"]).astype(float) * 1e2
    spacing_tensor = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2

    top_region_index_tensor = torch.from_numpy(top_region_index_tensor[np.newaxis, :]).half().to(device)
    bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor[np.newaxis, :]).half().to(device)
    spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).half().to(device)
    modality_tensor = args.diffusion_unet_inference["modality"] * torch.ones((len(spacing_tensor)), dtype=torch.long).to(device)

    return top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor


def run_inference(
    args: argparse.Namespace,
    device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: float,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    modality_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Run the inference to generate synthetic images."""
    from tqdm import tqdm

    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    noise = torch.randn(
        (
            1,
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    unet.eval()

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    progress_bar = tqdm(
        zip(all_timesteps, all_next_timesteps),
        total=min(len(all_timesteps), len(all_next_timesteps)),
    )
    cfg_guidance_scale = args.cfg_guidance_scale
    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in progress_bar:
            unet_inputs = {
                "x": image,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
            }

            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )

            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )

            if cfg_guidance_scale > 0:
                for k in unet_inputs.keys():
                    if k != "class_labels":
                        unet_inputs[k] = torch.cat([unet_inputs[k]] * 2)
                    else:
                        unet_inputs[k] = torch.cat([unet_inputs[k], torch.zeros_like(modality_tensor)])
            if cfg_guidance_scale == 0:
                model_output = unet(**unet_inputs)
            else:
                model_t, model_uncond = unet(**unet_inputs).chunk(2)
                model_output = model_uncond + cfg_guidance_scale * (model_t - model_uncond)

            if not isinstance(noise_scheduler, RFlowScheduler):
                image, _ = noise_scheduler.step(model_output, t, image)  # type: ignore
            else:
                image, _ = noise_scheduler.step(model_output, t, image, next_t)  # type: ignore

        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.4,
            sw_device=device,
            device=device,
        )
        synthetic_images = dynamic_infer(inferer, recon_model, image)
        data = synthetic_images.squeeze().cpu().detach().numpy()
        modality = int(modality_tensor.cpu().item())
        if modality >= 8:
            a_min, a_max, b_min, b_max = 0, 1000, 0, 1  # MR
            data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
            data = np.clip(data, a_min, None)
        else:
            a_min, a_max, b_min, b_max = -1000, 1000, 0, 1  # CT
            data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
            data = np.clip(data, a_min, a_max)
        return np.int16(data)


def augment_modality_label(modality_tensor: torch.Tensor, prob: float = 0.1) -> torch.Tensor:
    """Augments the modality tensor by randomly modifying certain elements based on `prob`."""
    mask_ct = (modality_tensor < 8) & (modality_tensor >= 2)
    prob_ct = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[mask_ct & prob_ct] = 1

    mask_mri = modality_tensor >= 9
    prob_mri = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[mask_mri & prob_mri] = 8

    mask_zero = torch.rand(modality_tensor.size(), device=modality_tensor.device) > prob
    modality_tensor = modality_tensor * mask_zero.long()

    return modality_tensor


# =============================================================================================
# Our own code again.
# =============================================================================================


def required_spatial_divisor(autoencoder, cfg_args) -> int:
    """The padding target for a volume before VAE encoding: the encoder's 2^n_downsample
    compression factor, times `num_splits` (`MaisiConvolution`'s memory-saving conv-splitting also
    requires divisibility by that, at the bottleneck resolution). See `models/README.md` for how
    this differs from the sampling-time divisor.
    """
    n_downsamples = sum(1 for m in autoencoder.encoder.modules() if isinstance(m, MaisiDownsample))
    compression = 2**n_downsamples
    num_splits = cfg_args.autoencoder_def.get("num_splits", 1)
    return compression * max(num_splits, 1)


def load_autoencoder(checkpoint_path, env_config=DEFAULT_ENV_CONFIG, model_config=DEFAULT_MODEL_CONFIG, network_config=DEFAULT_NETWORK_CONFIG, device: str = "cuda"):
    """Loads only the autoencoder (VAE), for callers that never need the diffusion UNet.
    Returns (autoencoder, cfg_args, required_divisor).
    """
    cfg_args = load_config(str(env_config), str(model_config), str(network_config))
    autoencoder = define_instance(cfg_args, "autoencoder_def").to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "unet_state_dict" in checkpoint:
        checkpoint = checkpoint["unet_state_dict"]
    missing, unexpected = autoencoder.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/architecture mismatch: missing={missing} unexpected={unexpected}")
    autoencoder.eval()
    return autoencoder, cfg_args, required_spatial_divisor(autoencoder, cfg_args)


def load_autoencoder_and_unet(env_config=DEFAULT_ENV_CONFIG, model_config=DEFAULT_MODEL_CONFIG,
                              network_config=DEFAULT_NETWORK_CONFIG, device: str = "cuda",
                              autoencoder_checkpoint_override=None, unet_checkpoint_override=None):
    """For generation (needs both nets). Returns (autoencoder, unet, scale_factor, cfg_args) via
    NVIDIA's own `load_models`, unmodified.

    Both checkpoint paths must be overridden with absolute paths: NVIDIA's shipped env config
    stores them relative to the current working directory (`model_dir="./models"`), which
    `load_models` resolves as-is. `unet_checkpoint_override` is split into the `model_dir`/
    `model_filename` pair `load_models` reads, with `existing_ckpt_filepath` kept consistent.
    """
    cfg_args = load_config(str(env_config), str(model_config), str(network_config))
    if autoencoder_checkpoint_override is not None:
        cfg_args.trained_autoencoder_path = str(autoencoder_checkpoint_override)
    if unet_checkpoint_override is not None:
        unet_path = Path(unet_checkpoint_override).resolve()
        if not unet_path.is_file():
            raise FileNotFoundError(f"diffusion UNet checkpoint not found: {unet_path}")
        cfg_args.model_dir = str(unet_path.parent)
        cfg_args.model_filename = unet_path.name
        cfg_args.existing_ckpt_filepath = str(unet_path)
    logger = logging.getLogger("nvidia_model")
    autoencoder, unet, scale_factor = load_models(cfg_args, device, logger)
    autoencoder.eval()
    unet.eval()
    return autoencoder, unet, scale_factor, cfg_args


def default_unet_filename(env_config=DEFAULT_ENV_CONFIG) -> str:
    """The UNet checkpoint filename NVIDIA's env config expects, so a caller can look for it next
    to the autoencoder checkpoint instead of hardcoding the name."""
    return json.loads(Path(env_config).read_text()).get(
        "model_filename", "diff_unet_3d_rflow-mr-brain_v0.pt")


__all__ = [
    "DEFAULT_CONFIGS_DIR", "DEFAULT_ENV_CONFIG", "DEFAULT_MODEL_CONFIG", "DEFAULT_NETWORK_CONFIG",
    "load_models", "prepare_tensors", "run_inference", "set_random_seed", "load_config", "define_instance",
    "augment_modality_label", "required_spatial_divisor", "load_autoencoder", "load_autoencoder_and_unet",
    "default_unet_filename",
]
