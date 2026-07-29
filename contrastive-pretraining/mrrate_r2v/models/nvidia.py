"""The ONLY module in this package that touches NVIDIA's model-loading code -- and it imports
exclusively from this repository's OWN vendored copy (`NV-Generate-CTMR/` at the repo root, added
by commit `cf5cf1f`, Apache-2.0 licensed, `LICENSE`/`LICENSE.weights` intact). No other file in
`evaluation/` imports `scripts.diff_model_infer` or anything else from that directory directly --
this is the single seam, so a future change to the vendored copy's layout only requires touching
this one file. This module never imports from any OTHER repository (in particular, never from
`~/NV-Generate-CTMR`, the sibling fork used only as reference material for this package's design
-- see docs/design/archive/09_older_evaluation_implementation_audit.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../MR-RATE
VENDORED_NVIDIA_ROOT = REPO_ROOT / "NV-Generate-CTMR"
DEFAULT_CONFIGS_DIR = VENDORED_NVIDIA_ROOT / "configs"

if not VENDORED_NVIDIA_ROOT.is_dir():
    raise FileNotFoundError(
        f"Vendored NVIDIA code not found at {VENDORED_NVIDIA_ROOT}. This repository is expected to "
        "carry its own copy (added by commit cf5cf1f, 'feat: add NVIDIA-MRI-Brain official code') -- "
        "this package never imports NVIDIA's model-loading code from any other repository."
    )
if str(VENDORED_NVIDIA_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_NVIDIA_ROOT))

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import MaisiDownsample  # noqa: E402
from scripts.diff_model_infer import load_models, prepare_tensors, run_inference, set_random_seed  # noqa: E402
from scripts.diff_model_setting import load_config  # noqa: E402
from scripts.utils import define_instance  # noqa: E402

DEFAULT_ENV_CONFIG = DEFAULT_CONFIGS_DIR / "environment_maisi_diff_model_rflow-mr-brain.json"
DEFAULT_MODEL_CONFIG = DEFAULT_CONFIGS_DIR / "config_maisi_diff_model_rflow-mr-brain.json"
DEFAULT_NETWORK_CONFIG = DEFAULT_CONFIGS_DIR / "config_network_rflow.json"


def required_spatial_divisor(autoencoder, cfg_args) -> int:
    """The encoder's raw 2^n_downsample compression factor is NOT a sufficient padding target --
    `MaisiConvolution`'s memory-saving conv-splitting also requires divisibility by `num_splits`
    at the bottleneck resolution (ported rationale, verified in the older implementation's own
    empirical test, docs/design/archive/09_older_evaluation_implementation_audit.md §15 -- this function's logic is unchanged from there,
    only its call site differs).
    """
    n_downsamples = sum(1 for m in autoencoder.encoder.modules() if isinstance(m, MaisiDownsample))
    compression = 2**n_downsamples
    num_splits = cfg_args.autoencoder_def.get("num_splits", 1)
    return compression * max(num_splits, 1)


def load_autoencoder(checkpoint_path, env_config=DEFAULT_ENV_CONFIG, model_config=DEFAULT_MODEL_CONFIG, network_config=DEFAULT_NETWORK_CONFIG, device: str = "cuda"):
    """Loads ONLY the autoencoder (VAE) -- for VAE-reconstruction evaluation, which never needs
    the diffusion UNet. Returns (autoencoder, cfg_args, required_divisor).
    """
    import torch

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


def load_autoencoder_and_unet(env_config=DEFAULT_ENV_CONFIG, model_config=DEFAULT_MODEL_CONFIG, network_config=DEFAULT_NETWORK_CONFIG, device: str = "cuda", autoencoder_checkpoint_override=None):
    """For unconditional-generation evaluation (needs both). Returns (autoencoder, unet,
    scale_factor, cfg_args) via NVIDIA's own `load_models`, unmodified.
    """
    import logging

    cfg_args = load_config(str(env_config), str(model_config), str(network_config))
    if autoencoder_checkpoint_override is not None:
        cfg_args.trained_autoencoder_path = str(autoencoder_checkpoint_override)
    logger = logging.getLogger("nvidia_model")
    autoencoder, unet, scale_factor = load_models(cfg_args, device, logger)
    autoencoder.eval()
    unet.eval()
    return autoencoder, unet, scale_factor, cfg_args


__all__ = [
    "VENDORED_NVIDIA_ROOT", "DEFAULT_ENV_CONFIG", "DEFAULT_MODEL_CONFIG", "DEFAULT_NETWORK_CONFIG",
    "load_models", "prepare_tensors", "run_inference", "set_random_seed", "load_config", "define_instance",
    "required_spatial_divisor", "load_autoencoder", "load_autoencoder_and_unet",
]
