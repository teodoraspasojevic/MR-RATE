"""The only module in this package that imports NVIDIA's model-loading code -- and only from this
repo's own vendored copy (`NV-Generate-CTMR/` at the repo root). See `models/README.md` for why
this is kept as a single seam.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../MR-RATE
VENDORED_NVIDIA_ROOT = REPO_ROOT / "NV-Generate-CTMR"
DEFAULT_CONFIGS_DIR = VENDORED_NVIDIA_ROOT / "configs"

if not VENDORED_NVIDIA_ROOT.is_dir():
    raise FileNotFoundError(f"Vendored NVIDIA code not found at {VENDORED_NVIDIA_ROOT}.")
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
    import logging
    from pathlib import Path as _Path

    cfg_args = load_config(str(env_config), str(model_config), str(network_config))
    if autoencoder_checkpoint_override is not None:
        cfg_args.trained_autoencoder_path = str(autoencoder_checkpoint_override)
    if unet_checkpoint_override is not None:
        unet_path = _Path(unet_checkpoint_override).resolve()
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
    import json

    return json.loads(Path(env_config).read_text()).get(
        "model_filename", "diff_unet_3d_rflow-mr-brain_v0.pt")


__all__ = [
    "VENDORED_NVIDIA_ROOT", "DEFAULT_ENV_CONFIG", "DEFAULT_MODEL_CONFIG", "DEFAULT_NETWORK_CONFIG",
    "load_models", "prepare_tensors", "run_inference", "set_random_seed", "load_config", "define_instance",
    "required_spatial_divisor", "load_autoencoder", "load_autoencoder_and_unet",
    "default_unet_filename",
]
