"""Report-adapter training. Structurally `NV-Generate-CTMR/scripts/diff_model_train.py`, with three
changes and no others.

What is taken from the official trainer unchanged:

| official (`scripts/diff_model_train.py`)          | here                                    |
|---------------------------------------------------|-----------------------------------------|
| `augment_modality_label` (:34-66)                 | imported, via `conditioning.py`         |
| `noise_scheduler.sample_timesteps(images)` (:328) | `official_timesteps`                    |
| `add_noise(...)` (:303)                           | `train_step`                            |
| `model_gt = images - noise` for RFlow (:328-329)  | `official_target`                       |
| `torch.nn.L1Loss()` (:478)                        | `build_loss`                            |
| `Adam(lr=...)` (:198), `PolynomialLR(power=2)` (:220) | `build_optimizer` / `build_scheduler`|
| `GradScaler` + `autocast("cuda")` (:479, :292)    | `train_step`                            |
| `set_float32_matmul_precision("highest")` (:480)  | `MRRateAdapterTrainer.__init__`         |
| DDP with `find_unused_parameters=True` (:158)     | `wrap_distributed`                      |
| `unet.train()` (:275)                             | `train_step`; safe -- the base UNet has |
|                                                   | no BatchNorm and every Dropout is p=0.0 |

The three deliberate differences:

1. **The optimizer sees only the adapter.** Official passes `unet.parameters()`; here the parameter
   set comes from `models/adapter.py` and is asserted before the first step. The forward pass is
   *not* wrapped in `no_grad` -- autograd has to cross the frozen convolutions to reach an adapter
   sitting in the middle of the network.
2. **`scale_factor` comes from the base checkpoint, not from `1/std(z)` of the first batch.**
   Official recomputes it because it trains from scratch (`calculate_scale_factor`, :173-195); the
   frozen denoiser here was trained at the value stored in its own checkpoint, which is also what
   official *inference* uses (`diff_model_infer.py:73`). Recomputing would rescale the latents out
   from under a model that cannot adapt. `--scale-factor recompute` restores the official behaviour.
3. **Latents are encoded on the fly** from MR-RATE volumes by the frozen autoencoder, instead of
   being read from `*_emb.nii.gz` files baked by `diff_model_create_training_data.py`. Same call
   (`encode_stage_2_inputs`), same preprocessing (MR-RATE's `percentile` normalizer is
   `ScaleIntensityRangePercentilesd(0.0, 99.5, 0.0, 1.0, clip=False)`, i.e. NVIDIA's own MRI
   transform, `scripts/transforms.py:64`), one fewer on-disk stage.

There is **no EMA** here because there is none in the official code: a repo-wide search for `ema`
over `NV-Generate-CTMR/scripts/*.py` and `configs/*.json` returns nothing, and the released
checkpoint carries only `unet_state_dict` / `optimizer_state_dict` / `scheduler_state_dict`. Adding
one would be inventing a behaviour to preserve.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from torch.amp import GradScaler, autocast

from .conditioning import ConditioningConfig, ModalityEncoder, augment_modality_label, sample_report_drop_mask
from .models.adapter import (
    assert_only_adapter_trainable,
    freeze_to_adapter_only,
    save_adapter_checkpoint,
    save_full_unet_checkpoint,
    sha256_file,
)

log = logging.getLogger("mrrate_r2v.training")


# --------------------------------------------------------------------------- official pieces


def official_timesteps(noise_scheduler, images: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """`diff_model_train.py:327-301`: RFlow samples its own timesteps; DDPM draws uniformly."""
    if isinstance(noise_scheduler, RFlowScheduler):
        return noise_scheduler.sample_timesteps(images)
    return torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()


def official_target(noise_scheduler, images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
    """`diff_model_train.py:327-341`, transcribed branch for branch.

    For the rectified-flow scheduler the mr-brain model actually uses, the target is the velocity
    along the linear path, `x0 - eps` -- not a DDPM prediction type.
    """
    if isinstance(noise_scheduler, RFlowScheduler):
        return images - noise
    if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
        return noise
    if noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
        return images
    if noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
        return noise_scheduler.get_velocity(images, noise, timesteps)
    raise ValueError(f"unsupported prediction type {noise_scheduler.prediction_type}")


def build_loss() -> torch.nn.Module:
    """`diff_model_train.py:478`. L1 on the velocity target, unweighted and unpreconditioned -- the
    objective NVIDIA's checkpoint was trained with. Not swapped for MSE, and no auxiliary
    report/image alignment term is added: the adapter has to earn its keep on this loss alone."""
    return torch.nn.L1Loss()


def build_optimizer(parameters, lr: float) -> torch.optim.Optimizer:
    """`diff_model_train.py:198` -- Adam, same defaults; only the parameter set differs."""
    return torch.optim.Adam(params=list(parameters), lr=lr)


def build_scheduler(optimizer, total_steps: int):
    """`diff_model_train.py:220` -- PolynomialLR, power 2."""
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=max(int(total_steps), 1), power=2.0)


def wrap_distributed(model, device):
    """`diff_model_train.py:155-159`. `find_unused_parameters=True` matters more here than there:
    with report dropout, an adapter can go unused within a step."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return model
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)


# --------------------------------------------------------------------------- latents


class LatentEncoder:
    """MR-RATE volume `[B, 1, X, Y, Z]` -> the frozen autoencoder's latent, the same call
    `diff_model_create_training_data.py:170` makes (`encode_stage_2_inputs`, which samples from the
    posterior). Padding to the encoder's required divisor reuses this package's own
    `required_spatial_divisor` and `pad_to_divisible`, so it matches `cli/predict_vae.py`.
    """

    def __init__(self, autoencoder, divisor: int, scale_factor: float, amp: bool = True) -> None:
        self.autoencoder = autoencoder
        self.divisor = int(divisor)
        self.scale_factor = float(scale_factor)
        self.amp = amp
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        autoencoder.eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        from .eval.geometry_contract import pad_to_divisible

        # End-only padding, the same primitive and the same convention `cli/predict_vae.py` uses, so
        # the encoder sees a training volume padded exactly as an evaluated one is.
        _padded_shape, record = pad_to_divisible(tuple(images.shape[2:]), self.divisor)
        if record is not None:
            pads = []
            for axis in reversed(record.per_axis):  # F.pad takes the last spatial axis first
                pads.extend([int(axis["before"]), int(axis["after"])])
            images = torch.nn.functional.pad(images, pads)
        device_type = "cuda" if images.is_cuda else "cpu"
        with autocast(device_type, enabled=self.amp and images.is_cuda):
            latent = self.autoencoder.encode_stage_2_inputs(images.float())
        return latent.float() * self.scale_factor


# --------------------------------------------------------------------------- trainer


@dataclass
class TrainingConfig:
    """Everything the trainer needs that is not a model or a dataloader."""

    lr: float = 1e-5
    n_epochs: int = 1
    max_steps: Optional[int] = None
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    grad_clip_norm: Optional[float] = None
    amp: bool = True
    seed: int = 0
    log_every: int = 1
    save_every_steps: Optional[int] = None
    validate_every_steps: Optional[int] = None
    save_format: str = "adapter"  # adapter | full | both
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)

    def __post_init__(self) -> None:
        if self.save_format not in ("adapter", "full", "both"):
            raise ValueError(f"save_format must be adapter|full|both, got {self.save_format!r}")
        if self.grad_accumulation_steps < 1:
            raise ValueError("grad_accumulation_steps must be >= 1")


class MRRateAdapterTrainer:
    """Trains the report adapter of a `ReportConditionedUNetMaisi` on MR-RATE."""

    def __init__(
        self,
        unet,
        text_embedder,
        noise_scheduler,
        latent_encoder: Optional[LatentEncoder],
        config: TrainingConfig,
        device,
        output_dir,
        base_checkpoint: Optional[dict] = None,
        num_train_timesteps: int = 1000,
        modality_encoder: Optional[ModalityEncoder] = None,
        local_rank: int = 0,
    ) -> None:
        self.unet = unet
        self.text_embedder = text_embedder
        self.noise_scheduler = noise_scheduler
        self.latent_encoder = latent_encoder
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.base_checkpoint = base_checkpoint or {}
        self.num_train_timesteps = num_train_timesteps
        self.modality_encoder = modality_encoder or ModalityEncoder()
        self.local_rank = local_rank

        torch.set_float32_matmul_precision("highest")  # diff_model_train.py:480
        torch.manual_seed(config.seed)

        self.freeze_report = freeze_to_adapter_only(unet, text_embedder)
        self.loss_fn = build_loss()
        adapter_params = [p for p in unet.parameters() if p.requires_grad]
        self.optimizer = build_optimizer(adapter_params, config.lr)
        self.lr_scheduler = None
        self.scaler = GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        # The gate that makes "only adapters train" a checked fact rather than an intention.
        assert_only_adapter_trainable(unet, self.optimizer, text_embedder)
        log.info("adapter training: %s", self.freeze_report.format())
        self.step = 0
        self.epoch = 0
        # Report dropout draws from its own generator so a resumed run reproduces the same drops
        # regardless of what else consumed the global RNG.
        self.dropout_generator = torch.Generator().manual_seed(config.seed + 12345)

    # -- one step ------------------------------------------------------------------------

    def prepare_batch(self, batch) -> dict:
        """MR-RATE batch -> the official `unet_inputs` dict, plus the report context.

        `spacing_tensor` is scaled by 1e2 exactly as the official transform does
        (`diff_model_train.py:117`), and modality goes through NVIDIA's own dropout.
        """
        images = batch["image"].to(self.device)
        if self.latent_encoder is not None:
            latents = self.latent_encoder.encode(images.float())
        else:  # already-latent input (tests, or a precomputed-latent dataset)
            latents = images.float()

        spacing = batch["target_spacing_mm"].to(self.device).float() * 1e2
        modality = self.modality_encoder.encode(batch["modality"], device=self.device)
        modality = augment_modality_label(modality, prob=self.config.conditioning.modality_dropout_probability)

        conditioning = self.text_embedder.encode(batch["report_text"], self.device)
        drop = sample_report_drop_mask(
            latents.shape[0], self.config.conditioning.report_dropout_probability,
            device=self.device, generator=self.dropout_generator,
        )
        return {
            "latents": latents,
            "spacing_tensor": spacing,
            "class_labels": modality,
            "context": conditioning.token_embeddings.to(self.device),
            "context_mask": conditioning.attention_mask.to(self.device),
            "context_drop_mask": drop,
        }

    def train_step(self, batch) -> dict:
        """One official training step, with the report threaded through."""
        self.unet.train()
        if hasattr(self.text_embedder, "eval"):
            self.text_embedder.eval()

        prepared = self.prepare_batch(batch)
        latents = prepared["latents"]
        amp_enabled = self.config.amp and self.device.type == "cuda"
        with autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=amp_enabled):
            noise = torch.randn_like(latents)
            timesteps = official_timesteps(self.noise_scheduler, latents, self.num_train_timesteps)
            noisy = self.noise_scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
            model_output = self.unet(
                x=noisy,
                timesteps=timesteps,
                spacing_tensor=prepared["spacing_tensor"],
                class_labels=prepared["class_labels"],
                context=prepared["context"],
                context_mask=prepared["context_mask"],
                context_drop_mask=prepared["context_drop_mask"],
            )
            target = official_target(self.noise_scheduler, latents, noise, timesteps)
            loss = self.loss_fn(model_output.float(), target.float())

        scaled = loss / self.config.grad_accumulation_steps
        if self.scaler.is_enabled():
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        stepped = False
        if (self.step + 1) % self.config.grad_accumulation_steps == 0:
            if self.config.grad_clip_norm:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.unet.parameters() if p.requires_grad], self.config.grad_clip_norm
                )
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            stepped = True

        self.step += 1
        return {
            "loss": float(loss.detach()),
            "lr": self.optimizer.param_groups[0]["lr"],
            "stepped": stepped,
            "n_dropped_reports": int(prepared["context_drop_mask"].sum()),
            "timestep_mean": float(timesteps.float().mean()),
        }

    # -- loop ----------------------------------------------------------------------------

    def fit(self, train_loader, validate=None) -> dict:
        total_steps = self.config.max_steps or int(self.config.n_epochs * max(len(train_loader), 1))
        if self.lr_scheduler is None:
            self.lr_scheduler = build_scheduler(self.optimizer, total_steps)
        history = []
        start = time.time()
        for epoch in range(self.config.n_epochs):
            self.epoch = epoch
            if hasattr(getattr(train_loader, "batch_sampler", None), "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)
            for batch in train_loader:
                metrics = self.train_step(batch)
                history.append(metrics)
                if self.local_rank == 0 and self.step % max(self.config.log_every, 1) == 0:
                    log.info(
                        "epoch %d step %d/%s loss %.5f lr %.3e dropped %d",
                        epoch + 1, self.step, total_steps, metrics["loss"], metrics["lr"],
                        metrics["n_dropped_reports"],
                    )
                if (self.config.validate_every_steps and validate is not None
                        and self.step % self.config.validate_every_steps == 0):
                    validation = validate(self)
                    log.info("validation at step %d: %s", self.step, validation)
                if (self.config.save_every_steps and self.local_rank == 0
                        and self.step % self.config.save_every_steps == 0):
                    self.save(self.output_dir / f"adapter_step{self.step:07d}.pt", loss=metrics["loss"])
                if self.config.max_steps and self.step >= self.config.max_steps:
                    break
            if self.config.max_steps and self.step >= self.config.max_steps:
                break
        if self.local_rank == 0:
            self.save(self.output_dir / "adapter_last.pt", loss=history[-1]["loss"] if history else None)
        return {
            "steps": self.step,
            "epochs": self.epoch + 1,
            "seconds": time.time() - start,
            "final_loss": history[-1]["loss"] if history else None,
            "mean_loss": sum(h["loss"] for h in history) / len(history) if history else None,
            "history": history,
        }

    # -- checkpoints ---------------------------------------------------------------------

    def _unwrapped(self):
        return getattr(self.unet, "module", self.unet)

    def config_payload(self) -> dict:
        model = self._unwrapped()
        return {
            "context_dim": model.context_proj.context_dim,
            "cross_attention_dim": model.cross_attention_dim,
            "conditioning_levels": list(model.conditioning_levels),
            "condition_mid": model.mid_cross_attn is not None,
            "training": {
                "lr": self.config.lr, "batch_size": self.config.batch_size,
                "grad_accumulation_steps": self.config.grad_accumulation_steps,
                "grad_clip_norm": self.config.grad_clip_norm, "amp": self.config.amp,
                "seed": self.config.seed,
            },
            "conditioning": vars(self.config.conditioning),
            "num_train_timesteps": self.num_train_timesteps,
        }

    def save(self, path, loss: float | None = None) -> Path:
        model = self._unwrapped()
        scale_factor = self.latent_encoder.scale_factor if self.latent_encoder else None
        written = None
        if self.config.save_format in ("adapter", "both"):
            written = save_adapter_checkpoint(
                path, model,
                step=self.step, epoch=self.epoch, config=self.config_payload(),
                base_checkpoint=self.base_checkpoint,
                text_encoder=dict(self.text_embedder.identity),
                scale_factor=scale_factor,
                optimizer=self.optimizer, lr_scheduler=self.lr_scheduler, scaler=self.scaler,
                loss=loss,
                rng_state={
                    "torch": torch.get_rng_state(),
                    "dropout_generator": self.dropout_generator.get_state(),
                },
            )
            log.info("saved adapter checkpoint %s", written)
        if self.config.save_format in ("full", "both"):
            full_path = Path(path).with_name(Path(path).stem + "_full_unet.pt")
            save_full_unet_checkpoint(
                full_path, model, epoch=self.epoch, loss=loss or 0.0,
                num_train_timesteps=self.num_train_timesteps, scale_factor=scale_factor,
            )
            log.info("saved full-UNet checkpoint %s (official layout)", full_path)
            written = written or full_path
        return written

    def load_for_resume(self, path) -> dict:
        from .models.adapter import load_adapter_checkpoint

        payload = load_adapter_checkpoint(
            path, self._unwrapped(),
            base_checkpoint_sha256=self.base_checkpoint.get("sha256"),
        )
        if payload.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("lr_scheduler_state_dict"):
            if self.lr_scheduler is None:
                self.lr_scheduler = build_scheduler(self.optimizer, 1)
            self.lr_scheduler.load_state_dict(payload["lr_scheduler_state_dict"])
        if payload.get("scaler_state_dict") and self.scaler.is_enabled():
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        rng = payload.get("rng_state") or {}
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        if "dropout_generator" in rng:
            self.dropout_generator.set_state(rng["dropout_generator"])
        self.step = int(payload.get("step", 0))
        self.epoch = int(payload.get("epoch", 0))
        log.info("resumed from %s at step %d epoch %d", path, self.step, self.epoch)
        return payload


def resolve_scale_factor(base_checkpoint_path, mode: str = "auto", train_loader=None,
                         latent_encoder_factory=None, device=None) -> float:
    """`auto` (default): the value stored in NVIDIA's own checkpoint, which is what their inference
    uses (`diff_model_infer.py:73`). `recompute`: `1/std(z)` over the first batch, the official
    *training* behaviour (`diff_model_train.py:188`) -- only correct when the denoiser can adapt to
    the new scale, which a frozen one cannot.
    """
    if mode == "auto":
        from .models.report_conditioned_unet import _allow_maisi_checkpoint_globals

        _allow_maisi_checkpoint_globals()
        payload = torch.load(str(base_checkpoint_path), map_location="cpu", weights_only=True)
        if "scale_factor" not in payload:
            raise RuntimeError(
                f"{base_checkpoint_path} has no scale_factor; pass --scale-factor recompute or a "
                "literal value"
            )
        return float(payload["scale_factor"])
    if mode == "recompute":
        if train_loader is None or latent_encoder_factory is None:
            raise ValueError("recompute needs a train_loader and a latent encoder")
        from monai.utils import first

        batch = first(train_loader)
        encoder = latent_encoder_factory(1.0)
        latent = encoder.encode(batch["image"].to(device).float())
        return float(1.0 / torch.std(latent))
    return float(mode)


__all__ = [
    "LatentEncoder",
    "MRRateAdapterTrainer",
    "TrainingConfig",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "official_target",
    "official_timesteps",
    "resolve_scale_factor",
    "sha256_file",
    "wrap_distributed",
]
