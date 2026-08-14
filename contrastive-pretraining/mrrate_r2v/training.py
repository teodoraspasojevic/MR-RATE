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
from .text import encode_reports
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


def set_loader_epoch(train_loader, epoch: int) -> None:
    """Propagate `epoch` to every component of a DataLoader that reseeds on it.

    **The Dataset must come first.** `MRReportToVolumeDataset.set_epoch` *replaces* its `samples`
    list under `series_selection="one_per_study_random"`, and `GeometryBucketBatchSampler` groups
    those samples into shape-compatible buckets. Reseeding only the batch sampler -- what this
    loop did before -- left `one_per_study_random` frozen at its epoch-0 draw for the whole run.
    Reseeding only the Dataset leaves the sampler's bucket index pointing at the previous epoch's
    samples, which yields batches spanning two geometry buckets and a `collate_fn_r2v` shape
    error; the sampler now rebuilds off `dataset.samples_version`, so both orders are safe, but
    this one also gets the sampler's `__len__` right.

    `sampler` is included for a future `DistributedSampler`, which needs the same call. Anything
    without a `set_epoch` (a plain list of batches, as `--dry-run` uses) is skipped.
    """
    for component in (getattr(train_loader, "dataset", None),
                      getattr(train_loader, "batch_sampler", None),
                      getattr(train_loader, "sampler", None)):
        if hasattr(component, "set_epoch"):
            component.set_epoch(epoch)


def wrap_distributed(model, device):
    """`diff_model_train.py:155-159`. `find_unused_parameters=True` matters more here than there:
    with report dropout, an adapter can go unused within a step.

    `device_ids` is passed **only for a CUDA module**. DDP rejects it for a CPU module -- "device_ids
    and output_device arguments only work with single-device/multiple-device GPU modules or CPU
    modules" -- which broke the gloo/CPU path that exists so the whole distributed wiring can be
    smoke-tested without booking GPUs.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return model
    from torch.nn.parallel import DistributedDataParallel

    on_cuda = next(model.parameters()).is_cuda
    kwargs = {"device_ids": [device]} if on_cuda else {}
    return DistributedDataParallel(model, find_unused_parameters=True, **kwargs)


# --------------------------------------------------------------------------- latents


class LatentEncoder:
    """MR-RATE volume `[B, 1, X, Y, Z]` -> the frozen autoencoder's latent, the same call
    `diff_model_create_training_data.py:170` makes (`encode_stage_2_inputs`, which samples from the
    posterior). Padding to the encoder's required divisor reuses this package's own
    `required_spatial_divisor` and `pad_to_divisible`, so it matches `cli/evaluate.py:reconstruct`.
    """

    def __init__(self, autoencoder, divisor: int, scale_factor: float, amp: bool = True,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        self.autoencoder = autoencoder
        self.divisor = int(divisor)
        self.scale_factor = float(scale_factor)
        self.amp = amp
        self.dtype = dtype
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        autoencoder.eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        from .eval.geometry_contract import pad_to_divisible

        # End-only padding, the same primitive and the same convention `cli/evaluate.py:reconstruct` uses, so
        # the encoder sees a training volume padded exactly as an evaluated one is.
        _padded_shape, record = pad_to_divisible(tuple(images.shape[2:]), self.divisor)
        if record is not None:
            pads = []
            for axis in reversed(record.per_axis):  # F.pad takes the last spatial axis first
                pads.extend([int(axis["before"]), int(axis["after"])])
            images = torch.nn.functional.pad(images, pads)
        device_type = "cuda" if images.is_cuda else "cpu"
        # Same dtype as the training step: a float16 VAE encode can overflow on an unusual
        # volume and hand the denoiser inf latents, which looks like a diverging adapter.
        with autocast(device_type, enabled=self.amp and images.is_cuda, dtype=self.dtype):
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
    #: Gradient-norm clip. **On by default, unlike the official trainer.** With `grad_clip_norm=None`
    #: and no GradScaler, a single non-finite gradient writes NaN into the weights and every later
    #: step, checkpoint and validation number is garbage -- observed in jobs 690054-57. The official
    #: trainer gets away without it because float16 + GradScaler silently *skips* such steps; that
    #: masking is what made the same failure look like a step-240 divergence under float16.
    grad_clip_norm: Optional[float] = 1.0
    amp: bool = True
    #: Autocast dtype. **bfloat16 by default, deliberately diverging from the official trainer.**
    #: `torch.amp.autocast("cuda")` with no dtype silently means *float16*, and float16 overflows:
    #: an LR sweep (jobs 689068-71) produced NaN at the identical step for every learning rate
    #: from 1e-5 to 3e-4, with losses matching to four decimals -- i.e. not divergence, an
    #: overflow. bfloat16 has float32's exponent range, is native on H200 (bf16_supported=True),
    #: and needs no GradScaler. Set "float16" to reproduce NVIDIA's exact setup.
    amp_dtype: str = "bfloat16"
    #: Consecutive non-finite losses tolerated before the run is stopped. One is survivable (the
    #: scaler skips the step); a run of them means the weights are gone and every later number,
    #: including every validation metric, is meaningless. Failing fast beats burning the walltime.
    max_consecutive_nonfinite: int = 5
    seed: int = 0
    log_every: int = 1
    # All three count **optimizer** steps, not micro-steps, so their meaning does not change with
    # grad_accumulation_steps. `self.step` remains the micro-step counter for resume compatibility.
    save_every_steps: Optional[int] = None
    #: Epochs between end-of-epoch checkpoints (`adapter_epoch<N>.pt`, absolute epoch number so a
    #: resumed run continues the numbering). Counted in epochs, not steps, because an epoch boundary
    #: does not land on a multiple of `save_every_steps` in general -- 2 epochs of configuration D
    #: is 4493 optimizer steps, so the boundary sits at 2246.5. Never pruned by `keep_last_n`.
    save_every_epochs: Optional[int] = None
    validate_every_steps: Optional[int] = None
    validate_full_every_steps: Optional[int] = None
    validate_at_end: bool = True
    validate_full_at_end: bool = False
    save_format: str = "adapter"  # adapter | full | both
    #: How many periodic step checkpoints to keep. `last`/`best_*` are never counted or deleted.
    keep_last_n: Optional[int] = 3
    #: Recorded in the checkpoint so a resume or an inference run can verify what it is loading.
    conditioning_name: Optional[str] = None
    report_format: Optional[str] = None
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)

    def __post_init__(self) -> None:
        if self.save_format not in ("adapter", "full", "both"):
            raise ValueError(f"save_format must be adapter|full|both, got {self.save_format!r}")
        if self.grad_accumulation_steps < 1:
            raise ValueError("grad_accumulation_steps must be >= 1")
        if self.amp_dtype not in ("bfloat16", "float16"):
            raise ValueError(f"amp_dtype must be bfloat16|float16, got {self.amp_dtype!r}")
        if self.keep_last_n is not None and self.keep_last_n < 1:
            raise ValueError("keep_last_n must be >= 1 or None (keep everything)")


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
        wandb_run=None,
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
        #: Rank 0 only. A `WandbRun` that failed to init is already a no-op, so this needs no
        #: further guarding beyond the rank check in `_log_metrics`.
        self.wandb_run = wandb_run

        torch.set_float32_matmul_precision("highest")  # diff_model_train.py:480
        torch.manual_seed(config.seed)

        # Every adapter introspection goes through `_unwrapped()`, so the trainer accepts a model
        # that is already DDP-wrapped. `freeze_to_adapter_only` and `assert_only_adapter_trainable`
        # read `context_proj` / `CONDITIONING_PREFIXES` off the module, and DDP does not forward
        # attribute access -- passing the wrapper raised
        # "'DistributedDataParallel' object has no attribute 'context_proj'".
        self.freeze_report = freeze_to_adapter_only(self._unwrapped(), text_embedder)
        self.loss_fn = build_loss()
        adapter_params = [p for p in unet.parameters() if p.requires_grad]
        self.optimizer = build_optimizer(adapter_params, config.lr)
        self.lr_scheduler = None
        self.amp_dtype = getattr(torch, config.amp_dtype)
        # A GradScaler exists to stop float16 gradients underflowing. bfloat16 has float32's
        # exponent range, so scaling it is pointless and PyTorch recommends against it.
        self.scaler = GradScaler(
            "cuda",
            enabled=config.amp and self.device.type == "cuda" and self.amp_dtype is torch.float16,
        )
        self._nonfinite_streak = 0
        self.skipped_steps = 0
        #: The reference constants are identical at every validation; the Table goes out once.
        self._reference_table_logged = False
        # The gate that makes "only adapters train" a checked fact rather than an intention.
        assert_only_adapter_trainable(self._unwrapped(), self.optimizer, text_embedder)
        log.info("adapter training: %s", self.freeze_report.format())
        self.step = 0            # micro-steps (one per forward/backward)
        self.optimizer_step = 0  # optimizer steps -- what every interval and every log is keyed on
        self.epoch = 0
        #: First epoch index `fit` will run. Advanced past the resumed epoch by `load_for_resume` so
        #: a continuation does not replay the original run's epoch-0 shuffle (and, under
        #: `series_selection="one_per_study_random"`, its epoch-0 series draw) as its first epoch.
        self.start_epoch = 0
        # Lower-is-better and higher-is-better tracked separately, both persisted, so a resumed run
        # does not overwrite a better checkpoint with a worse one on its first validation.
        self.best_metrics: dict = {"fvd": None, "fid_2p5d": None, "ssim": None}
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

        # The shared seam: dispatches to per-section encoding for a sectioned-fusion embedder and
        # to plain `report_text` otherwise, so training/validation/sampling cannot diverge.
        conditioning = encode_reports(self.text_embedder, batch, self.device)
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
        with autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=amp_enabled,
                      dtype=self.amp_dtype):
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

        if not torch.isfinite(loss):
            self._nonfinite_streak += 1
            # Name the first non-finite tensor: overflow in the frozen VAE encode, a degenerate
            # input volume, and a diverging adapter are three different bugs with three different
            # fixes, and the loss value alone cannot tell them apart.
            culprit = next(
                (name for name, tensor in (("image", batch.get("image")), ("latents", latents),
                                           ("context", prepared["context"]),
                                           ("model_output", model_output), ("target", target))
                 if tensor is not None and not torch.isfinite(tensor).all()),
                "none (loss only)",
            )
            log.error("non-finite loss at optimizer step %d (streak %d/%d); first non-finite "
                      "tensor: %s. amp=%s dtype=%s", self.optimizer_step, self._nonfinite_streak,
                      self.config.max_consecutive_nonfinite, culprit, self.config.amp,
                      self.config.amp_dtype)
            if self._nonfinite_streak >= self.config.max_consecutive_nonfinite:
                raise RuntimeError(
                    f"loss has been non-finite for {self._nonfinite_streak} consecutive steps "
                    f"(first non-finite tensor: {culprit}). Every metric from here on is "
                    f"meaningless. With amp_dtype=float16 this is usually overflow -- use "
                    f"bfloat16 (the default)."
                )
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            return {"loss": float("nan"), "lr": self.optimizer.param_groups[0]["lr"],
                    "stepped": False, "n_dropped_reports": 0, "timestep_mean": 0.0}
        self._nonfinite_streak = 0

        scaled = loss / self.config.grad_accumulation_steps
        if self.scaler.is_enabled():
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        stepped = False
        grad_norm = None
        if (self.step + 1) % self.config.grad_accumulation_steps == 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)   # needed before inspecting or clipping
            trainable = [p for p in self.unet.parameters() if p.requires_grad]
            # `clip_grad_norm_` returns the pre-clip total norm, so one call both clips and gives
            # the diagnostic. Called even when clipping is off, purely to inspect the norm.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                trainable, self.config.grad_clip_norm or float("inf")))

            if grad_norm != grad_norm or grad_norm == float("inf"):
                # Skip -- exactly what GradScaler does for float16, but dtype-independent. Without
                # this, bfloat16 and float32 (which have no scaler) write NaN straight into the
                # weights and never recover.
                self.skipped_steps += 1
                # Name the parameters whose gradient is non-finite. The loss is L1, so the gradient
                # *at the output* is bounded by 1/N -- a non-finite parameter gradient is therefore
                # produced somewhere in the backward pass (a normalisation layer dividing by a
                # near-zero activation std is the usual culprit), not by the loss. Which module it
                # is decides the fix, and only this tells us.
                culprits = [n for n, q in self._unwrapped().named_parameters()
                            if q.grad is not None and not torch.isfinite(q.grad).all()]
                log.warning("skipping optimizer step %d: gradient norm is %s (%d skipped so far). "
                            "Weights untouched. %d/%d trainable tensors have non-finite grads; "
                            "first: %s", self.optimizer_step + 1, grad_norm, self.skipped_steps,
                            len(culprits), len(trainable), culprits[:5] or "none (norm overflow only)")
                self.optimizer.zero_grad(set_to_none=True)
            else:
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                stepped = True
                self.optimizer_step += 1

        self.step += 1
        return {
            "loss": float(loss.detach()),
            "lr": self.optimizer.param_groups[0]["lr"],
            "stepped": stepped,
            "n_dropped_reports": int(prepared["context_drop_mask"].sum()),
            "timestep_mean": float(timesteps.float().mean()),
            "grad_norm": grad_norm,
        }

    # -- loop ----------------------------------------------------------------------------

    def _log_metrics(self, data: dict, step: int) -> None:
        """Rank 0 only, one W&B step definition: the optimizer step.

        Micro-steps are never used as the x-axis. Mixing the two is how a `grad_accumulation_steps`
        change silently rescales every curve in a project, making two runs incomparable.
        """
        if self.local_rank == 0 and self.wandb_run is not None:
            self.wandb_run.log(data, step=step)

    def _prune_checkpoints(self) -> None:
        """Keep the newest `keep_last_n` periodic step checkpoints. `adapter_last.pt` and the
        `adapter_best_*.pt` files are matched by a different glob and so are never deleted --
        retention must not be able to remove the two things a run exists to produce."""
        if self.config.keep_last_n is None or self.local_rank != 0:
            return
        periodic = sorted(self.output_dir.glob("adapter_step*.pt"))
        for stale in periodic[: max(0, len(periodic) - self.config.keep_last_n)]:
            for path in (stale, stale.with_name(stale.stem + "_full_unet.pt")):
                if path.exists():
                    path.unlink()
                    log.info("retention: removed %s", path.name)

    def _maybe_save_best(self, validation: dict) -> list:
        """Update the best-metric checkpoints. Lower FID is better, higher alignment is better."""
        saved = []
        if self.local_rank != 0:
            return saved
        # One entry per headline validation metric, with its direction stated rather than implied.
        # FVD and 2.5D FID are distributional distances (lower better, optimum 0); SSIM is a paired
        # similarity (higher better, max 1).
        candidates = [
            ("fvd", "val/fvd", min),
            ("fid_2p5d", "val/fid_2p5d", min),
            ("ssim", "val/ssim", max),
        ]
        for key, metric_name, better in candidates:
            value = validation.get(metric_name)
            if value is None or not isinstance(value, (int, float)) or value != value:  # NaN
                continue
            current = self.best_metrics.get(key)
            if current is None or better(value, current) == value:
                self.best_metrics[key] = float(value)
                path = self.save(self.output_dir / f"adapter_best_{key}.pt",
                                 loss=None, validation=validation)
                log.info("new best %s = %.5f -> %s", key, value, getattr(path, "name", path))
                saved.append(key)
        return saved

    def validate_now(self, validate, full: bool = False) -> dict:
        """Run validation and act on it. Every rank calls this -- `validate` is responsible for its
        own sharding and gathering -- but only rank 0 writes checkpoints or logs.

        **Exactly three curves reach W&B**, and only from the quick pass:

            val/quick/{fvd,fid_2p5d,ssim}

        A pass returns ~47 numbers (per plane, per bucket, rank flags, timings, counts, the
        sensitivity diagnostic); logging all of them produced a dashboard nobody could read.

        **Anything measured once is a summary entry, not a curve.** That covers the reference
        constants (`val/reference/*`, computed once by `cli.validation_reference` and identical at
        every step) and the full pass (`val/full/*`, which runs on the order of once per run). A
        one-point series renders as a lone marker or, worse, a flat line that reads as "tracked and
        unchanging" -- so those go to `run.summary`, which is a table.

        Quick and full are never merged into one series in any case: they measure the *same* metric
        at different N, and for a Frechet distance N is part of the number, so overlaying them would
        draw a step change that is a sample-size artefact rather than a model change.

        Everything else stays in the returned payload and lands in `train_summary.json`.
        """
        from .validation import HEADLINE_METRICS

        validation = validate(self, self.optimizer_step, full)
        is_full = bool(validation.get("val/full"))
        curves, once = {}, {}
        for metric in HEADLINE_METRICS:
            value = validation.get(f"val/{metric}")
            if isinstance(value, (int, float)) and value == value:      # present and not NaN
                (once if is_full else curves)[f"val/{'full' if is_full else 'quick'}/{metric}"] = float(value)
        if is_full:
            once["val/full/n_cases"] = validation.get("val/n_cases")
            once["val/full/optimizer_step"] = self.optimizer_step
        once.update({k: v for k, v in validation.items()
                     if k.startswith("val/reference/") and isinstance(v, (int, float))})
        if curves:
            self._log_metrics(curves, step=self.optimizer_step)
        if once and self.local_rank == 0 and self.wandb_run is not None:
            self.wandb_run.set_summary(once)
            # `set_summary` only populates the Overview tab. The reference constants are meant to be
            # read *together*, beside the curves they calibrate, so they also go out once as a real
            # Table panel -- otherwise "it is in the table" means a key/value list on another page.
            references = {k: v for k, v in once.items() if k.startswith("val/reference/")}
            if references and not self._reference_table_logged:
                self._reference_table_logged = True
                self.wandb_run.log_table(
                    "val/reference", ["reference", "value"],
                    sorted((k.removeprefix("val/reference/"), v) for k, v in references.items()),
                    step=self.optimizer_step)
        self._maybe_save_best(validation)
        return validation

    def fit(self, train_loader, validate=None) -> dict:
        """`validate(trainer, optimizer_step, full) -> dict`, or None to skip validation.

        `ValidationRunner.run` matches that signature. It is passed in rather than constructed
        here so the trainer keeps no dependency on the sampler or the feature extractor.
        """
        micro_per_epoch = max(len(train_loader), 1)
        # Both budgets count what THIS invocation does, not the model's whole history: `n_epochs` is
        # epochs to run and `max_steps` is micro-steps to run. Measuring `max_steps` against the
        # cumulative `self.step` instead made it meaningless after a resume -- the D checkpoint is at
        # micro-step 17972, so any smoke-sized cap was already exceeded before the first batch and
        # the job stopped one micro-step in while reporting that it had honoured the cap.
        start_step = self.step
        total_micro = self.config.max_steps or int(self.config.n_epochs * micro_per_epoch)
        # `PolynomialLR` is stepped once per *optimizer* step, so its horizon must be counted in
        # optimizer steps -- but both `max_steps` and `len(train_loader)` are micro-step counts.
        # Passing the micro count made the schedule decay `grad_accumulation_steps` times too
        # slowly: job 690962 (accum 2, max_steps 3000) reached only 64% of its base LR by its last
        # optimizer step instead of ~0, and at config B's accum 16 the LR would have been constant
        # to within 12% for the whole run. Every LR sweep before 2026-08-07 measured that schedule.
        total_steps = max(1, total_micro // self.config.grad_accumulation_steps)
        if self.lr_scheduler is None:
            self.lr_scheduler = build_scheduler(self.optimizer, total_steps)
        # A `PolynomialLR` restored from a run that finished is at `last_epoch == total_iters`, where
        # it returns exactly 0 and never rises again. Training on would be a silent no-op for the
        # whole job, so it is refused here rather than discovered in the loss curve afterwards.
        exhausted = getattr(self.lr_scheduler, "total_iters", None)
        if exhausted is not None and self.lr_scheduler.last_epoch >= exhausted:
            raise RuntimeError(
                f"the resumed LR schedule is exhausted (last_epoch="
                f"{self.lr_scheduler.last_epoch} >= total_iters={exhausted}), so it would hold the "
                f"learning rate at {self.lr_scheduler.get_last_lr()} for all {total_steps} optimizer "
                "steps of this run and train nothing. This is what extending a completed run looks "
                "like: pass --resume-lr-schedule restart (with --lr set to the peak you want the "
                "new schedule to start from) instead of continuing the old one."
            )
        history, validations = [], []
        start = time.time()
        # Intervals fire on the optimizer step *transition*, so an interval is never missed and
        # never fires twice for one optimizer step (which `step % N` on micro-steps would do
        # whenever grad_accumulation_steps > 1).
        # Anchored at the *current* optimizer step, not at 0. A resumed run starts at step 4493, and
        # `4493 // 600 > 0 // 600` is true, so anchoring at 0 fired a validation, a full validation
        # and a checkpoint save all on the first optimizer step after every resume.
        last_validated = last_saved = last_full = self.optimizer_step
        # `n_epochs` is how many epochs *this* invocation runs; `start_epoch` is where they sit in
        # the model's whole training history (0 for a fresh run, past the resumed epoch otherwise).
        # Every epoch number that leaves this loop -- the log line, the W&B curve, the checkpoint
        # filename, `train_summary.json` -- is the absolute one, so a continuation reads as epochs
        # 3-4 rather than as a second run of epochs 1-2.
        for offset in range(self.config.n_epochs):
            epoch = self.start_epoch + offset
            self.epoch = epoch
            set_loader_epoch(train_loader, epoch)
            for batch in train_loader:
                metrics = self.train_step(batch)
                history.append(metrics)
                if metrics["stepped"]:
                    if self.optimizer_step % max(self.config.log_every, 1) == 0:
                        if self.local_rank == 0:
                            # No skip counter here: a skip already emits its own WARNING at the
                            # moment it happens, and repeating a "skipped 0" on every healthy line
                            # is noise. The cumulative count lives on one W&B curve and in
                            # train_summary.json.
                            # `total_steps` is an optimizer-step count, so it is the denominator of
                            # `opt-step`, not of `micro`. Printing it next to the micro counter
                            # read as a progress fraction that was accum times too small.
                            log.info(
                                "epoch %d opt-step %d/%s (micro %d) loss %.5f lr %.3e dropped %d "
                                "grad_norm %s",
                                epoch + 1, self.optimizer_step, total_steps, self.step,
                                metrics["loss"], metrics["lr"], metrics["n_dropped_reports"],
                                ("%.3f" % metrics["grad_norm"]) if metrics.get("grad_norm") is not None else "-",
                            )
                        self._log_metrics({"train/loss": metrics["loss"],
                                           "train/lr": metrics["lr"],
                                           "train/epoch": epoch + 1,
                                           "train/dropped_reports": metrics["n_dropped_reports"],
                                           # No `train/skipped_steps` curve. In a healthy run it is
                                           # a flat line at 0 -- a panel that costs attention and
                                           # carries no information. A skip still emits its own
                                           # WARNING naming the offending parameters, the cumulative
                                           # count is in train_summary.json, and it is written to
                                           # the W&B run summary at the end.
                                           "train/grad_norm": metrics["grad_norm"]},
                                          step=self.optimizer_step)

                    interval = self.config.validate_every_steps
                    if (interval and validate is not None
                            and self.optimizer_step // interval > last_validated // interval):
                        last_validated = self.optimizer_step
                        full_interval = self.config.validate_full_every_steps
                        full = bool(full_interval
                                    and self.optimizer_step // full_interval > last_full // full_interval)
                        if full:
                            last_full = self.optimizer_step
                        validations.append(self.validate_now(validate, full=full))

                    save_interval = self.config.save_every_steps
                    if (save_interval and self.local_rank == 0
                            and self.optimizer_step // save_interval > last_saved // save_interval):
                        last_saved = self.optimizer_step
                        self.save(self.output_dir / f"adapter_step{self.optimizer_step:07d}.pt",
                                  loss=metrics["loss"])
                        self._prune_checkpoints()

                if self.config.max_steps and self.step - start_step >= self.config.max_steps:
                    break
            stopped_early = bool(self.config.max_steps
                                 and self.step - start_step >= self.config.max_steps)
            # Only for an epoch that actually ran to its end: a `max_steps` break leaves a partial
            # epoch, and a file named for it would claim a pass over the data that did not happen.
            if (self.config.save_every_epochs and self.local_rank == 0 and not stopped_early
                    and (offset + 1) % self.config.save_every_epochs == 0):
                self.save(self.output_dir / f"adapter_epoch{epoch + 1:03d}.pt",
                          loss=history[-1]["loss"] if history else None,
                          validation=validations[-1] if validations else None)
            if stopped_early:
                break

        if self.config.validate_at_end and validate is not None:
            validations.append(self.validate_now(validate, full=self.config.validate_full_at_end))
        if self.local_rank == 0:
            self.save(self.output_dir / "adapter_last.pt",
                      loss=history[-1]["loss"] if history else None,
                      validation=validations[-1] if validations else None)
        return {
            "steps": self.step,
            "optimizer_steps": self.optimizer_step,
            # Belongs in the summary, not only in the log: a run whose numbers look fine but which
            # skipped a third of its steps trained on a different dataset than it claims to have.
            "skipped_steps": self.skipped_steps,
            # Cumulative across resumes, not this job's `n_epochs`: with `start_epoch` beside it the
            # two are always separable, and the cumulative count is the one that describes the
            # weights the run produced.
            "epochs": self.epoch + 1,
            "start_epoch": self.start_epoch,
            "seconds": time.time() - start,
            "final_loss": history[-1]["loss"] if history else None,
            "mean_loss": sum(h["loss"] for h in history) / len(history) if history else None,
            "best_metrics": dict(self.best_metrics),
            "validations": validations,
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
            # What text the model actually saw. Without these two an inference run cannot tell
            # whether the adapter expects `impression_findings` or unformatted report text -- and
            # feeding it the wrong one is silent, not an error.
            "conditioning_name": self.config.conditioning_name,
            "report_format": self.config.report_format,
            "num_train_timesteps": self.num_train_timesteps,
        }

    def save(self, path, loss: float | None = None, validation: dict | None = None) -> Path:
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
                optimizer_step=self.optimizer_step,
                best_metrics=dict(self.best_metrics),
                validation=validation,
                rng_state={
                    "torch": torch.get_rng_state(),
                    "dropout_generator": self.dropout_generator.get_state(),
                    "numpy": None,
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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

    def load_for_resume(self, path, lr_schedule: str = "continue") -> dict:
        """Restore weights, optimizer, RNG and counters from an adapter checkpoint.

        `lr_schedule` decides what happens to the learning rate, and it is the one thing a resume
        cannot get right on its own:

        * ``continue`` -- restore the stored `PolynomialLR` and carry on down it. Correct when the
          original run was cut short (preemption, walltime) and the remaining horizon is the one the
          schedule was built for.
        * ``restart`` -- discard the stored schedule and LR, and build a fresh `PolynomialLR` from
          ``config.lr`` over *this* run's horizon. Required when extending a run that already ran to
          completion, because `PolynomialLR` reaches exactly 0 at `total_iters` and stays there:
          `r2v_final_D_report2ct_style/adapter_last.pt` stores
          ``{total_iters: 4493, last_epoch: 4493, _last_lr: [0.0]}`` and an optimizer whose
          ``param_groups[0]["lr"]`` is ``0.0``. Continuing that schedule trains at LR 0 -- a silent
          no-op that burns the whole walltime and writes a checkpoint identical to its input.

        `fit` refuses to run a restored schedule that is already exhausted, so the failure above is
        an error rather than a wasted job.
        """
        if lr_schedule not in ("continue", "restart"):
            raise ValueError(f"lr_schedule must be continue|restart, got {lr_schedule!r}")
        from .models.adapter import load_adapter_checkpoint

        payload = load_adapter_checkpoint(
            path, self._unwrapped(),
            base_checkpoint_sha256=self.base_checkpoint.get("sha256"),
            # Refuses a checkpoint trained under a different conditioning configuration, including
            # the two that share a width (cxr_bert_cls and radbert_mean are both 768x1) and would
            # otherwise load cleanly onto the wrong embedding space.
            text_encoder=dict(self.text_embedder.identity),
        )
        if payload.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if lr_schedule == "restart":
            # Adam's moments are kept -- they are a property of the loss surface, not of the
            # schedule -- but `lr` and `initial_lr` come back from the checkpoint too, and
            # `initial_lr` is what a newly built LRScheduler adopts as its `base_lrs`. Left alone,
            # a fresh schedule would restart from the *old* peak, or from 0.
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.lr
                group["initial_lr"] = self.config.lr
            # `fit` builds the new schedule, because only it knows the new horizon.
            self.lr_scheduler = None
            log.info("resume: LR schedule restarted at %.3e over this run's horizon "
                     "(stored schedule discarded)", self.config.lr)
        elif payload.get("lr_scheduler_state_dict"):
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
        if "cuda" in rng and rng["cuda"] and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as exc:  # noqa: BLE001 -- a different device count is not fatal
                log.warning("could not restore CUDA RNG state (%s); continuing", exc)
        self.step = int(payload.get("step", 0))
        self.optimizer_step = int(payload.get("optimizer_step", self.step))
        self.epoch = int(payload.get("epoch", 0))
        # The stored `epoch` is the index of the epoch in progress when the checkpoint was written,
        # so the next one to run is the one after it. A mid-epoch checkpoint therefore loses the
        # remainder of its epoch rather than replaying its first half -- the honest trade at an
        # epoch granularity this coarse (10.85 h), and it keeps every epoch's shuffle seed distinct.
        self.start_epoch = self.epoch + 1
        stored_best = payload.get("best_metrics") or {}
        # Carried forward so the first validation after a resume cannot overwrite a better
        # checkpoint with a worse one just because this process has not seen a score yet.
        self.best_metrics.update({k: v for k, v in stored_best.items() if v is not None})
        log.info("resumed from %s at optimizer step %d (micro %d) epoch %d, best=%s",
                 path, self.optimizer_step, self.step, self.epoch, self.best_metrics)
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
    "set_loader_epoch",
    "sha256_file",
    "wrap_distributed",
]
