"""What is trainable, and what a trained artefact contains.

Two jobs, both about the same invariant -- *only the report adapter is learned*:

- `freeze_to_adapter_only` + `assert_only_adapter_trainable`: turn `requires_grad` off everywhere
  else and then *prove* it, at startup, before an optimizer exists. A silent leak here means
  fine-tuning NVIDIA's 180M-parameter denoiser at lr 1e-5 on 88k studies instead of training a
  5M-parameter adapter.
- `save_adapter_checkpoint` / `load_adapter_checkpoint`: the adapter is ~8M of a 188M model, so the
  base weights are not duplicated. The checkpoint instead *identifies* the base checkpoint it was
  trained against (path + sha256) and refuses to load onto a different one unless told to.

Membership is decided by module identity, not by name matching: `adapter_parameter_names` walks the
actual `context_proj` / `{down,mid,up}_cross_attn` submodules and the `null_context` Parameter. The
name-prefix tuple `CONDITIONING_PREFIXES` is then cross-checked against that set, so if the two ever
disagree the mismatch is an error rather than a quietly frozen adapter.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

log = logging.getLogger("mrrate_r2v.adapter")

ADAPTER_CHECKPOINT_FORMAT = "mrrate_r2v_adapter_v1"


def adapter_modules(model) -> dict:
    """The adapter submodules, found by attribute, not by name string."""
    modules = {"context_proj": model.context_proj}
    for attribute in ("down_cross_attn", "up_cross_attn"):
        container = getattr(model, attribute, None)
        if container is not None:
            for key, module in container.items():
                modules[f"{attribute}.{key}"] = module
    if getattr(model, "mid_cross_attn", None) is not None:
        modules["mid_cross_attn"] = model.mid_cross_attn
    return modules


def adapter_parameter_names(model) -> set:
    """State-dict names of every adapter parameter, including the report-CFG null embedding.

    `null_context` is a bare Parameter on the UNet rather than a submodule, and it is part of the
    adapter: it *is* the null-report representation that report classifier-free guidance needs.
    """
    owned = {id(p) for module in adapter_modules(model).values() for p in module.parameters()}
    owned.add(id(model.null_context))
    names = {name for name, parameter in model.named_parameters() if id(parameter) in owned}

    by_prefix = {name for name, _ in model.named_parameters() if name.startswith(model.CONDITIONING_PREFIXES)}
    if names != by_prefix:
        raise RuntimeError(
            "adapter membership by module identity disagrees with CONDITIONING_PREFIXES: "
            f"only-by-identity={sorted(names - by_prefix)[:6]} "
            f"only-by-prefix={sorted(by_prefix - names)[:6]}. One of the two is stale; the "
            "checkpoint loader and the optimizer would then disagree about what is trained."
        )
    return names


@dataclass
class FreezeReport:
    trainable_parameters: int
    frozen_parameters: int
    trainable_tensors: int
    frozen_tensors: int
    adapter_names: tuple

    @property
    def total_parameters(self) -> int:
        return self.trainable_parameters + self.frozen_parameters

    def format(self) -> str:
        return (
            f"trainable {self.trainable_parameters:,} params in {self.trainable_tensors} tensors "
            f"({100 * self.trainable_parameters / max(self.total_parameters, 1):.2f}% of "
            f"{self.total_parameters:,}); frozen {self.frozen_parameters:,} in {self.frozen_tensors}"
        )


def freeze_to_adapter_only(model, text_embedder=None) -> FreezeReport:
    """Make exactly the report adapter trainable. Everything else -- NVIDIA's denoiser, its
    modality/spacing/timestep embeddings, and the text encoder -- gets `requires_grad=False`.

    Note what this does *not* do: it does not run any part of the forward pass under `no_grad`.
    Autograd still traverses the frozen convolutions, because that is the only route by which a
    gradient can reach an adapter sitting in the middle of the network.
    """
    names = adapter_parameter_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    if text_embedder is not None:
        for parameter in getattr(text_embedder, "parameters", lambda: [])():
            parameter.requires_grad_(False)
        if hasattr(text_embedder, "eval"):
            text_embedder.eval()

    trainable = [p for n, p in model.named_parameters() if n in names]
    frozen = [p for n, p in model.named_parameters() if n not in names]
    report = FreezeReport(
        trainable_parameters=sum(p.numel() for p in trainable),
        frozen_parameters=sum(p.numel() for p in frozen),
        trainable_tensors=len(trainable),
        frozen_tensors=len(frozen),
        adapter_names=tuple(sorted(names)),
    )
    log.info("freeze_to_adapter_only: %s", report.format())
    return report


def assert_only_adapter_trainable(model, optimizer=None, text_embedder=None) -> FreezeReport:
    """Startup gate. Raises unless every adapter parameter is trainable, no other model parameter
    is, no text-encoder parameter is, and the optimizer holds exactly the adapter set."""
    names = adapter_parameter_names(model)
    if not names:
        raise RuntimeError("no adapter parameters found -- there would be nothing to train")

    frozen_adapter = sorted(n for n, p in model.named_parameters() if n in names and not p.requires_grad)
    if frozen_adapter:
        raise RuntimeError(f"{len(frozen_adapter)} adapter parameters are frozen: {frozen_adapter[:8]}")
    trainable_base = sorted(n for n, p in model.named_parameters() if n not in names and p.requires_grad)
    if trainable_base:
        raise RuntimeError(
            f"{len(trainable_base)} base-model parameters are trainable, so NVIDIA's pretrained "
            f"weights would be modified: {trainable_base[:8]}"
        )
    if text_embedder is not None:
        leaking = [n for n, p in getattr(text_embedder, "named_parameters", lambda: [])() if p.requires_grad]
        if leaking:
            raise RuntimeError(f"{len(leaking)} text-encoder parameters are trainable: {leaking[:8]}")

    if optimizer is not None:
        in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
        expected = {id(p) for n, p in model.named_parameters() if n in names}
        if in_optimizer != expected:
            missing = sorted(n for n, p in model.named_parameters() if n in names and id(p) not in in_optimizer)
            extra = len(in_optimizer - expected)
            raise RuntimeError(
                f"optimizer parameter set != adapter set: {len(missing)} adapter params absent "
                f"({missing[:6]}), {extra} non-adapter params present"
            )

    trainable = [p for n, p in model.named_parameters() if n in names]
    frozen = [p for n, p in model.named_parameters() if n not in names]
    return FreezeReport(
        trainable_parameters=sum(p.numel() for p in trainable),
        frozen_parameters=sum(p.numel() for p in frozen),
        trainable_tensors=len(trainable),
        frozen_tensors=len(frozen),
        adapter_names=tuple(sorted(names)),
    )


def adapter_state_dict(model) -> dict:
    names = adapter_parameter_names(model)
    full = model.state_dict()
    # Adapter *buffers* would also belong here; the adapter has none today, and this comprehension
    # would silently drop them, so assert that stays true.
    module_prefixes = tuple(f"{key}." for key in adapter_modules(model)) + ("null_context",)
    buffers = [n for n, _ in model.named_buffers() if n.startswith(module_prefixes)]
    if buffers:
        raise RuntimeError(f"adapter gained buffers that this checkpoint format would drop: {buffers}")
    return {name: full[name].detach().cpu().clone() for name in sorted(names)}


def sha256_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def save_adapter_checkpoint(
    path,
    model,
    *,
    step: int,
    epoch: int,
    config: dict,
    base_checkpoint: dict,
    text_encoder: dict,
    scale_factor,
    optimizer=None,
    lr_scheduler=None,
    scaler=None,
    loss: float | None = None,
    rng_state: dict | None = None,
    optimizer_step: int | None = None,
    best_metrics: dict | None = None,
    validation: dict | None = None,
) -> Path:
    """Adapter-only checkpoint. Deliberately does *not* contain NVIDIA's 180M base weights: they are
    unchanged by construction, so storing them would make every checkpoint 700 MB of a file the
    workspace already has, and would let a stale copy silently diverge from the real base."""
    path = Path(path)
    payload = {
        "format": ADAPTER_CHECKPOINT_FORMAT,
        "adapter_state_dict": adapter_state_dict(model),
        "step": int(step),
        # The optimizer step is the run's real clock (every interval and every W&B x-axis uses it);
        # `step` stays the micro-step count so an older checkpoint still resumes.
        "optimizer_step": int(optimizer_step if optimizer_step is not None else step),
        "epoch": int(epoch),
        "loss": loss,
        "best_metrics": dict(best_metrics or {}),
        "validation": validation,
        "scale_factor": float(scale_factor) if scale_factor is not None else None,
        "config": config,
        "base_checkpoint": base_checkpoint,
        "text_encoder": text_encoder,
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "lr_scheduler_state_dict": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "rng_state": rng_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return path


def assert_conditioning_compatible(payload: dict, text_encoder: dict | None) -> None:
    """Refuse an adapter trained under a different conditioning configuration.

    Checked before any tensor is loaded, because the failure mode this prevents is *silent*: a
    768-wide pooled adapter and a 2560-wide fused adapter have different `context_proj` shapes and
    would be caught by the shape check, but two configurations that share a width (`cxr_bert_cls`
    and `radbert_mean` are both 768x1) load cleanly and generate confident nonsense -- the
    projection was fitted to a different encoder's embedding space.

    The comparison is on the recorded identity, not on the checkpoint filename.
    """
    if not text_encoder:
        return
    stored = payload.get("text_encoder") or {}
    if not stored:
        log.warning("checkpoint records no text-encoder identity; cannot verify conditioning")
        return

    def key(identity: dict) -> tuple:
        encoder = identity.get("encoder") or {}
        return (
            identity.get("kind"),
            identity.get("pooling"),
            int(identity.get("output_dim") or 0),
            int(identity.get("sequence_length") or 1),
            tuple(identity.get("sections") or ()),
            tuple(identity.get("encoder_order") or ()),
            tuple(identity.get("encoder_dims") or ()),
            # single-encoder configurations carry their checkpoint one level down
            encoder.get("name") or identity.get("name"),
            encoder.get("hf_repo") or identity.get("hf_repo"),
        )

    if key(stored) != key(text_encoder):
        raise RuntimeError(
            "adapter checkpoint was trained under a different conditioning configuration.\n"
            f"  checkpoint: {key(stored)}\n"
            f"  requested:  {key(text_encoder)}\n"
            "Load it with the configuration it was trained under (the checkpoint's "
            "config['conditioning_name'] names it), or pass allow_conditioning_mismatch=True for "
            "a deliberate transfer experiment."
        )


def load_adapter_checkpoint(
    path, model, *, base_checkpoint_sha256: str | None = None, allow_base_mismatch: bool = False,
    strict: bool = True, text_encoder: dict | None = None,
    allow_conditioning_mismatch: bool = False,
) -> dict:
    """Load an adapter checkpoint onto a model whose base weights are already loaded.

    `strict=True` (the default) means every adapter tensor in the model must be present in the file
    and vice versa -- a partially-trained adapter loaded silently would produce plausible garbage.

    `text_encoder` is the identity dict of the live embedder. Supplying it enables the conditioning
    compatibility check; omitting it skips the check rather than guessing.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format") != ADAPTER_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"{path} is not a {ADAPTER_CHECKPOINT_FORMAT} checkpoint (format="
            f"{payload.get('format')!r}). A full NVIDIA UNet checkpoint loads with "
            "load_pretrained_maisi_weights instead."
        )
    stored_sha = (payload.get("base_checkpoint") or {}).get("sha256")
    if base_checkpoint_sha256 and stored_sha and stored_sha != base_checkpoint_sha256:
        message = (
            f"adapter was trained against base checkpoint {stored_sha[:12]} but the loaded base is "
            f"{base_checkpoint_sha256[:12]}; the adapter's zero-point is a different frozen model"
        )
        if not allow_base_mismatch:
            raise RuntimeError(message)
        log.warning("%s (continuing: allow_base_mismatch=True)", message)

    if not allow_conditioning_mismatch:
        assert_conditioning_compatible(payload, text_encoder)

    expected = adapter_parameter_names(model)
    provided = set(payload["adapter_state_dict"])
    if strict and expected != provided:
        raise RuntimeError(
            f"adapter checkpoint does not match this model: missing={sorted(expected - provided)[:6]} "
            f"unexpected={sorted(provided - expected)[:6]}. Check context_dim, cross_attention_dim "
            "and conditioning_levels against the checkpoint's config."
        )
    model_state = model.state_dict()
    mismatched = [
        (name, tuple(tensor.shape), tuple(model_state[name].shape))
        for name, tensor in payload["adapter_state_dict"].items()
        if name in model_state and tuple(tensor.shape) != tuple(model_state[name].shape)
    ]
    if mismatched:
        raise RuntimeError(f"adapter tensor shapes differ from the model: {mismatched[:4]}")

    missing, unexpected = model.load_state_dict(payload["adapter_state_dict"], strict=False)
    unexpected = [name for name in unexpected]
    if unexpected:
        raise RuntimeError(f"adapter checkpoint has tensors the model does not: {unexpected[:6]}")
    still_missing = [name for name in missing if name in expected]
    if still_missing:
        raise RuntimeError(f"adapter tensors were not loaded: {still_missing[:6]}")
    log.info("loaded adapter checkpoint %s (step %s, %d tensors)", path, payload.get("step"), len(provided))
    return payload


def save_full_unet_checkpoint(path, model, *, epoch: int, loss: float, num_train_timesteps: int, scale_factor) -> Path:
    """The official layout (`diff_model_train.py:save_checkpoint`, lines 386-399), so NVIDIA's own
    tooling can read the result. Larger and redundant for adapter training; provided because the
    official inference path takes this format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch) + 1,
            "loss": loss,
            "num_train_timesteps": int(num_train_timesteps),
            "scale_factor": scale_factor,
            "unet_state_dict": model.state_dict(),
        },
        str(path),
    )
    return path


__all__ = [
    "ADAPTER_CHECKPOINT_FORMAT",
    "FreezeReport",
    "adapter_modules",
    "adapter_parameter_names",
    "adapter_state_dict",
    "assert_only_adapter_trainable",
    "freeze_to_adapter_only",
    "load_adapter_checkpoint",
    "save_adapter_checkpoint",
    "save_full_unet_checkpoint",
    "sha256_file",
]
