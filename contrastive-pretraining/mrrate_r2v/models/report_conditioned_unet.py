"""Report-conditioned NV-Generate-MR-Brain diffusion UNet.

`DiffusionModelUNetMaisi` already has a conditioning path (`with_conditioning=True`), but it
*replaces* each attention level's `SpatialAttentionBlock` with a `SpatialTransformer`: a different
module tree, so NVIDIA's pretrained weights no longer load and the frozen part is no longer the
architecture that was trained. This module keeps the pretrained tree byte-identical and *adds*
cross-attention as new top-level modules:

    context_proj                small MLP, any report embedding -> `cross_attention_dim`, so the
                                choice of text encoder is not baked into the architecture
    {down,mid,up}_cross_attn    one zero-initialised `ReportCrossAttentionAdapter` per conditioned
                                level, applied to the tensor *entering* that block

Properties that are the point of doing it this way:

- Pretrained state dicts load with `unexpected == []` and `missing` exactly the conditioning path
  (`load_pretrained_maisi_weights` enforces this, and the base geometry comes from NVIDIA's own
  `config_network_rflow.json` via `nvidia_unet_kwargs`, not from a second copy of the numbers).
- The adapter's `proj_out` is zero-initialised (`zero_module`, the same convention MAISI uses for its
  own output conv), so at init the conditioned UNet is numerically *identical* to the pretrained one
  -- fine-tuning starts *at* NVIDIA's model, not near it.
- Conditioning enters at each block's *input*, so the skip connections leaving that level carry it
  too; conditioning the block output would leave the decoder's skips report-blind.
- `context=None`, or a per-sample `context_drop_mask`, selects a learned null embedding -- what
  report dropout during training and the classifier-free-guidance branch of NVIDIA's inference loop
  both need.
- `context_mask` is honoured, so a padded batch of variable-length reports conditions on the real
  tokens only. This is why the adapter is not monai's `SpatialTransformer`: that block's
  `CrossAttentionBlock.forward` (monai/networks/blocks/crossattention.py:144) takes no mask, so
  padding would silently join the attention softmax and the conditioning of a sample would depend
  on the longest report that happened to share its batch. Dropping that block also drops its
  spatial self-attention and GEGLU feed-forward, which duplicate what the pretrained
  `SpatialAttentionBlock`s at those same levels already do, at O(voxels^2) attention cost.
"""

from __future__ import annotations

import contextlib
import inspect
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.networks.blocks import Convolution
from monai.networks.blocks.crossattention import CrossAttentionBlock
from monai.networks.nets.diffusion_model_unet import zero_module
from monai.utils.type_conversion import convert_to_tensor

DEFAULT_CROSS_ATTENTION_DIM = 512

_MAISI_PARAMS = inspect.signature(DiffusionModelUNetMaisi.__init__).parameters


# -- SDPA backend guard ------------------------------------------------------------------------
#
# `F.scaled_dot_product_attention` is one *interface* over four interchangeable CUDA kernels; torch
# picks one per call from the shapes, dtype and mask. Two places here reach it: the base model's
# `SpatialAttentionBlock`s (MONAI's `use_flash_attention=True` means "call SDPA", not "use
# FlashAttention"), and `MaskedCrossAttention` below, which calls it unconditionally.
#
# The cuDNN backend returns **non-finite gradients from a finite forward** at latent 48^3 -- the
# (T2w, CORONAL) bucket -- in bfloat16 and float16. Measured: 8/8 seeds, batch 1-8, with random
# Gaussian latents and no data, model or adapter involved; born on the query branch of
# `up_blocks.1.attentions.*` and reaching every adapter tensor. This is the same class as PyTorch
# issue #166211 (NaN in grad_q under CUDNN_ATTENTION, finite output), and cuDNN's own release notes
# document a backward defect at certain static sequence lengths plus problems at key/value length 1
# -- which is exactly what a pooled report embedding gives the adapter.
#
# Every other backend is correct here, so the fix is to take cuDNN out of the running rather than
# to abandon SDPA. MATH stays in the list purely as a last resort: `MaskedCrossAttention` passes an
# `attn_mask` that the fused kernels can refuse, and an empty candidate set is a hard
# "No available kernel" error, not a fallback.
def safe_sdpa_backends() -> list:
    """The SDPA backends this model is allowed to use: every one except cuDNN's."""
    from torch.nn.attention import SDPBackend

    return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def sdpa_backend_guard(on_cuda: bool = True):
    """Context manager restricting SDPA to `safe_sdpa_backends()`. A no-op off CUDA.

    Applied around the *forward* pass only, in `ReportConditionedUNetMaisi.forward`, which is
    enough: the backend is chosen during forward and the matching backward kernel is bound into
    the autograd node then, so backward inherits the choice (verified -- the 48^3 gradient is
    finite with the guard on forward alone). Activation checkpointing would break that assumption
    by re-running attention during backward under whatever context is live then; this model does
    not use it, and a future change that adds it must wrap the backward too.
    """
    if not on_cuda:
        return contextlib.nullcontext()
    from torch.nn.attention import sdpa_kernel

    return sdpa_kernel(safe_sdpa_backends())


def _maisi_arg(maisi_kwargs: dict, name: str):
    """The value the base `__init__` will actually use for `name`: the caller's, else the base
    signature's own default -- so the adapters cannot drift from the blocks they sit next to."""
    if name in maisi_kwargs:
        return maisi_kwargs[name]
    return _MAISI_PARAMS[name].default


class MaskedCrossAttention(CrossAttentionBlock):
    """monai's `CrossAttentionBlock` with a key-padding mask.

    Same parameters, same state-dict keys, same head/scale semantics -- only the attention call is
    re-implemented, because monai's `forward` has no mask argument and its non-flash branch
    materialises the full `(B, heads, voxels, tokens)` score matrix. `scaled_dot_product_attention`
    is used unconditionally: it takes `attn_mask`, it is exact, and unlike monai's
    `use_flash_attention` switch it does not require CUDA, so the adapters stay CPU-testable while
    the base UNet keeps NVIDIA's own flash-attention setting.

    `context_mask` is `True`/`1` = attend, matching a HuggingFace `attention_mask`.
    """

    # Attributes borrowed from the base class. monai is a floating dependency (`monai>=1.5.0`), so
    # check rather than discover a silent rename as wrong numbers.
    _BORROWED = ("to_q", "to_k", "to_v", "out_proj", "input_rearrange", "out_rearrange", "scale", "drop_output")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        missing = [name for name in self._BORROWED if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                f"monai's CrossAttentionBlock no longer provides {missing}; MaskedCrossAttention "
                "reimplements only its forward pass and relies on those attributes."
            )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = context if context is not None else x
        q = self.input_rearrange(self.to_q(x))  # (B, heads, queries, head_dim)
        k = self.input_rearrange(self.to_k(kv))
        v = self.input_rearrange(self.to_v(kv))

        if self.attention_dtype is not None:
            # monai upcasts q and k only, which then mismatches v's dtype; upcast all three.
            q, k, v = q.to(self.attention_dtype), k.to(self.attention_dtype), v.to(self.attention_dtype)

        attn_mask = None
        if context_mask is not None:
            # (B, tokens) -> (B, 1, 1, tokens): broadcast over heads and over every query voxel.
            attn_mask = context_mask.view(context_mask.shape[0], 1, 1, context_mask.shape[1])

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            scale=self.scale,
            dropout_p=self.dropout_rate if self.training else 0.0,
        )
        out = self.out_rearrange(out.to(x.dtype))
        return self.drop_output(self.out_proj(out))


class ReportCrossAttentionAdapter(nn.Module):
    """Cross-attention residual branch over a feature map: `x + proj_out(attn(proj_in(norm(x))))`.

    The shape of monai's `SpatialTransformer` minus its spatial self-attention and feed-forward,
    which the pretrained `SpatialAttentionBlock` at the same level already provides. `proj_out` is
    zero-initialised, so the branch is an exact identity at init and the whole module is a no-op
    until it is trained.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        cross_attention_dim: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        upcast_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.cross_attention_dim = cross_attention_dim
        inner_dim = num_attention_heads * num_head_channels
        self.inner_dim = inner_dim

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.proj_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=inner_dim,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.norm_attn = nn.LayerNorm(inner_dim)
        self.attn = MaskedCrossAttention(
            hidden_size=inner_dim,
            num_heads=num_attention_heads,
            hidden_input_size=inner_dim,
            context_input_size=cross_attention_dim,
            dim_head=num_head_channels,
            dropout_rate=dropout,
            attention_dtype=torch.float if upcast_attention else None,
        )
        self.proj_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inner_dim,
                out_channels=in_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        h = self.proj_in(self.norm(x))
        batch, channels = h.shape[:2]
        spatial = h.shape[2:]
        h = h.reshape(batch, channels, -1).transpose(1, 2)  # "b c ... -> b (...) c"
        h = self.attn(self.norm_attn(h), context=context, context_mask=context_mask)
        h = h.transpose(1, 2).reshape(batch, channels, *spatial)
        return residual + self.proj_out(h)


class ContextProjection(nn.Module):
    """Projects a report/text embedding of arbitrary width onto the UNet's cross-attention width.

    Accepts `(B, context_dim)` (a pooled embedding, treated as one token) or `(B, L, context_dim)`
    (per-token/per-sentence embeddings). The input LayerNorm is what makes the module indifferent to
    the scale of whichever encoder ends up producing the embedding.
    """

    def __init__(self, context_dim: int, cross_attention_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * cross_attention_dim
        self.context_dim = context_dim
        self.cross_attention_dim = cross_attention_dim
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim == 2:
            context = context.unsqueeze(1)
        if context.ndim != 3:
            raise ValueError(f"context must be (B, context_dim) or (B, L, context_dim), got {tuple(context.shape)}")
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"context width {context.shape[-1]} != context_dim {self.context_dim}")
        out: torch.Tensor = self.net(context)
        return out


class ReportConditionedUNetMaisi(DiffusionModelUNetMaisi):
    """MAISI diffusion UNet, structurally unchanged, plus cross-attention on a report embedding.

    Args:
        context_dim: width of the incoming report embedding (whatever encoder produces it).
        cross_attention_dim: width the embedding is projected to and attended over. This is the
            adapters' own width; the base class's identically named argument is not accepted, since
            setting it there is what would rebuild the pretrained attention blocks.
        conditioning_levels: per-level bool, which UNet levels get a cross-attention adapter.
            Defaults to `attention_levels` -- the low-resolution levels, where the pretrained model
            already spends its attention budget.
        condition_mid: also condition the middle block.
        conditioning_num_head_channels: head width for the adapters. Defaults to the base model's
            `num_head_channels` at that level, which is 0 at non-attention levels -- set this
            explicitly to condition a level the base model has no attention at.
        dropout_cattn, context_hidden_dim: adapter capacity knobs.
        maisi_kwargs: passed verbatim to `DiffusionModelUNetMaisi`; use `nvidia_unet_kwargs()` to get
            the pretrained model's own values.
    """

    CONDITIONING_PREFIXES = ("context_proj.", "null_context", "down_cross_attn.", "mid_cross_attn.", "up_cross_attn.")

    def __init__(
        self,
        context_dim: int,
        cross_attention_dim: int = DEFAULT_CROSS_ATTENTION_DIM,
        conditioning_levels: Sequence[bool] | None = None,
        condition_mid: bool = True,
        conditioning_num_head_channels: int | None = None,
        dropout_cattn: float = 0.0,
        context_hidden_dim: int | None = None,
        **maisi_kwargs,
    ) -> None:
        if maisi_kwargs.get("with_conditioning"):
            raise ValueError(
                "with_conditioning=True makes the base class swap its SpatialAttentionBlocks for "
                "SpatialTransformers, which changes the pretrained module tree. Conditioning here is "
                "added alongside those blocks instead."
            )
        super().__init__(**maisi_kwargs)

        num_channels = list(self.block_out_channels)
        levels = list(self.attention_levels if conditioning_levels is None else conditioning_levels)
        if len(levels) != len(num_channels):
            raise ValueError(f"conditioning_levels has length {len(levels)}, expected {len(num_channels)}")
        self.conditioning_levels = tuple(bool(v) for v in levels)
        self.cross_attention_dim = cross_attention_dim

        adapter_kwargs = dict(
            spatial_dims=_maisi_arg(maisi_kwargs, "spatial_dims"),
            norm_num_groups=_maisi_arg(maisi_kwargs, "norm_num_groups"),
            norm_eps=_maisi_arg(maisi_kwargs, "norm_eps"),
            cross_attention_dim=cross_attention_dim,
            upcast_attention=_maisi_arg(maisi_kwargs, "upcast_attention"),
            dropout=dropout_cattn,
        )

        def make_adapter(channels: int, level: int) -> ReportCrossAttentionAdapter:
            head_channels = conditioning_num_head_channels
            if head_channels is None:
                head_channels = self.num_head_channels[level]
            if head_channels <= 0 or channels % head_channels:
                raise ValueError(
                    f"cannot build a cross-attention adapter for {channels} channels with "
                    f"num_head_channels={head_channels} (level {level}); pass "
                    "conditioning_num_head_channels explicitly"
                )
            return ReportCrossAttentionAdapter(
                in_channels=channels,
                num_attention_heads=channels // head_channels,
                num_head_channels=head_channels,
                **adapter_kwargs,
            )

        # Channels of the tensor *entering* each block: a down block's input is the previous level's
        # output, and an up block's input is the previous up block's output (the first one being the
        # bottleneck). Both are keyed by the base model's level index, so `conditioning_levels[i]`
        # and `num_head_channels[i]` mean the same level on the way down and on the way up.
        n_levels = len(num_channels)
        reversed_channels = list(reversed(num_channels))
        self.down_cross_attn = nn.ModuleDict(
            {
                str(i): make_adapter(num_channels[max(i - 1, 0)], i)
                for i in range(n_levels)
                if self.conditioning_levels[i]
            }
        )
        self.up_cross_attn = nn.ModuleDict(
            {
                str(i): make_adapter(reversed_channels[max(i - 1, 0)], n_levels - 1 - i)
                for i in range(n_levels)
                if self.conditioning_levels[n_levels - 1 - i]
            }
        )
        self.mid_cross_attn = make_adapter(num_channels[-1], n_levels - 1) if condition_mid else None

        self.context_proj = ContextProjection(context_dim, cross_attention_dim, context_hidden_dim)
        # Learned "no report" token. Not zero: it is the unconditional branch the model trains on.
        self.null_context = nn.Parameter(torch.zeros(1, 1, cross_attention_dim))

    # -- conditioning path -------------------------------------------------------------------

    def null_context_for(self, batch_size: int) -> torch.Tensor:
        """The unconditional context, batched -- for the CFG branch of an inference loop."""
        return self.null_context.expand(batch_size, 1, -1)

    def prepare_context(
        self,
        batch_size: int,
        context: torch.Tensor | None,
        context_drop_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Raw report embedding -> `(context, context_mask)` for the adapters.

        `None`, an all-padding row, and a row selected by `context_drop_mask` all become the learned
        null embedding. `context_mask` is `True` = real token, `False` = padding.
        """
        if context is None:
            return self.null_context_for(batch_size), None

        if context.shape[0] != batch_size:
            raise ValueError(f"context batch {context.shape[0]} != input batch {batch_size}")
        if context.device != self.null_context.device:
            raise ValueError(
                f"context is on {context.device} but the model is on {self.null_context.device}"
            )
        projected = self.context_proj(context)
        n_tokens = projected.shape[1]

        mask = None
        if context_mask is not None:
            if context_mask.shape != (batch_size, n_tokens):
                raise ValueError(
                    f"context_mask shape {tuple(context_mask.shape)} != expected {(batch_size, n_tokens)}"
                )
            mask = context_mask.to(dtype=torch.bool, device=projected.device)

        drop = torch.zeros(batch_size, dtype=torch.bool, device=projected.device)
        if context_drop_mask is not None:
            if context_drop_mask.shape != (batch_size,):
                raise ValueError(
                    f"context_drop_mask shape {tuple(context_drop_mask.shape)} != expected {(batch_size,)}"
                )
            drop = context_drop_mask.to(dtype=torch.bool, device=projected.device)
        if mask is not None:
            # A row with no real tokens has nothing to condition on and would make the attention
            # softmax NaN; it is the unconditional model.
            drop = drop | ~mask.any(dim=1)

        if bool(drop.any()):
            # A null token repeated L times is attention-equivalent to a single null token (identical
            # keys/values), so per-sample dropout is a plain where() on a same-shaped tensor -- and
            # every one of those repeats has to be unmasked for that equivalence to hold.
            null = self.null_context.expand_as(projected).to(projected.dtype)
            projected = torch.where(drop.view(-1, 1, 1), null, projected)
            if mask is not None:
                mask = mask | drop.view(-1, 1)
        return projected, mask

    def conditioning_parameters(self):
        """The added parameters, for a separate optimizer group or a two-stage schedule."""
        return [p for name, p in self.named_parameters() if name.startswith(self.CONDITIONING_PREFIXES)]

    def base_parameter_names(self) -> list[str]:
        """State-dict keys that must come from NVIDIA's checkpoint."""
        return [name for name in self.state_dict() if not name.startswith(self.CONDITIONING_PREFIXES)]

    def freeze_base_unet(self) -> None:
        """Train the conditioning path only, leaving NVIDIA's weights exactly as loaded."""
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith(self.CONDITIONING_PREFIXES)

    @staticmethod
    def _condition(adapters: nn.ModuleDict, index: int, h: torch.Tensor, context, context_mask):
        key = str(index)
        if key not in adapters:
            return h
        return adapters[key](h, context=context, context_mask=context_mask)

    # -- forward -----------------------------------------------------------------------------

    def _apply_down_blocks(self, h, emb, context, down_block_additional_residuals, context_mask=None):
        """`DiffusionModelUNetMaisi._apply_down_blocks`, with an adapter before each conditioned
        block. The blocks themselves are called with `context=None`: they are the pretrained
        self-attention blocks and ignore it.

        The ControlNet residual add is out-of-place, where the base class's is in-place
        (diffusion_model_unet_maisi.py:353). The base's `h` is the same object as its last residual,
        so its in-place version adds that residual to `h` as well; this one does not. Only reachable
        with `down_block_additional_residuals`, i.e. with a ControlNet attached.
        """
        down_block_res_samples: list[torch.Tensor] = [h]
        for i, downsample_block in enumerate(self.down_blocks):
            h = self._condition(self.down_cross_attn, i, h, context, context_mask)
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=None)
            down_block_res_samples.extend(res_samples)

        if down_block_additional_residuals is not None:  # ControlNet residuals
            down_block_res_samples = [
                sample + residual
                for sample, residual in zip(down_block_res_samples, down_block_additional_residuals)
            ]
        return h, down_block_res_samples

    def _apply_up_blocks(self, h, emb, context, down_block_res_samples, context_mask=None):
        for i, upsample_block in enumerate(self.up_blocks):
            idx = -len(upsample_block.resnets)
            res_samples = down_block_res_samples[idx:]
            down_block_res_samples = down_block_res_samples[:idx]
            h = self._condition(self.up_cross_attn, i, h, context, context_mask)
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=None)
        return h

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
        context_drop_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """As `DiffusionModelUNetMaisi.forward`, except `context` is the *raw* report embedding
        (`(B, context_dim)` or `(B, L, context_dim)`; `None` = unconditional), `context_mask` is an
        optional `(B, L)` bool marking real (`True`) versus padding (`False`) tokens, and
        `context_drop_mask` is an optional `(B,)` bool selecting the null embedding per sample.

        The whole body runs under `sdpa_backend_guard`, so no caller -- trainer, sampler, either
        CLI -- can reach the cuDNN attention backend that produces non-finite gradients at some
        latent shapes. Putting it here rather than in the trainer is what makes that impossible to
        forget at a new call site.
        """
        with sdpa_backend_guard(x.is_cuda):
            context, context_mask = self.prepare_context(x.shape[0], context, context_drop_mask, context_mask)

            emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
            emb = self._get_input_embeddings(emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor)
            h = self.conv_in(x)
            h, res_samples = self._apply_down_blocks(
                h, emb, context, down_block_additional_residuals, context_mask=context_mask
            )

            if self.mid_cross_attn is not None:
                h = self.mid_cross_attn(h, context=context, context_mask=context_mask)
            h = self.middle_block(h, emb, None)
            if mid_block_additional_residual is not None:  # ControlNet residual
                h = h + mid_block_additional_residual

            h = self._apply_up_blocks(h, emb, context, res_samples, context_mask=context_mask)
            out: torch.Tensor = convert_to_tensor(self.out(h))
        return out


# The arguments this class handles itself. `build_report_conditioned_unet` needs them to route a
# shared name (cross_attention_dim, dropout_cattn) to the conditioning path.
_CONDITIONING_PARAMS = tuple(
    name
    for name in inspect.signature(ReportConditionedUNetMaisi.__init__).parameters
    if name not in ("self", "context_dim", "maisi_kwargs")
)


def nvidia_unet_kwargs(network_config: str | Path | None = None) -> dict:
    """The `diffusion_unet_def` kwargs from NVIDIA's network config, references resolved and
    `_target_` dropped -- so the pretrained geometry is read from their file rather than restated
    here. Imported lazily: this module must stay importable without the vendored NVIDIA scripts."""
    if network_config is None:
        from .nvidia import DEFAULT_NETWORK_CONFIG

        network_config = DEFAULT_NETWORK_CONFIG

    from monai.bundle import ConfigParser

    parser = ConfigParser(json.loads(Path(network_config).read_text()))
    parser.parse()
    kwargs = dict(parser.get_parsed_content("diffusion_unet_def", instantiate=False).get_config())
    kwargs.pop("_target_", None)
    return kwargs


def build_report_conditioned_unet(context_dim: int, network_config=None, **kwargs) -> ReportConditionedUNetMaisi:
    """`ReportConditionedUNetMaisi` on the pretrained NV-Generate-MR-Brain geometry.

    Conditioning arguments go to `ReportConditionedUNetMaisi`; anything else must be a
    `DiffusionModelUNetMaisi` argument and overrides the config -- in practice only
    `use_flash_attention=False`, which the config enables and which needs CUDA. Names the two
    signatures share (`cross_attention_dim`, `dropout_cattn`) always mean the conditioning one.
    """
    conditioning_kwargs = {k: v for k, v in kwargs.items() if k in _CONDITIONING_PARAMS}
    base_overrides = {k: v for k, v in kwargs.items() if k not in _CONDITIONING_PARAMS}
    unknown = sorted(set(base_overrides) - set(_MAISI_PARAMS))
    if unknown:
        raise TypeError(f"unknown argument(s) for either signature: {unknown}")

    base_kwargs = nvidia_unet_kwargs(network_config)
    base_kwargs.update(base_overrides)
    return ReportConditionedUNetMaisi(context_dim=context_dim, **conditioning_kwargs, **base_kwargs)


# -- pretrained loading ----------------------------------------------------------------------

# Where a checkpoint may keep the weights. Ordered: the first one present wins.
_STATE_DICT_KEYS = ("unet_state_dict", "state_dict", "model_state_dict")
_EMA_STATE_DICT_KEYS = ("ema_state_dict", "ema_unet_state_dict", "model_ema", "ema")
# Wrapper prefixes stripped only when *every* key carries them, so an ambiguous case cannot be
# silently mangled: DDP, torch.compile, Lightning-style wrappers, and ema-pytorch respectively.
_WRAPPER_PREFIXES = ("module.", "_orig_mod.", "model.", "unet.", "ema_model.")
# ema-pytorch bookkeeping that is not model state.
_EMA_BOOKKEEPING = ("initted", "step", "decay", "num_updates")

_safe_globals_allowed = False


def _allow_maisi_checkpoint_globals() -> None:
    """NVIDIA's released checkpoint pickles `scale_factor` as a monai `MetaTensor`, so a plain
    `weights_only=True` load of `diff_unet_3d_rflow-mr-brain_v0.pt` raises `UnpicklingError`
    (measured). Allow-listing those two monai container classes keeps the load restricted; NVIDIA's
    own loaders instead pass `weights_only=False`
    (NV-Generate-CTMR/scripts/diff_model_infer.py:69), which permits arbitrary unpickling.
    """
    global _safe_globals_allowed
    if _safe_globals_allowed:
        return
    from monai.data.meta_tensor import MetaTensor
    from monai.utils.enums import TraceKeys

    torch.serialization.add_safe_globals([MetaTensor, TraceKeys])
    _safe_globals_allowed = True


@dataclass
class PretrainedLoadReport:
    """What `load_pretrained_maisi_weights` did, so a run log can prove the base is pretrained."""

    checkpoint_path: str
    source_key: str | None
    stripped_prefixes: tuple[str, ...] = ()
    used_ema: bool = False
    ema_available: bool = False
    n_checkpoint_tensors: int = 0
    n_base_parameters: int = 0
    n_loaded: int = 0
    missing_by_component: dict[str, list[str]] = field(default_factory=dict)
    unexpected: list[str] = field(default_factory=list)
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    unequal_after_load: list[str] = field(default_factory=list)
    dtype_changes: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return [name for names in self.missing_by_component.values() for name in names]

    @property
    def loaded_fraction(self) -> float:
        return self.n_loaded / self.n_base_parameters if self.n_base_parameters else 0.0

    @property
    def all_shared_equal_checkpoint(self) -> bool:
        return not self.unequal_after_load

    def format(self) -> str:
        lines = [
            f"checkpoint            {self.checkpoint_path}",
            f"  tensors in file     {self.n_checkpoint_tensors}"
            f" (from {self.source_key or '<top level>'}{', EMA' if self.used_ema else ''}"
            f"{', EMA present but not used' if self.ema_available and not self.used_ema else ''})",
            f"  prefixes stripped   {list(self.stripped_prefixes) or 'none'}",
            f"  base tensors loaded {self.n_loaded}/{self.n_base_parameters}"
            f" ({100 * self.loaded_fraction:.2f}%)",
            f"  unexpected keys     {len(self.unexpected)}",
            f"  shape mismatches    {len(self.shape_mismatches)}",
            f"  loaded == file      {'yes, all shared tensors bit-equal' if self.all_shared_equal_checkpoint else f'NO, {len(self.unequal_after_load)} differ'}",
            f"  dtype widened       {len(self.dtype_changes)}",
            f"  left at init        {len(self.missing)} (conditioning path)",
        ]
        for component, names in sorted(self.missing_by_component.items()):
            lines.append(f"    {component:<24} {len(names)}")
        for name, ckpt_shape, model_shape in self.shape_mismatches:
            lines.append(f"    shape {name}: file {tuple(ckpt_shape)} vs model {tuple(model_shape)}")
        return "\n".join(lines)


def _select_state_dict(checkpoint, prefer_ema: bool) -> tuple[dict, str | None, bool, bool]:
    """The tensor dict inside a checkpoint, chosen explicitly rather than by falling through."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint is a {type(checkpoint).__name__}, expected a dict of tensors")

    ema_key = next((k for k in _EMA_STATE_DICT_KEYS if isinstance(checkpoint.get(k), dict)), None)
    if prefer_ema:
        if ema_key is None:
            raise RuntimeError(
                f"prefer_ema=True but the checkpoint has no EMA weights (looked for {list(_EMA_STATE_DICT_KEYS)}, "
                f"top-level keys are {sorted(map(str, checkpoint))[:12]})"
            )
        return dict(checkpoint[ema_key]), ema_key, True, True

    key = next((k for k in _STATE_DICT_KEYS if isinstance(checkpoint.get(k), dict)), None)
    if key is not None:
        return dict(checkpoint[key]), key, False, ema_key is not None
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return dict(checkpoint), None, False, ema_key is not None
    raise RuntimeError(
        f"cannot find the weights in this checkpoint: no {list(_STATE_DICT_KEYS)} entry, and the "
        f"top level is not all tensors (keys: {sorted(map(str, checkpoint))[:12]})"
    )


def _strip_wrapper_prefixes(state_dict: dict) -> tuple[dict, tuple[str, ...]]:
    """Remove wrapper prefixes shared by every key. Repeats, so `module._orig_mod.` also resolves."""
    stripped: list[str] = []
    while state_dict:
        prefix = next((p for p in _WRAPPER_PREFIXES if all(k.startswith(p) for k in state_dict)), None)
        if prefix is None:
            break
        state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
        stripped.append(prefix)
    return state_dict, tuple(stripped)


def _component_of(name: str) -> str:
    """The conditioning sub-module a parameter name belongs to, for grouping the report."""
    if name.startswith(("down_cross_attn.", "up_cross_attn.")):
        head, index = name.split(".")[:2]
        return f"{head}.{index}"
    return name.split(".")[0]


def load_pretrained_maisi_weights(
    model: ReportConditionedUNetMaisi,
    checkpoint_path: str | Path,
    state_dict_key: str | None = None,
    prefer_ema: bool = False,
) -> PretrainedLoadReport:
    """Load NVIDIA's pretrained diffusion-UNet weights into `model` and return a load report.

    Hard-fails unless every pretrained tensor lands, every unfilled parameter belongs to the
    conditioning path, and every shared tensor is bit-equal to the file afterwards: those conditions
    together are what "same architecture plus adapters" means, and a silent mismatch here is a
    fine-tune that quietly starts from noise. This is deliberately *not* NVIDIA's own
    `load_state_dict(..., strict=False)` (diff_model_infer.py:70), which accepts any subset.

    Args:
        state_dict_key: force a specific sub-dict instead of the searched order
            (`unet_state_dict`, `state_dict`, `model_state_dict`, then a bare tensor dict).
        prefer_ema: use the checkpoint's EMA weights; raises if it has none.
    """
    _allow_maisi_checkpoint_globals()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

    if state_dict_key is not None:
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get(state_dict_key), dict):
            raise RuntimeError(f"checkpoint has no dict under {state_dict_key!r}")
        state_dict, source_key, used_ema = dict(checkpoint[state_dict_key]), state_dict_key, False
        ema_available = any(isinstance(checkpoint.get(k), dict) for k in _EMA_STATE_DICT_KEYS)
    else:
        state_dict, source_key, used_ema, ema_available = _select_state_dict(checkpoint, prefer_ema)

    state_dict = {k: v for k, v in state_dict.items() if k not in _EMA_BOOKKEEPING}
    state_dict, stripped = _strip_wrapper_prefixes(state_dict)
    non_tensor = sorted(k for k, v in state_dict.items() if not isinstance(v, torch.Tensor))
    if non_tensor:
        raise RuntimeError(f"checkpoint entries are not tensors: {non_tensor[:8]}")

    model_state = model.state_dict()
    report = PretrainedLoadReport(
        checkpoint_path=str(checkpoint_path),
        source_key=source_key,
        stripped_prefixes=stripped,
        used_ema=used_ema,
        ema_available=ema_available,
        n_checkpoint_tensors=len(state_dict),
        n_base_parameters=len(model.base_parameter_names()),
    )
    report.unexpected = sorted(k for k in state_dict if k not in model_state)
    report.shape_mismatches = sorted(
        (k, tuple(v.shape), tuple(model_state[k].shape))
        for k, v in state_dict.items()
        if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)
    )
    missing = [k for k in model_state if k not in state_dict]
    for name in missing:
        report.missing_by_component.setdefault(_component_of(name), []).append(name)
    report.n_loaded = len(state_dict) - len(report.unexpected) - len(report.shape_mismatches)

    if report.unexpected:
        raise RuntimeError(
            f"checkpoint has {len(report.unexpected)} tensors with no home in the model, so it was "
            f"not trained on this architecture: {report.unexpected[:8]}\n{report.format()}"
        )
    if report.shape_mismatches:
        raise RuntimeError(
            f"{len(report.shape_mismatches)} tensors have the right name and the wrong shape:\n"
            f"{report.format()}"
        )
    not_conditioning = [name for name in missing if not name.startswith(model.CONDITIONING_PREFIXES)]
    if not_conditioning:
        raise RuntimeError(
            f"{len(not_conditioning)} base-UNet parameters were not in the checkpoint, so the base "
            f"is not the pretrained architecture: {sorted(not_conditioning)[:8]}\n{report.format()}"
        )

    torch_missing, torch_unexpected = model.load_state_dict(state_dict, strict=False)
    if torch_unexpected or sorted(torch_missing) != sorted(missing):
        raise RuntimeError(  # pragma: no cover - the pre-checks above make this unreachable
            f"load_state_dict disagreed with the pre-check: missing={len(torch_missing)} "
            f"unexpected={len(torch_unexpected)}"
        )

    loaded = model.state_dict()
    # `torch.equal` promotes, so a widening cast (fp16 file into an fp32 model) still compares equal
    # -- it is lossless. A narrowing one is not, and shows up as a difference.
    report.dtype_changes = sorted(k for k, v in state_dict.items() if loaded[k].dtype != v.dtype)
    report.unequal_after_load = sorted(
        k for k, v in state_dict.items() if not torch.equal(loaded[k].detach().cpu(), v.detach().cpu())
    )
    if report.unequal_after_load:
        raise RuntimeError(
            f"{len(report.unequal_after_load)} tensors differ from the checkpoint after loading "
            f"(a lossy dtype cast, or a parameter re-initialised after load): "
            f"{report.unequal_after_load[:8]}\n{report.format()}"
        )
    return report


__all__ = [
    "DEFAULT_CROSS_ATTENTION_DIM",
    "ContextProjection",
    "MaskedCrossAttention",
    "PretrainedLoadReport",
    "ReportConditionedUNetMaisi",
    "ReportCrossAttentionAdapter",
    "build_report_conditioned_unet",
    "load_pretrained_maisi_weights",
    "nvidia_unet_kwargs",
    "safe_sdpa_backends",
    "sdpa_backend_guard",
]
