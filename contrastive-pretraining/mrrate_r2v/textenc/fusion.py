"""Combining several frozen encoders into one `TextEmbedder`.

`MultiEncoderEmbedder` is itself a `TextEmbedder`, so the trainer, the sampler and
`ReportConditionedUNetMaisi` take it in place of a single encoder with no change: they only ever
read `output_dim` and call `encode`. The learned projection to `cross_attention_dim` stays where
it already is (`ContextProjection` inside the conditioned UNet), so fusion adds no second
trainable head by default.

Two fusion axes, because they answer different questions:

| `mode`    | Output                       | Use when |
|---|---|---|
| `"token"` | (B, sum(L_i), max(D_i))      | the denoiser should attend over each encoder's tokens |
| `"feature"` | (B, 1, sum(D_i))           | you want the 2025 CT-winner's pooled-concat recipe |

`"token"` keeps token-level detail -- the thing cross-attention exists to use -- and pads the
narrower encoders' features to the widest with zeros, which a linear projection can undo exactly.
`"feature"` reproduces Report2CT (VLM3D 2025 CT track, rank 1): masked-mean-pool each encoder,
concatenate the pooled vectors, condition on the single resulting vector. It throws away token
structure, so it is offered as a comparable baseline rather than as the recommended default.

`ProjectedConcatFusion` is the optional learned variant: per-encoder Linear to a shared width
before concatenation, so a 1024-wide and a 384-wide encoder contribute comparable magnitudes.
It is a plain `nn.Module` and is *not* coupled to any encoder pair -- it takes a list of input
dims.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ..text import TextConditioning, masked_mean

FUSION_MODES = ("token", "feature")


class ProjectedConcatFusion(nn.Module):
    """Per-encoder Linear -> LayerNorm, then concatenate along features.

    Built from a list of input dims, so it works for any number of encoders of any widths.
    """

    def __init__(self, input_dims: Sequence[int], projection_dim: int = 256) -> None:
        super().__init__()
        self.input_dims = list(input_dims)
        self.projection_dim = int(projection_dim)
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Linear(int(d), self.projection_dim), nn.LayerNorm(self.projection_dim))
            for d in self.input_dims
        )

    @property
    def output_dim(self) -> int:
        return self.projection_dim * len(self.input_dims)

    def forward(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(parts) != len(self.projections):
            raise ValueError(f"expected {len(self.projections)} inputs, got {len(parts)}")
        return torch.cat([p(x) for p, x in zip(self.projections, parts)], dim=-1)


class MultiEncoderEmbedder(nn.Module):
    """Several `TextEmbedder`s encoding the same reports, combined into one conditioning tensor.

    `fusion=None` (default) concatenates raw; pass a `ProjectedConcatFusion` to learn a projection
    first. With `mode="token"` and a fusion module, the fusion is applied per token position.
    """

    def __init__(self, embedders: Sequence, mode: str = "token",
                 fusion: Optional[ProjectedConcatFusion] = None, name: Optional[str] = None) -> None:
        super().__init__()
        if len(embedders) < 2:
            raise ValueError("MultiEncoderEmbedder needs at least two encoders; use one directly "
                             "otherwise.")
        if mode not in FUSION_MODES:
            raise ValueError(f"unknown fusion mode '{mode}'. Choose from: {FUSION_MODES}")
        self.embedders = nn.ModuleList(embedders)
        self.mode = mode
        self.fusion = fusion
        self._name = name or "+".join(e.identity.get("name", "?") for e in embedders)
        dims = [int(e.output_dim) for e in embedders]
        if fusion is not None:
            self._output_dim = fusion.output_dim
        elif mode == "token":
            self._output_dim = max(dims)   # narrower encoders are zero-padded up to the widest
        else:
            self._output_dim = sum(dims)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def identity(self) -> dict:
        return {
            "name": self._name,
            "kind": "multi_encoder",
            "mode": self.mode,
            "output_dim": self._output_dim,
            "fusion": None if self.fusion is None else
                      {"type": "projected_concat", "projection_dim": self.fusion.projection_dim},
            "members": [dict(e.identity) for e in self.embedders],
        }

    def train(self, mode: bool = True):
        super().train(mode)
        for embedder in self.embedders:          # each member enforces its own freeze policy
            embedder.train(mode)
        return self

    def encode(self, reports: Sequence[str], device) -> TextConditioning:
        parts = [e.encode(reports, device) for e in self.embedders]
        if self.mode == "feature":
            pooled = [p.pooled_embedding if p.pooled_embedding is not None
                      else masked_mean(p.token_embeddings, p.attention_mask) for p in parts]
            vector = (self.fusion(pooled) if self.fusion is not None else torch.cat(pooled, dim=-1))
            tokens = vector.unsqueeze(1)                                   # (B, 1, D)
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
            return TextConditioning(tokens, mask, vector, dict(self.identity))

        if self.fusion is not None:
            # Per-token projection needs a common token axis, so every member is padded to the
            # longest sequence in the batch before projecting.
            length = max(p.token_embeddings.shape[1] for p in parts)
            aligned = [_pad_tokens(p, length) for p in parts]
            tokens = self.fusion([t for t, _ in aligned])
            mask = torch.stack([m for _, m in aligned]).any(dim=0)
            return TextConditioning(tokens, mask, masked_mean(tokens, mask), dict(self.identity))

        width = self._output_dim
        tokens = torch.cat([_pad_features(p.token_embeddings, width) for p in parts], dim=1)
        mask = torch.cat([p.attention_mask for p in parts], dim=1)
        return TextConditioning(tokens, mask, masked_mean(tokens, mask), dict(self.identity))

    def log_truncation_summary(self) -> dict:
        return {e.identity.get("name", f"encoder{i}"): e.log_truncation_summary()
                for i, e in enumerate(self.embedders) if hasattr(e, "log_truncation_summary")}


def _pad_features(tokens: torch.Tensor, width: int) -> torch.Tensor:
    """Zero-pad the feature axis. A linear layer can undo this exactly, so no information is
    created and none is destroyed -- it only makes widths concatenable along the token axis."""
    missing = width - tokens.shape[-1]
    if missing == 0:
        return tokens
    if missing < 0:
        raise ValueError(f"token width {tokens.shape[-1]} exceeds fused width {width}")
    return torch.nn.functional.pad(tokens, (0, missing))


def _pad_tokens(part: TextConditioning, length: int):
    tokens, mask = part.token_embeddings, part.attention_mask
    missing = length - tokens.shape[1]
    if missing == 0:
        return tokens, mask
    tokens = torch.nn.functional.pad(tokens, (0, 0, 0, missing))
    mask = torch.nn.functional.pad(mask, (0, missing), value=False)
    return tokens, mask


__all__ = ["FUSION_MODES", "MultiEncoderEmbedder", "ProjectedConcatFusion"]
