"""Fusing several frozen encoders' pooled outputs into one wider vector, for
`textenc.conditioning.SectionedFusionEmbedder`.

`ProjectedConcatFusion` is the optional learned variant of that fusion: per-encoder Linear to a
shared width before concatenation, so a 1024-wide and a 384-wide encoder contribute comparable
magnitudes. It is a plain `nn.Module`, not coupled to any particular encoder pair -- it takes a
list of input dims.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


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


__all__ = ["ProjectedConcatFusion"]
