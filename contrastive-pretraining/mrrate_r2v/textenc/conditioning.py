"""The three supported report-conditioning configurations, as `TextEmbedder`s.

Everything here returns a `TextConditioning` with a **kept sequence axis**, so the trainer, the
sampler and `ReportConditionedUNetMaisi` need no branch on which configuration is in use:

    embeddings     (B, L, D)
    attention_mask (B, L)      True = attend

| configuration | class | L | D |
|---|---|---|---|
| A `cxr_bert` CLS | `PooledEmbedder` | 1 | 768 |
| B `radbert` masked mean | `PooledEmbedder` | 1 | 768 |
| C Report2CT-style fusion | `SectionedFusionEmbedder` | 2 | 2560 |
| E Report2CT-style + acquisition | `SectionedFusionEmbedder` | 3 | 2560 |
| A-tokens `cxr_bert` unpooled | `TokenSequenceEmbedder` | dynamic | 768 |
| B-tokens `radbert` unpooled | `TokenSequenceEmbedder` | dynamic | 768 |

**L=1 is a degenerate cross-attention.** Softmax over one key is identically 1, so the query and
key projections have no effect and get no gradient, and the adapter reduces to a per-channel bias
applied uniformly at every voxel. A and B are therefore *pooled-conditioning baselines*, not the
recommended path; the `_tokens` variants keep the token axis and make the attention real. See
`TokenSequenceEmbedder` for the measurements.

`build_conditioning_config(name)` is the only place these are assembled; `CONDITIONING_CONFIGS`
is the table `--conditioning` validates against. The projection to the UNet's
`cross_attention_dim` stays where it already was (`ContextProjection`), built from
`embedder.output_dim`, so no configuration hardcodes a width into the generative model.

**Configuration C is a Report2CT-*style* fusion, not a reproduction.** Verified from
`github.com/sinaamirrajab/report2ct@7b483a856ef159cfd0dada249b110d8f8eebf502`
(`vlm3d_inference.ipynb` cell 0 `encode_batch_multi`, and
`src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py:275-297`), the original uses:

- encoders, in this order: `abhinand/MedEmbed-large-v0.1` (1024), `medicalai/ClinicalBERT` (768),
  `microsoft/BiomedVLP-CXR-BERT-specialized` (768) -> **2560**;
- **masked mean** pooling of `last_hidden_state`, each encoder pooled *independently* and only
  then concatenated on the feature axis (`mean_pooling`, `mask.sum(dim=1).clamp(min=1e-9)`);
- findings and impression encoded **separately**, promoted to `(B, 1, 2560)` each and
  concatenated on the **sequence** axis, findings first -> `(B, 2, 2560)`;
- no LayerNorm, no learned projection: `cross_attention_dim` is set to 2560 outright
  (`vlm3D_work_dir/config_maisi_2560.json`).

Three deliberate differences here, each because the alternative is worse or impossible:

1. **`medicalai/ClinicalBERT` is substituted by `emilyalsentzer/Bio_ClinicalBERT`.** The original
   is a 6-layer DistilBERT; the substitute is 12-layer BERT-base. Both are 768-wide, so the fused
   width is 2560 either way, but they are *different checkpoints* -- which is why the name below
   is `report2ct_style` and not `report2ct`. Staging the original would make this exact.
2. **A conditioning mask is produced.** Report2CT has none (MAISI's `SpatialTransformer` takes no
   mask) and encodes a missing section as the empty string, so an absent impression contributes a
   real attention key. Impression is absent for 8.9% of MR-RATE studies, so here an empty section
   is masked out instead. With both sections empty the row is all-False, which
   `ReportConditionedUNetMaisi.prepare_context` already maps to the learned null embedding.
3. **The fused vector is projected, not consumed raw.** Report2CT set the UNet's
   `cross_attention_dim=2560`, which requires `with_conditioning=True` and therefore destroys
   NVIDIA's pretrained MR-Brain weight loading (see `models/report_conditioned_unet.py:263-268`).
   The existing `ContextProjection` maps 2560 -> `cross_attention_dim` instead.
"""
from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

import torch
from torch import nn

from ..text import TextConditioning, masked_mean
from .encoders import POOLING_MODES, build_encoder
from .formats import ACQUISITION_SECTION, ORDER_AGNOSTIC_META_SPEC
from .fusion import ProjectedConcatFusion

log = logging.getLogger("mrrate_r2v.textenc.conditioning")

#: Section order is explicit and part of the contract: index 0 is findings, index 1 impression.
#: Report2CT's own order (`torch.cat((context_f, context_i), dim=1)`).
REPORT2CT_SECTIONS = ("findings", "impression")

#: The same two sections plus a third token holding the acquisition metadata as text
#: (`[MODALITY] .. [PLANE] .. [SPACING] ..`), encoded by the same three encoders.
#:
#: **The gap it closes.** A/B/C put that prefix at the head of their joined string, so modality,
#: plane and spacing reach the text encoder as well as reaching the UNet numerically
#: (`class_labels`, `spacing_tensor`). A sectioned-fusion configuration never composes a joined
#: string, so under `REPORT2CT_SECTIONS` those three values are the one thing the report branch
#: cannot see. Appended, not inserted: findings and impression keep sequence indices 0 and 1, which
#: is what makes an E checkpoint's first two tokens the same quantity as a D checkpoint's.
REPORT2CT_META_SECTIONS = REPORT2CT_SECTIONS + (ACQUISITION_SECTION,)

#: Report2CT's verified encoder order, with `bio_clinicalbert` substituted for
#: `medicalai/ClinicalBERT` (see the module docstring). 1024 + 768 + 768 = 2560.
REPORT2CT_STYLE_ENCODERS = ("medembed_large", "bio_clinicalbert", "cxr_bert")


def _as_text(value) -> str:
    return "" if value is None else str(value)


class PooledEmbedder(nn.Module):
    """One encoder reduced to a single conditioning token: `(B, 1, D)` + `(B, 1)` mask.

    `pooling=None` takes the encoder's own `EncoderSpec.pooling`, which is the choice the
    text-encoder study was run under -- `cls` for `cxr_bert`, `mean` for everything else. Pass it
    explicitly to override.

    **What "CLS" means here**: `last_hidden_state[:, 0, :]`, the raw CLS state. Not a pooler
    output -- none of the staged checkpoints has a trained pooler, and `HFTextEncoder` passes
    `add_pooling_layer=False` precisely so a randomly initialised one cannot be mistaken for one
    (`encoders.py:266-269`). For `cxr_bert` this is also *pre*-projection: the checkpoint's
    CLIP `cls_projection_head` is dropped by the `bert_shim` loader.

    The mask is all-True. A pooled vector is a real token even for an empty report, and the
    encoder still emits `[CLS]`/`[SEP]`; a row that should be unconditional is selected by
    `context_drop_mask`, not by masking the only token away.
    """

    def __init__(self, inner, pooling: Optional[str] = None) -> None:
        super().__init__()
        pooling = pooling or getattr(getattr(inner, "spec", None), "pooling", "mean")
        if pooling not in POOLING_MODES:
            raise ValueError(f"unknown pooling '{pooling}'. Choose from: {POOLING_MODES}")
        self.inner = inner
        self.pooling = pooling

    @property
    def output_dim(self) -> int:
        return int(self.inner.output_dim)

    @property
    def identity(self) -> dict:
        return {
            "kind": "pooled",
            "pooling": self.pooling,
            "sequence_length": 1,
            "output_dim": self.output_dim,
            "encoder": dict(self.inner.identity),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.inner.train(mode)      # the member enforces its own freeze policy
        return self

    def encode(self, reports: Sequence[str], device) -> TextConditioning:
        part = self.inner.encode([_as_text(r) for r in reports], device)
        pooled = _pool(part, self.pooling)
        tokens = pooled.unsqueeze(1)
        if tokens.shape[1] != 1 or tokens.shape[-1] != self.output_dim:
            raise AssertionError(
                f"{type(self).__name__} produced {tuple(tokens.shape)}; expected "
                f"(B, 1, {self.output_dim}). This is a bug in the pooling path, not a config error."
            )
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return TextConditioning(tokens, mask, pooled, dict(self.identity))

    def log_truncation_summary(self) -> dict:
        report = getattr(self.inner, "log_truncation_summary", None)
        return report() if report else {}


class TokenSequenceEmbedder(nn.Module):
    """One encoder's **unpooled** `last_hidden_state`: `(B, L, D)` + its real padding mask.

    Why this exists: cross-attention over a *single* key is degenerate. Softmax over one element is
    identically 1 for every query, so `to_q` and `to_k` have no effect on the output and receive
    exactly zero gradient, and the attention output does not depend on the image at all -- the
    adapter collapses into a per-channel bias that is the same at every voxel. Measured on the real
    model: with L=1 the injected residual is spatially constant to 1.8e-05 and `to_q`/`to_k`/
    `proj_in`/`norm` gradients sit at 1e-11..1e-13 (roundoff), against 1e-4 for `to_v`. With L>1
    they come alive and the residual varies over space.

    Keeping the token axis is therefore not an optimisation, it is what makes the cross-attention a
    cross-attention. It is also the only configuration in which the report can express *where*
    something is rather than only *what*, because different voxels can attend to different tokens.

    `MaskedCrossAttention` was written for exactly this input: it honours `context_mask`, so the
    padding introduced by batching reports of different lengths cannot join the attention softmax.
    That is the property `tests/test_models_report_conditioned_unet.py::TestPadding` pins.

    `L` is dynamic -- the tokenizer pads to the longest report in the batch, not to `max_length` --
    so `sequence_length` in the identity records the *upper bound* (`max_length`), which is the
    stable quantity a checkpoint can be checked against.
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner

    @property
    def output_dim(self) -> int:
        return int(self.inner.output_dim)

    @property
    def sequence_length(self) -> int:
        """Upper bound, not the runtime length: batches are padded to their own longest report."""
        return int(getattr(self.inner, "max_length", 0))

    @property
    def identity(self) -> dict:
        return {
            # `kind` is what stops an adapter trained on pooled conditioning from loading onto this
            # one: both are 768-wide, so width alone cannot tell them apart and
            # `assert_conditioning_compatible` would pass a silently wrong checkpoint.
            "kind": "tokens",
            "pooling": None,
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
            "encoder": dict(self.inner.identity),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.inner.train(mode)
        return self

    def encode(self, reports: Sequence[str], device) -> TextConditioning:
        part = self.inner.encode([_as_text(r) for r in reports], device)
        tokens, mask = part.token_embeddings, part.attention_mask
        if tokens.ndim != 3 or tokens.shape[-1] != self.output_dim:
            raise AssertionError(
                f"{type(self).__name__} produced {tuple(tokens.shape)}; expected "
                f"(B, L, {self.output_dim}). This is a bug in the encoder, not a config error."
            )
        if tokens.shape[1] < 2:
            # Not fatal -- a one-token batch is legal -- but it is the degenerate case this class
            # exists to avoid, and silently training on it would waste the run.
            log.warning(
                "%s produced only %d token(s): cross-attention over a single key is degenerate "
                "(to_q/to_k get no gradient). Check the tokenizer and the report text.",
                type(self).__name__, tokens.shape[1],
            )
        # The pooled vector is carried for logging/probing only; the denoiser never reads it.
        return TextConditioning(tokens, mask, masked_mean(tokens, mask), dict(self.identity))

    def log_truncation_summary(self) -> dict:
        report = getattr(self.inner, "log_truncation_summary", None)
        return report() if report else {}


class SectionedFusionEmbedder(nn.Module):
    """Report2CT's conditioning shape: one token per report section, each token the feature-axis
    concatenation of every encoder's independently pooled vector.

    `(B, n_sections, sum(D_i))` + `(B, n_sections)` mask, with a section's mask entry False when
    that section's text is empty.

    Encoding order is fixed twice over and both orders are part of the checkpoint's identity:
    `sections` fixes the sequence axis, `embedders` fixes the feature axis. Neither is sorted or
    inferred, because a silently permuted 2560-vector is not detectable from a loss curve.
    """

    #: `encode` needs per-section text, so `text.encode_reports` routes the batch's
    #: `report_sections_text` here instead of its single `report_text` string.
    needs_sections = True

    def __init__(
        self,
        embedders: Sequence,
        sections: Sequence[str] = REPORT2CT_SECTIONS,
        pooling: str = "mean",
        fusion: Optional[ProjectedConcatFusion] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not embedders:
            raise ValueError("SectionedFusionEmbedder needs at least one encoder")
        if not sections:
            raise ValueError("SectionedFusionEmbedder needs at least one report section")
        if pooling not in POOLING_MODES:
            raise ValueError(f"unknown pooling '{pooling}'. Choose from: {POOLING_MODES}")
        self.embedders = nn.ModuleList(embedders)
        self.sections = tuple(sections)
        self.pooling = pooling
        self.fusion = fusion
        self._dims = [int(e.output_dim) for e in embedders]
        self._output_dim = fusion.output_dim if fusion is not None else sum(self._dims)
        self._name = name or "+".join(e.identity.get("name", "?") for e in embedders)

    @property
    def output_dim(self) -> int:
        return int(self._output_dim)

    @property
    def sequence_length(self) -> int:
        return len(self.sections)

    @property
    def identity(self) -> dict:
        return {
            "kind": "sectioned_fusion",
            "name": self._name,
            "sections": list(self.sections),           # sequence-axis order
            "encoder_order": [e.identity.get("name", "?") for e in self.embedders],
            "encoder_dims": list(self._dims),          # feature-axis slice widths, in order
            "pooling": self.pooling,
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
            "fusion": None if self.fusion is None else
                      {"type": "projected_concat", "projection_dim": self.fusion.projection_dim},
            # `pooling` is overwritten with the *effective* value. A member's own spec pooling is
            # ignored here -- the fusion pools every encoder the same way (Report2CT uses masked
            # mean for all three) -- and `cxr_bert`'s spec says "cls", so recording the member's
            # own value would put a flat contradiction in the checkpoint's provenance.
            "members": [dict(e.identity, pooling=self.pooling) for e in self.embedders],
        }

    def train(self, mode: bool = True):
        super().train(mode)
        for embedder in self.embedders:
            embedder.train(mode)
        return self

    def _encode_section(self, texts: Sequence[str], device) -> torch.Tensor:
        """One section, every encoder: `(B, sum(D_i))`. Each encoder is pooled on its *own*
        tokenisation before concatenation, which is what makes three different tokenizers
        composable at all -- there is no token-level correspondence between them to align."""
        pooled = [_pool(e.encode(texts, device), self.pooling) for e in self.embedders]
        if self.fusion is not None:
            return self.fusion(pooled)
        return torch.cat(pooled, dim=-1)

    def encode_sections(self, sections: Sequence[Mapping[str, str]], device) -> TextConditioning:
        batch = len(sections)
        if batch == 0:
            raise ValueError("encode_sections received an empty batch")
        per_section, present = [], []
        for name in self.sections:
            texts = [_as_text(item.get(name) if item else "").strip() for item in sections]
            per_section.append(self._encode_section(texts, device).unsqueeze(1))   # (B, 1, D)
            present.append([bool(t) for t in texts])

        tokens = torch.cat(per_section, dim=1)                                    # (B, S, D)
        mask = torch.tensor(present, dtype=torch.bool, device=tokens.device).t().contiguous()
        expected = (batch, self.sequence_length, self.output_dim)
        if tuple(tokens.shape) != expected:
            raise AssertionError(
                f"{type(self).__name__} produced {tuple(tokens.shape)}; expected {expected}. "
                f"Section order {self.sections}, encoder dims {self._dims}."
            )
        return TextConditioning(tokens, mask, masked_mean(tokens, mask), dict(self.identity))

    def encode(self, reports, device) -> TextConditioning:
        """Accepts per-section mappings, or plain strings as a documented fallback.

        A plain string cannot be split back into sections without parsing, so every string is
        routed to the *first* section and the rest are masked out. That keeps a stray caller
        working and shaped correctly rather than crashing, but it is not the intended path --
        `text.encode_reports` supplies `report_sections_text` and is what training, validation
        and sampling all use.
        """
        items = []
        for report in reports:
            if isinstance(report, Mapping):
                items.append(report)
            else:
                log.warning(
                    "%s.encode received a plain string; routing it to section '%s' and masking "
                    "the rest. Pass per-section text (batch['report_sections_text']) instead.",
                    type(self).__name__, self.sections[0],
                )
                items.append({self.sections[0]: _as_text(report)})
        return self.encode_sections(items, device)

    def log_truncation_summary(self) -> dict:
        return {e.identity.get("name", f"encoder{i}"): e.log_truncation_summary()
                for i, e in enumerate(self.embedders) if hasattr(e, "log_truncation_summary")}


def _pool(part: TextConditioning, pooling: str) -> torch.Tensor:
    """`(B, L, D)` -> `(B, D)`. Padding is excluded from the mean; `masked_mean` divides by the
    real-token count, never by `L`."""
    if pooling == "cls":
        return part.token_embeddings[:, 0]
    return masked_mean(part.token_embeddings, part.attention_mask)


# --------------------------------------------------------------------------- the three configs


#: name -> everything needed to build it and to describe it in a checkpoint. `report_format` is
#: the *recommended* format, applied unless the CLI overrides it, and is recorded either way.
#:
#: A and B recommend `ORDER_AGNOSTIC_META_SPEC` -- two formats, sampled per training sample. The
#: challenge's report layout is unknown, and a model trained on one fixed section order has no way
#: to signal at submission time that the order it was given is not the one it learned.
CONDITIONING_CONFIGS: dict[str, dict] = {
    "cxr_bert_cls": {
        "kind": "pooled",
        "encoders": ("cxr_bert",),
        "pooling": "cls",
        "report_format": ORDER_AGNOSTIC_META_SPEC,
        "sequence_length": 1,
        "output_dim": 768,
        "note": "Configuration A (pooled baseline). Raw last_hidden_state[:, 0, :]; matches the encoder spec's own "
                "pooling, under which the text-encoder study scored it best of the single encoders.",
    },
    "radbert_mean": {
        "kind": "pooled",
        "encoders": ("radbert",),
        "pooling": "mean",
        "report_format": ORDER_AGNOSTIC_META_SPEC,
        "sequence_length": 1,
        "output_dim": 768,
        "note": "Superseded, out of the lettered lineup (use C). Masked mean, per the encoder spec. RadBERT is a "
                "RobertaForMaskedLM with no pooler and no sentence-level objective, so its <s> "
                "state was never trained to summarise -- use --text-pooling cls only for an "
                "explicit ablation.",
    },
    "radbert_tokens": {
        "kind": "tokens",
        "encoders": ("radbert",),
        "pooling": None,                     # nothing is pooled; the token axis is kept
        "report_format": ORDER_AGNOSTIC_META_SPEC,
        "sequence_length": None,             # dynamic; the embedder records max_length
        "output_dim": 768,
        "note": "Configuration C. RadBERT's full last_hidden_state, unpooled. RadBERT is a "
                "RobertaForMaskedLM with no pooler and no sentence-level objective, so there is no "
                "trained summary vector to pool *to* -- masked-mean is an unweighted average over "
                "token states chosen for lack of an alternative. Keeping the tokens removes both "
                "that arbitrary reduction and the single-key attention collapse it caused.",
    },
    "cxr_bert_tokens": {
        "kind": "tokens",
        "encoders": ("cxr_bert",),
        "pooling": None,
        "report_format": ORDER_AGNOSTIC_META_SPEC,
        "sequence_length": None,
        "output_dim": 768,
        "note": "Configuration B. CXR-BERT's full last_hidden_state. Unlike RadBERT its CLS "
                "*was* trained to summarise (CLIP objective), so this is a genuine A/B against "
                "cxr_bert_cls rather than a strict improvement: it trades a supervised summary "
                "vector for a non-degenerate attention that can localise.",
    },
    "report2ct_style": {
        "kind": "sectioned_fusion",
        "encoders": REPORT2CT_STYLE_ENCODERS,
        "pooling": "mean",
        "sections": REPORT2CT_SECTIONS,
        "report_format": None,               # sections are encoded separately, never joined
        "sequence_length": 2,
        "output_dim": 2560,
        "note": "Configuration D. Report2CT-style: masked-mean each encoder, concat features to "
                "2560, one token per section, findings first. bio_clinicalbert substitutes for "
                "Report2CT's medicalai/ClinicalBERT -- same width, different checkpoint.",
    },
    "report2ct_style_meta": {
        "kind": "sectioned_fusion",
        "encoders": REPORT2CT_STYLE_ENCODERS,
        "pooling": "mean",
        "sections": REPORT2CT_META_SECTIONS,
        "report_format": None,               # sections are encoded separately, never joined
        "sequence_length": 3,
        "output_dim": 2560,
        "note": "Configuration E. Configuration D plus a third token holding "
                "[MODALITY]/[PLANE]/[SPACING] as text, so the acquisition metadata A/B/C get from "
                "their format prefix reaches the report branch here too. Identical encoders, "
                "pooling and feature-axis order to D; findings and impression keep indices 0 and 1.",
    },
}

DEFAULT_CONDITIONING = "radbert_mean"


def build_conditioning(
    name: str,
    *,
    max_length: Optional[int] = None,
    pooling: Optional[str] = None,
    checkpoints: Optional[Mapping[str, str]] = None,
    trainable: bool = False,
    dtype: Optional[torch.dtype] = None,
    pretrained_dir: Optional[str] = None,
    projection_dim: Optional[int] = None,
):
    """Assemble one named configuration. The only place the three are constructed.

    `pooling` overrides the config's own choice for every member encoder. `checkpoints` maps an
    encoder name to a local directory, for encoders staged somewhere other than
    `MRRATE_PRETRAINED_DIR`. `trainable=False` is the default and the study-recommended
    behaviour: every encoder goes to eval mode permanently and drops gradients, so no activation
    is retained for a backward pass that will never reach it.
    """
    if name not in CONDITIONING_CONFIGS:
        raise ValueError(
            f"unknown conditioning configuration '{name}'. Choose from: "
            f"{sorted(CONDITIONING_CONFIGS)}"
        )
    spec = CONDITIONING_CONFIGS[name]
    checkpoints = dict(checkpoints or {})

    def make(encoder_name: str):
        kwargs = {"trainable": trainable}
        if max_length is not None:
            kwargs["max_length"] = max_length
        if dtype is not None:
            kwargs["dtype"] = dtype
        if pretrained_dir is not None:
            kwargs["pretrained_dir"] = pretrained_dir
        if encoder_name in checkpoints:
            kwargs["checkpoint"] = str(checkpoints[encoder_name])
        return build_encoder(encoder_name, **kwargs)

    members = [make(n) for n in spec["encoders"]]
    resolved_pooling = pooling or spec["pooling"]

    if spec["kind"] == "tokens":
        if pooling:
            raise ValueError(
                f"conditioning '{name}' keeps the token axis, so --text-pooling {pooling!r} has "
                f"nothing to act on. Use a pooled configuration (e.g. "
                f"{'cxr_bert_cls' if 'cxr' in name else 'radbert_mean'}) to pool."
            )
        embedder = TokenSequenceEmbedder(members[0])
    elif spec["kind"] == "pooled":
        embedder = PooledEmbedder(members[0], pooling=resolved_pooling)
    else:
        fusion = None
        if projection_dim:
            fusion = ProjectedConcatFusion([m.output_dim for m in members], projection_dim)
        embedder = SectionedFusionEmbedder(
            members, sections=spec["sections"], pooling=resolved_pooling, fusion=fusion, name=name,
        )

    expected_dim = spec.get("output_dim")
    if expected_dim and not projection_dim and embedder.output_dim != expected_dim:
        raise AssertionError(
            f"conditioning '{name}' built with output_dim={embedder.output_dim}, but the config "
            f"table says {expected_dim}. A staged checkpoint's hidden size does not match what "
            f"this configuration was defined against -- check which snapshot is in "
            f"{pretrained_dir or 'MRRATE_PRETRAINED_DIR'}."
        )
    length = getattr(embedder, "sequence_length", 1)
    # For a token configuration `sequence_length` is the cap, not the runtime length: batches are
    # padded to their own longest report. Printing a bare "L=512" invites reading it as fixed.
    shown = f"L<={length} (variable, padded per batch)" if spec["kind"] == "tokens" else f"L={length}"
    log.info("conditioning '%s': %s D=%d (%s)", name, shown, embedder.output_dim, spec["note"])
    return embedder


__all__ = [
    "CONDITIONING_CONFIGS",
    "DEFAULT_CONDITIONING",
    "PooledEmbedder",
    "REPORT2CT_META_SECTIONS",
    "REPORT2CT_SECTIONS",
    "REPORT2CT_STYLE_ENCODERS",
    "SectionedFusionEmbedder",
    "TokenSequenceEmbedder",
    "build_conditioning",
]
