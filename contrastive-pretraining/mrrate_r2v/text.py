"""The report -> embedding seam. The trainer, the sampler and the conditioned denoiser depend on
`TextEmbedder`/`TextConditioning` only, never on a specific encoder, so swapping RadBERT for another
model changes one config value and nothing else.

    build_text_embedder("radbert", checkpoint=<dir>)  ->  RadBertEmbedder    (frozen, eval, no_grad)
    build_text_embedder("mock", output_dim=32)        ->  MockTextEmbedder   (deterministic, CPU tests)

What deliberately lives *outside* the embedder: the learned projection from `embedder.output_dim` to
the denoiser's `cross_attention_dim`. That is `ContextProjection` inside
`ReportConditionedUNetMaisi`, is trainable, and is built from `embedder.output_dim` at construction
time -- so a 1024-wide encoder needs no code change, only a rebuilt adapter.

`attention_mask` is HuggingFace's convention throughout: 1/True = real token, 0/False = padding.
`ReportConditionedUNetMaisi.forward(context_mask=...)` takes it verbatim.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import torch

log = logging.getLogger("mrrate_r2v.text")

DEFAULT_MAX_REPORT_TOKENS = 512


@dataclass
class TextConditioning:
    """One batch of encoded reports.

    token_embeddings: (B, L, output_dim) float -- what the adapter's K/V are computed from.
    attention_mask:   (B, L) bool -- True = real token. Never all-False for a row; an empty report
                      still carries its start/end tokens, and the denoiser treats an all-padding row
                      as unconditional anyway.
    pooled_embedding: (B, output_dim) or None -- a mask-aware mean, for logging/probing only. The
                      denoiser never reads it.
    metadata:         encoder identity and tokenisation settings; goes into the checkpoint and the
                      inference manifest so a run can be reproduced.
    """

    token_embeddings: torch.Tensor
    attention_mask: torch.Tensor
    pooled_embedding: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_embeddings.ndim != 3:
            raise ValueError(f"token_embeddings must be (B, L, D), got {tuple(self.token_embeddings.shape)}")
        expected = self.token_embeddings.shape[:2]
        if tuple(self.attention_mask.shape) != tuple(expected):
            raise ValueError(
                f"attention_mask {tuple(self.attention_mask.shape)} does not match token_embeddings "
                f"{tuple(self.token_embeddings.shape)} -- expected {tuple(expected)}"
            )
        if self.attention_mask.dtype != torch.bool:
            self.attention_mask = self.attention_mask.to(torch.bool)

    def to(self, device=None, dtype=None) -> "TextConditioning":
        return TextConditioning(
            token_embeddings=self.token_embeddings.to(device=device, dtype=dtype),
            attention_mask=self.attention_mask.to(device=device),
            pooled_embedding=None if self.pooled_embedding is None
            else self.pooled_embedding.to(device=device, dtype=dtype),
            metadata=dict(self.metadata),
        )


@runtime_checkable
class TextEmbedder(Protocol):
    """A frozen report encoder. `encode` must be side-effect free and gradient-free."""

    @property
    def output_dim(self) -> int:
        """Width of `token_embeddings`. The trainable projection head is built from this."""

    @property
    def identity(self) -> dict:
        """What this encoder is, for the checkpoint and the cache key."""

    def encode(self, reports: Sequence[str], device: torch.device) -> TextConditioning: ...


def masked_mean(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over real tokens only. `clamp(min=1)` is not a fudge: a row with no real tokens would
    divide by zero, and the model treats such a row as unconditional regardless."""
    weights = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


# --------------------------------------------------------------------------- mock


class MockTextEmbedder(torch.nn.Module):
    """Deterministic stand-in with no dependencies, for CPU tests and dry runs.

    The embedding is a hash of the report text, so the same report always gives the same vector and
    two different reports give different ones -- which is all a test of the conditioning path needs.
    Token count is `min(word_count + 2, max_length)`, so padding and truncation behave like a real
    tokeniser's.
    """

    def __init__(self, output_dim: int = 32, max_length: int = 16) -> None:
        super().__init__()
        self._output_dim = int(output_dim)
        self.max_length = int(max_length)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def identity(self) -> dict:
        return {"name": "mock", "output_dim": self._output_dim, "max_length": self.max_length,
                "tokenizer": "whitespace"}

    def _tokens(self, report: str) -> int:
        return max(1, min(len(report.split()) + 2, self.max_length))

    @torch.no_grad()
    def encode(self, reports: Sequence[str], device: torch.device) -> TextConditioning:
        reports = list(reports)
        lengths = [self._tokens(r) for r in reports]
        length = max(lengths)
        tokens = torch.zeros(len(reports), length, self._output_dim)
        mask = torch.zeros(len(reports), length, dtype=torch.bool)
        for i, (report, n) in enumerate(zip(reports, lengths)):
            for t in range(n):
                seed = int(hashlib.sha256(f"{t}:{report}".encode()).hexdigest()[:8], 16)
                generator = torch.Generator().manual_seed(seed)
                tokens[i, t] = torch.randn(self._output_dim, generator=generator)
            mask[i, :n] = True
        tokens, mask = tokens.to(device), mask.to(device)
        return TextConditioning(tokens, mask, masked_mean(tokens, mask), dict(self.identity))


# --------------------------------------------------------------------------- RadBERT


def ensure_local_safetensors(checkpoint_dir, force: bool = False) -> Optional[str]:
    """Write `model.safetensors` next to a `pytorch_model.bin`-only checkpoint, if needed.

    transformers >= 5 refuses to `torch.load` a `.bin` when torch < 2.6 (CVE-2025-32434), and the
    RadBERT snapshot ships no safetensors file. Converting once is the documented way out; the
    alternative -- upgrading torch -- would break this cluster's torchvision build. The `.bin` is
    left untouched, only a new file is added, and `roberta.`-prefixed keys are stripped so the file
    matches the encoder-only model that gets loaded (`lm_head.*` and the legacy `position_ids`
    buffer are dropped: nothing downstream uses them).

    Returns the path written, or None if nothing was needed.
    """
    from pathlib import Path

    checkpoint_dir = Path(checkpoint_dir)
    target = checkpoint_dir / "model.safetensors"
    if target.is_file() and not force:
        return None
    source = checkpoint_dir / "pytorch_model.bin"
    if not source.is_file():
        return None
    if torch.__version__ >= "2.6":
        return None  # transformers can read the .bin directly

    import safetensors.torch

    state = torch.load(str(source), map_location="cpu", weights_only=True)
    prefix = "roberta."
    encoder = {
        key[len(prefix):] if key.startswith(prefix) else key: value.clone()
        for key, value in state.items()
        if not key.startswith("lm_head.") and not key.endswith("position_ids")
    }
    if not encoder:
        raise RuntimeError(f"no encoder tensors found in {source}")
    safetensors.torch.save_file(encoder, str(target), metadata={"format": "pt"})
    log.warning(
        "converted %s -> %s (%d tensors): transformers>=5 will not torch.load a .bin under "
        "torch %s (<2.6). The .bin was not modified.",
        source.name, target.name, len(encoder), torch.__version__,
    )
    return str(target)


class RadBertEmbedder(torch.nn.Module):
    """RadBERT (`zzxslp/RadBERT-RoBERTa-4m`, a RoBERTa-base trained on 4M radiology reports),
    frozen, in eval mode, encoding under `no_grad`.

    Loaded with `local_files_only=True` and `trust_remote_code=False`: the snapshot is a stock
    `roberta` architecture, so no repository code is executed. `add_pooling_layer=False` is not
    cosmetic -- the checkpoint is a `RobertaForMaskedLM` and has no pooler, so a pooler would be
    *randomly initialised* and its `pooler_output` meaningless. `pooled_embedding` is a mask-aware
    mean of the token states instead.
    """

    def __init__(
        self,
        checkpoint: str,
        max_length: int = DEFAULT_MAX_REPORT_TOKENS,
        dtype: Optional[torch.dtype] = None,
        allow_safetensors_conversion: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(
                f"text encoder checkpoint directory not found: {checkpoint}. Pass --text-encoder "
                "mock for a CPU dry run, or stage the RadBERT snapshot first."
            )
        if allow_safetensors_conversion:
            ensure_local_safetensors(checkpoint)

        self.checkpoint = str(checkpoint)
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            checkpoint, config=config, local_files_only=True, trust_remote_code=False,
            add_pooling_layer=False,
        )
        self._output_dim = int(config.hidden_size)
        # RoBERTa spends 2 of its position embeddings on <s>/</s> offsets, so the usable token budget
        # is max_position_embeddings - 2. Silently exceeding it is a CUDA index error at step 1.
        self.encoder_max_length = int(min(config.max_position_embeddings - 2, self.tokenizer.model_max_length))
        if max_length > self.encoder_max_length:
            raise ValueError(
                f"max_length={max_length} exceeds what this encoder supports "
                f"({self.encoder_max_length}); reports longer than that must be truncated."
            )
        self.max_length = int(max_length)

        if dtype is not None:
            self.model = self.model.to(dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def identity(self) -> dict:
        return {
            "name": "radbert",
            "checkpoint": self.checkpoint,
            "model_type": self.model.config.model_type,
            "output_dim": self._output_dim,
            "max_length": self.max_length,
            "encoder_max_length": self.encoder_max_length,
            "tokenizer": type(self.tokenizer).__name__,
            "vocab_size": int(self.model.config.vocab_size),
        }

    def train(self, mode: bool = True):  # noqa: D102 - the encoder is frozen, permanently in eval
        return super().train(False)

    @torch.no_grad()
    def encode(self, reports: Sequence[str], device: torch.device) -> TextConditioning:
        reports = ["" if r is None else str(r) for r in reports]
        batch = self.tokenizer(
            reports, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        self.model.eval()
        if next(self.model.parameters()).device != torch.device(device):
            self.model.to(device)
        output = self.model(**batch)
        tokens = output.last_hidden_state
        mask = batch["attention_mask"].to(torch.bool)
        metadata = dict(self.identity)
        metadata["n_tokens"] = int(tokens.shape[1])
        return TextConditioning(tokens, mask, masked_mean(tokens, mask), metadata)


# --------------------------------------------------------------------------- registry

TEXT_EMBEDDERS = {"radbert": RadBertEmbedder, "mock": MockTextEmbedder}


def _textenc_names():
    """The `textenc` zoo's encoder names, or () if that subpackage's deps are unavailable.

    Imported lazily and defensively so this module keeps working exactly as before on an
    interpreter where the zoo cannot be imported -- `radbert` and `mock` never depend on it.
    """
    try:
        from .textenc.encoders import ENCODER_SPECS
    except Exception:  # noqa: BLE001 -- availability probe, never fatal
        return ()
    return tuple(ENCODER_SPECS)


def build_text_embedder(name: str, **kwargs) -> TextEmbedder:
    """The one place a concrete encoder class is named. Everything else takes a `TextEmbedder`.

    Names resolve in two tiers: this module's own `TEXT_EMBEDDERS` first (so `radbert` keeps its
    long-standing behaviour byte-for-byte), then `textenc.ENCODER_SPECS` for the encoder zoo.
    """
    if name not in TEXT_EMBEDDERS:
        if name in _textenc_names():
            from .textenc.encoders import build_encoder

            embedder = build_encoder(name, **kwargs)
            if embedder.output_dim <= 0:
                raise ValueError(f"text encoder '{name}' reported output_dim={embedder.output_dim}")
            return embedder
        raise ValueError(
            f"unknown text encoder '{name}'. Choose from: "
            f"{sorted(set(TEXT_EMBEDDERS) | set(_textenc_names()))}"
        )
    embedder = TEXT_EMBEDDERS[name](**kwargs)
    if embedder.output_dim <= 0:
        raise ValueError(f"text encoder '{name}' reported output_dim={embedder.output_dim}")
    return embedder


class ReportEncodingCache:
    """Optional memo for `TextEmbedder.encode` on single reports. Correctness never depends on it:
    a miss just re-encodes. The key covers the encoder identity, so two encoders, two max lengths or
    two tokenisers can never collide.
    """

    def __init__(self, embedder: TextEmbedder, max_entries: int = 4096) -> None:
        self.embedder = embedder
        self.max_entries = max_entries
        self._entries: dict[str, TextConditioning] = {}
        identity = sorted(embedder.identity.items())
        self._identity_key = hashlib.sha256(repr(identity).encode()).hexdigest()[:16]
        self.hits = self.misses = 0

    def key(self, report: str) -> str:
        return f"{self._identity_key}:{hashlib.sha256(report.encode()).hexdigest()}"

    def encode_one(self, report: str, device: torch.device) -> TextConditioning:
        key = self.key(report)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return cached.to(device=device)
        self.misses += 1
        encoded = self.embedder.encode([report], device=torch.device("cpu"))
        if len(self._entries) < self.max_entries:
            self._entries[key] = encoded
        return encoded.to(device=device)


__all__ = [
    "DEFAULT_MAX_REPORT_TOKENS",
    "MockTextEmbedder",
    "RadBertEmbedder",
    "ReportEncodingCache",
    "TEXT_EMBEDDERS",
    "TextConditioning",
    "TextEmbedder",
    "build_text_embedder",
    "ensure_local_safetensors",
    "masked_mean",
]
