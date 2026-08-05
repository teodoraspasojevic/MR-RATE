"""The text-encoder zoo. Every entry implements `mrrate_r2v.text.TextEmbedder`, so anything that
already accepts `RadBertEmbedder` accepts these unchanged.

    build_encoder("bioclinical_mbert")                  # cluster default checkpoint directory
    build_encoder("radbert", checkpoint="/some/dir")     # explicit path
    build_encoder("medembed_small", max_length=256, trainable=True)

One class (`HFTextEncoder`) covers every checkpoint; the differences live in `ENCODER_SPECS` as
data. Adding an encoder is one dict entry plus a staged snapshot -- no new class, no new branch.

**Three things this deliberately does that the code it generalises did not:**

1. **Truncation is counted, never silent.** `encode` tokenises once without truncation, records
   how many sequences overflowed and by how much, then truncates. `encoder.truncation` is a live
   counter and `log_truncation_summary()` prints it. A conditioning signal that quietly loses its
   last 200 tokens is the kind of bug that shows up only as slightly worse FID.
2. **No `trust_remote_code`, ever.** CXR-BERT ships a custom `cxr-bert` model type whose repo code
   would otherwise have to be executed; `loader="bert_shim"` loads its `bert.*` weights into a
   stock `BertModel` instead. The MLM head and the CLIP projection head are dropped -- neither is
   used for token or mean-pooled embeddings.
3. **Freezing is configurable but frozen is the default.** `trainable=False` puts the model in
   eval mode permanently and drops gradients; `trainable=True` leaves it to the caller.

`max_length` defaults to `min(spec default, what the checkpoint supports)` and is validated
against the checkpoint's real position budget, so an over-long request fails at construction
rather than as a CUDA index error at step 1.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from ..text import TextConditioning, masked_mean

log = logging.getLogger("mrrate_r2v.textenc")

#: Where staged checkpoints live on this cluster. Override per encoder with `checkpoint=`,
#: or globally with the MRRATE_PRETRAINED_DIR environment variable.
DEFAULT_PRETRAINED_DIR = os.environ.get(
    "MRRATE_PRETRAINED_DIR", "/hnvme/workspace/y100dc19-nvidia-mri-brain/pretrained"
)

POOLING_MODES = ("mean", "cls")


@dataclass(frozen=True)
class EncoderSpec:
    """What a checkpoint is and how to load it. `hidden` and `context` are the published values,
    used for planning and for the comparison table; the real ones are re-read from the checkpoint
    at construction and are what the code actually uses."""

    name: str
    directory: str          # subdirectory of DEFAULT_PRETRAINED_DIR
    hf_repo: str
    loader: str = "auto"    # "auto" = AutoModel/AutoTokenizer; "bert_shim" = force stock BERT
    pooling: str = "mean"
    hidden: int = 768
    context: int = 512      # published usable token budget
    default_max_length: int = 512
    domain: str = ""
    license: str = ""
    note: str = ""


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "radbert": EncoderSpec(
        name="radbert", directory="RadBERT-RoBERTa-4m", hf_repo="zzxslp/RadBERT-RoBERTa-4m",
        hidden=768, context=512, domain="radiology reports (4M, US VA)", license="apache-2.0",
        note="RoBERTa-base continued-pretrained on radiology reports; no pooler in the checkpoint",
    ),
    "bioclinical_mbert": EncoderSpec(
        name="bioclinical_mbert", directory="BioClinical-ModernBERT-base",
        hf_repo="thomas-sounack/BioClinical-ModernBERT-base",
        hidden=768, context=8192, default_max_length=1024,
        domain="biomedical + 20 clinical corpora incl. radiology and brain-MRI reports",
        license="mit", note="8192-token context: no truncation on MR-RATE at any format",
    ),
    "medembed_large": EncoderSpec(
        name="medembed_large", directory="MedEmbed-large-v0.1", hf_repo="abhinand/MedEmbed-large-v0.1",
        hidden=1024, context=512, domain="medical retrieval fine-tune of BGE-large",
        license="apache-2.0", note="one of the three encoders in the 2025 VLM3D CT-track winner",
    ),
    "medembed_small": EncoderSpec(
        name="medembed_small", directory="MedEmbed-small-v0.1", hf_repo="abhinand/MedEmbed-small-v0.1",
        hidden=384, context=512, domain="medical retrieval fine-tune of BGE-small",
        license="apache-2.0", note="lightweight candidate: 33M parameters, 384-wide",
    ),
    "bio_clinicalbert": EncoderSpec(
        name="bio_clinicalbert", directory="Bio_ClinicalBERT", hf_repo="emilyalsentzer/Bio_ClinicalBERT",
        hidden=768, context=512, domain="MIMIC-III clinical notes", license="mit",
        note="pretrained at sequence length 128; long reports are outside its pretraining regime",
    ),
    "cxr_bert": EncoderSpec(
        name="cxr_bert", directory="BiomedVLP-CXR-BERT-specialized",
        hf_repo="microsoft/BiomedVLP-CXR-BERT-specialized", loader="bert_shim", pooling="cls",
        hidden=768, context=512, domain="chest X-ray reports (CLIP-aligned)", license="mit",
        note="custom cxr-bert model type; loaded as stock BERT to avoid trust_remote_code",
    ),
    "modernbert": EncoderSpec(
        name="modernbert", directory="ModernBERT-base", hf_repo="answerdotai/ModernBERT-base",
        hidden=768, context=8192, default_max_length=1024, domain="general English (control)",
        license="apache-2.0",
        note="architecture-matched control for bioclinical_mbert: isolates the domain adaptation",
    ),
    "bge_base": EncoderSpec(
        name="bge_base", directory="bge-base-en-v1.5", hf_repo="BAAI/bge-base-en-v1.5",
        hidden=768, context=512, domain="general English retrieval (control)", license="mit",
        note="MedEmbed's own base model: isolates the medical fine-tune",
    ),
}


@dataclass
class TruncationCounter:
    """Live truncation bookkeeping. Reported, never swallowed."""

    n_sequences: int = 0
    n_truncated: int = 0
    max_length: int = 0
    longest_seen: int = 0
    dropped_tokens: int = 0
    _warned: bool = field(default=False, repr=False)

    @property
    def fraction(self) -> float:
        return self.n_truncated / self.n_sequences if self.n_sequences else 0.0

    def observe(self, lengths: Sequence[int], max_length: int) -> None:
        self.max_length = max_length
        self.n_sequences += len(lengths)
        for n in lengths:
            self.longest_seen = max(self.longest_seen, n)
            if n > max_length:
                self.n_truncated += 1
                self.dropped_tokens += n - max_length

    def as_dict(self) -> dict:
        return {"n_sequences": self.n_sequences, "n_truncated": self.n_truncated,
                "fraction_truncated": self.fraction, "max_length": self.max_length,
                "longest_seen": self.longest_seen, "dropped_tokens": self.dropped_tokens}

    def snapshot(self) -> "TruncationCounter":
        """A copy of the current totals. `since(snapshot)` then gives the rate for one stretch of
        encoding rather than for the encoder's whole lifetime -- which matters because one encoder
        object is reused across several formats and splits, and a cumulative percentage would
        attribute one format's truncation to the next."""
        return TruncationCounter(self.n_sequences, self.n_truncated, self.max_length,
                                 self.longest_seen, self.dropped_tokens)

    def since(self, earlier: "TruncationCounter") -> dict:
        n = self.n_sequences - earlier.n_sequences
        truncated = self.n_truncated - earlier.n_truncated
        return {"n_sequences": n, "n_truncated": truncated,
                "fraction_truncated": truncated / n if n else 0.0,
                "max_length": self.max_length, "longest_seen": self.longest_seen,
                "dropped_tokens": self.dropped_tokens - earlier.dropped_tokens}


class HFTextEncoder(torch.nn.Module):
    """A HuggingFace encoder behind the `TextEmbedder` interface.

    `encode(reports, device)` returns a `TextConditioning`:
        token_embeddings  (B, L, output_dim)  float
        attention_mask    (B, L)              bool, True = real token
        pooled_embedding  (B, output_dim)     mask-aware mean, or the CLS state
    """

    def __init__(
        self,
        spec: EncoderSpec,
        checkpoint: Optional[str] = None,
        max_length: Optional[int] = None,
        pooling: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trainable: bool = False,
        pretrained_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.checkpoint = str(checkpoint or os.path.join(pretrained_dir or DEFAULT_PRETRAINED_DIR,
                                                         spec.directory))
        if not os.path.isdir(self.checkpoint):
            raise FileNotFoundError(
                f"text encoder '{spec.name}' checkpoint directory not found: {self.checkpoint}. "
                f"Stage it with `python -m mrrate_r2v.cli.download_text_encoders --encoders "
                f"{spec.name}`, pass checkpoint=..., or set MRRATE_PRETRAINED_DIR."
            )
        pooling = pooling or spec.pooling
        if pooling not in POOLING_MODES:
            raise ValueError(f"unknown pooling '{pooling}'. Choose from: {POOLING_MODES}")
        self.pooling = pooling

        self.tokenizer, self.model, self.config = self._load(spec, self.checkpoint)
        self._output_dim = int(self.config.hidden_size)
        self.encoder_max_length = self._resolve_encoder_max_length()

        requested = int(max_length if max_length is not None else spec.default_max_length)
        if requested > self.encoder_max_length:
            raise ValueError(
                f"max_length={requested} exceeds what '{spec.name}' supports "
                f"({self.encoder_max_length}). Lower --max-report-tokens, or use an encoder with a "
                f"longer context (e.g. bioclinical_mbert)."
            )
        self.max_length = requested
        self.truncation = TruncationCounter()

        if dtype is not None:
            self.model = self.model.to(dtype)
        self.trainable = bool(trainable)
        if not self.trainable:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    # ------------------------------------------------------------------ loading

    @staticmethod
    def _load_tokenizer(spec: EncoderSpec, checkpoint: str):
        """The one place a tokenizer is constructed. Do not inline this anywhere else.

        It exists as a separate function because it was once duplicated -- `cli.analyze_reports`
        had its own copy so it could measure token lengths without paying for the weights -- and
        when the shim was fixed in `_load`, the copy kept the bug and kept reporting CXR-BERT as
        30% more token-efficient than every other encoder. It was not: it was emitting one [UNK]
        per word. Anything that needs a tokenizer calls this.
        """
        if spec.loader == "bert_shim":
            from transformers import BertTokenizerFast

            # `.from_pretrained` on the explicitly-named class, NOT `BertTokenizerFast(vocab_file=...)`.
            # The constructor form silently builds a WordPiece model that matches nothing under
            # tokenizers >= 0.22 and tokenises every word to [UNK] -- weights load fine, nothing
            # raises, and the only symptom is a probe score that looks like a domain mismatch.
            # Naming the class still bypasses the checkpoint's `auto_map`, so no repo code runs.
            return BertTokenizerFast.from_pretrained(checkpoint, local_files_only=True)

        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                             trust_remote_code=False)

    @staticmethod
    def _load(spec: EncoderSpec, checkpoint: str):
        """Never passes trust_remote_code=True. `bert_shim` exists so a checkpoint whose
        `model_type` names custom repo code can still be loaded, by naming the stock architecture
        its weights actually are."""
        tokenizer = HFTextEncoder._load_tokenizer(spec, checkpoint)
        if spec.loader == "bert_shim":
            from transformers import BertConfig, BertModel

            config = BertConfig.from_pretrained(checkpoint, local_files_only=True)
            model = BertModel.from_pretrained(
                checkpoint, config=config, local_files_only=True, add_pooling_layer=False,
            )
            return tokenizer, model, config

        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True,
                                            trust_remote_code=False)
        kwargs = {}
        if config.model_type in ("bert", "roberta"):
            # These checkpoints are MLM checkpoints with no trained pooler; letting transformers
            # add one would give a randomly initialised `pooler_output`.
            kwargs["add_pooling_layer"] = False
        model = AutoModel.from_pretrained(checkpoint, config=config, local_files_only=True,
                                          trust_remote_code=False, **kwargs)
        return tokenizer, model, config

    def _resolve_encoder_max_length(self) -> int:
        """The real usable token budget, from the checkpoint rather than from the spec table.

        Absolute-position models (BERT, RoBERTa) are hard-capped by `max_position_embeddings`, and
        RoBERTa additionally spends 2 of those on its padding-index offset. RoPE models
        (ModernBERT) are capped by the same field but do not pay that offset.
        """
        limit = getattr(self.config, "max_position_embeddings", None)
        if not limit:
            return int(getattr(self.tokenizer, "model_max_length", 512) or 512)
        if getattr(self.config, "model_type", "") == "roberta":
            limit -= 2
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None) or limit
        # HF uses a sentinel ~1e30 for "no limit"; anything above the model's own cap is not real.
        return int(min(limit, tokenizer_limit if tokenizer_limit < 10 ** 9 else limit))

    # ------------------------------------------------------------------ interface

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def identity(self) -> dict:
        return {
            "name": self.spec.name,
            "checkpoint": self.checkpoint,
            "hf_repo": self.spec.hf_repo,
            "model_type": getattr(self.config, "model_type", "unknown"),
            "output_dim": self._output_dim,
            "max_length": self.max_length,
            "encoder_max_length": self.encoder_max_length,
            "pooling": self.pooling,
            "tokenizer": type(self.tokenizer).__name__,
            "vocab_size": int(getattr(self.config, "vocab_size", 0)),
            "trainable": self.trainable,
        }

    def train(self, mode: bool = True):
        """A frozen encoder stays in eval mode whatever the enclosing module does to it."""
        return super().train(mode if self.trainable else False)

    def _tokenize(self, reports: Sequence[str]):
        """Tokenise twice: once unbounded to learn the true lengths, once truncated to use.

        Slicing the unbounded ids instead would be one pass, but it would cut off the trailing
        `[SEP]`/`</s>` that `truncation=True` preserves -- and a conditioning sequence missing its
        end token is a subtle distribution shift. The second pass is tokenizer-only work, which is
        a few percent of a transformer forward, so the honest count is nearly free.
        """
        unbounded = self.tokenizer(list(reports), add_special_tokens=True, truncation=False,
                                   padding=False, return_attention_mask=False)["input_ids"]
        self.truncation.observe([len(ids) for ids in unbounded], self.max_length)
        if self.truncation.n_truncated and not self.truncation._warned:
            self.truncation._warned = True
            log.warning(
                "%s: truncating reports at %d tokens (first offender was %d tokens). "
                "Truncation is counted in encoder.truncation and reported by "
                "log_truncation_summary().", self.spec.name, self.max_length,
                self.truncation.longest_seen,
            )
        return self.tokenizer(list(reports), add_special_tokens=True, truncation=True,
                              max_length=self.max_length, padding=True, return_tensors="pt")

    def encode(self, reports: Sequence[str], device) -> TextConditioning:
        reports = ["" if r is None else str(r) for r in reports]
        batch = self._tokenize(reports)
        device = torch.device(device)
        current = next(self.model.parameters()).device
        # Compare by type only. `torch.device("cuda") != torch.device("cuda:0")` even though they
        # are the same device, so comparing the objects would re-issue `.to()` on every batch.
        if current.type != device.type:
            self.model.to(device)
        batch = {key: value.to(device) for key, value in batch.items()
                 if key in ("input_ids", "attention_mask", "token_type_ids")}
        context = torch.enable_grad() if self.trainable else torch.no_grad()
        with context:
            tokens = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].to(torch.bool)
        pooled = tokens[:, 0] if self.pooling == "cls" else masked_mean(tokens, mask)
        metadata = dict(self.identity)
        metadata["n_tokens"] = int(tokens.shape[1])
        return TextConditioning(tokens, mask, pooled, metadata)

    def log_truncation_summary(self) -> dict:
        summary = self.truncation.as_dict()
        if summary["n_sequences"]:
            log.info("%s truncation: %d/%d sequences (%.2f%%) over %d tokens; longest seen %d; "
                     "%d tokens dropped", self.spec.name, summary["n_truncated"],
                     summary["n_sequences"], 100 * summary["fraction_truncated"],
                     summary["max_length"], summary["longest_seen"], summary["dropped_tokens"])
        return summary


def build_encoder(name: str, **kwargs) -> HFTextEncoder:
    """The only place a spec is turned into a live encoder."""
    if name not in ENCODER_SPECS:
        raise ValueError(f"unknown text encoder '{name}'. Choose from: {sorted(ENCODER_SPECS)}")
    return HFTextEncoder(ENCODER_SPECS[name], **kwargs)


def available_encoders(pretrained_dir: Optional[str] = None) -> dict[str, bool]:
    """{encoder name: is its checkpoint staged}. Cheap -- no model is loaded."""
    root = pretrained_dir or DEFAULT_PRETRAINED_DIR
    return {name: os.path.isdir(os.path.join(root, spec.directory))
            for name, spec in ENCODER_SPECS.items()}


__all__ = [
    "DEFAULT_PRETRAINED_DIR",
    "ENCODER_SPECS",
    "EncoderSpec",
    "HFTextEncoder",
    "POOLING_MODES",
    "TruncationCounter",
    "available_encoders",
    "build_encoder",
]
