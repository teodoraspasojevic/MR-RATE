"""Report text -> conditioning embeddings.

    from mrrate_r2v.textenc import build_encoder, format_report

    encoder = build_encoder("bioclinical_mbert")            # frozen, staged checkpoint
    text    = format_report(record, "findings_impression")
    cond    = encoder.encode([text], device)                # TextConditioning

`formats` is importable without torch; `encoders`/`fusion` are not, so they are imported lazily
here for the same reason `data/{storage,manifest,reports,geometry}.py` avoid torch: a pyarrow-only
or stdlib-only interpreter must still be able to build report strings.
"""
from .formats import (  # noqa: F401
    DEFAULT_REPORT_FORMAT,
    METADATA_DEPENDENT_FORMATS,
    REPORT_FORMATS,
    format_report,
)

__all__ = [
    "CONDITIONING_CONFIGS",
    "DEFAULT_CONDITIONING",
    "DEFAULT_REPORT_FORMAT",
    "ENCODER_SPECS",
    "METADATA_DEPENDENT_FORMATS",
    "REPORT_FORMATS",
    "MultiEncoderEmbedder",
    "PooledEmbedder",
    "ProjectedConcatFusion",
    "SectionedFusionEmbedder",
    "available_encoders",
    "build_conditioning",
    "build_encoder",
    "format_report",
]

_LAZY = {
    "ENCODER_SPECS": ("encoders", "ENCODER_SPECS"),
    "available_encoders": ("encoders", "available_encoders"),
    "build_encoder": ("encoders", "build_encoder"),
    "HFTextEncoder": ("encoders", "HFTextEncoder"),
    "MultiEncoderEmbedder": ("fusion", "MultiEncoderEmbedder"),
    "ProjectedConcatFusion": ("fusion", "ProjectedConcatFusion"),
    "CONDITIONING_CONFIGS": ("conditioning", "CONDITIONING_CONFIGS"),
    "DEFAULT_CONDITIONING": ("conditioning", "DEFAULT_CONDITIONING"),
    "PooledEmbedder": ("conditioning", "PooledEmbedder"),
    "SectionedFusionEmbedder": ("conditioning", "SectionedFusionEmbedder"),
    "build_conditioning": ("conditioning", "build_conditioning"),
}


def __getattr__(name):
    if name in _LAZY:
        module, attribute = _LAZY[name]
        from importlib import import_module

        return getattr(import_module(f".{module}", __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
