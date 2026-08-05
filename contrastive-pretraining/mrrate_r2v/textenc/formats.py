"""Report text -> conditioning string. Stdlib only, no torch, no transformers.

One named format = one deterministic function of a `ReportRecord` (`data/reports.py`) plus,
optionally, structured acquisition metadata. The name is recorded in every checkpoint and every
embedding cache, so a run can never be ambiguous about what text the model actually saw.

    format_report(record, "findings_impression")
    format_report(record, "findings_impression_meta", modality="T1w", plane="AXIAL")

**Why section markers are `[FINDINGS]`-style and not `Findings:`.** MR-RATE's own raw reports
already contain `Findings:` headings in 83% of cases and 845 distinct other `Word:` heads, so a
plain-prose heading is indistinguishable from report content. A bracketed marker is unambiguous
and survives every tokenizer in the zoo as a short, stable token sequence.

**Empty sections are dropped, never emitted as an empty marker.** `impression` is absent for 8.9%
of studies; emitting `[IMPRESSION]` with nothing after it would teach the model that the marker
itself carries no information.

**No format invents text.** Everything here is selection, ordering and marking of released
fields. `_meta` formats are the one exception and they add only values that are supplied by the
caller from structured metadata -- never parsed out of the report.
"""
from __future__ import annotations

from typing import Callable, Optional

SECTION_MARKERS = {
    "clinical_information": "CLINICAL",
    "technique": "TECHNIQUE",
    "findings": "FINDINGS",
    "impression": "IMPRESSION",
}


def _text(record, field: str) -> str:
    value = getattr(record, field, None)
    return value.strip() if value else ""


def _sectioned(record, fields) -> str:
    parts = [f"[{SECTION_MARKERS[f]}] {_text(record, f)}" for f in fields if _text(record, f)]
    return "\n".join(parts)


def _meta_prefix(modality: Optional[str], plane: Optional[str]) -> str:
    """Structured acquisition conditioning as text. Values come from the caller (the manifest /
    DICOM-derived metadata), never from the report."""
    parts = []
    if modality:
        parts.append(f"[MODALITY] {modality}")
    if plane:
        parts.append(f"[PLANE] {plane}")
    return " ".join(parts)


def _raw(record, **_):
    return _text(record, "raw")


def _findings(record, **_):
    return _text(record, "findings")


def _impression(record, **_):
    return _text(record, "impression")


def _findings_impression(record, **_):
    return _sectioned(record, ["findings", "impression"])


def _impression_findings(record, **_):
    """Impression first. Identical content to `findings_impression`; the order matters only
    under truncation, where a 512-token encoder keeps the head of the string. 8-10% of studies
    truncate at 512 tokens, and in those the impression is what survives here."""
    return _sectioned(record, ["impression", "findings"])


def _clinical_findings_impression(record, **_):
    return _sectioned(record, ["clinical_information", "findings", "impression"])


def _full_structured(record, **_):
    return _sectioned(record, ["clinical_information", "technique", "findings", "impression"])


def _findings_impression_meta(record, modality=None, plane=None, **_):
    prefix = _meta_prefix(modality, plane)
    body = _sectioned(record, ["findings", "impression"])
    return f"{prefix}\n{body}" if prefix and body else (prefix or body)


REPORT_FORMATS: dict[str, Callable] = {
    "raw": _raw,
    "findings": _findings,
    "impression": _impression,
    "findings_impression": _findings_impression,
    "impression_findings": _impression_findings,
    "clinical_findings_impression": _clinical_findings_impression,
    "full_structured": _full_structured,
    "findings_impression_meta": _findings_impression_meta,
}

#: Formats whose output depends on values that must be supplied from structured metadata.
#: Using one of these commits you to having that metadata at inference time.
METADATA_DEPENDENT_FORMATS = ("findings_impression_meta",)

DEFAULT_REPORT_FORMAT = "findings_impression"


def format_report(record, name: str = DEFAULT_REPORT_FORMAT, *,
                  modality: Optional[str] = None, plane: Optional[str] = None) -> str:
    """Render one `ReportRecord` under a named format. Returns "" if every requested section is
    empty -- callers decide whether that case is dropped or conditioned on as unconditional."""
    if name not in REPORT_FORMATS:
        raise ValueError(f"unknown report format '{name}'. Choose from: {sorted(REPORT_FORMATS)}")
    return REPORT_FORMATS[name](record, modality=modality, plane=plane)


__all__ = [
    "DEFAULT_REPORT_FORMAT",
    "METADATA_DEPENDENT_FORMATS",
    "REPORT_FORMATS",
    "SECTION_MARKERS",
    "format_report",
]
