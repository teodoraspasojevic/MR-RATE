"""Report text -> conditioning string. Stdlib only, no torch, no transformers.

One named format = one deterministic function of a `ReportRecord` (`data/reports.py`) plus,
optionally, structured acquisition metadata. The name is recorded in every checkpoint and every
embedding cache, so a run can never be ambiguous about what text the model actually saw.

    format_report(record, "findings_impression")
    format_report(record, "findings_impression_meta", modality="T1w", plane="AXIAL",
                  spacing_mm_xyz=(0.94, 0.94, 1.09))

**Every function in `REPORT_FORMATS` is deterministic, and must stay that way** -- the registry is
what a checkpoint and an embedding cache name. Sampling *between* formats is a separate concept:
`parse_format_spec` accepts a comma-separated spec and `choose_format` picks one from it by an
explicit key, so the randomness lives at the call site (the Dataset, keyed on seed/epoch/index)
and never inside a format.

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

import random
from typing import Callable, Optional, Sequence

SECTION_MARKERS = {
    "clinical_information": "CLINICAL",
    "technique": "TECHNIQUE",
    "findings": "FINDINGS",
    "impression": "IMPRESSION",
}

#: The name of the *pseudo*-section carrying acquisition metadata as text
#: (`[MODALITY] .. [PLANE] .. [SPACING] ..`, i.e. `meta_prefix_for`'s output).
#:
#: Not a released report field, which is why it is not in `REPORT_SECTION_NAMES`: the values come
#: from the manifest row and the resolved geometry, exactly as they do for the `*_meta` formats.
#: It exists because a sectioned-fusion configuration never composes a joined string and so has
#: nowhere to put that prefix -- it gets its own conditioning token instead (configuration E).
ACQUISITION_SECTION = "acquisition"


def _text(record, field: str) -> str:
    value = getattr(record, field, None)
    return value.strip() if value else ""


def _sectioned(record, fields) -> str:
    parts = [f"[{SECTION_MARKERS[f]}] {_text(record, f)}" for f in fields if _text(record, f)]
    return "\n".join(parts)


#: Decimals kept in a `[SPACING]` marker. Two is enough to separate every bucket in the geometry
#: table (they differ in the first or second decimal) while staying a short, stable token sequence:
#: the exact float would tokenise into ~8 tokens per axis and change if the FOV table were ever
#: re-derived, which would silently make a new run's text differ from an old checkpoint's.
SPACING_DECIMALS = 2


def _format_spacing(spacing_mm_xyz: Sequence[float]) -> str:
    if len(tuple(spacing_mm_xyz)) != 3:
        raise ValueError(f"spacing_mm_xyz must be 3 values (X, Y, Z), got {tuple(spacing_mm_xyz)!r}")
    return " ".join(f"{float(v):.{SPACING_DECIMALS}f}" for v in spacing_mm_xyz)


def _meta_prefix(modality: Optional[str], plane: Optional[str],
                 spacing_mm_xyz: Optional[Sequence[float]] = None) -> str:
    """Structured acquisition conditioning as text. Values come from the caller (the manifest /
    DICOM-derived metadata), never from the report.

    `spacing_mm_xyz` is **(X, Y, Z)**, the order that crosses this package's boundary and the order
    NVIDIA's own `--spacing` flag uses -- not the internal (D, H, W). It is also already a numeric
    conditioning input via the UNet's `spacing_tensor`; repeating it as text is not redundant,
    because the text encoder is the only path that sees modality and plane *and* spacing together,
    and it is the path a challenge submission can populate from a request that carries no volume.

    Each marker is emitted only when its value is supplied, so a caller that has no spacing
    produces byte-identical text to before this argument existed.
    """
    parts = []
    if modality:
        parts.append(f"[MODALITY] {modality}")
    if plane:
        parts.append(f"[PLANE] {plane}")
    if spacing_mm_xyz is not None:
        parts.append(f"[SPACING] {_format_spacing(spacing_mm_xyz)}")
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


def _with_meta(record, fields, modality, plane, spacing_mm_xyz) -> str:
    prefix = _meta_prefix(modality, plane, spacing_mm_xyz)
    body = _sectioned(record, fields)
    return f"{prefix}\n{body}" if prefix and body else (prefix or body)


def _findings_impression_meta(record, modality=None, plane=None, spacing_mm_xyz=None, **_):
    return _with_meta(record, ["findings", "impression"], modality, plane, spacing_mm_xyz)


def _impression_findings_meta(record, modality=None, plane=None, spacing_mm_xyz=None, **_):
    """The metadata prefix with the impression first. Pairs with `findings_impression_meta` as the
    two orderings `--report-format a,b` samples between during training -- see `choose_format`."""
    return _with_meta(record, ["impression", "findings"], modality, plane, spacing_mm_xyz)


REPORT_FORMATS: dict[str, Callable] = {
    "raw": _raw,
    "findings": _findings,
    "impression": _impression,
    "findings_impression": _findings_impression,
    "impression_findings": _impression_findings,
    "clinical_findings_impression": _clinical_findings_impression,
    "full_structured": _full_structured,
    "findings_impression_meta": _findings_impression_meta,
    "impression_findings_meta": _impression_findings_meta,
}

#: Formats whose output depends on values that must be supplied from structured metadata.
#: Using one of these commits you to having that metadata at inference time -- which for the
#: challenge means supplying a *default* modality/plane/spacing when the request carries none
#: (`cli/generate_r2v.py --modality/--plane/--spacing`), never omitting the markers, because the
#: model never saw a report without them.
METADATA_DEPENDENT_FORMATS = ("findings_impression_meta", "impression_findings_meta")

DEFAULT_REPORT_FORMAT = "findings_impression"

#: The training spec that makes section order something the model cannot rely on. Both orderings
#: carry the identical metadata prefix and the identical sections, so a model trained on this has
#: seen every section in first position and cannot be surprised by the challenge's ordering.
ORDER_AGNOSTIC_META_SPEC = "findings_impression_meta,impression_findings_meta"

FORMAT_SPEC_SEPARATOR = ","


def parse_format_spec(spec) -> tuple:
    """`"a"` or `"a,b,c"` (or an already-split sequence) -> a validated tuple of format names.

    A one-name spec is the ordinary deterministic case. A multi-name spec means "sample one of
    these per sample", which only the Dataset acts on; every other consumer (the embedding cache,
    the text benchmark, a cohort's pre-composed text) needs one fixed answer and must reject a
    multi-name spec rather than silently take the first.
    """
    names = tuple(spec) if not isinstance(spec, str) else tuple(
        part.strip() for part in spec.split(FORMAT_SPEC_SEPARATOR)
    )
    names = tuple(n for n in names if n)
    if not names:
        raise ValueError("empty report-format spec")
    unknown = [n for n in names if n not in REPORT_FORMATS]
    if unknown:
        raise ValueError(f"unknown report format(s) {unknown}. Choose from: {sorted(REPORT_FORMATS)}")
    if len(set(names)) != len(names):
        raise ValueError(f"report-format spec repeats a name: {names}. A repeat silently doubles "
                         "that format's sampling weight.")
    return names


def choose_format(names: Sequence[str], key) -> str:
    """Pick one format from `names`, deterministically from `key`.

    `key` is a caller-supplied identity, not process state: the Dataset passes
    `(seed, epoch, index)`. That matters twice over -- `__getitem__` runs in DataLoader worker
    processes, so a process-global RNG would give a different draw per worker count, and a resumed
    or re-run job must be able to reproduce the exact text a step saw.

    Uniform over `names`. A weighted variant is deliberately absent: the point is that the model
    can rely on neither ordering, and any weighting reintroduces a preferred one.
    """
    names = tuple(names)
    if len(names) == 1:
        return names[0]
    return random.Random(str(key)).choice(names)


def format_report(record, name: str = DEFAULT_REPORT_FORMAT, *,
                  modality: Optional[str] = None, plane: Optional[str] = None,
                  spacing_mm_xyz: Optional[Sequence[float]] = None) -> str:
    """Render one `ReportRecord` under a named format. Returns "" if every requested section is
    empty -- callers decide whether that case is dropped or conditioned on as unconditional.

    `name` must be a single format. Pass a multi-name spec through `parse_format_spec` +
    `choose_format` first; this function stays deterministic by construction.
    """
    if name not in REPORT_FORMATS:
        raise ValueError(f"unknown report format '{name}'. Choose from: {sorted(REPORT_FORMATS)}")
    return REPORT_FORMATS[name](record, modality=modality, plane=plane,
                                spacing_mm_xyz=spacing_mm_xyz)


def meta_prefix_for(modality: Optional[str], plane: Optional[str],
                    spacing_mm_xyz: Optional[Sequence[float]] = None) -> str:
    """The public form of `_meta_prefix`, for inference paths that hold text which was composed
    *without* metadata (a cohort's `reports.json`, a challenge request's raw string) and must
    prepend the markers the adapter was trained with. Using this instead of re-implementing the
    string is the whole point: the prefix is a trained token sequence, not a display detail."""
    return _meta_prefix(modality, plane, spacing_mm_xyz)


def with_acquisition_section(sections, modality: Optional[str], plane: Optional[str],
                             spacing_mm_xyz: Optional[Sequence[float]] = None) -> dict:
    """`sections` plus the `acquisition` pseudo-section, holding exactly what `meta_prefix_for`
    returns. The one place an *inference* path composes that section, so the text a submission
    sends and the text the Dataset produced during training cannot drift apart.

    Additive by construction: `SectionedFusionEmbedder` encodes the sections it was *built* with
    and ignores every other key, so a caller may call this unconditionally rather than branching on
    whether the loaded configuration declares the section.
    """
    out = dict(sections or {})
    out[ACQUISITION_SECTION] = _meta_prefix(modality, plane, spacing_mm_xyz)
    return out


__all__ = [
    "ACQUISITION_SECTION",
    "DEFAULT_REPORT_FORMAT",
    "FORMAT_SPEC_SEPARATOR",
    "METADATA_DEPENDENT_FORMATS",
    "ORDER_AGNOSTIC_META_SPEC",
    "REPORT_FORMATS",
    "SECTION_MARKERS",
    "SPACING_DECIMALS",
    "choose_format",
    "format_report",
    "meta_prefix_for",
    "parse_format_spec",
    "with_acquisition_section",
]
