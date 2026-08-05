"""Dataset-level report statistics. Stdlib + numpy; no torch, no transformers.

Everything the report-format decision rests on: how long reports are, which sections exist, how
standardised the headings are, what acquisition information the *text* actually contains (as
opposed to what the manifest contains), and how much of the content is negated or normal.

Emits counts and statistics only -- never a verbatim report, never an identifier.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

SECTIONS = ("clinical_information", "technique", "findings", "impression")

#: Regexes for "does the report text mention this acquisition property at all". Deliberately
#: generous: a high hit rate here is evidence the *text* carries the information, and a low one is
#: strong evidence it does not (a generous pattern that still misses is a real absence).
ACQUISITION_PROBES = {
    "modality_T1": r"\bT1[\s-]?(weighted|w)\b|\bT1\b",
    "modality_T2": r"\bT2[\s-]?(weighted|w)\b|\bT2\b",
    "modality_FLAIR": r"\bFLAIR\b",
    "modality_SWI_GRE": r"\bSWI\b|\bSWAN\b|\bgradient[- ]echo\b|\bGRE\b|\bT2\*",
    "modality_DWI_ADC": r"\bDWI\b|\bADC\b|diffusion",
    "plane_word": r"\baxial\b|\bsagittal\b|\bcoronal\b|\btransverse\b",
    "multiplanar": r"3[- ]plane|multiplanar|three[- ]plane",
    "slice_thickness": r"\bslice thickness\b|\b\d+(\.\d+)?\s?mm\s+(slice|section|thick)",
    "voxel_or_pixel_spacing": r"voxel\s+(size|spacing)|pixel\s+spacing|in[- ]plane resolution",
    "matrix_or_dimensions": r"\bmatrix\b|\b\d{3}\s?[x×]\s?\d{3}\b",
    "field_strength": r"\b[13](\.\d)?\s?T\b|\btesla\b",
    "contrast_agent": r"gadolinium|dotarem|gadovist|contrast|IV\s+\d+\s*ml",
    "scanner_vendor": r"siemens|philips|ge healthcare|toshiba|canon",
}

_SENTENCE = re.compile(r"(?<=[.;])\s+|\n+")
_HEADING = re.compile(r"^\s*([A-Za-z][A-Za-z /\-]{2,40})\s*:", re.M)
_STRICT_NEGATION = re.compile(
    r"\bno\b|\bnot\b|\bnone\b|\bwithout\b|\bnegative for\b|\babsent\b|\bnor\b|\bfree of\b", re.I)
_NORMAL = re.compile(r"\bnormal\b|\bunremarkable\b|\bwithin normal limits\b|\bnatural\b", re.I)
_ASSERTION = re.compile(
    r"\b(is|are|was|were|there is|there are|shows?|demonstrates?|reveals?|consistent with|"
    r"compatible with|suggestive of|noted|seen|observed|present)\b", re.I)
_HEDGE = re.compile(r"\bmay\b|\bpossib|\bprobabl|\bsuspicio|\bcannot be excluded|\bequivocal|"
                    r"\blikely\b|\bcould\b|\bsuggest", re.I)


def describe(values):
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "min": float(a.min()), "mean": float(a.mean()),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def _field(record, name):
    return (getattr(record, name, None) or "").strip()


def _normalize_for_match(text):
    """Collapse whitespace and dash-family characters, lowercase, then strip.

    The trailing strip matters: the structuring step re-renders the radiologist's "— " bullets, so
    a section's text often begins with one. Collapsing it leaves a leading space that would make an
    otherwise-exact substring match fail, understating how extractive the sections are.
    """
    return re.sub(r"[\s\u2014\u2013\-]+", " ", text).strip().lower()


def split_counts(records):
    out = defaultdict(Counter)
    for record in records:
        counter = out[record.split]
        counter["studies"] += 1
        if _field(record, "raw"):
            counter["nonempty_report"] += 1
        if record.labels:
            counter["has_labels"] += 1
        for section in SECTIONS:
            if _field(record, section):
                counter[f"section_{section}"] += 1
        for bucket in record.buckets:
            counter[f"bucket_{bucket}"] += 1
    return {split: dict(counter) for split, counter in out.items()}


def report_health(records):
    total = len(records)
    texts = [_field(r, "raw") for r in records]
    nonempty = [t for t in texts if t]
    normalised = [re.sub(r"\s+", " ", t).lower() for t in nonempty]
    duplicates = Counter(normalised)
    repeated = sum(count for count in duplicates.values() if count > 1)

    by_text = defaultdict(set)
    for record, text in zip([r for r, t in zip(records, texts) if t], normalised):
        by_text[text].add(record.split)

    return {
        "n_studies": total,
        "n_empty_report": total - len(nonempty),
        "empty_sections": {s: sum(1 for r in records if not _field(r, s)) for s in SECTIONS},
        "n_duplicate_reports": repeated,
        "n_duplicate_groups": sum(1 for c in duplicates.values() if c > 1),
        "max_repeat_count": max(duplicates.values()) if duplicates else 0,
        "n_texts_in_multiple_splits": sum(1 for splits in by_text.values() if len(splits) > 1),
        "n_reports_under_10_words": sum(1 for t in nonempty if len(t.split()) < 10),
    }


def length_statistics(records):
    """Characters and words per field, plus the findings+impression combination."""
    fields = {name: [_field(r, name) for r in records] for name in ("raw", *SECTIONS)}
    fields["findings+impression"] = [
        (_field(r, "findings") + "\n" + _field(r, "impression")).strip() for r in records]
    out = {}
    for name, values in fields.items():
        present = [v for v in values if v]
        out[name] = {
            "n_present": len(present),
            "characters": describe(len(v) for v in present),
            "words": describe(len(v.split()) for v in present),
        }
    return out


def heading_statistics(records, top=40):
    texts = [_field(r, "raw") for r in records]
    texts = [t for t in texts if t]
    headings = Counter()
    for text in texts:
        for match in _HEADING.finditer(text):
            headings[match.group(1).strip().lower()] += 1
    first_lines = Counter(
        re.sub(r"\[\w+_\d+\]", "[TOKEN]", t.splitlines()[0].strip())[:60]
        for t in texts if t.splitlines())

    verbatim = Counter()
    have = Counter()
    for record in records:
        raw = _normalize_for_match(_field(record, "raw"))
        if not raw:
            continue
        for section in SECTIONS:
            value = _normalize_for_match(_field(record, section))
            if not value:
                continue
            have[section] += 1
            if value in raw:
                verbatim[section] += 1
    return {
        "n_reports": len(texts),
        "n_distinct_headings": len(headings),
        "top_headings": headings.most_common(top),
        "n_distinct_first_lines": len(first_lines),
        "top_first_lines": first_lines.most_common(10),
        "section_verbatim_in_raw": {s: {"verbatim": verbatim[s], "present": have[s]}
                                    for s in SECTIONS},
    }


def acquisition_content(records):
    """How often each acquisition property is *mentioned in the text*, per field.

    Read this next to the manifest: a property with a high rate here is recoverable from text; a
    property near zero here must come from structured metadata or not at all.
    """
    fields = {"raw": [_field(r, "raw") for r in records],
              "technique": [_field(r, "technique") for r in records],
              "findings_impression": [(_field(r, "findings") + " " + _field(r, "impression"))
                                      for r in records]}
    out = {}
    for probe, pattern in ACQUISITION_PROBES.items():
        rx = re.compile(pattern, re.I)
        out[probe] = {}
        for field_name, values in fields.items():
            present = [v for v in values if v]
            hits = sum(1 for v in present if rx.search(v))
            out[probe][field_name] = {"hits": hits, "n": len(present),
                                      "pct": 100.0 * hits / max(len(present), 1)}
    return out


def polarity_statistics(records):
    """Sentence-level classification of findings+impression into negated / normal / asserted.

    **Method and its limits.** A sentence is `negated` if it contains an explicit negation cue,
    else `normal_statement` if it contains a normality word, else `positive_assertion` if it has
    an assertive verb, else `other`. This is lexical and order-blind, so it cannot scope a
    negation ("no acute infarct but chronic gliosis is present" counts once, as negated) and it
    cannot resolve double negation. It is reported because it is fully explainable and identical
    for every encoder -- not as a gold standard. `hedged` is counted independently and overlaps
    the other four.
    """
    counts = Counter()
    fractions = []
    for record in records:
        body = (_field(record, "findings") + "\n" + _field(record, "impression")).strip()
        if not body:
            continue
        sentences = [s.strip(" -—–\t") for s in _SENTENCE.split(body) if len(s.strip()) > 3]
        if not sentences:
            continue
        negative_or_normal = 0
        for sentence in sentences:
            if _STRICT_NEGATION.search(sentence):
                counts["negated"] += 1
                negative_or_normal += 1
            elif _NORMAL.search(sentence):
                counts["normal_statement"] += 1
                negative_or_normal += 1
            elif _ASSERTION.search(sentence):
                counts["positive_assertion"] += 1
            else:
                counts["other_descriptive"] += 1
            if _HEDGE.search(sentence):
                counts["hedged_overlapping"] += 1
        counts["sentences"] += len(sentences)
        fractions.append(negative_or_normal / len(sentences))
    array = np.asarray(fractions)
    return {
        "counts": dict(counts),
        "n_reports": len(fractions),
        "negative_or_normal_fraction": describe(array) if array.size else {},
        "n_reports_all_negative_or_normal": int((array == 1).sum()) if array.size else 0,
        "n_reports_no_negative_or_normal": int((array == 0).sum()) if array.size else 0,
    }


def label_statistics(records):
    with_labels = [r for r in records if r.labels]
    if not with_labels:
        return {}
    names = sorted({k for r in with_labels for k in r.labels})
    prevalence = Counter()
    per_study = []
    for record in with_labels:
        positives = [n for n in names if record.labels.get(n)]
        per_study.append(len(positives))
        prevalence.update(positives)
    return {
        "n_studies": len(with_labels), "n_labels": len(names),
        "positives_per_study": describe(per_study),
        "n_all_negative": int(sum(1 for k in per_study if k == 0)),
        "prevalence": {name: {"n": prevalence[name],
                              "pct": 100.0 * prevalence[name] / len(with_labels)}
                       for name in sorted(names, key=lambda n: -prevalence[n])},
    }


def analyze(records):
    """Every statistic above, in one JSON-serialisable dict."""
    return {
        "splits": split_counts(records),
        "health": report_health(records),
        "lengths": length_statistics(records),
        "headings": heading_statistics(records),
        "acquisition_content": acquisition_content(records),
        "polarity": polarity_statistics(records),
        "labels": label_statistics(records),
    }


__all__ = [
    "ACQUISITION_PROBES", "SECTIONS", "acquisition_content", "analyze", "describe",
    "heading_statistics", "label_statistics", "length_statistics", "polarity_statistics",
    "report_health", "split_counts",
]
