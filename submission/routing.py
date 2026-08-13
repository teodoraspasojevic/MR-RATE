"""Which (modality, plane) bucket to generate for a prompt that does not say.

The challenge hands us `{report, input_image_name}` and nothing else (the CT track's
published `data_schema`; the MR track's is still empty -- see README.md §"Unknowns").
Our model needs a modality and a plane regardless: they select the geometry bucket, they
reach the UNet as `class_labels` and `spacing_tensor`, and -- for adapters A/B/C, trained
on `findings_impression_meta` -- they are literal `[MODALITY]`/`[PLANE]`/`[SPACING]`
tokens in the conditioning text. There is no "unspecified" option the model has seen.

An MR-RATE report is **study-level**: it describes every series in the exam, so it is not
evidence for which single series a given prompt stands for. That is why the default
strategy does not try to read the modality out of the text. Instead it reproduces the
population marginal exactly, which is the right objective for the metrics that decide the
leaderboard: FID/FVD compare our *set* of volumes against the hidden *set* of real ones,
so the cheapest large win available without knowing per-case truth is to make the two
sets' modality/plane mixtures agree.

Strategies (`R2V_ROUTING`):
  ``marginal``  default. Largest-remainder allocation of the prompt list over the ten
                buckets so the emitted mixture matches `TEST_SPLIT_BUCKET_SHARE` as
                closely as an integer allocation can. Deterministic: the assignment is a
                function of the prompt set, not of iteration order or an RNG.
  ``report``    keyword-scan the prompt for a sequence/plane name and honour it when one
                is found unambiguously; every unresolved prompt falls back to
                ``marginal`` over the *remaining* quota, so the mixture is still filled.
  ``fixed``     one bucket for every prompt, from `R2V_FIXED_BUCKET` (e.g. `T1w:AXIAL`).
                For an ablation, or if the organizers eventually say the test set is
                single-modality.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Sequence, Tuple

Bucket = Tuple[str, str]

#: Share of each (modality, plane) bucket in MR-RATE's own **test** split, over the ten
#: buckets the adapters were trained on. Counted from
#: `r2v_manifest/manifest_shards_native.csv` (34,453 test series, 2026-08-13); the ten
#: buckets are the complete set present in that split, not a selection.
#:
#: This is an estimate of the hidden test set's mixture, not a measurement of it: the 2025
#: CT edition used an external hold-out rather than the public split, so the real mixture
#: may differ. It is still the best available prior -- same institution, same acquisition
#: protocols -- and any error is a mixture error, not a per-case error.
TEST_SPLIT_BUCKET_SHARE: Dict[Bucket, float] = {
    ("T1w", "AXIAL"): 0.1587,
    ("T2w", "AXIAL"): 0.1586,
    ("T1w", "SAGITTAL"): 0.1260,
    ("SWI", "AXIAL"): 0.1146,
    ("FLAIR", "SAGITTAL"): 0.1081,
    ("FLAIR", "AXIAL"): 0.0997,
    ("T1w", "CORONAL"): 0.0712,
    ("T2w", "CORONAL"): 0.0687,
    ("FLAIR", "CORONAL"): 0.0513,
    ("T2w", "SAGITTAL"): 0.0431,
}

STRATEGIES = ("marginal", "report", "fixed")

# Sequence keywords, longest-first so "t2 flair" resolves to FLAIR rather than T2w. Only
# used by the `report` strategy, and only when exactly one modality matches.
_MODALITY_PATTERNS: Sequence[Tuple[str, str]] = (
    ("FLAIR", r"\bflair\b|\bt2[\s\-_]*flair\b|fluid[\s\-]*attenuat"),
    ("SWI", r"\bswi\b|susceptibility[\s\-]*weighted|\bswan\b"),
    ("T1w", r"\bt1\b|\bt1w\b|t1[\s\-]*weighted"),
    ("T2w", r"\bt2\b|\bt2w\b|t2[\s\-]*weighted"),
)

_PLANE_PATTERNS: Sequence[Tuple[str, str]] = (
    ("AXIAL", r"\baxial\b|\btransvers"),
    ("SAGITTAL", r"\bsagittal\b|\bsag\b"),
    ("CORONAL", r"\bcoronal\b|\bcor\b"),
)


def parse_bucket(text: str) -> Bucket:
    """`"T1w:AXIAL"` -> `("T1w", "AXIAL")`, validated against the trained buckets."""
    parts = [p.strip() for p in str(text).replace("/", ":").split(":")]
    if len(parts) != 2:
        raise ValueError(f"expected MODALITY:PLANE, got {text!r}")
    bucket = (parts[0], parts[1].upper())
    if bucket not in TEST_SPLIT_BUCKET_SHARE:
        raise ValueError(
            f"{bucket} is not one of the trained buckets: "
            f"{sorted(TEST_SPLIT_BUCKET_SHARE)}"
        )
    return bucket


def _stable_rank(case_id: str) -> int:
    """A per-case ordering key that is independent of the order prompts arrive in.

    Allocating buckets in file order would correlate the modality mixture with whatever
    order the platform happens to write `prompts.json` in; hashing decorrelates it while
    staying reproducible across reruns and resumes.
    """
    return int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16], 16)


def _largest_remainder(total: int, shares: Dict[Bucket, float]) -> Dict[Bucket, int]:
    """Integer counts summing to exactly `total` whose proportions track `shares`.

    Floor every quota, then hand the leftover slots to the largest fractional parts. Ties
    break on the bucket name so the result does not depend on dict iteration order.
    """
    if total <= 0:
        return {b: 0 for b in shares}
    scale = sum(shares.values())
    exact = {b: total * s / scale for b, s in shares.items()}
    counts = {b: int(v) for b, v in exact.items()}
    leftover = total - sum(counts.values())
    order = sorted(shares, key=lambda b: (-(exact[b] - counts[b]), b))
    for bucket in order[:leftover]:
        counts[bucket] += 1
    return counts


def _scan(text: str, patterns: Sequence[Tuple[str, str]]) -> str | None:
    """The single label whose pattern matches, or None if zero or several match."""
    hits = [label for label, pattern in patterns if re.search(pattern, text, re.IGNORECASE)]
    return hits[0] if len(hits) == 1 else None


def route(
    case_ids: Sequence[str],
    reports: Sequence[str],
    strategy: str = "marginal",
    fixed_bucket: Bucket | None = None,
    shares: Dict[Bucket, float] | None = None,
) -> Dict[str, Bucket]:
    """`{case_id: (modality, plane)}` for a whole prompt set.

    Routing is set-level, not per-case, because `marginal` cannot be computed one case at a
    time -- the whole point is that the emitted mixture is right, which is a property of the
    set. Callers therefore route once, up front, and look up per case.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown routing strategy {strategy!r}; choose from {STRATEGIES}")
    if len(case_ids) != len(reports):
        raise ValueError("case_ids and reports must be the same length")
    shares = dict(shares or TEST_SPLIT_BUCKET_SHARE)

    if strategy == "fixed":
        if fixed_bucket is None:
            raise ValueError("strategy='fixed' needs fixed_bucket")
        return {cid: fixed_bucket for cid in case_ids}

    assigned: Dict[str, Bucket] = {}
    unresolved: List[str] = list(case_ids)

    if strategy == "report":
        unresolved = []
        for cid, report in zip(case_ids, reports):
            modality = _scan(report or "", _MODALITY_PATTERNS)
            plane = _scan(report or "", _PLANE_PATTERNS)
            if modality and plane and (modality, plane) in shares:
                assigned[cid] = (modality, plane)
            else:
                unresolved.append(cid)
        # Text-resolved cases already consumed part of the target mixture, so the fallback
        # allocation is over what is left rather than over the full marginal -- otherwise
        # the two mechanisms double-count and the emitted mixture drifts.
        used = {b: 0 for b in shares}
        for bucket in assigned.values():
            used[bucket] += 1
        wanted = _largest_remainder(len(case_ids), shares)
        shares = {b: max(wanted[b] - used[b], 0) for b in shares}
        if not any(shares.values()):  # text resolved everything, or over-filled every bucket
            shares = dict(TEST_SPLIT_BUCKET_SHARE)

    counts = _largest_remainder(len(unresolved), shares)
    queue: List[Bucket] = []
    for bucket in sorted(counts):
        queue.extend([bucket] * counts[bucket])
    for cid, bucket in zip(sorted(unresolved, key=_stable_rank), queue):
        assigned[cid] = bucket
    return assigned


def mixture(assignment: Dict[str, Bucket]) -> List[Tuple[Bucket, int, float]]:
    """`[(bucket, count, share)]`, most common first -- for the run log."""
    total = max(len(assignment), 1)
    counts: Dict[Bucket, int] = {}
    for bucket in assignment.values():
        counts[bucket] = counts.get(bucket, 0) + 1
    return [(b, n, n / total) for b, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


__all__ = ["Bucket", "STRATEGIES", "TEST_SPLIT_BUCKET_SHARE", "mixture", "parse_bucket", "route"]
