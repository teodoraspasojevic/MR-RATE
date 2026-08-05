"""Negation minimal pairs: a measurement probe for text encoders. **Not data.**

READ THIS FIRST, because this module deliberately constructs sentences that state the *opposite*
of what a radiologist wrote:

    THE COUNTERFACTUAL SENTENCES PRODUCED HERE ARE NEVER PERSISTED AS TEXT, NEVER ATTACHED TO A
    STUDY, NEVER USED AS A LABEL, AND NEVER USED AS CONDITIONING TEXT.

They exist for one purpose: to be encoded, compared against their original, and discarded. What
reaches disk is a float16 embedding array plus a one-word topic tag (`cli/embed_reports.py`) --
no sentence. Conditioning text comes from `textenc.formats`, which performs no substitution of
any kind; `tests/test_textenc_formats.py` asserts that every format preserves negation cues, and
`tests/test_textbench.py::test_production_modules_never_import_the_benchmark` asserts that no
module on the training or sampling path can import this one.

Why the probe is needed: "no acute infarct" and "acute infarct" share almost every token, and an
encoder that places them in nearly the same spot will condition the generator on a pathology the
report explicitly ruled out. 45.7% of MR-RATE finding sentences are normal statements and a
further 17.2% carry an explicit negation cue, so this is the majority of the signal, not an edge
case.

**Construction, and why it is rule-based rather than model-generated.** Every pair is one real
sentence plus the same sentence with only its negation cue deleted:

    "There is no evidence of acute infarction."  ->  "There is evidence of acute infarction."
    "No pathological contrast enhancement."      ->  "Pathological contrast enhancement."

Nothing is paraphrased, nothing is generated, and the affirmed member is never claimed to be a
real radiologist sentence -- it is a controlled counterfactual, and it is clinically false by
construction. That is the point: the *only* difference between the two members is polarity, so
any embedding distance between them is attributable to polarity alone. It is also exactly why
the containment rule at the top of this file is absolute -- a clinically false sentence is a
useful ruler and a catastrophic training example.

**Stated limitations.** (1) The affirmed member can be mildly ungrammatical; this affects all
encoders equally and is measured, not hidden. (2) Cue deletion cannot handle double negation or
"not excluded"-style hedges, so those patterns are excluded by construction rather than
mis-transformed. (3) The pairs come from the same corpus the encoders are scored on, so this is
an intrinsic probe, not an independent test set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

#: (pattern, replacement) applied to the *start* of a sentence. Ordered: first match wins.
_CUE_RULES = (
    (re.compile(r"^There (is|are) no evidence of\s+", re.I), r"There \1 evidence of "),
    (re.compile(r"^There (is|are) no\s+", re.I),             r"There \1 "),
    (re.compile(r"^No evidence of\s+", re.I),                ""),
    (re.compile(r"^No significant\s+", re.I),                "Significant "),
    (re.compile(r"^No\s+", re.I),                            ""),
)

#: Patterns whose polarity cannot be flipped by deleting a cue. Excluded, never transformed.
_EXCLUDE = re.compile(
    r"\bnot excluded\b|\bcannot be excluded\b|\bno longer\b|\bnot only\b|\bnor\b|"
    r"\bunremarkable\b|\bwithin normal limits\b|\bnot significantly\b|"
    r"\b(no)\b.*\b(no)\b",
    re.I,
)

_SENTENCE = re.compile(r"(?<=[.;])\s+|\n+")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]+")


@dataclass(frozen=True)
class NegationPair:
    """One minimal pair. `topic` is the content head used to keep folds term-disjoint."""

    negated: str
    affirmed: str
    topic: str
    study_uid: str


def _flip(sentence: str) -> Optional[str]:
    for pattern, replacement in _CUE_RULES:
        flipped, n = pattern.subn(replacement, sentence, count=1)
        if n:
            flipped = flipped.strip()
            if not flipped or len(flipped.split()) < 2:
                return None
            return flipped[0].upper() + flipped[1:]
    return None


def _topic(sentence: str) -> Optional[str]:
    """The first content word after the negation cue -- a coarse stand-in for "what was ruled
    out". Used only to group pairs into disjoint folds, never as a label."""
    stop = {"there", "is", "are", "was", "were", "no", "not", "evidence", "of", "the", "a", "an",
            "any", "significant", "and", "or", "in", "on", "at", "with", "seen", "detected",
            "observed", "present", "identified", "noted"}
    for word in _WORD.findall(sentence.lower()):
        if word not in stop and len(word) > 3:
            return word
    return None


def mine_negation_pairs(records, max_pairs: int = 4000, min_words: int = 4,
                        max_words: int = 40, seed: int = 0) -> list[NegationPair]:
    """Mine minimal pairs from a corpus, at most one per study so no study is over-represented.

    Deterministic: records are visited in order and the cap is a plain prefix, so the same corpus
    always yields the same pairs.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    pairs, seen_topics = [], {}
    for index in order:
        record = records[index]
        body = ((record.findings or "") + "\n" + (record.impression or "")).strip()
        if not body:
            continue
        for sentence in _SENTENCE.split(body):
            sentence = sentence.strip(" -—–\t")
            if not sentence or _EXCLUDE.search(sentence):
                continue
            words = sentence.split()
            if not (min_words <= len(words) <= max_words):
                continue
            affirmed = _flip(sentence)
            if affirmed is None or affirmed.lower() == sentence.lower():
                continue
            topic = _topic(sentence)
            if topic is None:
                continue
            # Cap per topic so one very common phrase cannot dominate the metric.
            if seen_topics.get(topic, 0) >= max(4, max_pairs // 50):
                continue
            seen_topics[topic] = seen_topics.get(topic, 0) + 1
            pairs.append(NegationPair(negated=sentence, affirmed=affirmed, topic=topic,
                                      study_uid=record.study_uid))
            break                                   # at most one pair per study
        if len(pairs) >= max_pairs:
            break
    return pairs


__all__ = ["NegationPair", "mine_negation_pairs"]
