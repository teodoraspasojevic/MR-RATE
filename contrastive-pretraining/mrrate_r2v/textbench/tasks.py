"""The metrics. Five, chosen so each answers a question the challenge actually scores on.

| Metric | Question | Label source | Strength |
|---|---|---|---|
| `pathology_probe_auroc` | is pathology content linearly recoverable? | `labels.json` | **weak** -- LLM-derived from the same report text |
| `bucket_probe_auroc` | is acquisition (modality, plane) recoverable? | manifest / DICOM | independent of the text |
| `nn_jaccard_delta` | are clinically similar reports close? | `labels.json` | **weak**, same caveat |
| `sim_spearman` | does cosine track clinical similarity? | `labels.json` | **weak**, same caveat |
| `negation_delta` | how far does flipping polarity move the embedding? | rule-based minimal pairs | construction, not annotation |

**Why these and not FID/CLIP-score.** The challenge scores generated volumes with a feature-based
FID-like metric plus a *blinded classifier consistency* check -- "does a classifier trained on
real data assign the generated volume the labels its conditioning report described"
(`docs/challange_docs/MRI_Report_to_Volume.md`). The conditioning embedding is upstream of both.
`pathology_probe_auroc` is the direct frozen-encoder proxy for the consistency check: a label the
embedding cannot linearly express is one the denoiser has no way to render. `negation_delta` is
the same question restricted to the failure mode that matters most clinically. `bucket_probe_auroc`
covers the acquisition side, which the FID-like metric is sensitive to because a T1w and a FLAIR
have very different feature distributions.

**The weak-label caveat, stated once and meant everywhere.** `labels.json` was produced by an LLM
reading the same report the encoder reads. High probe AUROC therefore proves the embedding
retained information the labeller also extracted from the text -- not that the label is clinically
correct. These numbers are valid for *ranking encoders against each other* (every encoder faces
the identical labels) and invalid as absolute measures of clinical accuracy. `bucket_probe_auroc`
and `negation_delta` are the two metrics not affected by this.

**Splits.** Probes train on the `train` split and are scored on `test`. MR-RATE's splits are
patient-isolated (`validation_report.json:patient_split_isolation`, 0 violations), so a probe
cannot see the same patient twice. Nothing here ever fits on `test`.
"""
from __future__ import annotations

import numpy as np

MIN_PREVALENCE = 0.01          # a label rarer than this has too few test positives for a stable AUROC
DEFAULT_C = 1.0


def _standardize(train, test):
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / scale, (test - mean) / scale


def _auroc(y_true, score):
    """Rank-based AUROC. Returns nan when a class is absent, rather than a misleading 0.5."""
    y_true = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(np.asarray(score, dtype=float), kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties, so a constant score gives exactly 0.5
    values = np.asarray(score, dtype=float)[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            ranks[order[start:i]] = (start + i + 1) / 2.0
            start = i
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fit_logistic(x_train, y_train, x_test, C=DEFAULT_C, seed=0, max_iter=1000):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, solver="lbfgs")
    model.fit(x_train, y_train)
    return model.decision_function(x_test)


# --------------------------------------------------------------------------- multi-label probes


def multilabel_probe(x_train, y_train, x_test, y_test, names, C=DEFAULT_C, seed=0,
                     min_prevalence=MIN_PREVALENCE):
    """One-vs-rest linear probe. Returns (macro AUROC, {label: AUROC}, n_labels_scored).

    A label is scored only if it clears `min_prevalence` in *train* and has both classes present
    in *test*; skipped labels are reported by name rather than silently averaged over.
    """
    x_train, x_test = _standardize(x_train, x_test)
    per_label, skipped = {}, []
    for j, name in enumerate(names):
        column_train = y_train[:, j].astype(bool)
        column_test = y_test[:, j].astype(bool)
        if column_train.mean() < min_prevalence or column_train.all():
            skipped.append(name)
            continue
        if column_test.sum() == 0 or column_test.all():
            skipped.append(name)
            continue
        scores = _fit_logistic(x_train, column_train, x_test, C=C, seed=seed)
        per_label[name] = _auroc(column_test, scores)
    values = [v for v in per_label.values() if np.isfinite(v)]
    macro = float(np.mean(values)) if values else float("nan")
    return macro, per_label, skipped


# --------------------------------------------------------------------------- similarity


def _l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def nearest_neighbour_label_agreement(x_test, y_test, seed=0, chunk=512):
    """Mean Jaccard of the label vectors of a report and its nearest neighbour, minus the same
    quantity for random pairs.

    Positive means the embedding puts clinically similar reports together. The random baseline is
    subtracted because MR-RATE is 44% all-negative: a model that collapsed every report to one
    point would score a high raw Jaccard and a delta of exactly zero.
    """
    x = _l2_normalize(x_test.astype(np.float32))
    y = y_test.astype(bool)
    n = x.shape[0]
    rng_ties = np.random.default_rng(seed)
    neighbour = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk):
        block = x[start:start + chunk] @ x.T
        # Break ties uniformly rather than by index. Without this a collapsed embedding (every
        # report at the same point) would have argmax return the same fixed record for everyone,
        # scoring far *below* random instead of at it -- an artifact of tie-breaking, not a
        # property of the embedding.
        block = block + 1e-6 * rng_ties.random(block.shape).astype(block.dtype)
        for row in range(block.shape[0]):
            block[row, start + row] = -np.inf          # never retrieve yourself
        neighbour[start:start + chunk] = block.argmax(axis=1)

    def jaccard(a, b):
        inter = np.logical_and(a, b).sum(axis=1)
        union = np.logical_or(a, b).sum(axis=1)
        both_empty = union == 0                        # two all-negative reports: perfect agreement
        return np.where(both_empty, 1.0, inter / np.maximum(union, 1))

    nn_score = jaccard(y, y[neighbour])
    rng = np.random.default_rng(seed)
    random_score = jaccard(y, y[rng.permutation(n)])
    return float(nn_score.mean()), float(random_score.mean()), float(nn_score.mean() - random_score.mean())


def similarity_label_spearman(x_test, y_test, n_pairs=200_000, seed=0):
    """Spearman correlation between embedding cosine and label-vector Jaccard over random pairs.

    Pairs are sampled rather than exhaustive: 200k pairs give a standard error under 0.003 and
    the full 5,554-report matrix would be 15M pairs for no extra resolution.
    """
    from scipy.stats import spearmanr

    x = _l2_normalize(x_test.astype(np.float32))
    y = y_test.astype(bool)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    cosine = np.einsum("ij,ij->i", x[i], x[j])
    inter = np.logical_and(y[i], y[j]).sum(axis=1)
    union = np.logical_or(y[i], y[j]).sum(axis=1)
    jaccard = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    rho = spearmanr(cosine, jaccard).statistic
    return float(rho)


# --------------------------------------------------------------------------- negation


def negation_probe(negated, affirmed, topics, seed=0, n_folds=5, C=DEFAULT_C):
    """How much does flipping a finding's polarity move its embedding?

    `negated`/`affirmed` are (P, D) arrays of embeddings for the two members of each minimal pair
    (`negation.py`). Three numbers, in decreasing order of usefulness:

    **`negation_delta` (the one to rank on).** How far a polarity flip moves the embedding, in
    units of how far a *topic* change moves it:

        (1 - cos(negated_i, affirmed_i)) / (1 - cos(negated_i, negated_j))     j: different topic

    0 means polarity is ignored; 1 means flipping "no infarct" to "infarct" moves the embedding as
    far as changing the finding entirely. Both terms are cosines from the same anchor, so the ratio
    is free of the encoder's own anisotropy -- which matters a lot here: RadBERT's mean pairwise
    cosine is ~0.95 and MedEmbed's ~0.71, so raw `cos_pair` values are not comparable between them.

    **`negation_dominance`.** The fraction of pairs where the polarity flip moves the embedding
    *further* than a full topic change does. Expected to be small for every encoder (the affirmed
    counterpart shares almost every token); it is reported because a near-zero value is a concrete
    statement that topic overwhelms polarity in that embedding space.

    **`negation_auroc`.** Whether a topic-grouped linear probe can decode polarity at all. Folds
    are grouped by topic so the probe cannot memorise "infarct means negated". **This saturates
    near 1.0 for every encoder in the zoo** -- the negation cue is a literal token, so it is
    trivially decodable. It is kept as a floor check ("polarity is present in the embedding at
    all"), not as a discriminator; rank on `negation_delta`.
    """
    negated = negated.astype(np.float32)
    affirmed = affirmed.astype(np.float32)
    topics = np.asarray(topics)
    x = np.concatenate([negated, affirmed])
    y = np.concatenate([np.ones(len(negated), bool), np.zeros(len(affirmed), bool)])
    groups = np.concatenate([topics, topics])

    unique = np.array(sorted(set(topics.tolist())))
    rng = np.random.default_rng(seed)
    assignment = {t: int(f) for t, f in zip(unique, rng.integers(0, n_folds, size=len(unique)))}
    fold = np.array([assignment[g] for g in groups])

    scores = np.full(len(y), np.nan)
    for f in range(n_folds):
        train_mask, test_mask = fold != f, fold == f
        if test_mask.sum() == 0 or len(set(y[train_mask])) < 2:
            continue
        x_train, x_test = _standardize(x[train_mask], x[test_mask])
        scores[test_mask] = _fit_logistic(x_train, y[train_mask], x_test, C=C, seed=seed)
    valid = np.isfinite(scores)
    auroc = _auroc(y[valid], scores[valid])

    # Centre before measuring. Sentence-encoder spaces carry a large shared mean component whose
    # size differs enormously between checkpoints (measured on MR-RATE: RadBERT's mean pairwise
    # cosine ~0.95, MedEmbed's ~0.71). That component is common to both members of every pair, so
    # it contributes nothing to polarity while compressing every cosine toward 1 -- and it does so
    # by a different amount per encoder, which would make the ranking a ranking of anisotropy.
    # Subtracting the corpus mean removes exactly that and makes the measure invariant to it.
    centre = np.concatenate([negated, affirmed]).mean(axis=0, keepdims=True)
    a = _l2_normalize(negated - centre)
    b = _l2_normalize(affirmed - centre)
    cos_pair = np.einsum("ij,ij->i", a, b)

    # Partner selection is explicitly different-topic: a same-topic partner would understate the
    # denominator and inflate negation_delta for no good reason.
    order = rng.permutation(len(a))
    partner = np.empty(len(a), dtype=np.int64)
    for i, j in enumerate(order):
        step, candidate = 0, int(j)
        while topics[candidate] == topics[i] and step < len(a):
            candidate = (candidate + 1) % len(a)
            step += 1
        partner[i] = candidate
    cos_other = np.einsum("ij,ij->i", a, a[partner])

    # Ratio of means, NOT mean of ratios. Per-pair `(1-cos_pair)/(1-cos_other)` is unbounded: one
    # unlucky pair whose different-topic partner happens to sit almost on top of it has a
    # denominator near zero and dominates the average. Measured on the real corpus, that turned
    # cxr_bert's delta into 16,927 while every other encoder sat near 0.2 -- a single pair, not a
    # property of the encoder. Both forms have the same units and the same invariance to the
    # shared mean component; only this one is robust.
    mean_polarity_distance = float(np.mean(1.0 - cos_pair))
    mean_topic_distance = float(np.mean(1.0 - cos_other))
    delta = mean_polarity_distance / max(mean_topic_distance, 1e-9)
    # Median of the per-pair ratios, as a distribution check on the ratio of means. A large gap
    # between the two means the pairs are heavy-tailed and `delta` should be read with care.
    per_pair = (1.0 - cos_pair) / np.maximum(1.0 - cos_other, 1e-6)
    delta_median = float(np.median(per_pair))
    dominance = float(np.mean(cos_pair < cos_other))
    # The raw, uncentred cosines are reported too: they are what a downstream cosine-similarity
    # consumer would actually see, and their gap from the centred ones is the anisotropy itself.
    raw_pair = float(np.einsum("ij,ij->i", _l2_normalize(negated),
                               _l2_normalize(affirmed)).mean())
    raw_other = float(np.einsum("ij,ij->i", _l2_normalize(negated),
                                _l2_normalize(negated)[partner]).mean())
    return dict(negation_delta=delta, negation_delta_median=delta_median,
                negation_dominance=dominance, negation_auroc=auroc,
                cos_pair=float(cos_pair.mean()), cos_other_negated=float(cos_other.mean()),
                cos_pair_uncentred=raw_pair, cos_other_uncentred=raw_other,
                n_pairs=int(len(negated)), n_topics=int(len(unique)))


__all__ = [
    "MIN_PREVALENCE",
    "multilabel_probe",
    "nearest_neighbour_label_agreement",
    "negation_probe",
    "similarity_label_spearman",
]
