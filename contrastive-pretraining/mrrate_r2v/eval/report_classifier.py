"""The blinded pathology classifier, and the consistency metric built on it.

This is the local stand-in for the VLM3D challenge's **Blinded Classifier Consistency**: "whether a
classifier trained on real data assigns consistent clinical labels to generated volumes matching
the conditioning report." Nothing else in this package measures whether a generated volume says
what its report said -- `eval/runner.report_image_similarity` is a hook that has never had a model
behind it, and the training loop's `ssim_advantage` is a *structural* swap test, not a semantic one.

    real train-split cohort ─┐
                             ├─► cli.train_report_classifier ─► classifier.pt
    mrrate_merged_labels.csv ┘                                       │
                                                                    ▼
    prediction set + its cohort ─────────────────────► cli.evaluate --task report2volume
                                                        report_consistency metrics

**How it works.** A frozen MedicalNet ResNet-10 (already staged, already this package's 3D FID
backbone) turns a volume into 512 features; a small head maps those to the 14 merged clinical
labels. The head is fitted on real volumes from the **train** split only, so no generation is ever
scored by a classifier that saw its patient. At evaluation the head is run blind on generated
volumes and its predictions are compared against the labels of the report each volume was
conditioned on.

**Four design choices that decide whether the number means anything:**

1. **The classifier gets the image and nothing else.** No bucket, no modality, no spacing. A head
   that knew the bucket could reach a respectable score purely from `SWI AXIAL -> hemorrhage is
   common`, without ever looking at a voxel -- and it would score a *degenerate* generator just as
   well as a good one. `prevalence_baseline_auroc` is reported for exactly this reason: it is what
   a bucket-prior guesser gets, and the classifier has to beat it.

2. **Every consistency number is reported next to the same classifier's score on the REAL
   volumes** (`real_reference`). The classifier is a weak instrument on some labels; on those, the
   generated score is uninformative rather than bad. Reading a generated AUROC of 0.58 against a
   real AUROC of 0.61 is a completely different conclusion from reading it against 0.95, and only
   one of those two readings is available without the reference.

3. **Labels are never imputed.** A case whose study carries no label row is excluded, not counted
   negative (see `report_labels.py`).

4. **A label the classifier cannot do on real data is marked, not dropped.** `usable=False` when
   `real_reference` AUROC is below `MIN_REAL_AUROC`, and the aggregate reports both the all-label
   and the usable-only mean. Dropping quietly would let a weak classifier flatter a model by
   scoring it only where it happens to work.

sklearn is deliberately not used: neither apptainer image has it (`slurm/_common.sh`), so AUROC and
average precision are computed here from ranks, in numpy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

log = logging.getLogger("mrrate_r2v.eval.report_classifier")

CLASSIFIER_FORMAT = "mrrate_r2v.report_classifier/1"

#: Below this real-data AUROC the classifier cannot separate the label on real volumes, so its
#: verdict on generated ones carries no information. 0.60 is a judgement call, recorded rather
#: than hidden: it is reported per label so a reader can apply their own threshold.
MIN_REAL_AUROC = 0.60

#: Fewer positives than this and a per-label AUROC is noise. Reported as `n_positive` either way.
MIN_POSITIVES = 20


# --------------------------------------------------------------------------- metrics (no sklearn)


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUROC (the Mann-Whitney U form), ties averaged. None if either class is empty.

    Identical to `sklearn.metrics.roc_auc_score` including tie handling, which matters here: a
    degenerate model can emit the same score for every case, and the honest answer for that is
    0.5, not 1.0 or 0.0.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks within tied groups
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Area under the precision-recall curve, step-interpolated (sklearn's `average_precision_score`).

    Reported alongside AUROC because most of these labels are rare (1.6-41% prevalence); AUROC is
    optimistic under heavy imbalance in a way AP is not.

    Thresholds are the **distinct** scores, not the individual samples. The difference only shows
    up under ties -- but ties are exactly what a degenerate classifier produces, and the
    per-sample form rewards it: a constant-score model scores its own prevalence the sklearn way
    and noticeably above it the per-sample way.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    hits = y_true[order].astype(np.float64)
    tp = np.cumsum(hits)
    counted = np.arange(1, len(hits) + 1, dtype=np.float64)
    # Keep only the last index of each tied group: one operating point per distinct score.
    sorted_scores = scores[order]
    last_of_group = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp, counted = tp[last_of_group], counted[last_of_group]
    precision = tp / counted
    recall = tp / n_pos
    return float((np.diff(np.r_[0.0, recall]) * precision).sum())


# --------------------------------------------------------------------------- the head


class PathologyHead(nn.Module):
    """Feature vector -> one logit per label. One hidden layer, because a linear probe on 512
    pooled features underfits and a deeper net overfits 5,000 volumes.

    Deliberately small (~140k parameters): this is an *instrument*, not a contribution. A stronger
    classifier would make the metric more sensitive, and the way to get one is more labelled real
    volumes or a better backbone -- not more capacity on the same 512-d pooled feature.
    """

    def __init__(self, in_dim: int, n_labels: int, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        # Recorded so `ReportPathologyClassifier.load` rebuilds the architecture it is about to
        # load weights into, rather than the default one. Without this a non-default --hidden
        # trains fine and then cannot be loaded at all.
        self.config = {"in_dim": int(in_dim), "n_labels": int(n_labels),
                       "hidden": int(hidden), "dropout": float(dropout)}
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class ClassifierProvenance:
    """What this classifier was fitted on. Written into the checkpoint and copied into every
    results directory that uses it, so a consistency number is always traceable to an instrument."""

    train_cohort_id: str = ""
    train_split: str = ""
    n_train_cases: int = 0
    val_cohort_id: str = ""
    val_split: str = ""
    n_val_cases: int = 0
    labels_csv: str = ""
    feature_extractor: str = "medicalnet_resnet10"
    feature_checkpoint_sha256: str = ""
    epochs: int = 0
    seed: int = 0
    label_prevalence_train: dict = field(default_factory=dict)


class ReportPathologyClassifier:
    """A fitted head plus the feature standardisation it was fitted under.

    The feature extractor itself is *not* stored in the checkpoint (it is a 200 MB frozen file
    already on disk); its sha256 is, so a mismatch is detectable.
    """

    def __init__(self, labels, head: PathologyHead, feature_mean: np.ndarray,
                 feature_std: np.ndarray, provenance: ClassifierProvenance,
                 real_reference: dict | None = None, device: str = "cpu") -> None:
        self.labels = tuple(labels)
        self.head = head.eval().to(device)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.provenance = provenance
        self.real_reference = real_reference or {}
        self.device = device

    # -- persistence ---------------------------------------------------------------------

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": CLASSIFIER_FORMAT,
            "labels": list(self.labels),
            "head_state_dict": self.head.state_dict(),
            "head_config": dict(self.head.config),
            "in_dim": int(self.feature_mean.shape[0]),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "provenance": vars(self.provenance),
            "real_reference": self.real_reference,
        }, str(path))
        return path

    @classmethod
    def load(cls, path, device: str = "cpu") -> "ReportPathologyClassifier":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("format") != CLASSIFIER_FORMAT:
            raise SystemExit(
                f"{path} is not a {CLASSIFIER_FORMAT} checkpoint (format={payload.get('format')!r}). "
                f"Build one with `python -m mrrate_r2v.cli.train_report_classifier`."
            )
        labels = payload["labels"]
        # `head_config` is authoritative; the fallback covers a checkpoint written before it was
        # recorded, and a width mismatch then surfaces as a loud load_state_dict error.
        config = payload.get("head_config") or {"in_dim": int(payload["in_dim"]),
                                                "n_labels": len(labels)}
        head = PathologyHead(int(config["in_dim"]), int(config.get("n_labels", len(labels))),
                             hidden=int(config.get("hidden", 256)),
                             dropout=float(config.get("dropout", 0.2)))
        head.load_state_dict(payload["head_state_dict"])
        return cls(
            labels=labels, head=head,
            feature_mean=payload["feature_mean"], feature_std=payload["feature_std"],
            provenance=ClassifierProvenance(**payload.get("provenance", {})),
            real_reference=payload.get("real_reference", {}),
            device=device,
        )

    # -- inference -----------------------------------------------------------------------

    def standardize(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float32) - self.feature_mean) / np.maximum(self.feature_std, 1e-6)

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """`(N, D)` features -> `(N, n_labels)` probabilities."""
        x = torch.from_numpy(self.standardize(np.atleast_2d(features))).to(self.device)
        return torch.sigmoid(self.head(x)).cpu().numpy()


# --------------------------------------------------------------------------- the metric


def evaluate_consistency(
    probabilities: np.ndarray,
    truth: np.ndarray,
    labels,
    real_reference: dict | None = None,
    prevalence_baseline: dict | None = None,
) -> dict:
    """Per-label agreement between a classifier's verdict and the conditioning reports' labels.

    `probabilities` and `truth` are both `(N, n_labels)`, row-aligned by case. The return is
    per-label and then aggregated; nothing is averaged before a reader can see the per-label
    support, because a mean over 14 labels of wildly different prevalence is not a quantity
    anyone should quote on its own.

    `real_reference` is the same computation run on the REAL volumes of the same cohort -- the
    ceiling this instrument can reach. `prevalence_baseline` is what a guesser that ignores the
    image gets. A generated AUROC is only evidence of report-consistency insofar as it sits above
    the baseline; how *good* it is, is judged against the reference.
    """
    real_reference = real_reference or {}
    prevalence_baseline = prevalence_baseline or {}
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth).astype(int)

    per_label = {}
    for i, name in enumerate(labels):
        y, p = truth[:, i], probabilities[:, i]
        n_pos = int(y.sum())
        reference = real_reference.get(name, {}).get("auroc") if real_reference else None
        entry = {
            "auroc": auroc(y, p),
            "average_precision": average_precision(y, p),
            "prevalence": float(y.mean()) if len(y) else None,
            "n": int(len(y)),
            "n_positive": n_pos,
            "mean_predicted_probability": float(p.mean()) if len(p) else None,
            "real_reference_auroc": reference,
            "prevalence_baseline_auroc": prevalence_baseline.get(name),
            "low_support": n_pos < MIN_POSITIVES,
            # "usable" is a property of the INSTRUMENT on this cohort, decided by the real-data
            # reference, never by how well the generated volumes happened to score.
            "usable": bool(reference is not None and reference >= MIN_REAL_AUROC),
            # How much of the classifier's real-data discrimination survives on generated volumes.
            # 1.0 = the generated set is as separable as the real one; 0.0 = chance.
            #
            # **Always present, None when undefined.** Every per-label key is unconditional: a
            # label with no positives in this cohort has `auroc is None` and therefore no
            # retention, and a dict that sometimes lacks the key turns a legitimately
            # unmeasurable label into a KeyError that loses the whole evaluation. That is not
            # hypothetical -- it killed job 718982 after 14 minutes of feature extraction.
            "retention": None,
        }
        if entry["auroc"] is not None and reference:
            entry["retention"] = (entry["auroc"] - 0.5) / max(reference - 0.5, 1e-6)
        per_label[name] = entry

    def _mean(key, only_usable):
        # `.get`, not `[...]`: an aggregate must never be the thing that raises. A key missing from
        # a per-label entry means "not measurable for this label", which is a value to skip.
        vals = [e.get(key) for e in per_label.values()
                if e.get(key) is not None and not e["low_support"]
                and (e["usable"] or not only_usable)]
        return float(np.mean(vals)) if vals else None

    usable = [n for n, e in per_label.items() if e["usable"] and not e["low_support"]]
    return {
        "per_label": per_label,
        "labels_usable": usable,
        "n_labels_usable": len(usable),
        "macro_auroc_all_labels": _mean("auroc", only_usable=False),
        "macro_auroc_usable_labels": _mean("auroc", only_usable=True),
        "macro_average_precision_all_labels": _mean("average_precision", only_usable=False),
        "macro_retention_usable_labels": _mean("retention", only_usable=True),
        "min_real_auroc_for_usable": MIN_REAL_AUROC,
        "min_positives_for_support": MIN_POSITIVES,
        "interpretation": (
            "macro_auroc_usable_labels is the headline: how well a classifier trained only on real "
            "volumes recovers, from a generated volume, the findings its conditioning report "
            "described. Read it against real_reference_auroc (the same classifier on the real "
            "volumes -- the ceiling) and above prevalence_baseline_auroc (an image-blind guesser -- "
            "the floor). macro_retention_usable_labels expresses the same thing as a fraction of "
            "the reference's above-chance margin."
        ),
    }


def per_case_consistency(probabilities: np.ndarray, truth: np.ndarray, labels,
                         usable_labels=None) -> np.ndarray:
    """One agreement score per case, for the challenge's case-level permutation test.

    Mean over labels of `p` when the report says positive and `1 - p` when it says negative --
    i.e. the probability the classifier assigns to the report's own answer. Higher is more
    consistent, it is bounded in [0, 1], and unlike a thresholded accuracy it needs no operating
    point chosen after the fact.

    Restricted to `usable_labels` when given, since a label the classifier cannot do on real data
    contributes only noise to a per-case score.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth).astype(int)
    keep = [i for i, name in enumerate(labels)
            if usable_labels is None or name in set(usable_labels)]
    if not keep:
        return np.full(len(probabilities), np.nan)
    p, y = probabilities[:, keep], truth[:, keep]
    return (p * y + (1.0 - p) * (1 - y)).mean(axis=1)


def prevalence_baseline_auroc(labels) -> dict:
    """An image-blind guesser's AUROC: 0.5 for every label, by definition.

    Spelled out rather than left implicit because it is the number the classifier has to beat, and
    a reader should see it in the same table rather than have to remember it.
    """
    return {name: 0.5 for name in labels}


def load_classifier_or_none(path, device: str = "cpu"):
    """`ReportPathologyClassifier` or None, with the reason logged. The evaluator treats a missing
    classifier the way it treats a missing image-text model: the group is recorded unavailable with
    a reason, never silently skipped and never faked."""
    if not path:
        return None, "no --report-classifier passed"
    path = Path(path)
    if not path.is_file():
        return None, f"classifier checkpoint not found: {path}"
    try:
        return ReportPathologyClassifier.load(path, device=device), None
    # SystemExit, not just Exception: `load` raises SystemExit for a wrong-format file (the house
    # style for a user-facing message), and SystemExit is a BaseException -- so a bare
    # `except Exception` would let it kill the whole evaluation instead of recording the group as
    # unavailable, throwing away every other metric of a long run.
    except (Exception, SystemExit) as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


__all__ = [
    "CLASSIFIER_FORMAT",
    "ClassifierProvenance",
    "MIN_POSITIVES",
    "MIN_REAL_AUROC",
    "PathologyHead",
    "ReportPathologyClassifier",
    "auroc",
    "average_precision",
    "evaluate_consistency",
    "load_classifier_or_none",
    "per_case_consistency",
    "prevalence_baseline_auroc",
]
