#!/usr/bin/env python3
"""Fit the blinded pathology classifier the report-consistency metric is built on.

    python -m mrrate_r2v.cli.train_report_classifier \\
        --manifest <data>/r2v_manifest/manifest_shards_native.csv \\
        --report-index <data>/r2v_manifest/report_index_shards_native.csv \\
        --medicalnet-checkpoint <ws>/pretrained/medicalnet/resnet_10_23dataset_statedict.pth \\
        --out <ws>/models/report_classifier_v2.pt

The instrument, not a contribution: a frozen MedicalNet ResNet-10 turns each volume into 512
features, and a ~140k-parameter head maps those to the 14 merged clinical labels. See
`eval/report_classifier.py` for why it takes the image and nothing else, and why every number it
later produces is reported next to its own real-data reference.

**Like `cli.evaluate`, this reads the Dataset directly -- there are no cohorts.** It builds the
same `R2VDatasetConfig` from the same flags, so the classifier is fitted on volumes preprocessed
exactly the way the volumes it will later score were. That is not a convention: a head fitted on
differently-normalised features measures a domain gap rather than a pathology, and under the old
cohort pipeline the two could drift apart with nothing able to see it.

**It fits on `--split train` and selects on `--val-split val`.** Fitting on `test` would make every
consistency number downstream circular -- the generations are scored on that split -- so it is
refused unless `--allow-test-split` is passed for a deliberate ceiling experiment.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..cohort import sha256_file
from ..eval.report_classifier import (
    ClassifierProvenance,
    PathologyHead,
    ReportPathologyClassifier,
    auroc,
    average_precision,
)
from ..eval.report_labels import ReportLabels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_report_classifier")


def extract_features(dataset, cases, medicalnet_checkpoint, device: str,
                     cache: Path | None = None) -> np.ndarray:
    """`(N, 512)` MedicalNet features, in `cases` order, streamed from the Dataset.

    One volume is held at a time: the classifier's training set is thousands of volumes and none of
    them needs to outlive its own forward pass.

    Cached keyed by the run fingerprint and the case count, so a cache built from a different split,
    a different normalizer or a different case list cannot be silently reused. The cache exists to
    make re-fitting the head (seconds) cheap relative to re-extracting (minutes).
    """
    if cache is not None and cache.is_file():
        stored = np.load(cache, allow_pickle=False)
        if list(stored["case_ids"]) == [c.case_id for c in cases]:
            log.info("features: reusing %s (%s)", cache, stored["features"].shape)
            return stored["features"]
        log.warning("%s does not match this case list -- re-extracting", cache)

    from ..eval.distribution import MedicalNetFeatureExtractor

    extractor = MedicalNetFeatureExtractor(Path(medicalnet_checkpoint), device)
    out = np.zeros((len(cases), 512), dtype=np.float32)
    t0 = time.time()
    for i, case in enumerate(cases):
        volume = dataset[case.index]["image"].squeeze(0).float().numpy().astype(np.float32)
        out[i] = extractor.extract(volume)
        del volume
        if (i + 1) % 200 == 0 or i + 1 == len(cases):
            log.info("features [%d/%d] %.1fs", i + 1, len(cases), time.time() - t0)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=out,
                            case_ids=np.array([c.case_id for c in cases]))
    return out


def prepare(args, split: str, n_per_bucket, labels: ReportLabels, cache_dir):
    """`(view, cases, features, truth)` for one split, restricted to labelled cases.

    Builds the dataset the way `cli.evaluate` and `cli.train_r2v` build it, and selects cases with
    the same deterministic no-RNG rule -- so which volumes the classifier is fitted on is a
    function of the split and the sample cap, and of nothing else.
    """
    from types import SimpleNamespace

    from ..eval.live import LiveCohortView, build_cases, run_fingerprint, select_eval_cases
    from .evaluate import build_dataset, population_bucket_counts

    dataset, config = build_dataset(SimpleNamespace(
        manifest=args.manifest, report_index=args.report_index, split=split,
        report_sections=args.report_sections, report_format=args.report_format,
        geometry_mode=args.geometry_mode, posterior_shift_mm=args.posterior_shift_mm,
        normalizer=args.normalizer, seed=args.seed,
    ))
    cases = build_cases(dataset, select_eval_cases(dataset, n_per_bucket))
    run_id = run_fingerprint(split=split, cases=cases, geometry=config.geometry_fingerprint(),
                             task="report_classifier", n_per_bucket=n_per_bucket, seed=args.seed,
                             model_identity={"name": "medicalnet_resnet10"})
    view = LiveCohortView(cases=cases, split=split, geometry=config.geometry_fingerprint(),
                          population_bucket_counts=population_bucket_counts(dataset), run_id=run_id)

    joined = labels.for_cohort(view)
    coverage = labels.cohort_coverage(view)
    log.info("split=%s run_id=%s: %d cases, %d labelled (%d unlabelled -- excluded, never imputed)",
             split, run_id, coverage["n_cases"], coverage["n_labelled"], coverage["n_unlabelled"])
    if not joined:
        raise SystemExit(f"split {split}: no case's study appears in {labels.path}")

    labelled = [c for c in cases if c.case_id in joined]
    truth = np.array([joined[c.case_id] for c in labelled], dtype=np.int64)
    cache = (Path(cache_dir) / f"medicalnet_{run_id}_{len(labelled)}.npz") if cache_dir else None
    features = extract_features(dataset, labelled, args.medicalnet_checkpoint, args.device, cache)
    return view, labelled, features, truth


def score_all(probabilities: np.ndarray, truth: np.ndarray, label_names) -> dict:
    """Per-label AUROC / AP / support, the form the checkpoint stores as `real_reference`."""
    return {
        name: {
            "auroc": auroc(truth[:, i], probabilities[:, i]),
            "average_precision": average_precision(truth[:, i], probabilities[:, i]),
            "prevalence": float(truth[:, i].mean()),
            "n": int(truth.shape[0]),
            "n_positive": int(truth[:, i].sum()),
        }
        for i, name in enumerate(label_names)
    }


def fit(features_train, truth_train, features_val, truth_val, label_names, args) -> tuple:
    """Fit the head. Returns `(head, mean, std, best_val_reference, history)`.

    Selection is on **macro AUROC over the validation split**, not on the training loss: the labels
    are heavily imbalanced (1.6-41%), so a loss that looks fine can belong to a head that has
    learned only the prior. `pos_weight` counteracts the same imbalance in the objective itself.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    mean = features_train.mean(axis=0)
    std = features_train.std(axis=0)
    xt = torch.from_numpy((features_train - mean) / np.maximum(std, 1e-6)).float().to(device)
    yt = torch.from_numpy(truth_train).float().to(device)
    xv = torch.from_numpy((features_val - mean) / np.maximum(std, 1e-6)).float().to(device)

    n_pos = truth_train.sum(axis=0)
    # clamp: a label with almost no positives would otherwise get a weight in the hundreds and
    # dominate the gradient for a column the head cannot learn anyway.
    pos_weight = torch.tensor(
        np.clip((len(truth_train) - n_pos) / np.maximum(n_pos, 1), 1.0, 20.0), dtype=torch.float32
    ).to(device)
    log.info("pos_weight per label: %s", np.round(pos_weight.cpu().numpy(), 2).tolist())

    head = PathologyHead(features_train.shape[1], len(label_names), hidden=args.hidden,
                         dropout=args.dropout).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"macro_auroc": -1.0, "epoch": -1, "state": None, "reference": None}
    history = []
    n = len(xt)
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)

        head.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(head(xv)).cpu().numpy()
        reference = score_all(probabilities, truth_val, label_names)
        aurocs = [v["auroc"] for v in reference.values() if v["auroc"] is not None]
        macro = float(np.mean(aurocs)) if aurocs else 0.0
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_auroc": macro})
        log.info("epoch %3d  train_loss %.4f  val macro AUROC %.4f%s",
                 epoch, total / n, macro, "  *" if macro > best["macro_auroc"] else "")
        if macro > best["macro_auroc"]:
            best = {"macro_auroc": macro, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "reference": reference}

    if best["state"] is None:
        raise SystemExit("no epoch completed -- nothing to save")
    head.load_state_dict(best["state"])
    log.info("selected epoch %d (val macro AUROC %.4f)", best["epoch"], best["macro_auroc"])
    return head, mean, std, best["reference"], history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- data: the same flags cli.train_r2v and cli.evaluate take, read off the same dataclass,
    # so the classifier cannot be fitted on volumes preprocessed differently from the ones it will
    # later score.
    from ..data import R2VDatasetConfig

    p.add_argument("--manifest", type=Path, required=True, help="MR-RATE manifest CSV")
    p.add_argument("--report-index", type=Path, required=True, help="report index CSV")
    p.add_argument("--split", default="train", help="split to FIT on")
    p.add_argument("--val-split", default="val", help="split to select the epoch on")
    p.add_argument("--n-per-bucket", type=int, default=500,
                   help="cases per (modality, plane) to fit on. Deterministic prefix, no RNG. "
                        "Unlike cli.evaluate this defaults to a cap rather than the whole split: "
                        "the head is ~140k parameters and saturates long before 575k volumes")
    p.add_argument("--val-n-per-bucket", type=int, default=100)
    p.add_argument("--report-sections", nargs="+", default=["findings", "impression"])
    p.add_argument("--report-format", default=None)
    p.add_argument("--geometry-mode", default="per_modality_plane",
                   choices=["per_modality_plane", "fixed"])
    p.add_argument("--posterior-shift-mm", type=float,
                   default=R2VDatasetConfig.posterior_shift_mm)
    p.add_argument("--normalizer", default=R2VDatasetConfig.normalizer,
                   choices=["percentile", "zscore", "minmax"])
    p.add_argument("--medicalnet-checkpoint", type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, default=None,
                   help="default: scripts/eval_labels/splits_merged_majority/mrrate_merged_labels.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--feature-cache-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--allow-test-split", action="store_true",
                   help="fit on the test split. Only for a deliberate ceiling experiment -- a "
                        "classifier fitted on test makes every consistency number circular")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    labels = ReportLabels(args.labels_csv)
    log.info("labels: %d classes over %d studies (%s)", len(labels.labels), len(labels), labels.path)

    # Refused BEFORE any feature extraction: fitting on the split the generations are scored
    # against makes every consistency number circular, and that is not something to discover after
    # an hour of MedicalNet forward passes.
    if args.split == "test" and not args.allow_test_split:
        raise SystemExit(
            "--split test would fit the blinded classifier on the same split the generations are "
            "scored against, which makes every consistency number circular. Use --split train, or "
            "pass --allow-test-split for a deliberate ceiling experiment."
        )
    train_view, train_cases, features_train, truth_train = prepare(
        args, args.split, args.n_per_bucket, labels, args.feature_cache_dir)
    val_view, val_cases, features_val, truth_val = prepare(
        args, args.val_split, args.val_n_per_bucket, labels, args.feature_cache_dir)

    prevalence = labels.prevalence(labels.for_cohort(train_view))
    log.info("train prevalence: %s", {k: round(v, 4) for k, v in prevalence.items()})

    head, mean, std, reference, history = fit(
        features_train, truth_train, features_val, truth_val, labels.labels, args)

    classifier = ReportPathologyClassifier(
        labels=labels.labels, head=head, feature_mean=mean, feature_std=std,
        provenance=ClassifierProvenance(
            # `train_cohort_id` now carries the live run fingerprint -- same guarantee (equal
            # value means the same cases at the same geometry under the same preprocessing), and
            # the field name is kept so an existing checkpoint still loads.
            train_cohort_id=train_view.cohort_id, train_split=args.split,
            n_train_cases=len(train_cases),
            val_cohort_id=val_view.cohort_id, val_split=args.val_split,
            n_val_cases=len(val_cases),
            labels_csv=str(labels.path),
            feature_checkpoint_sha256=sha256_file(args.medicalnet_checkpoint),
            epochs=args.epochs, seed=args.seed, label_prevalence_train=prevalence,
        ),
        real_reference=reference, device=args.device,
    )
    out = classifier.save(args.out)

    report = {"history": history, "val_reference": reference,
              "provenance": vars(classifier.provenance)}
    Path(str(out) + ".report.json").write_text(json.dumps(report, indent=2, default=str))

    log.info("wrote %s", out)
    log.info("%-42s %8s %8s %6s", "label", "AUROC", "AP", "n_pos")
    for name, v in reference.items():
        a = "n/a" if v["auroc"] is None else f"{v['auroc']:.4f}"
        ap = "n/a" if v["average_precision"] is None else f"{v['average_precision']:.4f}"
        log.info("%-42s %8s %8s %6d", name, a, ap, v["n_positive"])
    log.info("Read these as the CEILING: a generated volume's consistency can never exceed what "
             "this classifier achieves on real volumes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
