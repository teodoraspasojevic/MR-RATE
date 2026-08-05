#!/usr/bin/env python3
"""Score cached embeddings into the encoder x report-format x metric matrix. CPU only.

    python -m mrrate_r2v.cli.eval_text_encoders \\
        --corpus       /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/report_analysis/reports_all.jsonl \\
        --manifest-csv /hnvme/workspace/y100dc19-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv \\
        --cache        /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/textbench/embeddings \\
        --out          /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/textbench/results

Writes `metrics_matrix.csv`, `per_label_auroc.csv` and `summary.json`. The corpus, the train
limit and the seed must match the ones `cli.embed_reports` ran with -- the runner checks the
cached study ids against the corpus and fails loudly if they disagree.

`--fusion a+b` additionally scores concatenated pooled embeddings from two cached encoders, which
is the cheap frozen-encoder analogue of the multi-encoder conditioning used by the 2025 VLM3D
CT-track winner. Pass it more than once for several pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Score the text-encoder selection benchmark from cached embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoders", nargs="+", default=None, help="default: whatever is cached")
    parser.add_argument("--formats", nargs="+", default=None, help="default: whatever is cached")
    parser.add_argument("--fusion", action="append", default=[],
                        help="'encA+encB' -- score the concatenation too; repeatable")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-limit", type=int, default=20000,
                        help="must match cli.embed_reports")
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--min-prevalence", type=float, default=0.01)
    parser.add_argument("--mean-only", action="store_true",
                        help="probe on the mean-pooled vector alone instead of concat(mean, max)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _cached_pairs(cache_dir: Path, split: str):
    pairs = set()
    for path in cache_dir.glob(f"*__{split}.npz"):
        stem = path.name[: -len(f"__{split}.npz")]
        if "__" in stem and not stem.startswith("negation"):
            encoder, report_format = stem.split("__", 1)
            pairs.add((encoder, report_format))
    return pairs


def _build_fusion_caches(cache_dir: Path, pairs, formats, splits):
    """Materialise `encA+encB` caches by concatenating two existing ones.

    Feature-axis concatenation of pooled vectors, i.e. the fusion the runner can score without a
    model. Written into the same cache directory under the fused name so nothing downstream needs
    to know fusion happened.
    """
    import numpy as np

    from ..textbench.embed import cache_paths, read_cache, write_cache

    made = []
    for spec in pairs:
        members = spec.split("+")
        if len(members) < 2:
            raise SystemExit(f"--fusion needs at least two encoders, got '{spec}'")
        for report_format in formats:
            for split in splits:
                out_npz, _ = cache_paths(str(cache_dir), spec, report_format, split)
                if Path(out_npz).is_file():
                    made.append((spec, report_format, split))
                    continue
                try:
                    parts = [read_cache(str(cache_dir), m, report_format, split) for m in members]
                except FileNotFoundError:
                    continue
                uids = parts[0][2]
                if any(p[2] != uids for p in parts):
                    raise SystemExit(f"fusion '{spec}': member caches disagree on study order")
                mean = np.concatenate([p[0] for p in parts], axis=1)
                maximum = np.concatenate([p[1] for p in parts], axis=1)
                stats = {
                    "encoder": {"name": spec, "kind": "fusion_concat",
                                "members": [p[3].get("encoder", {}) for p in parts],
                                "output_dim": int(mean.shape[1])},
                    "report_format": report_format, "split": split,
                    # A fused run costs the sum of its members: throughput is the harmonic sum,
                    # truncation is the worst member's.
                    "reports_per_second": 1.0 / sum(
                        1.0 / max(p[3].get("reports_per_second", float("inf")), 1e-9) for p in parts),
                    "truncation": max((p[3].get("truncation") or {} for p in parts),
                                      key=lambda t: t.get("fraction_truncated", 0.0)),
                }
                write_cache(str(cache_dir), spec, report_format, split, mean, maximum, uids, stats)
                made.append((spec, report_format, split))
    return made


def main(argv=None):
    args = parse_args(argv)
    import os

    # Cap the BLAS thread pool *before* numpy/scikit-learn import -- they read these at load time.
    #
    # Two separate failures are being avoided here, and the second is the counter-intuitive one:
    #  1. Left alone, BLAS sizes its pool from the node's core count (384 on a Helma CPU node)
    #     rather than the job's cgroup, and oversubscribes.
    #  2. Handing it the *whole* allocation is also wrong. This workload is hundreds of small
    #     logistic fits on a 20,000 x 1,536 matrix, where per-call thread setup dominates the
    #     arithmetic. Measured: one scoring row took 18 s at 8 threads and roughly 6 minutes at
    #     48 -- a 20x slowdown from asking for more cores.
    # Hence a low cap, not the allocation. Override with MRRATE_BLAS_THREADS if profiling says so.
    threads = os.environ.get("MRRATE_BLAS_THREADS") or "8"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = threads

    import numpy as np

    from ..textbench.corpus import load_corpus
    from ..textbench.runner import run_benchmark
    from ..textbench.tasks import negation_probe

    manifest = str(args.manifest_csv) if args.manifest_csv else None
    train_records = load_corpus(str(args.corpus), splits=[args.train_split], manifest_csv=manifest,
                                limit_per_split=args.train_limit or None, seed=args.seed)
    test_records = load_corpus(str(args.corpus), splits=[args.test_split], manifest_csv=manifest,
                               limit_per_split=args.test_limit or None, seed=args.seed)
    print(f"corpus: {len(train_records)} train / {len(test_records)} test studies")

    cached = _cached_pairs(args.cache, args.test_split) & _cached_pairs(args.cache, args.train_split)
    encoders = args.encoders or sorted({e for e, _ in cached})
    formats = args.formats or sorted({f for _, f in cached})

    if args.fusion:
        made = _build_fusion_caches(args.cache, args.fusion, formats,
                                    [args.train_split, args.test_split])
        print(f"fusion caches: {len(made)} (encoder, format, split) combinations")
        encoders = list(encoders) + [f for f in args.fusion if f not in encoders]

    # Negation is scored per encoder, not per format: the minimal pairs are sentences, not reports.
    negation_cache = {}
    for encoder in encoders:
        members = encoder.split("+")
        vectors = []
        for member in members:
            path = args.cache / f"negation__{member}.npz"
            if not path.is_file():
                vectors = []
                break
            with np.load(path, allow_pickle=True) as data:
                vectors.append((data["negated"].astype(np.float32),
                                data["affirmed"].astype(np.float32),
                                [str(t) for t in data["topic"]]))
        if not vectors:
            continue
        negated = np.concatenate([v[0] for v in vectors], axis=1)
        affirmed = np.concatenate([v[1] for v in vectors], axis=1)
        result = negation_probe(negated, affirmed, vectors[0][2], seed=args.seed)
        for report_format in formats:
            negation_cache[f"{encoder}__{report_format}"] = result
        print(f"  negation[{encoder}]: delta={result['negation_delta']:.4f} "
              f"dominance={result['negation_dominance']:.4f} auroc={result['negation_auroc']:.4f} "
              f"cos(pair)={result['cos_pair']:.4f} cos(other topic)={result['cos_other_negated']:.4f}")

    print(f"scoring {len(encoders)} encoders x {len(formats)} formats")
    summary = run_benchmark(
        str(args.cache), train_records, test_records, encoders, formats, str(args.out),
        negation_cache=negation_cache, use_max_pooling=not args.mean_only, seed=args.seed,
        min_prevalence=args.min_prevalence, train_split=args.train_split,
        test_split=args.test_split)
    print(f"\n{len(summary['rows'])} rows -> {args.out}/metrics_matrix.csv")
    print(json.dumps({"weak_label_warning": summary["weak_label_warning"]}, indent=1))


if __name__ == "__main__":
    main()
