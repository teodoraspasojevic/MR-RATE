#!/usr/bin/env python3
"""Cache frozen-encoder embeddings for every (encoder, report format, split). GPU-bound stage.

This is the only expensive step of the text-encoder benchmark; scoring afterwards is CPU-seconds.
No label is baked into the cache, so a new metric or a new label set never triggers a re-encode.

    python -m mrrate_r2v.cli.embed_reports \\
        --corpus       /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/report_analysis/reports_all.jsonl \\
        --manifest-csv /hnvme/workspace/y100dc19-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv \\
        --out          /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/textbench/embeddings \\
        --encoders radbert bioclinical_mbert medembed_large medembed_small \\
        --formats  findings_impression impression findings \\
        --train-limit 20000 --max-length 512 --batch-size 64

`--max-length` is applied to every encoder, so the matrix compares encoders at one token budget.
Leave it unset to use each encoder's own default (which lets the 8192-context models read every
report in full -- a different, also useful, comparison).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("embed_reports")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cache text-encoder embeddings for the selection benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoders", nargs="+", default=None, help="default: every staged encoder")
    parser.add_argument("--formats", nargs="+", default=None, help="default: every report format")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--train-limit", type=int, default=20000,
                        help="seeded subsample of the train split; 0 = all 88,985 studies")
    parser.add_argument("--test-limit", type=int, default=0, help="0 = the whole split")
    parser.add_argument("--max-length", type=int, default=None,
                        help="one token budget for every encoder; unset = each encoder's default")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--negation-pairs", type=int, default=4000,
                        help="0 disables the negation minimal-pair cache")
    parser.add_argument("--pooling", default=None, choices=["mean", "cls"],
                        help="override the encoder spec's pooling. Only cxr_bert defaults to 'cls' "
                             "(its CLIP-aligned vector); use this to test that choice rather than "
                             "assume it")
    parser.add_argument("--pretrained-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch

    from ..textbench.corpus import load_corpus
    from ..textbench.embed import cache_paths, embed_corpus, write_cache
    from ..textbench.negation import mine_negation_pairs
    from ..textenc.encoders import DEFAULT_PRETRAINED_DIR, available_encoders, build_encoder
    from ..textenc.formats import REPORT_FORMATS

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # torch sizes its thread pool from the *node's* core count, not from the cgroup it was given.
    # Several of these jobs on one 384-core CPU node then each spawn 384 threads onto their own 48
    # cores, and throughput collapses by roughly the oversubscription factor. Measured on Helma:
    # 8 concurrent jobs went from <7 reports/s each to the rates in the run log after this fix.
    allotted = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS")
    if allotted and allotted.isdigit():
        torch.set_num_threads(int(allotted))
    log.info("torch threads: %d (SLURM_CPUS_PER_TASK=%s)", torch.get_num_threads(),
             os.environ.get("SLURM_CPUS_PER_TASK"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    root = str(args.pretrained_dir or DEFAULT_PRETRAINED_DIR)

    staged = available_encoders(root)
    encoders = args.encoders or [name for name, ok in staged.items() if ok]
    unstaged = [e for e in encoders if not staged.get(e)]
    if unstaged:
        raise SystemExit(f"not staged in {root}: {unstaged}. "
                         f"Run `python -m mrrate_r2v.cli.download_text_encoders --encoders "
                         f"{' '.join(unstaged)}` first.")
    formats = args.formats or list(REPORT_FORMATS)
    unknown = [f for f in formats if f not in REPORT_FORMATS]
    if unknown:
        raise SystemExit(f"unknown format(s) {unknown}; choose from {sorted(REPORT_FORMATS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = str(args.manifest_csv) if args.manifest_csv else None

    by_split = {}
    for split in args.splits:
        limit = args.train_limit if split == "train" else args.test_limit
        records = load_corpus(str(args.corpus), splits=[split], manifest_csv=manifest,
                              limit_per_split=limit or None, seed=args.seed)
        by_split[split] = records
        log.info("split %s: %d studies", split, len(records))

    negation_pairs = []
    if args.negation_pairs:
        source = by_split.get("test") or next(iter(by_split.values()))
        negation_pairs = mine_negation_pairs(source, max_pairs=args.negation_pairs, seed=args.seed)
        log.info("mined %d negation minimal pairs over %d topics",
                 len(negation_pairs), len({p.topic for p in negation_pairs}))
        (args.out / "negation_pairs_meta.json").write_text(json.dumps({
            "n_pairs": len(negation_pairs), "n_topics": len({p.topic for p in negation_pairs}),
            "topics": sorted({p.topic for p in negation_pairs}), "seed": args.seed,
        }, indent=1))

    for encoder_name in encoders:
        kwargs = {"pretrained_dir": root, "dtype": dtype}
        if args.max_length is not None:
            kwargs["max_length"] = args.max_length
        if args.pooling is not None:
            kwargs["pooling"] = args.pooling
        encoder = build_encoder(encoder_name, **kwargs)
        encoder.model.to(device)
        log.info("=== %s: dim=%d max_length=%d (supports %d) on %s",
                 encoder_name, encoder.output_dim, encoder.max_length,
                 encoder.encoder_max_length, device)
        for report_format in formats:
            for split, records in by_split.items():
                npz_path, _ = cache_paths(str(args.out), encoder_name, report_format, split)
                if os.path.isfile(npz_path) and not args.overwrite:
                    log.info("  [skip] %s/%s/%s cached", encoder_name, report_format, split)
                    continue
                log.info("  [run ] %s/%s/%s (%d studies)", encoder_name, report_format, split,
                         len(records))
                mean, maximum, uids, stats = embed_corpus(
                    encoder, records, report_format, device, batch_size=args.batch_size)
                stats["split"] = split
                stats["device"] = str(device)
                stats["dtype"] = args.dtype
                if device.type == "cuda":
                    stats["peak_gpu_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
                    torch.cuda.reset_peak_memory_stats(device)
                write_cache(str(args.out), encoder_name, report_format, split, mean, maximum,
                            uids, stats)
                log.info("        %.0f reports/s, truncated %.2f%%",
                         stats["reports_per_second"],
                         100 * stats["truncation"].get("fraction_truncated", 0.0))

        if negation_pairs:
            path = args.out / f"negation__{encoder_name}.npz"
            if path.is_file() and not args.overwrite:
                log.info("  [skip] negation pairs cached")
            else:
                started = time.time()
                vectors = {}
                for field in ("negated", "affirmed"):
                    sentences = [getattr(p, field) for p in negation_pairs]
                    chunks = []
                    for start in range(0, len(sentences), args.batch_size):
                        conditioning = encoder.encode(sentences[start:start + args.batch_size], device)
                        chunks.append(conditioning.pooled_embedding.to(torch.float32)
                                      .detach().cpu().numpy())
                    vectors[field] = np.concatenate(chunks).astype(np.float16)
                np.savez_compressed(path, negated=vectors["negated"], affirmed=vectors["affirmed"],
                                    topic=np.array([p.topic for p in negation_pairs], dtype=object),
                                    allow_pickle=True)
                log.info("  [run ] negation pairs: %d in %.1fs", len(negation_pairs),
                         time.time() - started)
        summary = encoder.log_truncation_summary()
        log.info("=== %s truncation over the whole run: %s", encoder_name, summary)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log.info("caches in %s", args.out)


if __name__ == "__main__":
    main()
