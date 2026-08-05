#!/usr/bin/env python3
"""Dataset-level report analysis: lengths, sections, headings, acquisition content, polarity.

Stage 0 of the text-encoder work. CPU only, no model, no volume read.

    # once per storage location: pull the report/label sidecars out of the shard tars (~3 min)
    python -m mrrate_r2v.cli.analyze_reports build-corpus \\
        --shards-root /hnvme/workspace/y100dc19-MR-Rate-raw \\
        --out /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/report_analysis/reports_all.jsonl

    # then, as often as you like
    python -m mrrate_r2v.cli.analyze_reports analyze \\
        --corpus /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/report_analysis/reports_all.jsonl \\
        --manifest-csv /hnvme/workspace/y100dc19-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv \\
        --out /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/report_analysis/analysis.json

    # token lengths and truncation rates per encoder tokenizer (needs transformers; still CPU)
    python -m mrrate_r2v.cli.analyze_reports tokens \\
        --corpus .../reports_all.jsonl --sample 20000 \\
        --out .../token_lengths.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MR-RATE report analysis for text-encoder selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-corpus", help="extract report/label sidecars from the shard tars")
    build.add_argument("--shards-root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    build.add_argument("--workers", type=int, default=16)

    analyze = sub.add_parser("analyze", help="dataset statistics")
    analyze.add_argument("--corpus", type=Path, required=True)
    analyze.add_argument("--manifest-csv", type=Path, default=None,
                         help="attaches (modality, plane) buckets from the manifest")
    analyze.add_argument("--splits", nargs="+", default=None)
    analyze.add_argument("--out", type=Path, required=True)

    tokens = sub.add_parser("tokens", help="token lengths and truncation rate per encoder x format")
    tokens.add_argument("--corpus", type=Path, required=True)
    tokens.add_argument("--encoders", nargs="+", default=None, help="default: every staged encoder")
    tokens.add_argument("--formats", nargs="+", default=None, help="default: every report format")
    tokens.add_argument("--sample", type=int, default=20000,
                        help="0 = all studies; a seeded sample is +-0.3%% on a percentile")
    tokens.add_argument("--seed", type=int, default=0)
    tokens.add_argument("--pretrained-dir", type=Path, default=None)
    tokens.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def cmd_build_corpus(args):
    from ..textbench.corpus import build_corpus

    n = build_corpus(str(args.shards_root), str(args.out), splits=tuple(args.splits),
                     workers=args.workers)
    print(f"{n} studies -> {args.out}")


def cmd_analyze(args):
    from ..textbench.analysis import analyze
    from ..textbench.corpus import load_corpus

    records = load_corpus(str(args.corpus), splits=args.splits,
                          manifest_csv=str(args.manifest_csv) if args.manifest_csv else None)
    print(f"loaded {len(records)} studies")
    result = analyze(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float))
    print(f"wrote {args.out}")

    health = result["health"]
    print(f"  empty reports: {health['n_empty_report']}, "
          f"duplicate reports: {health['n_duplicate_reports']}, "
          f"texts spanning >1 split: {health['n_texts_in_multiple_splits']}")
    for name in ("raw", "findings", "impression", "findings+impression"):
        words = result["lengths"][name]["words"]
        if words:
            print(f"  {name:<22} words: median={words['median']:.0f} p95={words['p95']:.0f} "
                  f"max={words['max']:.0f}")


def cmd_tokens(args):
    import numpy as np

    from ..textbench.corpus import load_corpus
    from ..textenc.encoders import (
        DEFAULT_PRETRAINED_DIR,
        ENCODER_SPECS,
        HFTextEncoder,
        available_encoders,
    )
    from ..textenc.formats import REPORT_FORMATS, format_report

    root = str(args.pretrained_dir or DEFAULT_PRETRAINED_DIR)
    staged = available_encoders(root)
    encoders = args.encoders or [name for name, ok in staged.items() if ok]
    missing = [e for e in encoders if not staged.get(e)]
    if missing:
        sys.exit(f"not staged in {root}: {missing}. Run cli.download_text_encoders first.")
    formats = args.formats or list(REPORT_FORMATS)

    records = load_corpus(str(args.corpus))
    if args.sample and args.sample < len(records):
        order = np.random.default_rng(args.seed).permutation(len(records))[:args.sample]
        records = [records[i] for i in sorted(order)]
    print(f"{len(records)} studies, {len(encoders)} encoders, {len(formats)} formats")

    texts = {name: [t for t in (format_report(r.as_report_record(), name) for r in records) if t]
             for name in formats}

    out = {}
    for encoder in encoders:
        spec = ENCODER_SPECS[encoder]
        path = Path(root) / spec.directory
        # Same constructor the encoder itself uses -- never a second copy. See _load_tokenizer.
        tokenizer = HFTextEncoder._load_tokenizer(spec, str(path))
        out[encoder] = {"context": spec.context, "default_max_length": spec.default_max_length,
                        "tokenizer": type(tokenizer).__name__, "formats": {}}
        print(f"\n=== {encoder} ({type(tokenizer).__name__}, context {spec.context})")
        print(f"{'format':<30} {'mean':>7} {'median':>7} {'p95':>6} {'p99':>7} {'max':>7} "
              f"{'>512':>8} {'>context':>9}")
        for name in formats:
            lengths = np.array([len(ids) for ids in tokenizer(
                texts[name], add_special_tokens=True, truncation=False, padding=False,
                return_attention_mask=False)["input_ids"]])
            row = {
                "n": int(lengths.size), "mean": float(lengths.mean()),
                "median": float(np.median(lengths)), "p90": float(np.percentile(lengths, 90)),
                "p95": float(np.percentile(lengths, 95)), "p99": float(np.percentile(lengths, 99)),
                "max": int(lengths.max()),
                "over_512_pct": float(100.0 * (lengths > 512).mean()),
                "over_context_pct": float(100.0 * (lengths > spec.context).mean()),
            }
            out[encoder]["formats"][name] = row
            print(f"{name:<30} {row['mean']:>7.1f} {row['median']:>7.0f} {row['p95']:>6.0f} "
                  f"{row['p99']:>7.0f} {row['max']:>7} {row['over_512_pct']:>7.2f}% "
                  f"{row['over_context_pct']:>8.2f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


def main(argv=None):
    args = parse_args(argv)
    {"build-corpus": cmd_build_corpus, "analyze": cmd_analyze, "tokens": cmd_tokens}[args.command](args)


if __name__ == "__main__":
    main()
