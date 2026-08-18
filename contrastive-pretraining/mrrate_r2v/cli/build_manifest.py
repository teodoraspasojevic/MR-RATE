#!/usr/bin/env python3
"""Stage 0: build the manifest. One-time, per storage location.

The manifest is one CSV row per eligible (study, series) pair -- what exists and where. It is
independent of split, geometry, report source, and sampling, so you build it once and every
cohort afterwards reads from it.

Builds from SHARDS_PATH's WebDataset-style `shard-*.tar` + `series.parquet` layout (needs
pyarrow, not torch). Opens no archives -- locators come purely from `series.parquet`'s own
index, so a full build takes seconds. Always pass `--verify-sample N` afterwards: it resolves
N random rows for real and confirms the filename convention still holds.

    # also writing the report index that ShardReportStore needs
    python -m mrrate_r2v.cli.build_manifest \\
        --shards-root <workspace>/MR-Rate-raw \\
        --out-csv <workspace>/r2v_manifest/manifest_shards_native.csv \\
        --out-report-index-csv <workspace>/r2v_manifest/report_index_shards_native.csv \\
        --verify-sample 20

`--dry-run` reports what would be built without writing.

(Two other sources -- an already-extracted directory tree, and DATA_PATH's `batchNN.tar` of
per-study zips -- were supported here and removed 2026-08-18: nothing in this repo built a
manifest from either. Re-add one only if a real storage location needs it; `git log` has the
removed implementations if so.)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..data.manifest import (
    DEFAULT_EXCLUDED_MODALITIES,
    build_manifest_rows_from_shards_parquet,
    build_shard_report_index,
    verify_archive_locators_sample,
    write_manifest_csv,
    write_report_index_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_manifest")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-csv", type=Path, default=None, help="required unless --dry-run")
    p.add_argument("--shards-root", type=Path, default=None)
    p.add_argument("--out-report-index-csv", type=Path, default=None,
                    help="also write the (study_uid, archive_path) index ShardReportStore needs")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--excluded-modalities", nargs="*", default=sorted(DEFAULT_EXCLUDED_MODALITIES),
                   help="pass with no values to exclude nothing")
    p.add_argument("--include-derived", action="store_true")
    p.add_argument("--include-localizer", action="store_true")
    p.add_argument("--verify-sample", type=int, default=20,
                   help="resolve N random archive rows for real afterwards (0 to skip)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def build(args):
    if not args.shards_root:
        raise SystemExit("--shards-root is required")
    policy = dict(
        excluded_modalities=frozenset(args.excluded_modalities),
        exclude_derived=not args.include_derived,
        exclude_localizer=not args.include_localizer,
    )
    return build_manifest_rows_from_shards_parquet(
        str(args.shards_root), splits=tuple(args.splits), **policy)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.dry_run and not args.out_csv:
        raise SystemExit("--out-csv is required unless --dry-run")

    rows = build(args)
    if not rows:
        raise SystemExit("0 rows built -- check the source paths and the split/eligibility filters")

    by_split, by_modality = {}, {}
    for r in rows:
        by_split[r.split] = by_split.get(r.split, 0) + 1
        by_modality[r.modality] = by_modality.get(r.modality, 0) + 1
    log.info("%d rows  by split: %s", len(rows), dict(sorted(by_split.items(), key=lambda kv: str(kv[0]))))
    log.info("           by modality: %s", dict(sorted(by_modality.items(), key=lambda kv: str(kv[0]))))

    n_archive = sum(1 for r in rows if r.backend == "archive")
    if args.verify_sample and n_archive:
        n_ok, failures = verify_archive_locators_sample(rows, n=args.verify_sample)
        log.info("locator spot-check: %d/%d resolved", n_ok, min(args.verify_sample, n_archive))
        for idx, locator, err in failures:
            log.error("  sample %d: %s -- %s", idx, locator, err)
        if failures:
            raise SystemExit(
                f"{len(failures)} of {min(args.verify_sample, n_archive)} sampled locators did not "
                f"resolve. The filename convention or the archives themselves have changed -- do "
                f"not use this manifest until that is understood."
            )

    if args.dry_run:
        log.info("--dry-run: nothing written")
        return 0

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(rows, str(args.out_csv))
    log.info("wrote %s", args.out_csv)

    if args.out_report_index_csv:
        index_rows = build_shard_report_index(str(args.shards_root), splits=tuple(args.splits))
        args.out_report_index_csv.parent.mkdir(parents=True, exist_ok=True)
        write_report_index_csv(index_rows, str(args.out_report_index_csv))
        log.info("wrote %s (%d studies with reports)", args.out_report_index_csv, len(index_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
