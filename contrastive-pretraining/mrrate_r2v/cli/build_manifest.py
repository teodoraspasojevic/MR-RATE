#!/usr/bin/env python3
"""Stage 0: build the manifest. One-time, per storage location.

The manifest is one CSV row per eligible (study, series) pair -- what exists and where. It is
independent of split, geometry, report source, and sampling, so you build it once and every
cohort afterwards reads from it.

Pick `--source` to match how MR-RATE is stored on your machine:

    shards_parquet   WebDataset-style `shard-*.tar` + `series.parquet`   (needs pyarrow, no torch)
    data_path_archive  un-extracted `batchNN.tar` of per-study zips      (needs a metadata + splits CSV)
    extracted_dir    an already-extracted directory tree                 (needs torch + nibabel)

The two archive sources open no archives at all -- they build locators purely from each root's
own index, so a full build takes seconds. Always pass `--verify-sample N` afterwards: it
resolves N random rows for real and confirms the filename convention still holds.

    # shards layout, also writing the report index that ShardReportStore needs
    python -m mrrate_r2v.cli.build_manifest --source shards_parquet \\
        --shards-root <workspace>/MR-Rate-raw \\
        --out-csv <workspace>/r2v_manifest/manifest_shards_native.csv \\
        --out-report-index-csv <workspace>/r2v_manifest/report_index_shards_native.csv \\
        --verify-sample 20

    # DATA_PATH layout
    python -m mrrate_r2v.cli.build_manifest --source data_path_archive \\
        --data-root <workspace>/MR-RATE --metadata-csv <...>/metadata.tar.gz \\
        --splits-csv <...>/splits.csv --out-csv <...>/manifest.csv --verify-sample 20

    # extracted tree
    python -m mrrate_r2v.cli.build_manifest --source extracted_dir \\
        --data-folder /path/to/MR-RATE --metadata-csv <...>/metadata.csv \\
        --splits-csv <...>/splits.csv --out-csv <...>/manifest.csv

`--dry-run` reports what would be built without writing. Use it first on a source you have not
built from before.

**Interpreter note.** `shards_parquet` needs pyarrow but not torch, and `extracted_dir` needs
torch but not pyarrow. This script imports neither at module level, so whichever interpreter
you have works for the source you are using -- there is no separate standalone script anymore.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..data.manifest import (
    DEFAULT_EXCLUDED_MODALITIES,
    build_manifest_rows,
    build_manifest_rows_from_data_path_zips,
    build_manifest_rows_from_shards_parquet,
    build_shard_report_index,
    verify_archive_locators_sample,
    write_manifest_csv,
    write_report_index_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_manifest")

SOURCES = ("shards_parquet", "data_path_archive", "extracted_dir")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="shards_parquet", choices=list(SOURCES))
    p.add_argument("--out-csv", type=Path, default=None, help="required unless --dry-run")

    sh = p.add_argument_group("shards_parquet")
    sh.add_argument("--shards-root", type=Path, default=None)
    sh.add_argument("--out-report-index-csv", type=Path, default=None,
                    help="also write the (study_uid, archive_path) index ShardReportStore needs")

    dp = p.add_argument_group("data_path_archive")
    dp.add_argument("--data-root", type=Path, default=None)
    dp.add_argument("--batch-tar-pattern", default="{batch_id}.tar")

    ed = p.add_argument_group("extracted_dir")
    ed.add_argument("--data-folder", type=Path, default=None)
    ed.add_argument("--space", default="native_space",
                    choices=["native_space", "coreg_space", "atlas_space"])

    sc = p.add_argument_group("shared")
    sc.add_argument("--metadata-csv", type=Path, default=None, help="CSV or .tar.gz of per-batch CSVs")
    sc.add_argument("--splits-csv", type=Path, default=None)
    sc.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    sc.add_argument("--excluded-modalities", nargs="*", default=sorted(DEFAULT_EXCLUDED_MODALITIES),
                    help="pass with no values to exclude nothing")
    sc.add_argument("--include-derived", action="store_true")
    sc.add_argument("--include-localizer", action="store_true")
    sc.add_argument("--verify-sample", type=int, default=20,
                    help="resolve N random archive rows for real afterwards (0 to skip)")
    sc.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def build(args):
    policy = dict(
        excluded_modalities=frozenset(args.excluded_modalities),
        exclude_derived=not args.include_derived,
        exclude_localizer=not args.include_localizer,
    )
    if args.source == "shards_parquet":
        if not args.shards_root:
            raise SystemExit("--source shards_parquet requires --shards-root")
        return build_manifest_rows_from_shards_parquet(
            str(args.shards_root), splits=tuple(args.splits), **policy)
    if args.source == "data_path_archive":
        if not (args.data_root and args.metadata_csv and args.splits_csv):
            raise SystemExit("--source data_path_archive requires --data-root, --metadata-csv, --splits-csv")
        return build_manifest_rows_from_data_path_zips(
            str(args.data_root), str(args.metadata_csv), str(args.splits_csv),
            batch_tar_pattern=args.batch_tar_pattern, **policy)
    if not args.data_folder:
        raise SystemExit("--source extracted_dir requires --data-folder")
    return build_manifest_rows(
        str(args.data_folder), space=args.space,
        metadata_csv=str(args.metadata_csv) if args.metadata_csv else None,
        splits_csv=str(args.splits_csv) if args.splits_csv else None, **policy)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.dry_run and not args.out_csv:
        raise SystemExit("--out-csv is required unless --dry-run")

    log.info("source=%s", args.source)
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
        if args.source != "shards_parquet":
            raise SystemExit("--out-report-index-csv is only available for --source shards_parquet")
        index_rows = build_shard_report_index(str(args.shards_root), splits=tuple(args.splits))
        args.out_report_index_csv.parent.mkdir(parents=True, exist_ok=True)
        write_report_index_csv(index_rows, str(args.out_report_index_csv))
        log.info("wrote %s (%d studies with reports)", args.out_report_index_csv, len(index_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
