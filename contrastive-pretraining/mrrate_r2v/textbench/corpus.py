"""The report corpus the benchmark runs on. Stdlib + numpy only; no torch, no transformers.

One study = one `CorpusRecord`. Built once from the shard tars into a JSONL, then reloaded
cheaply. Nothing here reads a volume: the whole benchmark is text-only, which is what makes it
runnable on a CPU node in minutes.

Privacy: `study_uid` is already the released anonymised identifier and is the only key kept. No
patient id, no date, no series id, and no raw report text ever leaves this package -- the
analysis emits counts and statistics, never verbatim clinical text.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, Optional

SPLIT_DIRECTORIES = {"train": "train", "validation": "val", "test": "test"}
SIDECARS = ("report.json", "labels.json")


@dataclass
class CorpusRecord:
    study_uid: str
    split: str
    raw: Optional[str] = None
    clinical_information: Optional[str] = None
    technique: Optional[str] = None
    findings: Optional[str] = None
    impression: Optional[str] = None
    labels: dict = field(default_factory=dict)
    buckets: tuple = ()          # ("T1w__AXIAL", ...) -- from the manifest, not from the text

    def as_report_record(self):
        """A `data.reports.ReportRecord` view, so `textenc.format_report` applies unchanged."""
        from ..data.reports import ReportRecord

        return ReportRecord(raw=self.raw, clinical_information=self.clinical_information,
                            technique=self.technique, findings=self.findings,
                            impression=self.impression)


# --------------------------------------------------------------------------- building


def _one_shard(job):
    split_directory, split, tar_path = job
    studies: dict[str, dict] = {}
    try:
        with tarfile.open(tar_path) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                if len(parts) != 2 or parts[1] not in SIDECARS:
                    continue
                try:
                    studies.setdefault(parts[0], {})[parts[1][:-5]] = json.loads(
                        tar.extractfile(member).read())
                except Exception:  # noqa: BLE001 -- one bad sidecar must not kill the shard
                    continue
    except Exception as exc:  # noqa: BLE001
        return split, tar_path, [], repr(exc)
    rows = []
    for study_uid, found in studies.items():
        report = found.get("report") or {}
        rows.append({
            "study_uid": study_uid, "split": split,
            "raw": report.get("report"),
            "clinical_information": report.get("clinical_information"),
            "technique": report.get("technique"),
            "findings": report.get("findings"),
            "impression": report.get("impression"),
            "labels": found.get("labels") or {},
        })
    return split, tar_path, rows, None


def build_corpus(shards_root, out_jsonl, splits=("train", "validation", "test"), workers=16):
    """Scan the shard tars for `report.json`/`labels.json` and write one JSON object per study.

    Measured ~0.25 s per shard (header seek + two small member reads, the volumes are never
    touched), so the full 3,762-shard release takes a couple of minutes at 16 workers.
    """
    jobs = []
    for directory in splits:
        split = SPLIT_DIRECTORIES[directory]
        pattern = os.path.join(shards_root, directory)
        if not os.path.isdir(pattern):
            raise FileNotFoundError(f"split directory not found: {pattern}")
        for name in sorted(os.listdir(pattern)):
            if name.startswith("shard-") and name.endswith(".tar"):
                jobs.append((directory, split, os.path.join(pattern, name)))
    print(f"[corpus] {len(jobs)} shards -> {out_jsonl}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)) or ".", exist_ok=True)
    written, errors = 0, []
    with open(out_jsonl, "w") as handle, ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (_split, tar_path, rows, error) in enumerate(pool.map(_one_shard, jobs, chunksize=8)):
            if error:
                errors.append((tar_path, error))
            for row in rows:
                handle.write(json.dumps(row) + "\n")
                written += 1
            if (i + 1) % 500 == 0:
                print(f"[corpus]   {i+1}/{len(jobs)} shards, {written} studies", flush=True)
    print(f"[corpus] wrote {written} studies", flush=True)
    if errors:
        print(f"[corpus] {len(errors)} unreadable shards, first: {errors[0]}", file=sys.stderr)
    return written


def read_bucket_index(manifest_csv):
    """{study_uid: ("T1w__AXIAL", ...)} from the manifest CSV.

    The manifest is the repo's own DICOM-derived series inventory, so a bucket label here is
    structured metadata -- independent of whatever the report happens to say about technique.
    """
    buckets: dict[str, set] = {}
    with open(manifest_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            uid = row.get("study_uid")
            if not uid:
                continue
            modality, plane = row.get("modality") or "", row.get("plane") or ""
            if modality and plane:
                buckets.setdefault(uid, set()).add(f"{modality}__{plane}")
    return {uid: tuple(sorted(values)) for uid, values in buckets.items()}


# --------------------------------------------------------------------------- loading


def load_corpus(jsonl_path, splits: Optional[Iterable[str]] = None,
                manifest_csv: Optional[str] = None, limit_per_split: Optional[int] = None,
                seed: int = 0):
    """Load the corpus, optionally attaching bucket labels and subsampling deterministically.

    Subsampling is a seeded permutation per split, never a head slice: the JSONL is written in
    shard order, and shards are batch-ordered, so the first N studies are not a random sample.
    """
    import numpy as np

    wanted = set(splits) if splits else None
    per_split: dict[str, list] = {}
    with open(jsonl_path) as handle:
        for line in handle:
            row = json.loads(line)
            if wanted and row["split"] not in wanted:
                continue
            per_split.setdefault(row["split"], []).append(row)

    bucket_index = read_bucket_index(manifest_csv) if manifest_csv else {}
    records = []
    for split, rows in sorted(per_split.items()):
        if limit_per_split and len(rows) > limit_per_split:
            order = np.random.default_rng(seed).permutation(len(rows))[:limit_per_split]
            rows = [rows[i] for i in sorted(order)]
        for row in rows:
            records.append(CorpusRecord(
                study_uid=row["study_uid"], split=split, raw=row.get("raw"),
                clinical_information=row.get("clinical_information"),
                technique=row.get("technique"), findings=row.get("findings"),
                impression=row.get("impression"), labels=row.get("labels") or {},
                buckets=tuple(bucket_index.get(row["study_uid"], ())),
            ))
    return records


__all__ = ["CorpusRecord", "SPLIT_DIRECTORIES", "build_corpus", "load_corpus", "read_bucket_index"]
