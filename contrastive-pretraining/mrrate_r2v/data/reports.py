"""Where the conditioning text comes from. Stdlib only, no torch.

Three interchangeable report sources, all duck-typed the same way -- `study_uid in store`
and `store[study_uid] -> ReportRecord`. Implement that pair and the Dataset accepts your
source too.

| Source | Reads | Use it when |
|---|---|---|
| `ShardReportStore` | per-study `report.json` inside the shard tars | training from SHARDS_PATH (self-contained; preferred) |
| `StructuredReportStore` | MR-RATE's official `reports.csv` / `reports.tar.gz` | you have DATA_PATH's release CSVs |
| `SentenceJSONLReportStore` | `findings_sentences.jsonl` | fallback only -- section boundaries are lost |

`ReportRecord` keeps the five released sections separate, so the caller picks which to
condition on via `R2VDatasetConfig.report_sections` instead of getting a pre-flattened blob.
"""
import json
from dataclasses import dataclass
from typing import Optional

from .storage import ArchiveReader, Locator, iter_csv_dict_rows

REPORT_SECTION_NAMES = ("raw", "clinical_information", "technique", "findings", "impression")


@dataclass
class ReportRecord:
    """One study's report, section-separated. Any field is None if unavailable."""

    raw: Optional[str] = None
    clinical_information: Optional[str] = None
    technique: Optional[str] = None
    findings: Optional[str] = None
    impression: Optional[str] = None

    def compose(self, sections):
        """Join the requested sections in the given order, skipping empty ones.

        Fully deterministic and never truncated -- the whole selected text is returned
        every call. (Contrast `MRReportDataset`, which randomly subsamples sentences per
        `__getitem__`; that is a contrastive-training trick, wrong for generation.)
        """
        parts = []
        for name in sections:
            if name not in REPORT_SECTION_NAMES:
                raise ValueError(f"Unknown report section '{name}'. Choose from: {REPORT_SECTION_NAMES}")
            text = getattr(self, name, None)
            if text:
                parts.append(f"{name.replace('_', ' ').capitalize()}: {text.strip()}")
        return "\n".join(parts)


class StructuredReportStore:
    """MR-RATE's official structured-reports schema: a CSV (or `.tar.gz` of per-batch CSVs,
    read without extracting) with columns study_uid, report, clinical_information,
    technique, findings, impression. Loaded eagerly -- a few seconds for the full release.
    """

    def __init__(self, reports_csv):
        self.records = {}
        for row in iter_csv_dict_rows(reports_csv):
            uid = row.get("study_uid")
            if not uid:
                continue
            self.records[uid] = ReportRecord(
                raw=row.get("report") or None,
                clinical_information=row.get("clinical_information") or None,
                technique=row.get("technique") or None,
                findings=row.get("findings") or None,
                impression=row.get("impression") or None,
            )

    def __contains__(self, study_uid):
        return study_uid in self.records

    def __getitem__(self, study_uid):
        return self.records[study_uid]


class SentenceJSONLReportStore:
    """Fallback source: the pre-extracted-sentence JSONL the contrastive pipeline uses
    (one JSON object per line: volume_name, valid_json, extracted_sentences).

    This format has no section boundaries, so all sentences are joined into one string
    exposed as both `raw` and `findings`. Strictly lower fidelity than
    `StructuredReportStore` -- only use it when the structured CSV is unavailable.
    """

    def __init__(self, jsonl_file):
        self.records = {}
        with open(jsonl_file, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if not row.get("valid_json", False):
                        continue
                    sentences = row.get("extracted_sentences") or []
                    if not sentences:
                        continue
                    uid = row["volume_name"]
                except Exception:
                    continue
                joined = " ".join(sentences)
                self.records[uid] = ReportRecord(raw=joined, findings=joined)

    def __contains__(self, study_uid):
        return study_uid in self.records

    def __getitem__(self, study_uid):
        return self.records[study_uid]


class ShardReportStore:
    """Reads reports from SHARDS_PATH's own per-study `report.json` sidecars -- the same
    shard tar the study's series already live in. Preferred when training from SHARDS_PATH:
    no dependency on the DATA_PATH workspace, and verified to carry the identical 5-section
    schema and content.

    Two load-bearing design choices:

    1. **Presence is index-driven, content is not.** `__contains__` answers from a small
       index CSV (study_uid, archive_path) built from the shards' own `has_report` column,
       without opening a tar. That keeps the Dataset's upfront "N dropped: no matching
       report" filter as cheap as every other store's.
    2. **Content is read lazily, once per study, then cached.** Eagerly reading report.json
       for ~5,000 studies measured ~54s (~91 studies/s), so a 90,000-study train split
       would cost 15+ minutes of Dataset construction. Lazy reads amortize that over the
       first epoch instead.

    Member path is always `{study_uid}/report.json` (verified against real shard tars), so
    it is derived rather than stored in the index.
    """

    def __init__(self, report_index_csv, archive_reader=None):
        self._locators = {}
        for row in iter_csv_dict_rows(report_index_csv):
            study_uid = row.get("study_uid")
            archive_path = row.get("archive_path")
            if not study_uid or not archive_path:
                continue
            self._locators[study_uid] = Locator(
                kind="archive", archive_path=archive_path,
                member_chain=(f"{study_uid}/report.json",),
            )
        self._archive_reader = archive_reader or ArchiveReader()
        self._cache = {}

    def __contains__(self, study_uid):
        return study_uid in self._locators

    def __getitem__(self, study_uid):
        if study_uid in self._cache:
            return self._cache[study_uid]
        raw = self._archive_reader.read_bytes(self._locators[study_uid])
        obj = json.loads(raw)
        record = ReportRecord(
            raw=obj.get("report") or None,
            clinical_information=obj.get("clinical_information") or None,
            technique=obj.get("technique") or None,
            findings=obj.get("findings") or None,
            impression=obj.get("impression") or None,
        )
        self._cache[study_uid] = record
        return record
