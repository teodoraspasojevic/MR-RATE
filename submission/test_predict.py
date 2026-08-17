"""Tests for predict.py's filename parsing and prompt/report helpers. Stdlib-only apart
from numpy (predict.py imports it at module level), so `python -m pytest
submission/test_predict.py -v` runs with no GPU, checkpoints, or conda env beyond that.
"""
from __future__ import annotations

import json

import pytest

from predict import (
    case_stem,
    checkpoint_paths,
    load_done,
    mark_done,
    modality_plane_for,
    read_prompts,
    split_sections,
)


# ─────────────────────────── modality_plane_for ───────────────────────────

DOC_EXAMPLES = [
    ("WNPYIQCPIN_flair-raw-sag", ("FLAIR", "SAGITTAL")),
    ("WNPYIQCPIN_swi-raw-axi", ("SWI", "AXIAL")),
    ("WNPYIQCPIN_t1w-raw-sag-2", ("T1w", "SAGITTAL")),
    ("WNPYIQCPIN_t1w-raw-sag", ("T1w", "SAGITTAL")),
    ("WNPYIQCPIN_t2w-raw-axi", ("T2w", "AXIAL")),
    ("AKXQOSLHTM_t2w-raw-obl", ("T2w", "OBLIQUE")),
]


@pytest.mark.parametrize("case_id, expected", DOC_EXAMPLES)
def test_parses_documented_examples(case_id, expected):
    assert modality_plane_for(case_id) == expected


@pytest.mark.parametrize("case_id, _expected", DOC_EXAMPLES)
def test_output_filename_matches_input_image_name(case_id, _expected):
    """The evaluator pairs by exact filename, so `case_stem` must round-trip an
    extensionless id (as in every published example) unchanged."""
    assert case_stem(case_id) == case_id


def test_case_insensitive():
    assert modality_plane_for("STUDY_T1W-RAW-SAG") == ("T1w", "SAGITTAL")
    assert modality_plane_for("STUDY_Flair-Raw-Cor") == ("FLAIR", "CORONAL")


def test_duplicate_series_suffix_ignored():
    assert modality_plane_for("STUDY_t1w-raw-sag-1") == ("T1w", "SAGITTAL")
    assert modality_plane_for("STUDY_t1w-raw-sag-17") == ("T1w", "SAGITTAL")


def test_study_uid_with_underscores_does_not_confuse_split():
    assert modality_plane_for("STUDY_WITH_UNDERSCORES_t1w-raw-sag") == ("T1w", "SAGITTAL")


def test_untrained_modality_still_parses():
    """MRA and SWI+SAGITTAL are real codes the adapter wasn't trained on, but the id is
    still the truth about the reference volume -- parsing must not coerce it."""
    assert modality_plane_for("STUDY_mra-raw-axi") == ("MRA", "AXIAL")
    assert modality_plane_for("STUDY_swi-raw-sag") == ("SWI", "SAGITTAL")


@pytest.mark.parametrize("case_id", [
    "case_001.mha",
    "case_001",
    "WNPYIQCPIN",
    "WNPYIQCPIN_t1w",
    "WNPYIQCPIN-raw-sag",
    "STUDY_xyz-raw-sag",   # unknown modality code
    "STUDY_t1w-raw-xyz",   # unknown plane code
])
def test_unparseable_ids_raise(case_id):
    with pytest.raises(ValueError):
        modality_plane_for(case_id)


def test_extension_is_stripped_before_parsing():
    stem = case_stem("WNPYIQCPIN_flair-raw-sag.nii.gz")
    assert stem == "WNPYIQCPIN_flair-raw-sag"
    assert modality_plane_for(stem) == ("FLAIR", "SAGITTAL")


# ─────────────────────────── read_prompts ───────────────────────────

def test_read_prompts_reads_array_and_strips_extension(tmp_path):
    (tmp_path / "prompts.json").write_text(json.dumps([
        {"input_image_name": "A_t1w-raw-sag.mha", "report": "Findings: normal."},
        {"input_image_name": "A_t1w-raw-sag-2", "report": "Findings: normal."},
    ]))
    prompts = read_prompts(tmp_path)
    assert prompts == [
        ("A_t1w-raw-sag", "Findings: normal."),
        ("A_t1w-raw-sag-2", "Findings: normal."),
    ]


def test_read_prompts_rejects_duplicate_case_ids(tmp_path):
    (tmp_path / "prompts.json").write_text(json.dumps([
        {"input_image_name": "A_t1w-raw-sag", "report": "x"},
        {"input_image_name": "A_t1w-raw-sag.mha", "report": "y"},  # same stem once stripped
    ]))
    with pytest.raises(SystemExit):
        read_prompts(tmp_path)


def test_read_prompts_requires_a_json_file(tmp_path):
    with pytest.raises(SystemExit):
        read_prompts(tmp_path)


# ─────────────────────────── split_sections ───────────────────────────

def test_split_sections_recovers_findings_and_impression():
    report = "44-year-old female: Findings: Normal brain. Impression: No acute pathology."
    sections = split_sections(report)
    assert "Normal brain." in sections["findings"]
    assert "No acute pathology." in sections["impression"]
    assert "44-year-old female" in sections["clinical_information"]


def test_split_sections_falls_back_to_findings_when_unsectioned():
    assert split_sections("Just some free text.") == {"findings": "Just some free text."}


def test_split_sections_empty_report():
    assert split_sections("") == {}
    assert split_sections(None) == {}


# ─────────────────────────── checkpoint helpers ───────────────────────────

def test_checkpoint_paths_unscoped_for_single_process():
    done_file, backup_dir = checkpoint_paths(rank=0, world_size=1)
    assert done_file.name == "done.json"
    assert backup_dir.name == "outputs"


def test_checkpoint_paths_rank_scoped_under_ddp():
    done_file, _ = checkpoint_paths(rank=1, world_size=2)
    assert done_file.name == "done_rank1.json"


def test_mark_done_and_load_done_round_trip(tmp_path, monkeypatch):
    import predict

    output_dir, checkpoint_dir = tmp_path / "output", tmp_path / "ckpt"
    backup_dir, done_file = checkpoint_dir / "outputs", checkpoint_dir / "done.json"
    output_dir.mkdir()
    monkeypatch.setattr(predict, "OUTPUT_DIR", output_dir)

    (output_dir / "case1.nii.gz").write_bytes(b"volume-bytes")
    done = []
    mark_done("case1.nii.gz", done, done_file, backup_dir)
    assert done == ["case1.nii.gz"]
    assert (backup_dir / "case1.nii.gz").exists()

    # Simulate /output being wiped by a restart: only the backup survives.
    (output_dir / "case1.nii.gz").unlink()
    restored = load_done(done_file, backup_dir)
    assert restored == ["case1.nii.gz"]
    assert (output_dir / "case1.nii.gz").exists()  # copied back from the backup


def test_load_done_drops_entries_missing_from_both_locations(tmp_path, monkeypatch):
    import predict

    output_dir, checkpoint_dir = tmp_path / "output", tmp_path / "ckpt"
    backup_dir, done_file = checkpoint_dir / "outputs", checkpoint_dir / "done.json"
    output_dir.mkdir()
    checkpoint_dir.mkdir()
    monkeypatch.setattr(predict, "OUTPUT_DIR", output_dir)

    done_file.write_text(json.dumps(["ghost.nii.gz"]))
    assert load_done(done_file, backup_dir) == []  # neither /output nor the backup has it


# ─────────────────────────── ddp_setup ───────────────────────────

def test_ddp_setup_is_noop_without_torchrun_env(monkeypatch):
    import predict

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert predict.ddp_setup() == (0, 1, 0, False)
    predict.ddp_cleanup(False)  # must not touch torch.distributed when not DDP
