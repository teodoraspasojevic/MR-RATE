"""Which modalities the challenge scores. Ported from the official `modality_filter.py`.

MR-RATE ground-truth filenames are `{study_uid}_{modality}-raw-{plane}.nii.gz`, e.g.
`UWCMTFCZ47_t1w-raw-axi.nii.gz`. Organizer decision: only T1w/T2w/FLAIR/SWI are scored; MRA/DWI/ADC/
etc. are excluded from scoring (not from the released data, only from this evaluation).
"""
from __future__ import annotations

#: Lower-cased, as the official code keeps them.
ALLOWED_MODALITIES = frozenset({"t1w", "t2w", "flair", "swi"})


def extract_modality(input_image_name: str) -> str:
    """'UWCMTFCZ47_t1w-raw-axi' -> 't1w'. Format: `{study_uid}_{modality}-raw-{plane}`.

    `study_uid` never contains '_', so the first '_' safely separates it from `sequence_name`.
    """
    name = input_image_name.strip()
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        name = name[: -len(".nii")]

    if "_" not in name:
        raise ValueError(f"unexpected filename format (no underscore): {input_image_name!r}")

    _study_uid, sequence_name = name.split("_", 1)
    if "-" not in sequence_name:
        raise ValueError(f"unexpected sequence_name format (no hyphen): {sequence_name!r} "
                         f"(from {input_image_name!r})")
    return sequence_name.split("-", 1)[0].lower()


def is_scored_modality(input_image_name: str) -> bool:
    """Should this entry be included in scoring?"""
    try:
        modality = extract_modality(input_image_name)
    except ValueError:
        # An unparseable name is excluded rather than silently swallowed -- the caller logs it.
        return False
    return modality in ALLOWED_MODALITIES
