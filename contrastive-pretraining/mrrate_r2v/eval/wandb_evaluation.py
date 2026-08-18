"""The one shim `eval/live.py` needs to render an example-volume panel with `figures.py`'s
training-time renderer, kept separate so `eval/live.py` doesn't need to know `figures.py`'s
attribute contract.
"""
from __future__ import annotations


class _PanelCase:
    """The attribute surface `figures.validation_panel_html` expects, backed by a `LiveCase`.

    A shim rather than a refactor of the panel renderer: the renderer is shared with training and
    its inputs are already right, so the cheap and non-breaking move is to present an evaluation
    case in the shape it already accepts. `case_id` is a hash and `study_key`/`series_key` are not
    carried, so a panel still contains no identifier.
    """

    def __init__(self, case, target, report_text: str, report_sections):
        self.index = 0
        self.case_id = case.case_id
        self.report_text = report_text or ""
        self.report_sections = report_sections or {}
        self.modality = case.sequence
        self.plane = case.acquisition_plane
        self.shape_xyz = tuple(case.shape)
        self.spacing_xyz = tuple(case.spacing_mm)
        self.study_hash = ""
        self.target = target
