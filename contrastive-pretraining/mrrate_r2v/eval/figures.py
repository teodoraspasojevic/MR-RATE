"""Example figures saved alongside the metrics, so you can *look* at what a number means.

A metric tells you a volume is worse; a picture tells you how. Every evaluation writes a handful
of orthogonal-slice montages into `<results>/figures/`.

    paired tasks   rows = ground truth / prediction / |difference|,  columns = sagittal / coronal / axial
    generation     rows = generated / an unpaired real reference,    same three columns

Which cases get rendered is deterministic and diagnostic rather than arbitrary: cases are ranked
by the primary metric and sampled at evenly-spaced ranks, so with `--save-figures 3` you get the
worst, the median, and the best. Failure modes show up in the worst one, which is the point.

Rendered with PIL rather than matplotlib: PIL is present in both the container and the login-node
test environment, so this code is actually covered by tests, and the evaluator picks up no new
heavy dependency. The cost is no colorbar -- intensity windows are printed in the labels instead.

Privacy: filenames and captions use `case_id` (a hash), never `study_uid`/`series_id`, and report
text is never drawn into an image.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("mrrate_r2v.eval")

FIGURES_DIR = "figures"
PLANE_NAMES = ("sagittal", "coronal", "axial")

_PAD = 4          # px between tiles
_LABEL_H = 14     # px reserved for each row's own label strip
# Title, subtitle, then the column headings -- three stacked 14 px lines. The column headings live
# in the header, NOT in the first row's label strip, or they collide with that row's label.
_HEADER_H = 46


def orthogonal_slices(volume: np.ndarray):
    """Mid-slices through an (X, Y, Z) = (R, A, S) volume, as display-oriented 2D arrays.

    Returns (sagittal, coronal, axial). Each is transposed and flipped so that superior is up and
    the in-plane axis reads left-to-right -- i.e. what a radiologist expects to see, not the raw
    array layout.
    """
    xm, ym, zm = (s // 2 for s in volume.shape[:3])
    sagittal = np.flipud(volume[xm, :, :].T)   # (Y=A, Z=S) -> (S, A), S up
    coronal = np.flipud(volume[:, ym, :].T)    # (X=R, Z=S) -> (S, R), S up
    axial = np.flipud(volume[:, :, zm].T)      # (X=R, Y=A) -> (A, R), A up
    return sagittal, coronal, axial


def _window(volume: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0):
    """Display intensity window from percentiles. Shared between ground truth and prediction so
    the two rows are visually comparable rather than each independently auto-scaled."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    return (float(lo), float(hi)) if hi > lo else (float(lo), float(lo) + 1e-6)


def _to_uint8(slice_2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = (np.nan_to_num(slice_2d, nan=vmin) - vmin) / max(vmax - vmin, 1e-12)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def _montage(rows, row_labels, column_labels, title, subtitle=""):
    """Tile `rows` (each a list of uint8 2D arrays) into one labelled PIL image."""
    from PIL import Image, ImageDraw

    tile_h = max(a.shape[0] for row in rows for a in row)
    tile_w = max(a.shape[1] for row in rows for a in row)
    n_rows, n_cols = len(rows), max(len(r) for r in rows)

    width = n_cols * tile_w + (n_cols + 1) * _PAD
    height = _HEADER_H + n_rows * (tile_h + _LABEL_H) + (n_rows + 1) * _PAD
    canvas = Image.new("L", (width, height), color=0)

    for r, row in enumerate(rows):
        y = _HEADER_H + _PAD + r * (tile_h + _LABEL_H + _PAD) + _LABEL_H
        for c, arr in enumerate(row):
            x = _PAD + c * (tile_w + _PAD)
            # centre tiles that are smaller than the largest (anisotropic shapes)
            canvas.paste(Image.fromarray(arr),
                         (x + (tile_w - arr.shape[1]) // 2, y + (tile_h - arr.shape[0]) // 2))

    img = canvas.convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((_PAD, 2), title, fill=(255, 255, 120))
    if subtitle:
        draw.text((_PAD, 16), subtitle, fill=(160, 200, 255))
    for c, label in enumerate(column_labels[:n_cols]):
        draw.text((_PAD + c * (tile_w + _PAD) + 2, 32), label, fill=(180, 180, 180))
    for r, label in enumerate(row_labels):
        y = _HEADER_H + _PAD + r * (tile_h + _LABEL_H + _PAD)
        draw.text((_PAD + 2, y + 1), label, fill=(120, 255, 160))
    return img


def save_paired_figure(gt: np.ndarray, pred: np.ndarray, out_path, *, case_id: str,
                       sequence: str, plane: str, caption: str = "") -> None:
    """Ground truth / prediction / |difference| across three planes.

    Ground truth and prediction share one intensity window so the rows are comparable. The
    difference row is scaled to its own maximum, which is printed -- an auto-scaled difference
    image with no stated range is easy to misread as worse (or better) than it is.
    """
    vmin, vmax = _window(gt)
    diff = np.abs(gt.astype(np.float64) - pred.astype(np.float64))
    dmax = float(diff.max()) if diff.size else 0.0

    rows = [
        [_to_uint8(s, vmin, vmax) for s in orthogonal_slices(gt)],
        [_to_uint8(s, vmin, vmax) for s in orthogonal_slices(pred)],
        [_to_uint8(s, 0.0, max(dmax, 1e-12)) for s in orthogonal_slices(diff)],
    ]
    img = _montage(
        rows,
        row_labels=["ground truth", "prediction", f"|diff| 0-{dmax:.3f}"],
        column_labels=PLANE_NAMES,
        title=f"{sequence} {plane}  case {case_id}",
        subtitle=f"window {vmin:.3f}-{vmax:.3f}   {caption}".strip(),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_unpaired_figure(generated: np.ndarray, reference: np.ndarray | None, out_path, *,
                         prediction_id: str, sequence: str) -> None:
    """A generated volume, with a real volume beneath it for visual scale.

    The reference is explicitly captioned as unpaired: it is not this generated volume's
    counterpart and no voxelwise comparison between the two rows is meaningful.
    """
    vmin, vmax = _window(generated)
    rows = [[_to_uint8(s, vmin, vmax) for s in orthogonal_slices(generated)]]
    row_labels = ["generated"]
    if reference is not None:
        rmin, rmax = _window(reference)
        rows.append([_to_uint8(s, rmin, rmax) for s in orthogonal_slices(reference)])
        row_labels.append("real (unpaired)")

    img = _montage(
        rows, row_labels=row_labels, column_labels=PLANE_NAMES,
        title=f"{sequence} generated  {prediction_id}",
        subtitle="rows are NOT counterparts -- unconditional generation has no ground truth",
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def select_ranked(rows, metric: str, n: int) -> list:
    """`n` rows sampled at evenly-spaced ranks of `metric`, worst first.

    n=1 -> median. n=2 -> worst, best. n=3 -> worst, ~median, best. Larger n fills in between.
    For an even number of candidates the interior ranks land on the lower of the two middle
    positions -- arbitrary, but deterministic. Ties break on `case_id`, so the same run always
    renders the same cases.
    Rows missing a finite value for `metric` are excluded from ranking (never silently ranked
    as if the metric were 0).
    """
    usable = [r for r in rows
              if isinstance(r.get(metric), (int, float)) and np.isfinite(r[metric])]
    if not usable or n <= 0:
        return []
    usable.sort(key=lambda r: (r[metric], r.get("case_id", "")))
    if n >= len(usable):
        return usable
    if n == 1:
        return [usable[len(usable) // 2]]
    positions = np.linspace(0, len(usable) - 1, n).round().astype(int)
    return [usable[p] for p in dict.fromkeys(positions.tolist())]


def save_example_nifti(volume: np.ndarray, spacing_mm, out_path) -> None:
    """One volume as .nii.gz, for inspection in a real viewer.

    The affine is synthesized from spacing alone (a cohort stores bare arrays, no affine), so it
    carries orientation and voxel size but no true patient-space origin. Fine for looking at;
    not a substitute for the original file's geometry.
    """
    import nibabel as nib

    from .geometry_contract import synthesize_diagonal_affine

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    affine = synthesize_diagonal_affine(tuple(float(s) for s in spacing_mm))
    nib.save(nib.Nifti1Image(np.ascontiguousarray(volume, dtype=np.float32), affine), str(out_path))
