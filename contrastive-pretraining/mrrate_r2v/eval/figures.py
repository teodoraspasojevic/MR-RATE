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


#: Slices encoded per plane per source in a validation panel. 12 x 3 planes x 2 sources = 72 tiles,
#: which lands at roughly 250-500 KB of PNG per case -- small enough to log every validation step
#: without the run becoming an artifact store. Raise it and check the upload size.
PANEL_SLICES_PER_PLANE = 12


def _plane_stack(volume: np.ndarray, axis: int, n_slices: int):
    """`n_slices` evenly-spaced slices along `axis`, display-oriented like `orthogonal_slices`.

    Indices are returned too, because ground truth and prediction must be sampled at *the same*
    indices for the comparison to mean anything -- the caller computes them once from the ground
    truth and passes them back in for the prediction.
    """
    size = volume.shape[axis]
    indices = np.linspace(0, size - 1, min(n_slices, size)).round().astype(int)
    out = []
    for index in indices:
        if axis == 0:
            out.append(np.flipud(volume[index, :, :].T))
        elif axis == 1:
            out.append(np.flipud(volume[:, index, :].T))
        else:
            out.append(np.flipud(volume[:, :, index].T))
    return out, indices


def _png_data_uri(tiles, vmin: float, vmax: float) -> str:
    """Stack `tiles` vertically into one PNG, returned as a `data:` URI.

    One strip per (plane, source) rather than one PNG per slice: an HTML panel with 72 separate
    data URIs is mostly base64 overhead, and a single strip lets the slider be a CSS offset with no
    per-slice decode.
    """
    import base64
    import io

    from PIL import Image

    height = max(t.shape[0] for t in tiles)
    width = max(t.shape[1] for t in tiles)
    canvas = np.zeros((height * len(tiles), width), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        as_uint8 = _to_uint8(tile, vmin, vmax)
        canvas[i * height:i * height + as_uint8.shape[0], :as_uint8.shape[1]] = as_uint8
    buffer = io.BytesIO()
    Image.fromarray(canvas, mode="L").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def validation_panel_html(case, generated: np.ndarray, step: int,
                          n_slices: int = PANEL_SLICES_PER_PLANE,
                          epoch: int = 0, validation_index: int = 0,
                          full: bool = False) -> str:
    """A self-contained interactive ground-truth vs generated panel for one validation case.

    Returns HTML with no external requests -- every image is an inline `data:` URI -- so it renders
    inside a `wandb.Html` panel offline as well as online.

    Guarantees the comparison depends on:

    - **Matched slice indices.** Both sources are sampled at the indices computed from the ground
      truth, so the slider moves through the same anatomy on both sides.
    - **One shared intensity window**, taken from the *ground truth* (`_window`), never
      per-source auto-scaling. A degenerate prediction therefore looks degenerate instead of being
      rescaled into looking plausible -- the same reason the evaluator's foreground mask always
      comes from the ground truth.
    - **Undistorted aspect ratio.** Each plane's tile is given the physical aspect implied by
      `case.spacing_xyz`, so a 0.5 x 0.5 x 5 mm acquisition is not silently drawn as isotropic.
    - **No identifiers.** `case.case_id` is a hash; `study_key`/`series_key` never reach here.

    Report text *is* included, so this panel must not be logged to a public W&B project -- the
    caller gates that (`--wandb-log-reports`).

    `validation_index`, `step` and `epoch` are rendered in the panel heading and as fields, so a
    panel is self-describing: which validation produced it, at which optimizer step, in which epoch,
    and whether the pass was quick or full. `validation_index` is not redundant with `step` -- two
    passes can share a step (an interval pass and the end-of-training pass both fire at the last
    one), and only the index separates them.
    """
    vmin, vmax = _window(np.asarray(case.target, dtype=np.float32))
    spacing = tuple(float(s) for s in (case.spacing_xyz or (1.0, 1.0, 1.0)))
    # (axis, in-plane pixel spacing as (horizontal, vertical)) for sagittal/coronal/axial.
    plane_spec = [
        ("sagittal", 0, (spacing[1], spacing[2])),
        ("coronal", 1, (spacing[0], spacing[2])),
        ("axial", 2, (spacing[0], spacing[1])),
    ]
    panels, count = [], n_slices
    for name, axis, (sx, sy) in plane_spec:
        gt_tiles, indices = _plane_stack(np.asarray(case.target, dtype=np.float32), axis, n_slices)
        gen_tiles, _ = _plane_stack(np.asarray(generated, dtype=np.float32), axis, n_slices)
        gen_tiles = gen_tiles[: len(gt_tiles)]
        if len(gen_tiles) < len(gt_tiles):     # shape mismatch: pad rather than resize
            gen_tiles += [np.zeros_like(gt_tiles[0])] * (len(gt_tiles) - len(gen_tiles))
        count = min(count, len(gt_tiles))
        height, width = gt_tiles[0].shape
        panels.append({
            "name": name,
            "n": len(gt_tiles),
            "indices": [int(i) for i in indices],
            "w": int(width), "h": int(height),
            "aspect": float((width * sx) / max(height * sy, 1e-6)),
            "gt": _png_data_uri(gt_tiles, vmin, vmax),
            "gen": _png_data_uri(gen_tiles, vmin, vmax),
        })

    import html as _html
    import json as _json

    def field(label, value):
        return (f'<div class="f"><span class="k">{_html.escape(label)}</span>'
                f'<span class="v">{_html.escape(str(value or "-"))}</span></div>')

    sections = case.report_sections or {}
    meta = "".join([
        field("validation #", validation_index),
        field("optimizer step", step),
        field("epoch", epoch),
        field("pass", "full" if full else "quick"),
        field("case", case.case_id),
        field("modality", case.modality),
        field("plane", case.plane),
        field("shape (X,Y,Z)", "x".join(str(s) for s in case.shape_xyz)),
        field("spacing mm", " x ".join(f"{s:.3f}" for s in spacing)),
        field("window", f"[{vmin:.4g}, {vmax:.4g}] (from ground truth)"),
    ])
    reports = "".join(
        f'<div class="sec"><div class="secname">{_html.escape(name.upper())}</div>'
        f'<div class="sectext">{_html.escape(text) or "<i>absent</i>"}</div></div>'
        for name, text in sections.items()
    ) or f'<div class="sec"><div class="sectext">{_html.escape(case.report_text[:4000])}</div></div>'

    heading = (f"validation #{validation_index} &middot; optimizer step {step} &middot; "
               f"epoch {epoch} &middot; {'full' if full else 'quick'} pass &middot; "
               f"{_html.escape(case.modality)} {_html.escape(case.plane)}")
    return _PANEL_TEMPLATE.replace("__PANELS__", _json.dumps(panels)) \
                          .replace("__COUNT__", str(max(count, 1))) \
                          .replace("__HEADING__", heading) \
                          .replace("__META__", meta) \
                          .replace("__REPORTS__", reports)


#: Kept as one string rather than assembled per call: the markup is fixed, only the payload varies.
#: `image-rendering: pixelated` matters -- browser smoothing of a 12-slice strip invents texture
#: that is not in the volume, which is exactly what you must not do when judging image quality.
_PANEL_TEMPLATE = """
<style>
 .r2v{font:13px/1.45 ui-sans-serif,system-ui,sans-serif;color:#e8e8ea;background:#16161a;
      padding:14px;border-radius:8px}
 .r2v .grid{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}
 .r2v .col{flex:1 1 200px;min-width:180px}
 .r2v .ttl{font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em;
           color:#9aa0a6;margin-bottom:5px}
 .r2v .pair{display:flex;gap:6px}
 .r2v .box{flex:1;position:relative;overflow:hidden;background:#000;border:1px solid #2a2a30;
           border-radius:4px}
 .r2v .box img{position:absolute;left:0;top:0;width:100%;image-rendering:pixelated;
               transform-origin:top left}
 .r2v .tag{position:absolute;left:3px;top:2px;z-index:2;font-size:10px;padding:1px 4px;
           border-radius:3px;background:#000a;color:#fff}
 .r2v .f{display:flex;gap:8px;font-size:12px}
 .r2v .k{color:#9aa0a6;min-width:110px}
 .r2v .v{color:#e8e8ea;font-variant-numeric:tabular-nums}
 .r2v .sec{margin-top:8px}
 .r2v .secname{font-size:10px;letter-spacing:.08em;color:#7cc4ff}
 .r2v .sectext{white-space:pre-wrap;font-size:12px;color:#cfd2d6;max-height:150px;overflow:auto}
 .r2v input[type=range]{width:100%;margin-top:6px}
 .r2v .bar{display:flex;align-items:center;gap:10px;margin-top:8px}
 .r2v .hdr{font-weight:600;font-size:13px;color:#7cc4ff;letter-spacing:.02em;margin-bottom:2px}
</style>
<div class="r2v">
  <div class="hdr">__HEADING__</div>
  <div class="grid" id="planes"></div>
  <div class="bar">
    <span class="k" style="min-width:auto">slice</span>
    <input type="range" id="sl" min="0" max="__COUNT__" value="0">
    <span class="v" id="slv" style="min-width:70px"></span>
  </div>
  <div style="margin-top:10px">__META__</div>
  <div style="margin-top:8px">__REPORTS__</div>
</div>
<script>
(function(){
  var panels = __PANELS__, count = __COUNT__;
  var host = document.getElementById('planes');
  panels.forEach(function(p, pi){
    var col = document.createElement('div'); col.className = 'col';
    col.innerHTML = '<div class="ttl">' + p.name + '</div><div class="pair">' +
      ['gt','gen'].map(function(kind){
        // Each box is one viewport onto a vertical strip; the slider translates the strip.
        return '<div class="box" data-kind="' + kind + '" data-pi="' + pi + '"' +
               ' style="aspect-ratio:' + p.aspect.toFixed(4) + '">' +
               '<span class="tag">' + (kind === 'gt' ? 'ground truth' : 'generated') + '</span>' +
               '<img src="' + p[kind] + '" style="height:' + (p.n * 100) + '%">' +
               '</div>';
      }).join('') + '</div>';
    host.appendChild(col);
  });
  var slider = document.getElementById('sl'), label = document.getElementById('slv');
  slider.max = String(count - 1);
  function draw(){
    var i = parseInt(slider.value, 10);
    host.querySelectorAll('.box').forEach(function(box){
      var p = panels[parseInt(box.dataset.pi, 10)];
      var j = Math.min(i, p.n - 1);
      box.querySelector('img').style.top = '-' + (j * 100) + '%';
    });
    var p0 = panels[0];
    label.textContent = (i + 1) + '/' + count + (p0 ? ' (idx ' + p0.indices[Math.min(i, p0.n - 1)] + ')' : '');
  }
  slider.addEventListener('input', draw);
  draw();
})();
</script>
"""


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
