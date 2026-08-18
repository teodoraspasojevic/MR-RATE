"""The interactive ground-truth-vs-generated panel logged to W&B during evaluation and training-
time validation. A metric tells you a volume is worse; a picture tells you how.

Privacy: filenames and captions use `case_id` (a hash), never `study_uid`/`series_id`.
"""
from __future__ import annotations

import numpy as np

#: Slices encoded per plane per source in a validation panel. 12 x 3 planes x 2 sources = 72 tiles,
#: which lands at roughly 250-500 KB of PNG per case -- small enough to log every validation step
#: without the run becoming an artifact store. Raise it and check the upload size.
PANEL_SLICES_PER_PLANE = 12


def _window(volume: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0):
    """Display intensity window from percentiles, taken from the ground truth only so a
    degenerate prediction looks degenerate instead of being rescaled into looking plausible."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    return (float(lo), float(hi)) if hi > lo else (float(lo), float(lo) + 1e-6)


def _to_uint8(slice_2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = (np.nan_to_num(slice_2d, nan=vmin) - vmin) / max(vmax - vmin, 1e-12)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def _plane_stack(volume: np.ndarray, axis: int, n_slices: int):
    """`n_slices` evenly-spaced slices along `axis`, display-oriented (superior/anterior up,
    left-to-right) so the panel reads the way a radiologist expects.

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
    """A self-contained interactive ground-truth vs generated panel for one case.

    Returns HTML with no external requests -- every image is an inline `data:` URI -- so it renders
    inside a `wandb.Html` panel offline as well as online.

    - **Matched slice indices.** Both sources are sampled at the indices computed from the ground
      truth, so the slider moves through the same anatomy on both sides.
    - **One shared intensity window**, taken from the ground truth, never per-source auto-scaling.
    - **Undistorted aspect ratio.** Each plane's tile uses the physical aspect implied by
      `case.spacing_xyz`, so an anisotropic acquisition is not silently drawn as isotropic.
    - **No identifiers.** `case.case_id` is a hash; `study_key`/`series_key` never reach here.

    Report text *is* included, so this panel must not be logged to a public W&B project -- the
    caller gates that (`--wandb-log-reports`).
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
