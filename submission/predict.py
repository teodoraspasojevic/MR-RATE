#!/usr/bin/env python3
"""VLM3D `mr-volume-generation` submission entry point: `/input/prompts.json` -> `/output/*.nii.gz`.

One report in, one synthetic 3D brain MRI volume out, for every prompt. Runs offline on the
platform's GPU with no arguments -- everything is read from the container layout and a handful
of `R2V_*` environment variables (see README.md).

**This file deliberately contains no model code.** It reuses
`mrrate_r2v.cli.generate_r2v.build_sampler` and `.conditioning_text_for` unchanged, so the
container loads checkpoints, rebuilds the text encoder, and composes conditioning text through
exactly the code path `cli.evaluate` was validated with. A submission that scored differently from
the local run because the container reimplemented the loading logic is the specific failure this
avoids.

What is genuinely new here, and lives here rather than in the package:

* the platform's I/O contract (`prompts.json` in, loose `.nii.gz` out, `/checkpoint` resume,
  SIGTERM within 30 s) -- see the organizers' `VLM3D-Dockers` README §6/§9;
* choosing a (modality, plane) for a prompt that carries none (`routing.py`);
* recovering `findings`/`impression` from one flat prompt string, which adapter D needs.

Fail-loud by design: the per-prompt loop has no try/except. The organizers' checklist asks for a
non-zero exit on failure, and a silently skipped prompt scores as a missing output -- the worst
possible outcome under the published ranking ("Missing outputs are treated as invalid and receive
the lowest possible score").
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("predict")

# ─────────────────────────── container layout ───────────────────────────
# Paths the platform mounts. `FORITHMUS_*` are exported by the platform's entrypoint
# trampoline and are authoritative when present; the literals are the documented defaults
# and what a local `docker run -v ...` test uses.
INPUT_DIR = Path(os.environ.get("FORITHMUS_INPUT", "/input"))
OUTPUT_DIR = Path(os.environ.get("FORITHMUS_OUTPUT", "/output"))
CHECKPOINT_DIR = Path(os.environ.get("FORITHMUS_CHECKPOINT", "/checkpoint"))
WEIGHTS_DIR = Path(os.environ.get("FORITHMUS_WEIGHTS", "/weights"))
MODELS_DIR = Path(os.environ.get("R2V_MODELS_DIR", "/opt/app/models"))

CHECKPOINT_FILE = CHECKPOINT_DIR / "progress.json"
OUTPUT_BACKUP = CHECKPOINT_DIR / "outputs"
RUN_MANIFEST = CHECKPOINT_DIR / "run_manifest.json"

# Volume suffixes a case id may arrive wearing. The CT track's `input_image_name` is an
# `.mha` filename; `Path.stem` would leave `.nii` on a `.nii.gz`, hence the explicit list.
VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".npy", ".npz")


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def env_int(name: str, default: int) -> int:
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env_str(name, str(default)))


def env_flag(name: str, default: bool) -> bool:
    return env_str(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


# ─────────────────────────── prompt parsing ───────────────────────────

def case_stem(name: str) -> str:
    """`"abc.mha"` / `"abc.nii.gz"` / `"abc"` -> `"abc"`. The output file is `<stem>.nii.gz`."""
    text = str(name).strip()
    lowered = text.lower()
    for suffix in VOLUME_SUFFIXES:
        if lowered.endswith(suffix):
            return text[: -len(suffix)]
    return text


def read_prompts(input_dir: Path) -> list[tuple[str, str]]:
    """`[(case_stem, report_text)]` from `/input/prompts.json`.

    Any `*.json` under `/input` is accepted, not just the documented `prompts.json`, because
    the MR phase's `data_schema` is still empty and the filename is the least certain part of
    the contract. Both the CT track's array-of-objects layout and a `{"prompts": [...]}`
    wrapper are read; the report key is looked up over the plausible spellings rather than
    assumed to be `report`.
    """
    candidates = sorted(input_dir.rglob("*.json"))
    if not candidates:
        listing = sorted(p.name for p in input_dir.rglob("*"))[:20]
        raise SystemExit(f"no JSON prompt file under {input_dir}; found: {listing}")
    # Preference order, because the platform documents an optional host-supplied
    # `/input/metadata.json` that would otherwise sort first and be parsed as the prompt list:
    # the exact documented name, then anything named like a prompt file, then whatever is left
    # once metadata.json is set aside.
    exact = [p for p in candidates if p.name == "prompts.json"]
    promptish = [p for p in candidates if "prompt" in p.name.lower()]
    other = [p for p in candidates if p.name.lower() != "metadata.json"]
    path = (exact or promptish or other or candidates)[0]
    if not exact:
        log.warning("no prompts.json under %s; falling back to %s (candidates: %s)",
                    input_dir, path.name, [p.name for p in candidates])
    log.info("reading prompts from %s", path)

    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        for key in ("prompts", "cases", "data", "inputs"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise SystemExit(f"{path} is not a JSON array (or an object wrapping one)")

    id_keys = ("input_image_name", "case_id", "id", "name", "output_image_name")
    text_keys = ("report", "text", "prompt", "findings", "report_text")
    prompts: list[tuple[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise SystemExit(f"{path} entry {index} is {type(item).__name__}, expected an object")
        raw_id = next((item[k] for k in id_keys if item.get(k)), None)
        if raw_id is None:
            raise SystemExit(f"{path} entry {index} has no case-id field (tried {id_keys})")
        report = next((item[k] for k in text_keys if item.get(k)), "")
        prompts.append((case_stem(raw_id), str(report)))

    stems = [s for s, _ in prompts]
    duplicates = {s for s in stems if stems.count(s) > 1}
    if duplicates:
        raise SystemExit(f"duplicate case ids in {path}: {sorted(duplicates)[:5]}")
    log.info("%d prompt(s); report length min/median/max = %d/%d/%d characters",
             len(prompts), *_length_stats([r for _, r in prompts]))
    return prompts


def _length_stats(texts: list[str]) -> tuple[int, int, int]:
    if not texts:
        return (0, 0, 0)
    lengths = sorted(len(t) for t in texts)
    return (lengths[0], lengths[len(lengths) // 2], lengths[-1])


#: Section headings MR-RATE's structuring step emits, and that the CT track's sample prompt
#: shows verbatim in the delivered text ("... Findings: ... Impression: ...").
_SECTION_HEADINGS = {
    "clinical_information": r"clinical(?:\s+information|\s+history)?|history|indication",
    "technique": r"technique|protocol",
    "findings": r"findings?|description",
    "impression": r"impression|conclusion|summary|assessment",
}


def split_sections(report: str) -> dict[str, str]:
    """One flat prompt string -> `{section: text}` for a sectioned-fusion adapter (arm D).

    Adapter D encodes `findings` and `impression` on separate encoders and receives two
    context tokens; handing it a single joined string routes everything to `findings` and
    masks `impression` out, which is silent and only shows up as a worse score. So the
    headings are recovered by regex here.

    If no heading is found the whole prompt becomes `findings` and `impression` is left
    empty -- an honest representation of an unsectioned request, and the same thing the
    embedder's documented string fallback would do, but explicit.
    """
    text = str(report or "").strip()
    if not text:
        return {}
    pattern = "|".join(f"(?P<{name}>{alts})" for name, alts in _SECTION_HEADINGS.items())
    matches = list(re.finditer(rf"(?<![A-Za-z])(?:{pattern})\s*:", text, re.IGNORECASE))
    if not matches:
        return {"findings": text}
    sections: dict[str, str] = {}
    for position, match in enumerate(matches):
        name = match.lastgroup
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections[name] = f"{sections[name]} {body}".strip() if name in sections else body
    # Text before the first heading is usually the age/sex preamble the CT track ships
    # ("58-year-old male: Findings: ..."). It is clinical context, so it is prepended to
    # clinical_information rather than dropped.
    preamble = text[: matches[0].start()].strip().rstrip(":").strip()
    if preamble:
        existing = sections.get("clinical_information", "")
        sections["clinical_information"] = f"{preamble} {existing}".strip()
    return sections or {"findings": text}


# ─────────────────────────── checkpoint / resume ───────────────────────────

class Progress:
    """`/checkpoint` bookkeeping for the platform's timeout-and-continue flow.

    `/output` is cleared between runs and only `/checkpoint` persists, so a completed volume
    is copied into `/checkpoint/outputs` and copied back on resume. That doubles the disk
    cost of a run (volumes are 20-40 MB each); the alternative -- regenerating from scratch
    after a timeout -- costs GPU minutes instead, which are the scarcer resource.

    **The copy happens in `record`, as each volume is finished, never in the signal
    handler.** SIGTERM gives 30 seconds before SIGKILL, and copying a full run's worth of
    volumes does not fit in it -- a handler that tried to would lose the whole run's
    progress rather than the last case's. So the handler only rewrites the small index.
    """

    def __init__(self) -> None:
        self.done: list[str] = []
        self.shutting_down = False
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_BACKUP.mkdir(parents=True, exist_ok=True)

    def restore(self) -> None:
        if not CHECKPOINT_FILE.exists():
            return
        recorded = list(json.loads(CHECKPOINT_FILE.read_text()).get("processed", []))
        restored, lost = 0, []
        for filename in recorded:
            backup, output = OUTPUT_BACKUP / filename, OUTPUT_DIR / filename
            if output.exists():
                self.done.append(filename)
            elif backup.exists():
                shutil.copy2(backup, output)
                self.done.append(filename)
                restored += 1
            else:
                # Recorded as done but present in neither place: regenerate rather than
                # trust the index, or the run finishes "successfully" with a missing output.
                lost.append(filename)
        log.info("resuming: %d case(s) already done, %d output(s) restored to %s",
                 len(self.done), restored, OUTPUT_DIR)
        if lost:
            log.warning("%d case(s) recorded as done but absent from both /output and the "
                        "checkpoint backup; regenerating them: %s", len(lost), lost[:5])

    def record(self, filename: str) -> None:
        """Mark one finished volume as done and back it up immediately."""
        source, target = OUTPUT_DIR / filename, OUTPUT_BACKUP / filename
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
        self.done.append(filename)

    def save(self) -> None:
        """Rewrite the index. Cheap enough for the signal handler, by construction."""
        CHECKPOINT_FILE.write_text(json.dumps({"processed": self.done}, indent=2))

    def install_sigterm_handler(self) -> None:
        def handle(signum, _frame):  # noqa: ANN001
            self.shutting_down = True
            log.warning("signal %d received after %d case(s); saving index and exiting",
                        signum, len(self.done))
            self.save()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle)
        signal.signal(signal.SIGINT, handle)


# ─────────────────────────── model ───────────────────────────

def resolve_weight(name_env: str, default_name: str) -> Path:
    """A weights file, looked up in `/opt/app/models` then `/weights`.

    `entrypoint.sh` symlinks `/weights/*` into `/opt/app/models`, so the first location
    normally wins; the second is the fallback for a `docker run` that mounts `/weights`
    directly without the entrypoint.
    """
    name = env_str(name_env, default_name)
    if Path(name).is_absolute():
        candidates = [Path(name)]
    else:
        candidates = [MODELS_DIR / name, WEIGHTS_DIR / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"{name_env}={name!r} not found. Looked in: {[str(c) for c in candidates]}. "
        f"/weights contains: {sorted(p.name for p in WEIGHTS_DIR.glob('*'))[:20]}"
    )


def build(seed: int):
    """`(sampler, embedder, payload)` via the package's own loader -- see the module docstring."""
    from mrrate_r2v.cli.generate_r2v import build_sampler

    args = SimpleNamespace(
        base_checkpoint=resolve_weight("R2V_BASE_CHECKPOINT", "diff_unet_3d_rflow-mr-brain_v0.pt"),
        vae_checkpoint=resolve_weight("R2V_VAE_CHECKPOINT", "autoencoder_v1.pt"),
        adapter=resolve_weight("R2V_ADAPTER", "adapter.pt"),
        network_config=None,
        # Both None: the adapter checkpoint names its own text encoder, and the encoder
        # directories are resolved from MRRATE_PRETRAINED_DIR. Setting --text-checkpoint here
        # would apply one path to whichever encoder the configuration names, which is how a
        # CXR-BERT adapter ends up running on RadBERT weights at equal width.
        text_encoder=None,
        text_checkpoint=None,
        max_report_tokens=env_int("R2V_MAX_REPORT_TOKENS", 0) or None,
        report_guidance_scale=env_float("R2V_REPORT_GUIDANCE_SCALE", 4.0),
        modality_guidance_scale=env_float("R2V_MODALITY_GUIDANCE_SCALE", 10.0),
        batched_guidance=env_flag("R2V_BATCHED_GUIDANCE", True),
        num_inference_steps=env_int("R2V_NUM_INFERENCE_STEPS", 30),
        seed=seed,
        device=env_str("R2V_DEVICE", "cuda"),
        latent_only=False,
        allow_base_mismatch=False,
    )
    log.info("adapter=%s base=%s vae=%s", args.adapter, args.base_checkpoint, args.vae_checkpoint)
    return build_sampler(args)


def stable_seed(base_seed: int, case_id: str) -> int:
    """Per-case sampler seed. Byte-identical to `mrrate_r2v.eval.live.stable_seed`, copied
    rather than imported so this file does not pull the evaluation module's dependencies into
    the container. A function of the case, so a resume regenerates the same volume.
    """
    return int(hashlib.sha256(f"{base_seed}:{case_id}".encode()).hexdigest()[:8], 16)


def geometry_for(modality: str, plane: str) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """`(dim_xyz, spacing_mm_xyz)` -- the grid this bucket was trained on.

    Not a 256^3 @ 1 mm cube: `spacing` is a conditioning input to the UNet and appears in the
    `[SPACING]` marker, so the decoded grid, the `spacing_tensor` and the text must all agree
    or the model is being asked for a configuration it never saw.
    """
    from mrrate_r2v.data.geometry import GeometryPolicy, dhw_to_xyz

    spec = GeometryPolicy(mode="per_modality_plane").resolve(modality, plane)
    dim = tuple(int(v) for v in dhw_to_xyz(spec.target_shape))
    spacing = tuple(float(v) for v in dhw_to_xyz(spec.target_spacing))
    return dim, spacing


# ─────────────────────────── main ───────────────────────────

def main() -> int:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(INPUT_DIR)
    progress = Progress()
    progress.install_sigterm_handler()
    progress.restore()

    from routing import mixture, parse_bucket, route

    strategy = env_str("R2V_ROUTING", "marginal")
    fixed = env_str("R2V_FIXED_BUCKET", "")
    assignment = route(
        [stem for stem, _ in prompts],
        [report for _, report in prompts],
        strategy=strategy,
        fixed_bucket=parse_bucket(fixed) if fixed else None,
    )
    log.info("routing strategy=%s; emitted mixture:", strategy)
    for (modality, plane), count, share in mixture(assignment):
        log.info("    %-6s %-9s %4d  %5.1f%%", modality, plane, count, 100 * share)

    seed = env_int("R2V_SEED", 1234)
    sampler, embedder, payload = build(seed)
    needs_sections = bool(getattr(embedder, "needs_sections", False))
    log.info("conditioning=%s needs_sections=%s trained report_format=%r",
             (payload.get("config") or {}).get("conditioning_name"), needs_sections,
             (payload.get("config") or {}).get("report_format"))

    dtype = np.dtype(env_str("R2V_OUTPUT_DTYPE", "float32"))
    # A resume rewrites this file, so the earlier run's per-case rows are carried forward rather
    # than lost -- the manifest is the only record of which bucket each volume was generated in.
    previous_cases: list = []
    if RUN_MANIFEST.exists():
        try:
            previous_cases = list(json.loads(RUN_MANIFEST.read_text()).get("cases", []))
        except (ValueError, OSError) as exc:
            log.warning("could not read the previous run manifest (%s); starting a fresh one", exc)
    manifest = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompts": len(prompts),
        "routing": {"strategy": strategy, "fixed_bucket": fixed or None,
                    "mixture": [{"modality": m, "plane": p, "count": n}
                                for (m, p), n, _ in mixture(assignment)]},
        "seed": seed,
        "output_dtype": str(dtype),
        "adapter_step": payload.get("step"),
        "conditioning_name": (payload.get("config") or {}).get("conditioning_name"),
        "text_encoder": embedder.identity,
        "guidance": {"report": sampler.conditioning.report_guidance_scale,
                     "modality": sampler.conditioning.modality_guidance_scale},
        "num_inference_steps": sampler.config.num_inference_steps,
        "cases": previous_cases,
    }

    from mrrate_r2v.cli.generate_r2v import conditioning_text_for
    from mrrate_r2v.sampling import save_volume

    for index, (stem, report) in enumerate(prompts, start=1):
        filename = f"{stem}.nii.gz"
        if filename in progress.done:
            continue
        if progress.shutting_down:
            break

        modality, plane = assignment[stem]
        dim, spacing = geometry_for(modality, plane)
        # A/B/C were trained on a metadata format, so the [MODALITY]/[PLANE]/[SPACING] prefix
        # is part of what they learned; text without it is out of distribution and silently
        # so. D records no format and gets the text unchanged.
        text, prefix = conditioning_text_for(report, payload, modality, plane, spacing)
        sections = split_sections(report) if needs_sections else None

        case_started = time.time()
        volume = sampler.generate(
            text, dim, spacing,
            seed=stable_seed(seed, stem),
            modality=modality,
            report_sections=sections,
            # int16 [0, 1000] -- NVIDIA's own MR output range, and the space our local
            # evaluation of these adapters was calibrated in. Cast below, values preserved.
            postprocess=True,
        )
        save_volume(volume.astype(dtype, copy=False), spacing, OUTPUT_DIR / filename)

        elapsed = time.time() - case_started
        progress.record(filename)
        manifest["cases"].append({
            "case_id": stem, "modality": modality, "plane": plane,
            "dim_xyz": list(dim), "spacing_mm_xyz": [round(s, 6) for s in spacing],
            "seconds": round(elapsed, 2), "meta_prefix": prefix,
            "sections": sorted(sections) if sections else None,
        })
        log.info("[%d/%d] %s %s %s dim=%s %.1fs", index, len(prompts), stem, modality, plane,
                 list(dim), elapsed)

        if len(progress.done) % max(1, env_int("R2V_CHECKPOINT_EVERY", 5)) == 0:
            progress.save()
            RUN_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))

    progress.save()
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["total_seconds"] = round(time.time() - started, 1)
    RUN_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))

    written = sorted(p.name for p in OUTPUT_DIR.glob("*.nii.gz"))
    log.info("wrote %d/%d volume(s) to %s in %.1f min",
             len(written), len(prompts), OUTPUT_DIR, (time.time() - started) / 60)
    missing = [f"{stem}.nii.gz" for stem, _ in prompts if f"{stem}.nii.gz" not in written]
    if missing:
        # Not a warning: a missing output scores as invalid, so an incomplete run must fail
        # loudly rather than hand the platform a partial set that still gets ranked.
        raise SystemExit(f"{len(missing)} prompt(s) produced no volume, e.g. {missing[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
