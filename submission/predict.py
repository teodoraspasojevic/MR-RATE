#!/usr/bin/env python3
"""VLM3D `mr-volume-generation` entry point: `/input/prompts.json` -> `/output/*.nii.gz`.

No model code here -- `mrrate_r2v.cli.generate_r2v.build_sampler`/`.conditioning_text_for`
are reused unchanged so the container runs exactly the code path `cli.evaluate` was
validated with. What's genuinely new: the platform I/O contract, decoding the (modality,
plane) each `input_image_name` encodes, multi-GPU launch, and recovering
findings/impression for adapter D.

Fail-loud: no try/except around the main loop. A missing output scores as invalid, which
is worse than crashing loudly.
"""
from __future__ import annotations

import json
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

# ---- container layout (FORITHMUS_* env vars are authoritative when the platform sets them) ----
INPUT_DIR = Path(os.environ.get("FORITHMUS_INPUT", "/input"))
OUTPUT_DIR = Path(os.environ.get("FORITHMUS_OUTPUT", "/output"))
CHECKPOINT_DIR = Path(os.environ.get("FORITHMUS_CHECKPOINT", "/checkpoint"))
WEIGHTS_DIR = Path(os.environ.get("FORITHMUS_WEIGHTS", "/weights"))
MODELS_DIR = Path(os.environ.get("R2V_MODELS_DIR", "/opt/app/models"))

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


# ─────────────────────────── multi-GPU (torchrun) ───────────────────────────

def ddp_setup() -> tuple[int, int, int, bool]:
    """`(rank, world_size, local_rank, is_ddp)`. Inference-only process-group coordination
    (rank/world_size + a final barrier), not gradient-synchronizing DDP -- each rank loads
    its own full model and generates a disjoint slice of the prompt list. Only active under
    `torchrun` (which sets `RANK`/`WORLD_SIZE`); plain `python predict.py` returns
    `(0, 1, 0, False)` and nothing about the single-GPU path changes.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise SystemExit("RANK/WORLD_SIZE are set (torchrun launch) but no GPU is visible")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    return 0, 1, 0, False


def ddp_cleanup(is_ddp: bool) -> None:
    if not is_ddp:
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────── case id -> modality, plane ───────────────────────────
# `input_image_name` is `{study_uid}_{modality}-raw-{plane}`, optionally with a `-{n}`
# duplicate-series suffix (e.g. `WNPYIQCPIN_t1w-raw-sag-2`) -- see
# docs/challange_docs/MRI_Report_to_Volume.md and data-preprocessing's own naming scheme.
MODALITY_CODES = {
    "t1w": "T1w", "t2w": "T2w", "flair": "FLAIR", "t2star": "T2star",
    "swi": "SWI", "swan": "SWI", "dwi": "DWI", "mra": "MRA", "asl": "ASL", "pdw": "PDw",
}
# "obl" = OBLIQUE: present in the challenge's own name vocabulary (7 of 690 entries) and a
# real value elsewhere in this package (mrrate_r2v/data/geometry.py's fallback FOV,
# mrrate_r2v/data/README.md's acquisition_plane enum) -- not an edge case to reject.
PLANE_CODES = {"axi": "AXIAL", "sag": "SAGITTAL", "cor": "CORONAL", "obl": "OBLIQUE"}
NAME_PATTERN = re.compile(
    r"[-_](?P<modality>[a-z0-9]+)-raw-(?P<plane>[a-z]+)(?:-\d+)?$", re.IGNORECASE
)


def modality_plane_for(case_id: str) -> tuple[str, str]:
    """Decode `(modality, plane)` from a case id's own `..._<modality>-raw-<plane>` suffix."""
    match = NAME_PATTERN.search(case_id)
    if not match:
        raise ValueError(f"case id {case_id!r} does not encode '..._<modality>-raw-<plane>'")
    modality = MODALITY_CODES.get(match.group("modality").lower())
    plane = PLANE_CODES.get(match.group("plane").lower())
    if modality is None or plane is None:
        raise ValueError(f"case id {case_id!r} names an unrecognised modality/plane code")
    return modality, plane


# ─────────────────────────── prompt parsing ───────────────────────────

def case_stem(name: str) -> str:
    """Strip a known volume extension; the output file is `<stem>.nii.gz`."""
    text = str(name).strip()
    lowered = text.lower()
    for suffix in VOLUME_SUFFIXES:
        if lowered.endswith(suffix):
            return text[: -len(suffix)]
    return text


def read_prompts(input_dir: Path) -> list[tuple[str, str]]:
    """`[(case_id, report_text)]` from the prompt file under `input_dir`."""
    candidates = sorted(input_dir.rglob("*.json"))
    if not candidates:
        raise SystemExit(f"no JSON prompt file under {input_dir}")
    # `prompts.json` if present; anything prompt-ish otherwise, skipping a possible metadata.json.
    exact = [p for p in candidates if p.name == "prompts.json"]
    promptish = [p for p in candidates if "prompt" in p.name.lower()]
    other = [p for p in candidates if p.name.lower() != "metadata.json"]
    path = (exact or promptish or other or candidates)[0]
    print(f"reading prompts from {path}")

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

    print(f"{len(prompts)} prompt(s)")
    return prompts


# Section headings MR-RATE's structuring step emits.
_SECTION_HEADINGS = {
    "clinical_information": r"clinical(?:\s+information|\s+history)?|history|indication",
    "technique": r"technique|protocol",
    "findings": r"findings?|description",
    "impression": r"impression|conclusion|summary|assessment",
}


def split_sections(report: str) -> dict[str, str]:
    """One flat report string -> `{section: text}`, for adapter D's separate findings/impression encoders."""
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
    # Preamble before the first heading (e.g. "58-year-old male:") is clinical context.
    preamble = text[: matches[0].start()].strip().rstrip(":").strip()
    if preamble:
        existing = sections.get("clinical_information", "")
        sections["clinical_information"] = f"{preamble} {existing}".strip()
    return sections or {"findings": text}


# ─────────────────────────── checkpoint / resume ───────────────────────────
# The platform can preempt or time out a long batch job, so finished volumes are backed up
# to /checkpoint (which survives a restart, unlike /output) and re-copied back on resume.
# `done.json` is rank-scoped under multi-GPU (each rank owns a disjoint case slice);
# `outputs/` stays shared -- filenames never collide across ranks.

def checkpoint_paths(rank: int, world_size: int) -> tuple[Path, Path]:
    suffix = f"_rank{rank}" if world_size > 1 else ""
    return CHECKPOINT_DIR / f"done{suffix}.json", CHECKPOINT_DIR / "outputs"


def load_done(done_file: Path, backup_dir: Path) -> list[str]:
    """Finished output filenames from a prior run, verified to still exist somewhere."""
    if not done_file.exists():
        return []
    done = []
    for filename in json.loads(done_file.read_text()):
        backup, output = backup_dir / filename, OUTPUT_DIR / filename
        if output.exists():
            done.append(filename)
        elif backup.exists():
            shutil.copy2(backup, output)
            done.append(filename)
        # else: neither copy survived -- let the main loop regenerate it.
    return done


def save_done(done_file: Path, done: list[str]) -> None:
    done_file.write_text(json.dumps(done))


def mark_done(filename: str, done: list[str], done_file: Path, backup_dir: Path) -> None:
    """Back up one finished volume and record it as done."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_DIR / filename, backup_dir / filename)
    done.append(filename)
    save_done(done_file, done)


def install_shutdown_handler(done: list[str], done_file: Path, rank: int) -> None:
    """SIGTERM gives 30s before SIGKILL -- just persist the index, not a full volume copy."""
    def handle(signum, _frame):
        print(f"[rank {rank}] signal {signum} received after {len(done)} case(s); saving and exiting")
        save_done(done_file, done)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


# ─────────────────────────── model ───────────────────────────

def resolve_weight(name_env: str, default_name: str) -> Path:
    """A weights file, looked up in `/opt/app/models` (symlinked by entrypoint.sh) then `/weights`."""
    name = env_str(name_env, default_name)
    candidates = [Path(name)] if Path(name).is_absolute() else [MODELS_DIR / name, WEIGHTS_DIR / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"{name_env}={name!r} not found. Looked in: {[str(c) for c in candidates]}")


def build(device: str):
    """`(sampler, embedder, payload)` via the package's own loader -- see the module docstring.

    `device` is explicit (not read from `R2V_DEVICE` here) so a DDP rank can pass its own
    `cuda:{local_rank}`.
    """
    from mrrate_r2v.cli.generate_r2v import build_sampler

    args = SimpleNamespace(
        base_checkpoint=resolve_weight("R2V_BASE_CHECKPOINT", "diff_unet_3d_rflow-mr-brain_v0.pt"),
        vae_checkpoint=resolve_weight("R2V_VAE_CHECKPOINT", "autoencoder_v1.pt"),
        adapter=resolve_weight("R2V_ADAPTER", "adapter.pt"),
        network_config=None,
        # Both None: the adapter checkpoint names its own text encoder, resolved from
        # MRRATE_PRETRAINED_DIR, so no single --text-checkpoint path is forced on it.
        text_encoder=None,
        text_checkpoint=None,
        max_report_tokens=env_int("R2V_MAX_REPORT_TOKENS", 0) or None,
        report_guidance_scale=env_float("R2V_REPORT_GUIDANCE_SCALE", 4.0),
        modality_guidance_scale=env_float("R2V_MODALITY_GUIDANCE_SCALE", 10.0),
        batched_guidance=env_flag("R2V_BATCHED_GUIDANCE", True),
        num_inference_steps=env_int("R2V_NUM_INFERENCE_STEPS", 30),
        # `SamplerConfig.random_seed` is stored but never read again anywhere in
        # `mrrate_r2v.sampling` -- every actual draw is unseeded (see `main()`) to match the
        # platform's own reference baseline (`mrgen_example_docker/inference.py`), which never
        # seeds either. This field is inert; any value here is equivalent.
        seed=None,
        device=device,
        latent_only=False,
        allow_base_mismatch=False,
    )
    print(f"adapter={args.adapter} base={args.base_checkpoint} vae={args.vae_checkpoint} device={device}")
    return build_sampler(args)


def geometry_for(modality: str, plane: str) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """`(dim_xyz, spacing_mm_xyz)` for this (modality, plane) bucket."""
    from mrrate_r2v.data.geometry import GeometryPolicy, dhw_to_xyz

    spec = GeometryPolicy(mode="per_modality_plane").resolve(modality, plane)
    dim = tuple(int(v) for v in dhw_to_xyz(spec.target_shape))
    spacing = tuple(float(v) for v in dhw_to_xyz(spec.target_spacing))
    return dim, spacing


def baseline_geometry(divisor: int) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """The platform's own reference baseline (`mrgen_example_docker/inference.py`) ignores
    modality/plane and always samples a 64^3 latent at spacing (1.5, 1.9, 1.9)mm. `divisor`
    (`sampler.divisor`, 4 for the mr-brain model) converts that latent size to the matching
    output voxel shape -- reusing our own model's divisor rather than guessing theirs.
    """
    dim = (64 * divisor,) * 3
    spacing = (1.5, 1.9, 1.9)
    return dim, spacing


# ─────────────────────────── main ───────────────────────────

def main() -> int:
    started = time.time()
    rank, world_size, local_rank, is_ddp = ddp_setup()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(INPUT_DIR)  # same file, same parse, identical on every rank
    done_file, backup_dir = checkpoint_paths(rank, world_size)
    done = load_done(done_file, backup_dir)
    install_shutdown_handler(done, done_file, rank)
    if done:
        print(f"[rank {rank}] resuming: {len(done)} case(s) already done")

    device = f"cuda:{local_rank}" if is_ddp else env_str("R2V_DEVICE", "cuda")
    sampler, embedder, payload = build(device)
    needs_sections = bool(getattr(embedder, "needs_sections", False))
    print(f"[rank {rank}] conditioning={(payload.get('config') or {}).get('conditioning_name')} "
          f"needs_sections={needs_sections}")

    dtype = np.dtype(env_str("R2V_OUTPUT_DTYPE", "float32"))
    # "policy" (default): per-modality/plane geometry from GeometryPolicy, as trained. "baseline":
    # the platform's own reference container's fixed 64^3-latent grid, to A/B against in a future
    # submission -- see `baseline_geometry`.
    geometry_mode = env_str("R2V_GEOMETRY_MODE", "policy")
    if geometry_mode not in ("policy", "baseline"):
        raise SystemExit(f"R2V_GEOMETRY_MODE={geometry_mode!r} must be 'policy' or 'baseline'")
    print(f"[rank {rank}] geometry_mode={geometry_mode}")

    from mrrate_r2v.cli.generate_r2v import conditioning_text_for
    from mrrate_r2v.sampling import save_volume
    from mrrate_r2v.textenc.formats import with_acquisition_section

    # Striped, disjoint slice of the (1-indexed) prompt list -- rank 0 gets prompts 1, 1+N, ...;
    # rank 1 gets 2, 2+N, ... A single process (world_size=1) gets the whole list, unchanged.
    my_prompts = list(enumerate(prompts, start=1))[rank::world_size]
    for index, (case_id, report) in my_prompts:
        filename = f"{case_id}.nii.gz"
        if filename in done:
            continue

        modality, plane = modality_plane_for(case_id)
        dim, spacing = (baseline_geometry(sampler.divisor) if geometry_mode == "baseline"
                        else geometry_for(modality, plane))
        # A/B/C were trained on a metadata format, so the [MODALITY]/[PLANE]/[SPACING] prefix
        # is part of what they learned; text without it is out of distribution and silently
        # so. D records no format and gets the text unchanged.
        text, prefix = conditioning_text_for(report, payload, modality, plane, spacing)
        # E carries the same metadata as its own conditioning token instead of as a text prefix, so
        # the assigned modality/plane/spacing have to reach the sections too. Additive: D declares
        # no `acquisition` section and encodes findings/impression exactly as before.
        sections = (with_acquisition_section(split_sections(report), modality, plane, spacing)
                    if needs_sections else None)

        case_started = time.time()
        volume = sampler.generate(
            text, dim, spacing,
            # No seed: matches the platform's own reference baseline, which draws unseeded
            # noise every run. A prior version derived a per-case seed for resume-reproducibility;
            # since a stopped run is now restarted from scratch rather than resumed in place,
            # that reproducibility isn't needed and there is no other benefit to fixing a seed.
            modality=modality,
            report_sections=sections,
            postprocess=True,  # int16 [0, 1000], NVIDIA's own MR output range
        )
        save_volume(volume.astype(dtype, copy=False), spacing, OUTPUT_DIR / filename)
        mark_done(filename, done, done_file, backup_dir)

        elapsed = time.time() - case_started
        print(f"[rank {rank}][{index}/{len(prompts)}] {case_id} {modality} {plane} "
              f"dim={dim} {elapsed:.1f}s")

    if is_ddp:
        import torch.distributed as dist
        # device_ids pins the barrier to this rank's own GPU -- omitting it makes NCCL guess.
        dist.barrier(device_ids=[local_rank])

    exit_code = 0
    if rank == 0:
        written = {p.name for p in OUTPUT_DIR.glob("*.nii.gz")}
        print(f"wrote {len(written)}/{len(prompts)} volume(s) to {OUTPUT_DIR} "
              f"in {(time.time() - started) / 60:.1f} min")
        missing = [f"{cid}.nii.gz" for cid, _ in prompts if f"{cid}.nii.gz" not in written]
        if missing:
            print(f"ERROR: {len(missing)} prompt(s) produced no volume, e.g. {missing[:5]}")
            exit_code = 1

    # Cleanup before the exit check, on every rank, so a failure on rank 0 doesn't leave the
    # others' process groups torn down uncleanly.
    ddp_cleanup(is_ddp)
    if exit_code:
        # Not a warning: a missing output scores as invalid, so an incomplete run must fail
        # loudly rather than hand the platform a partial set that still gets ranked.
        raise SystemExit(exit_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
