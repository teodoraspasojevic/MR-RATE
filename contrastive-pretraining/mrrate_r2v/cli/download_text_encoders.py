#!/usr/bin/env python3
"""Stage text-encoder checkpoints into the pretrained directory. Idempotent.

    python -m mrrate_r2v.cli.download_text_encoders --list
    python -m mrrate_r2v.cli.download_text_encoders --encoders bioclinical_mbert medembed_small
    python -m mrrate_r2v.cli.download_text_encoders --all

A directory that already carries a `download_metadata.json` naming the same revision is left
untouched, and one that exists *without* provenance is never overwritten -- it is reported and
skipped, so a hand-staged snapshot can never be silently replaced.

Revisions are pinned: the current hub SHA is resolved once and both the download and the recorded
provenance use it, so two people running this on different days get identical bytes or a loud
mismatch.

Only the files an encoder-only forward pass needs are fetched. Two checkpoints need a step the
hub cannot provide, and both are handled here rather than at load time:

- `bio_clinicalbert` ships no safetensors at all; its `pytorch_model.bin` is converted once
  (transformers >= 5 refuses to `torch.load` a .bin under torch < 2.6, CVE-2025-32434).
- `radbert` ships a .bin as well; `text.ensure_local_safetensors` already handles it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Checkpoints with no safetensors on the hub: (encoder, weight file, key prefix to strip,
#: key prefixes to drop). Converted locally, once.
BIN_ONLY = {
    "bio_clinicalbert": ("pytorch_model.bin", "bert.", ("cls.",)),
}

ALLOW_PATTERNS = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                  "vocab.json", "merges.txt", "special_tokens_map.json", "model.safetensors",
                  "spiece.model", "README.md"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage text-encoder snapshots for mrrate_r2v.textenc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--encoders", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="stage every encoder in the zoo")
    parser.add_argument("--list", action="store_true", help="print the zoo and what is staged")
    parser.add_argument("--pretrained-dir", type=Path, default=None,
                        help="default: MRRATE_PRETRAINED_DIR, else the cluster path")
    return parser.parse_args(argv)


def _convert_bin(directory: Path, weight_file: str, strip: str, drop: tuple) -> int:
    import safetensors.torch
    import torch

    state = torch.load(str(directory / weight_file), map_location="cpu", weights_only=True)
    tensors = {}
    for key, value in state.items():
        if key.startswith(drop) or key.endswith("position_ids"):
            continue
        tensors[key[len(strip):] if key.startswith(strip) else key] = value.contiguous().clone()
    if not tensors:
        raise RuntimeError(f"no encoder tensors found in {directory / weight_file}")
    safetensors.torch.save_file(tensors, str(directory / "model.safetensors"),
                                metadata={"format": "pt"})
    return len(tensors)


def main(argv=None):
    args = parse_args(argv)
    from ..textenc.encoders import DEFAULT_PRETRAINED_DIR, ENCODER_SPECS, available_encoders

    root = Path(args.pretrained_dir or DEFAULT_PRETRAINED_DIR)
    if args.list:
        staged = available_encoders(str(root))
        print(f"pretrained dir: {root}\n")
        print(f"{'encoder':<20} {'staged':<7} {'dim':>5} {'ctx':>6} {'license':<12} repo")
        for name, spec in ENCODER_SPECS.items():
            print(f"{name:<20} {'yes' if staged[name] else 'NO':<7} {spec.hidden:>5} "
                  f"{spec.context:>6} {spec.license:<12} {spec.hf_repo}")
        return

    names = list(ENCODER_SPECS) if args.all else (args.encoders or [])
    if not names:
        sys.exit("nothing to do: pass --encoders, --all or --list")
    unknown = [n for n in names if n not in ENCODER_SPECS]
    if unknown:
        sys.exit(f"unknown encoder(s) {unknown}; choose from {sorted(ENCODER_SPECS)}")

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = ENCODER_SPECS[name]
        target = root / spec.directory
        metadata_path = target / "download_metadata.json"
        revision = api.model_info(spec.hf_repo).sha
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text()).get("revision")
            verdict = "already at" if existing == revision else f"pinned to {existing}, hub is"
            print(f"[skip] {name}: {verdict} {revision[:12]}")
            continue
        if target.exists() and any(target.iterdir()):
            print(f"[skip] {name}: {target} exists without download_metadata.json -- not "
                  f"overwriting a hand-staged snapshot", file=sys.stderr)
            continue
        print(f"[get ] {name} <- {spec.hf_repo}@{revision[:12]}")
        snapshot_download(spec.hf_repo, revision=revision, local_dir=str(target),
                          allow_patterns=ALLOW_PATTERNS)
        note = None
        if name in BIN_ONLY and not (target / "model.safetensors").is_file():
            weight_file, strip, drop = BIN_ONLY[name]
            hf_hub_download(spec.hf_repo, weight_file, revision=revision, local_dir=str(target))
            n = _convert_bin(target, weight_file, strip, drop)
            note = (f"hub ships no safetensors; model.safetensors converted locally from "
                    f"{weight_file} ({n} tensors, '{strip}' prefix stripped, {drop} dropped)")
            print(f"       converted {weight_file} -> model.safetensors ({n} tensors)")
        files = sorted(p.name for p in target.iterdir() if p.is_file())
        size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
        metadata_path.write_text(json.dumps({
            "repo_id": spec.hf_repo, "revision": revision, "snapshot_path": str(target),
            "encoder": name, "files": files, "bytes": size, "note": note,
        }, indent=1))
        print(f"       {size/1e6:.0f} MB, {len(files)} files")


if __name__ == "__main__":
    main()
