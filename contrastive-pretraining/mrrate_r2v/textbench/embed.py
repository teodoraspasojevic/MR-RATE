"""Encode a corpus once per (encoder, format, split) and cache it.

Same split of concerns as the contrastive pipeline's `extract_features.py` + `linear_probe.py`:
the expensive frozen-encoder pass runs once and no label is baked into the cache, so re-scoring
with a different metric or a different label set is a seconds-long CPU job.

Cache layout, one directory per run:

    <out>/<encoder>__<format>__<split>.npz     mean, max, study_uid
    <out>/<encoder>__<format>__<split>.json    encoder identity, truncation stats, timings

`mean` is the mask-aware mean over token states -- the encoder's own `pooled_embedding`.
`max` is a mask-aware max over the same states. Probes use `concat(mean, max)`: the denoiser
attends over *tokens*, so a purely mean-pooled probe would over-report how much a mean-pooling
encoder discards. Both are stored so either can be scored later without re-encoding.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from ..textenc.formats import format_report


def cache_paths(out_dir, encoder: str, report_format: str, split: str):
    stem = os.path.join(out_dir, f"{encoder}__{report_format}__{split}")
    return stem + ".npz", stem + ".json"


def _masked_max(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    filled = tokens.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    out = filled.max(dim=1).values
    return torch.nan_to_num(out, neginf=0.0)       # an all-padding row would otherwise be -inf


def embed_corpus(encoder, records, report_format: str, device, batch_size: int = 64,
                 dtype=torch.float32, log_every: int = 20):
    """Encode every record under one format. Returns (mean, max, study_uids, stats).

    Records whose formatted text is empty are still encoded -- the tokenizer emits the
    start/end tokens, which is exactly what the model sees for an empty report at inference.
    """
    texts = [format_report(record.as_report_record(), report_format) for record in records]
    n_empty = sum(1 for t in texts if not t)
    means, maxes = [], []
    # One encoder object serves every format and split, so the lifetime counter would report the
    # previous format's truncation here. Snapshot, and report only this call's stretch.
    before = encoder.truncation.snapshot() if hasattr(encoder, "truncation") else None
    started = time.time()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        conditioning = encoder.encode(chunk, device)
        tokens = conditioning.token_embeddings.to(torch.float32)
        mask = conditioning.attention_mask
        pooled = (conditioning.pooled_embedding.to(torch.float32)
                  if conditioning.pooled_embedding is not None
                  else tokens.mean(dim=1))
        means.append(pooled.detach().cpu().numpy().astype(np.float32))
        maxes.append(_masked_max(tokens, mask).detach().cpu().numpy().astype(np.float32))
        if log_every and (start // batch_size) % log_every == 0:
            done = start + len(chunk)
            rate = done / max(time.time() - started, 1e-6)
            print(f"    {done}/{len(texts)} ({rate:.0f} reports/s)", flush=True)
    elapsed = time.time() - started
    stats = {
        "n_records": len(texts), "n_empty_text": n_empty, "seconds": elapsed,
        "reports_per_second": len(texts) / max(elapsed, 1e-6),
        "truncation": encoder.truncation.since(before) if before is not None else {},
        "truncation_cumulative": (encoder.truncation.as_dict()
                                  if hasattr(encoder, "truncation") else {}),
        "encoder": dict(encoder.identity), "report_format": report_format,
    }
    return np.concatenate(means), np.concatenate(maxes), [r.study_uid for r in records], stats


def write_cache(out_dir, encoder_name, report_format, split, mean, maximum, study_uids, stats):
    """Write one cache atomically: full file to a temporary name, then rename into place.

    `os.replace` is atomic within a filesystem, so a reader never sees a half-written `.npz` and
    two jobs that happen to target the same cache cannot interleave into a corrupt file -- the
    later rename simply wins, and both wrote identical content. Worth the four extra lines: these
    jobs run for hours, get preempted, and are deliberately fanned out over several nodes writing
    into one directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    npz_path, json_path = cache_paths(out_dir, encoder_name, report_format, split)
    temporary = f"{npz_path}.{os.getpid()}.tmp"
    # np.savez_compressed appends .npz unless the name already ends in it, hence the explicit
    # handle rather than a path.
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, mean=mean.astype(np.float16), max=maximum.astype(np.float16),
                            study_uid=np.array(study_uids, dtype=object), allow_pickle=True)
    os.replace(temporary, npz_path)
    temporary_json = f"{json_path}.{os.getpid()}.tmp"
    with open(temporary_json, "w") as handle:
        json.dump(stats, handle, indent=1, default=str)
    os.replace(temporary_json, json_path)
    return npz_path


def read_cache(out_dir, encoder_name, report_format, split):
    """(mean, max, study_uids, stats). float16 on disk is widened back to float32: the probes are
    solved in float64 by sklearn anyway, and float16 costs nothing in probe accuracy while halving
    a 25k x 1024 cache."""
    npz_path, json_path = cache_paths(out_dir, encoder_name, report_format, split)
    with np.load(npz_path, allow_pickle=True) as data:
        mean = data["mean"].astype(np.float32)
        maximum = data["max"].astype(np.float32)
        study_uids = [str(u) for u in data["study_uid"]]
    stats = json.load(open(json_path)) if os.path.isfile(json_path) else {}
    return mean, maximum, study_uids, stats


__all__ = ["cache_paths", "embed_corpus", "read_cache", "write_cache"]
