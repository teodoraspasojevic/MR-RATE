#!/usr/bin/env python3
"""Measure what this trainer actually does on the GPU, per conditioning configuration.

Reports the environment, then a short controlled benchmark per (configuration, batch size): step
time, throughput, peak allocated/reserved memory, and where the time goes (text encode, VAE
encode, denoiser forward/backward). **Never launches a training run** -- every measurement is a
few dozen steps with synthetic volumes, so nothing here touches the manifest or writes a checkpoint.

    python -m mrrate_r2v.cli.benchmark_h200 --env-only
    python -m mrrate_r2v.cli.benchmark_h200 --configs cxr_bert_cls report2ct_style \\
        --batch-sizes 1 2 4 --out bench.json

Batch sizes are probed upward and an out-of-memory result is *recorded, not fatal*: the largest
size that fits is data, and the recommendation is derived from the measurements rather than
asserted. The recommendation deliberately leaves headroom, because validation generation runs a
full diffusion sampler plus a MedicalNet forward pass in the same process and its peak is not
covered by a training-step measurement.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("benchmark_h200")


def environment() -> dict:
    """Everything that determines whether the optimisations below are available at all."""
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        import torch.distributed as dist

        info["nccl_available"] = dist.is_nccl_available()
        info["nccl_version"] = ".".join(str(v) for v in torch.cuda.nccl.version()) \
            if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        info["nccl_available"], info["nccl_error"] = False, str(exc)

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": properties.name,
                "total_memory_gb": round(properties.total_memory / 1024 ** 3, 1),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "multiprocessors": properties.multi_processor_count,
            })
        info["devices"] = devices
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        info["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
        info["flash_sdp_enabled"] = torch.backends.cuda.flash_sdp_enabled()
        info["mem_efficient_sdp_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        info["math_sdp_enabled"] = torch.backends.cuda.math_sdp_enabled()
        try:
            import flash_attn

            info["flash_attn_package"] = getattr(flash_attn, "__version__", "present")
        except Exception:  # noqa: BLE001
            info["flash_attn_package"] = None

    import os

    info["slurm"] = {k: v for k, v in os.environ.items()
                     if k.startswith(("SLURM_JOB", "SLURM_NTASKS", "SLURM_GPUS", "SLURM_CPUS"))}
    info["cpu_count"] = os.cpu_count()
    return info


def sdpa_backend_probe(device: str = "cuda") -> dict:
    """Which SDPA kernel actually serves the adapters' attention shape.

    `MaskedCrossAttention` calls `scaled_dot_product_attention` with an `attn_mask`, and a mask is
    what disqualifies the flash kernel on most versions -- so the answer for the shape this code
    really uses is worth measuring rather than assuming from a feature flag.
    """
    if not torch.cuda.is_available():
        return {"skipped": "no CUDA"}
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # (B, heads, queries=voxels, head_dim) against 2 context tokens: the report2ct_style shape.
    q = torch.randn(2, 16, 512, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 16, 2, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones(2, 1, 1, 2, dtype=torch.bool, device=device)
    out = {}
    for name, backend in [("flash", SDPBackend.FLASH_ATTENTION),
                          ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("math", SDPBackend.MATH)]:
        for label, attn_mask in (("with_mask", mask), ("no_mask", None)):
            try:
                with sdpa_kernel(backend):
                    torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
                out[f"{name}_{label}"] = "ok"
            except Exception as exc:  # noqa: BLE001
                out[f"{name}_{label}"] = f"unavailable: {type(exc).__name__}"
    return out


def synthetic_batch(batch_size: int, shape, sections: bool) -> dict:
    """One collated batch's worth of volumes in the model's own (X, Y, Z) order.

    Volumes, not latents: the VAE encode is a real and substantial part of a training step here
    (latents are computed on the fly rather than baked to disk), so benchmarking from latents would
    understate the step time and overstate the batch size that fits.
    """
    reports = [f"[IMPRESSION] Case {i}.\n[FINDINGS] " + "Detail sentence. " * 40
               for i in range(batch_size)]
    batch = {
        "image": torch.randn(batch_size, 1, *shape),
        "report_text": reports,
        "modality": ["T1w"] * batch_size,
        "target_spacing_mm": torch.tensor([[1.0, 1.0, 1.0]] * batch_size),
    }
    if sections:
        batch["report_sections_text"] = [
            {"findings": "Detail sentence. " * 40, "impression": f"Case {i}." if i % 8 else ""}
            for i in range(batch_size)
        ]
    return batch


def benchmark_configuration(name, batch_size, shape, steps, warmup, args) -> dict:
    """One (configuration, batch size) measurement. Returns a row, including an OOM row."""
    from ..conditioning import ConditioningConfig
    from ..models.nvidia import (
        DEFAULT_ENV_CONFIG,
        DEFAULT_MODEL_CONFIG,
        DEFAULT_NETWORK_CONFIG,
        define_instance,
        load_autoencoder,
        load_config,
    )
    from ..models.report_conditioned_unet import build_report_conditioned_unet, load_pretrained_maisi_weights
    from ..text import encode_reports
    from ..textenc.conditioning import CONDITIONING_CONFIGS, build_conditioning
    from ..training import LatentEncoder, MRRateAdapterTrainer, TrainingConfig, resolve_scale_factor

    device = torch.device("cuda")
    row = {"conditioning": name, "batch_size": batch_size, "shape_xyz": list(shape),
           "steps": steps, "amp": bool(args.amp)}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        started = time.time()
        embedder = build_conditioning(name, max_length=args.max_report_tokens)
        row["text_encoder_load_seconds"] = round(time.time() - started, 2)
        row["context_dim"] = embedder.output_dim
        row["context_length"] = CONDITIONING_CONFIGS[name]["sequence_length"]
        needs_sections = bool(getattr(embedder, "needs_sections", False))

        network_config = args.network_config or DEFAULT_NETWORK_CONFIG
        unet = build_report_conditioned_unet(
            context_dim=embedder.output_dim, network_config=network_config,
            cross_attention_dim=args.cross_attention_dim, use_flash_attention=True,
        ).to(device)
        load_pretrained_maisi_weights(unet, args.base_checkpoint)

        cfg_args = load_config(str(DEFAULT_ENV_CONFIG), str(DEFAULT_MODEL_CONFIG), str(network_config))
        noise_scheduler = define_instance(cfg_args, "noise_scheduler")
        scale_factor = resolve_scale_factor(args.base_checkpoint, "auto")
        autoencoder, _cfg, divisor = load_autoencoder(
            args.base_vae, DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, network_config, device=str(device)
        )
        latent_encoder = LatentEncoder(autoencoder, divisor, scale_factor, amp=args.amp)

        trainer = MRRateAdapterTrainer(
            unet=unet, text_embedder=embedder, noise_scheduler=noise_scheduler,
            latent_encoder=latent_encoder,
            config=TrainingConfig(amp=args.amp, batch_size=batch_size, validate_at_end=False),
            device=device, output_dir=Path(args.scratch), base_checkpoint={},
            num_train_timesteps=int(cfg_args.noise_scheduler["num_train_timesteps"]),
        )
        row["trainable_parameters"] = trainer.freeze_report.trainable_parameters

        batch = synthetic_batch(batch_size, shape, needs_sections)

        # Component timings on a warmed-up model, measured separately from the step loop so the
        # step number stays a clean end-to-end figure.
        for _ in range(warmup):
            trainer.train_step(batch)
        torch.cuda.synchronize()

        def timed(fn, repeats=5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - start) / repeats

        with torch.no_grad():
            row["text_encode_seconds"] = round(
                timed(lambda: encode_reports(embedder, batch, device)), 4)
            row["vae_encode_seconds"] = round(
                timed(lambda: latent_encoder.encode(batch["image"].to(device).float())), 4)

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(steps):
            trainer.train_step(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        row.update({
            "ok": True,
            "step_seconds": round(elapsed / steps, 4),
            "volumes_per_second": round(batch_size * steps / elapsed, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
            "total_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
        })
        row["denoiser_seconds"] = round(
            max(row["step_seconds"] - row["text_encode_seconds"] - row["vae_encode_seconds"], 0.0), 4)
        # Which of the three dominates decides what to optimise; guessing here is what leads to
        # tuning the dataloader when the denoiser is 90% of the step.
        row["dominant_stage"] = max(
            [("text_encode", row["text_encode_seconds"]), ("vae_encode", row["vae_encode_seconds"]),
             ("denoiser", row["denoiser_seconds"])], key=lambda pair: pair[1])[0]
    except torch.cuda.OutOfMemoryError as exc:
        row.update({"ok": False, "error": "OutOfMemoryError", "detail": str(exc)[:200]})
        log.warning("%s @ batch %d: OOM (recorded, not fatal)", name, batch_size)
    except Exception as exc:  # noqa: BLE001
        row.update({"ok": False, "error": type(exc).__name__, "detail": str(exc)[:400]})
        log.warning("%s @ batch %d failed: %s: %s", name, batch_size, type(exc).__name__, exc)
    finally:
        for variable in ("trainer", "unet", "autoencoder", "latent_encoder", "embedder"):
            if variable in dir():
                pass
        torch.cuda.empty_cache()
    return row


def recommend(rows: list) -> list:
    """Turn measurements into a per-configuration recommendation.

    The safe batch size is the largest measured one whose peak *reserved* memory stays under
    `--memory-headroom` of the card, not the largest that merely did not OOM. Reserved rather than
    allocated because the caching allocator's reserved figure is what the next allocation actually
    contends with, and headroom because validation generation runs a full sampler in this same
    process and its peak is not in these numbers.
    """
    out = []
    for name in dict.fromkeys(r["conditioning"] for r in rows):
        mine = [r for r in rows if r["conditioning"] == name]
        working = [r for r in mine if r.get("ok")]
        if not working:
            out.append({"conditioning": name, "status": "no batch size succeeded"})
            continue
        total = working[0].get("total_memory_gb") or 0
        budget = total * 0.75
        safe = [r for r in working if r["peak_reserved_gb"] <= budget]
        best = max(safe or working, key=lambda r: r["batch_size"])
        fastest = max(working, key=lambda r: r["volumes_per_second"])
        out.append({
            "conditioning": name,
            "max_tested_ok": max(r["batch_size"] for r in working),
            "max_attempted": max(r["batch_size"] for r in mine),
            "recommended_batch_size": best["batch_size"],
            "memory_budget_gb": round(budget, 1),
            "peak_reserved_gb_at_recommended": best["peak_reserved_gb"],
            "step_seconds_at_recommended": best["step_seconds"],
            "volumes_per_second_at_recommended": best["volumes_per_second"],
            "fastest_batch_size": fastest["batch_size"],
            "dominant_stage": best.get("dominant_stage"),
            "text_encode_share": (round(best["text_encode_seconds"] / best["step_seconds"], 3)
                                  if best.get("step_seconds") else None),
            "oom_at": sorted(r["batch_size"] for r in mine
                             if r.get("error") == "OutOfMemoryError") or None,
        })
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-only", action="store_true", help="print the environment and exit")
    p.add_argument("--configs", nargs="+",
                   default=["cxr_bert_cls", "radbert_mean", "report2ct_style"])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--shape", nargs=3, type=int, default=[256, 256, 256],
                   help="model input shape (X Y Z); the NVIDIA default bucket")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--base-checkpoint", type=Path, default=None)
    p.add_argument("--base-vae", type=Path, default=None)
    p.add_argument("--network-config", type=Path, default=None)
    p.add_argument("--cross-attention-dim", type=int, default=512)
    p.add_argument("--max-report-tokens", type=int, default=512)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--scratch", type=Path, default=Path("/tmp/r2v_bench"))
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.scratch.mkdir(parents=True, exist_ok=True)
    result = {"environment": environment()}
    log.info("environment:\n%s", json.dumps(result["environment"], indent=2, default=str))

    # Probed under --env-only too: which SDPA kernel serves a masked 2-token context is an
    # environment fact, and it is the main thing you want to know before booking a long run.
    if torch.cuda.is_available():
        result["sdpa_backends"] = sdpa_backend_probe()
        log.info("SDPA backends for the adapters' shape:\n%s",
                 json.dumps(result["sdpa_backends"], indent=2))

    if args.env_only:
        if args.out:
            args.out.write_text(json.dumps(result, indent=2, default=str))
        return 0
    if not torch.cuda.is_available():
        log.error("no CUDA device: run this on a GPU node (partition h200)")
        return 1
    if args.base_checkpoint is None or args.base_vae is None:
        log.error("--base-checkpoint and --base-vae are required for the timing runs")
        return 2

    # TF32 on: the denoiser is convolution- and matmul-bound, and its own training used
    # `set_float32_matmul_precision("highest")` only for the fp32 path -- under bf16 autocast the
    # matmuls are bf16 anyway, so TF32 affects the residual fp32 ops rather than the loss-bearing
    # ones. Recorded in the environment block either way.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    rows = []
    for name in args.configs:
        for batch_size in sorted(args.batch_sizes):
            log.info("=== %s @ batch %d ===", name, batch_size)
            row = benchmark_configuration(name, batch_size, tuple(args.shape),
                                          args.steps, args.warmup, args)
            rows.append(row)
            log.info("%s", json.dumps(row, default=str))
            if row.get("error") == "OutOfMemoryError":
                log.info("stopping the sweep for %s: larger batches cannot fit either", name)
                break
    result["rows"] = rows
    result["recommendations"] = recommend(rows)
    log.info("recommendations:\n%s", json.dumps(result["recommendations"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, default=str))
        log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
