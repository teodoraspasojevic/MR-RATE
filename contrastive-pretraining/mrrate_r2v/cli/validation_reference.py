#!/usr/bin/env python3
"""Compute the fixed reference values the validation curves are read against, once.

None of these is a model score. They answer "what would good even look like?" and "how much of the
gap is the metric's own noise?", so they are computed once per (validation subset, geometry) and
cached to JSON; `cli.train_r2v --validation-reference <json>` then logs them as flat lines beside
the moving curves and never recomputes them.

    python -m mrrate_r2v.cli.validation_reference \\
        --manifest <data>/r2v_manifest/manifest_shards_native.csv \\
        --report-index <data>/r2v_manifest/report_index_shards_native.csv \\
        --vae-checkpoint <ws>/models/autoencoder_v1.pt \\
        --split val --n-samples 64 --out <ws>/cache/r2v/validation_reference.json

What it produces, and what each one means:

| key | meaning | interpretation |
|---|---|---|
| `ssim_identity` | `SSIM(GT, GT)` | **implementation sanity check only.** Must be ~1.0. Says nothing about achievable performance. |
| `ssim_shift_1vox`, `ssim_blur`, `ssim_noise`, `ssim_intensity_scale` | SSIM under controlled degradations | confirms the implementation *responds* -- each must be clearly below 1.0 and ordered sensibly |
| `ssim_autoencoder` | `SSIM(GT, decode(encode(GT)))` | **autoencoder reconstruction reference.** The practical structural-fidelity ceiling imposed by the frozen VAE and decoder. Not a guaranteed upper bound on all models. |
| `fvd_real_vs_real`, `fid_2p5d_real_vs_real` | metric on two disjoint halves of the real set | the **finite-sample noise floor**. Lower is better for both and the theoretical optimum is 0, but two samples of the *same* distribution do not give 0 at finite N -- they give this. |

Deliberately **not** produced:

- **A preprocessing round-trip ceiling.** It would require an inverse of the resample/crop/pad
  chain, and this pipeline has none by design: `cli.preprocess` freezes a target grid and the
  evaluator compares *on* that grid, never resampling back (`eval/geometry_contract.py`). There is
  no round trip to measure, so reporting a number for one would be inventing a transform. The
  autoencoder reference below is the meaningful ceiling and subsumes the part that is real.
- **A repeated-acquisition reference.** MR-RATE ships no field identifying repeat scans of the same
  patient under a compatible protocol and geometry, so any pairing would be a guess. Stated rather
  than silently skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("validation_reference")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Cached reference values for the validation curves.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--report-index", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n-samples", type=int, default=64,
                   help="must match --val-quick-samples for the noise floor to be comparable")
    p.add_argument("--seed", type=int, default=0, help="must match --val-seed")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vae-checkpoint", type=Path, default=None,
                   help="enables the autoencoder reconstruction reference; skipped if omitted")
    p.add_argument("--network-config", type=Path, default=None)
    p.add_argument("--medicalnet-checkpoint", type=Path, default=None,
                   help="enables the MedicalNet FVD diagnostic noise floor")
    p.add_argument("--torch-home", type=Path, default=None,
                   help="where torchvision's r3d_18 weights are staged")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-fvd", action="store_true")
    p.add_argument("--skip-fid", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- SSIM references


def ssim_sanity_and_perturbations(volumes, log) -> dict:
    """Identity plus four controlled degradations, averaged over the subset.

    The perturbations are not calibrated to anything -- their only job is to show the metric moves
    in the right direction and in a sensible order. If `ssim_identity` is not ~1.0, or if any
    perturbation does not lower it, the implementation is wrong and every curve is meaningless.
    """
    from scipy.ndimage import gaussian_filter

    from ..eval.validation_metrics import ssim_volume

    out: dict = {}
    rows: dict = {name: [] for name in
                  ("identity", "shift_1vox", "blur_sigma1", "noise_sigma0p05", "intensity_scale_1p1")}
    rng = np.random.default_rng(0)
    for volume in volumes:
        gt = np.asarray(volume, dtype=np.float32)
        rows["identity"].append(ssim_volume(gt, gt.copy())["ssim"])
        rows["shift_1vox"].append(ssim_volume(gt, np.roll(gt, 1, axis=0))["ssim"])
        rows["blur_sigma1"].append(ssim_volume(gt, gaussian_filter(gt, sigma=1.0))["ssim"])
        rows["noise_sigma0p05"].append(
            ssim_volume(gt, gt + rng.normal(0, 0.05, gt.shape).astype(np.float32))["ssim"])
        rows["intensity_scale_1p1"].append(ssim_volume(gt, gt * 1.1)["ssim"])
    for name, values in rows.items():
        clean = [v for v in values if v is not None]
        out[f"ssim_{name}"] = float(np.mean(clean)) if clean else None

    identity = out.get("ssim_identity")
    if identity is None or abs(identity - 1.0) > 1e-4:
        log.error("SSIM IDENTITY CHECK FAILED: SSIM(GT, GT) = %s, expected ~1.0. The SSIM "
                  "implementation is wrong and every validation curve using it is meaningless.",
                  identity)
    else:
        log.info("SSIM identity check passed: %.6f", identity)
    for name in ("shift_1vox", "blur_sigma1", "noise_sigma0p05"):
        value = out.get(f"ssim_{name}")
        if value is not None and value >= identity - 1e-6:
            log.error("SSIM PERTURBATION CHECK FAILED: %s gave %.6f, not below identity %.6f",
                      name, value, identity)
    return out


def autoencoder_reference(volumes, args, log) -> dict:
    """`SSIM(GT, decode(encode(GT)))` -- the frozen autoencoder's structural-fidelity ceiling.

    The most useful practical reference here, because every generated volume is produced *through*
    this decoder: no report-to-volume model built on it can exceed this by more than noise. It is
    **not** a guaranteed upper bound on all possible models (a different decoder would move it), so
    it is labelled a reference, not a bound.

    Uses the same `encode_stage_2_inputs` / `decode_stage_2_outputs` calls and the same padding
    divisor as training and sampling, so the number describes the real path rather than an idealised
    one.
    """
    from ..eval.geometry_contract import pad_to_divisible
    from ..eval.validation_metrics import ssim_volume
    from ..models.nvidia import (
        DEFAULT_ENV_CONFIG,
        DEFAULT_MODEL_CONFIG,
        DEFAULT_NETWORK_CONFIG,
        load_autoencoder,
    )

    network_config = args.network_config or DEFAULT_NETWORK_CONFIG
    autoencoder, _cfg, divisor = load_autoencoder(
        args.vae_checkpoint, DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, network_config,
        device=str(args.device),
    )
    autoencoder.eval()
    scores = []
    with torch.no_grad():
        for volume in volumes:
            x = torch.from_numpy(np.asarray(volume, dtype=np.float32))[None, None].to(args.device)
            padded_shape, record = pad_to_divisible(tuple(x.shape[2:]), divisor)
            if record is not None:
                pads = []
                for axis in reversed(record.per_axis):
                    pads.extend([int(axis["before"]), int(axis["after"])])
                x = torch.nn.functional.pad(x, pads)
            latent = autoencoder.encode_stage_2_inputs(x)
            recon = autoencoder.decode_stage_2_outputs(latent)
            # Crop the padding back off so the comparison is on the original grid -- the one place a
            # round trip really is invertible, because end-only padding is exactly recorded.
            recon = recon[:, :, :volume.shape[0], :volume.shape[1], :volume.shape[2]]
            score = ssim_volume(np.asarray(volume, dtype=np.float32),
                                recon[0, 0].float().cpu().numpy())["ssim"]
            if score is not None:
                scores.append(score)
    if not scores:
        return {}
    log.info("autoencoder reconstruction reference: SSIM = %.4f over %d volumes",
             float(np.mean(scores)), len(scores))
    return {"ssim_autoencoder": float(np.mean(scores)),
            "ssim_autoencoder_std": float(np.std(scores)),
            "ssim_autoencoder_n": len(scores)}


# --------------------------------------------------------------------------- noise floors


def distribution_noise_floors(volumes, args, log) -> dict:
    """Real-vs-real FVD and 2.5D FID: the metrics' finite-sample noise floor."""
    from ..eval.validation_metrics import real_vs_real_baseline
    from ..eval.video_features import PLANE_AXES

    out: dict = {}
    if not args.skip_fvd:
        from ..eval.video_features import R3D18SequenceExtractor

        extractor = R3D18SequenceExtractor(device=str(args.device),
                                           torch_home=str(args.torch_home) if args.torch_home else None)
        per_plane = {name: [] for name, _ in PLANE_AXES}
        for volume in volumes:
            for name, vector in extractor.extract(volume).items():
                per_plane[name].append(vector)
        values = {}
        for name, vectors in per_plane.items():
            baseline = real_vs_real_baseline(vectors, seed=args.seed)
            if baseline.get("value") is not None:
                values[name] = baseline["value"]
                out[f"fvd_real_vs_real_{name}"] = baseline["value"]
        if values:
            # Aggregated the same way the live metric is, so the floor and the curve are comparable.
            out["fvd_real_vs_real"] = float(np.mean(list(values.values())))
            log.info("FVD real-vs-real noise floor: %.3f (per plane %s)",
                     out["fvd_real_vs_real"], {k: round(v, 2) for k, v in values.items()})

    if not args.skip_fid:
        from ..eval.distribution import InceptionFeatureExtractor, extract_2p5d_inception_features

        inception = InceptionFeatureExtractor(device=str(args.device))
        per_plane = {name: [] for name, _ in PLANE_AXES}
        for volume in volumes:
            for name, vector in extract_2p5d_inception_features(volume, inception).items():
                per_plane[name].append(vector)
        values = {}
        for name, vectors in per_plane.items():
            baseline = real_vs_real_baseline(vectors, seed=args.seed)
            if baseline.get("value") is not None:
                values[name] = baseline["value"]
                out[f"fid_2p5d_real_vs_real_{name}"] = baseline["value"]
        if values:
            out["fid_2p5d_real_vs_real"] = float(np.mean(list(values.values())))
            log.info("2.5D FID real-vs-real noise floor: %.3f (per plane %s)",
                     out["fid_2p5d_real_vs_real"], {k: round(v, 2) for k, v in values.items()})
    return out


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    args = parse_args(argv)
    from ..validation import ValidationConfig, select_validation_cases
    from .train_r2v import build_dataset

    # The *same* selection the trainer uses, from the same seed, so the noise floor is measured on
    # the very cases the curve is measured on -- not on a different sample of the same split.
    class _Args:
        report_sections = ["findings", "impression"]
        geometry_mode = "per_modality_plane"
        seed = args.seed
        manifest = args.manifest
        report_index = args.report_index

    dataset = build_dataset(_Args(), args.split, None)
    config = ValidationConfig(n_quick=args.n_samples, seed=args.seed, n_visualize=0)
    indices = select_validation_cases(dataset, config, args.n_samples)
    log.info("loading %d %s volumes (seed %d)", len(indices), args.split, args.seed)
    volumes = [dataset[i]["image"].squeeze(0).numpy().astype(np.float32) for i in indices]
    log.info("loaded; shapes %s", sorted({tuple(v.shape) for v in volumes}))

    reference: dict = {}
    reference.update(ssim_sanity_and_perturbations(volumes, log))
    if args.vae_checkpoint:
        reference.update(autoencoder_reference(volumes, args, log))
    else:
        log.info("no --vae-checkpoint: skipping the autoencoder reconstruction reference")
    reference.update(distribution_noise_floors(volumes, args, log))

    from ..eval.validation_metrics import ssim_parameters

    payload = {
        "reference": reference,
        "provenance": {
            "split": args.split, "n_samples": len(indices), "seed": args.seed,
            "shapes": sorted(str(tuple(v.shape)) for v in {tuple(x.shape) for x in volumes}),
            "buckets": sorted({f"{dataset.samples[i].modality}_{dataset.samples[i].plane}"
                               for i in indices}),
            "ssim_parameters": ssim_parameters(),
            "vae_checkpoint": str(args.vae_checkpoint) if args.vae_checkpoint else None,
        },
        "interpretation": {
            "ssim_identity": "implementation sanity check only; must be ~1.0; NOT an achievable "
                             "performance estimate",
            "ssim_perturbations": "confirms SSIM responds to shift/blur/noise/intensity error",
            "ssim_autoencoder": "autoencoder reconstruction reference -- the practical structural "
                                "ceiling imposed by the frozen VAE/decoder. Not a guaranteed upper "
                                "bound on all models.",
            "real_vs_real": "finite-sample noise floor for FVD and 2.5D FID. Lower is better and "
                            "the optimum is 0, but two disjoint samples of the same distribution "
                            "do not give 0 at finite N. A reference baseline, not a model score.",
            "not_computed": {
                "report_volume_alignment_matched": "needs a frozen, independent, validated "
                    "cross-modal MRI report-volume model. None is adopted, so neither the matched "
                    "ground-truth reference nor the mismatched reference is computed -- rather than "
                    "produce them from an unvalidated substitute. Best candidate on file: HLIP "
                    "(zch0414/clip-vit_base-scan_study-dualdinotxt1568, arXiv:2505.21862, MIT), "
                    "trained on MR-RATE's TRAIN split so this project's val split is unseen.",
                "report_volume_alignment_mismatched": "same reason. The deterministic "
                    "different-study permutation this would use is implemented and tested "
                    "(validation.ValidationRunner.shuffled_report_pairing); only the cross-modal "
                    "scorer is missing.",
                "preprocessing_round_trip": "this pipeline has no inverse resample by design (the "
                                            "evaluator compares on the frozen cohort grid), so "
                                            "there is no round trip to measure",
                "repeated_acquisition": "MR-RATE ships no field identifying protocol- and "
                                        "geometry-compatible repeat scans, so any pairing would be "
                                        "a guess",
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote %s", args.out)
    print(json.dumps(reference, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
