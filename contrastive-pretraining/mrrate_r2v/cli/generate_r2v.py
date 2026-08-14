#!/usr/bin/env python3
"""Report-to-volume inference. `NV-Generate-CTMR/scripts/diff_model_infer.py` with a report.

Loads the frozen NVIDIA base checkpoint strictly, the adapter checkpoint strictly, and the text
encoder named in the adapter checkpoint; then runs NVIDIA's own sampler, guidance point, decoder and
output postprocessing (see `mrrate_r2v/sampling.py` for the line-by-line mapping).

One report from a string:

    python -m mrrate_r2v.cli.generate_r2v \\
        --base-checkpoint <ws>/models/diff_unet_3d_rflow-mr-brain_v0.pt \\
        --vae-checkpoint  <ws>/models/autoencoder_v1.pt \\
        --adapter <ws>/runs/r2v_adapter_v1/adapter_last.pt \\
        --text-checkpoint <ws>/pretrained/RadBERT-RoBERTa-4m \\
        --report "Findings: 12 mm enhancing lesion in the right frontal lobe." \\
        --modality T1w --plane AXIAL \\
        --report-guidance-scale 4 --out <ws>/samples/case001.nii.gz

`--dim`/`--spacing` default to the (modality, plane) bucket's own trained grid; pass them only to
override it. When the adapter was trained on a `*_meta` format, the `[MODALITY]/[PLANE]/[SPACING]`
prefix is prepended from those same values -- so a challenge request that carries no acquisition
metadata just takes the defaults, and the text and the numeric conditioning still agree.

Every case in a cohort's validation split, from its paired MR-RATE report:

    python -m mrrate_r2v.cli.generate_r2v --cohort <ws>/cohorts/val_v1 --out-dir <ws>/samples/val_v1 ...

Modality-only generation (NVIDIA's original behaviour, no report term):

    ... --report "" --report-guidance-scale 0 --modality-guidance-scale 10
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
log = logging.getLogger("generate_r2v")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a 3D brain MRI volume from a radiology report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--vae-checkpoint", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True, help="adapter checkpoint from cli.train_r2v")
    p.add_argument("--network-config", type=Path, default=None)
    p.add_argument("--text-encoder", default=None, choices=[None, "radbert", "mock"],
                   help="default: whatever the adapter checkpoint recorded")
    p.add_argument("--text-checkpoint", type=Path, default=None)
    p.add_argument("--max-report-tokens", type=int, default=None)

    source = p.add_argument_group("report source (exactly one)")
    source.add_argument("--report", default=None, help="report text")
    source.add_argument("--report-file", type=Path, default=None, help="file containing the report text")
    source.add_argument("--cohort", type=Path, default=None,
                        help="generate one volume per case from the cohort's paired reports")
    source.add_argument("--allow-report-format-mismatch", action="store_true",
                        help="generate even when the cohort's report_format differs from the one "
                             "the adapter was trained on. For a deliberate ablation only -- the "
                             "mismatch is otherwise silent")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=None, help="output .nii.gz (single report)")
    out.add_argument("--out-dir", type=Path, default=None, help="output directory (--cohort)")
    out.add_argument("--modality", default="T1w", choices=["T1w", "T2w", "FLAIR", "SWI", "unknown"],
                     help="the modality to generate. Also the [MODALITY] marker when the adapter "
                          "was trained on a metadata format -- there is no 'unspecified' option, "
                          "because the model never saw a report without one")
    out.add_argument("--plane", default="AXIAL", choices=["AXIAL", "SAGITTAL", "CORONAL", "unknown"],
                     help="acquisition plane. Selects the geometry bucket and the [PLANE] marker")
    out.add_argument("--dim", nargs=3, type=int, default=None, metavar=("X", "Y", "Z"),
                     help="output shape (X Y Z). Default: the (modality, plane) bucket's own shape, "
                          "the grid that bucket was trained on")
    out.add_argument("--spacing", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                     help="mm (X Y Z). Default: the (modality, plane) bucket's own spacing")

    sampler = p.add_argument_group("sampler and guidance")
    sampler.add_argument("--num-inference-steps", type=int, default=30, help="NVIDIA's own default")
    sampler.add_argument("--report-guidance-scale", type=float, default=4.0,
                         help="0 disables the report term and reproduces NVIDIA's guidance exactly")
    sampler.add_argument("--modality-guidance-scale", type=float, default=10.0,
                         help="NVIDIA's cfg_guidance_scale for mr-brain; 0 disables modality guidance")
    sampler.add_argument("--no-batched-guidance", dest="batched_guidance", action="store_false",
                         help="run guidance branches as separate forwards (slower, same numbers)")
    sampler.add_argument("--seed", type=int, default=1234, help="NVIDIA's own random_seed default")
    sampler.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sampler.add_argument("--latent-only", action="store_true",
                         help="stop after the diffusion loop; skip the VAE decode (cheap smoke test)")
    sampler.add_argument("--allow-base-mismatch", action="store_true",
                         help="load an adapter trained against a different base checkpoint")

    args = p.parse_args(argv)
    sources = [args.report is not None, args.report_file is not None, args.cohort is not None]
    if sum(sources) != 1:
        p.error("give exactly one of --report, --report-file, --cohort")
    if args.cohort is not None and args.out_dir is None:
        p.error("--cohort needs --out-dir")
    if args.cohort is None and args.out is None:
        p.error("--report/--report-file needs --out")
    args.dim, args.spacing, args.geometry_source = resolve_output_geometry(args)
    return args


def resolve_output_geometry(args):
    """`(dim_xyz, spacing_mm_xyz, where_it_came_from)` for a free-form generation.

    Unset `--dim`/`--spacing` resolve to the training bucket for `(--modality, --plane)`, not to a
    256^3 @ 1 mm cube. Three things have to agree or the conditioning is a configuration the model
    never saw: the numeric `spacing_tensor`, the `[SPACING]` marker in the text, and the grid the
    volume is actually decoded onto. Defaulting to 1 mm while training T1w AXIAL at 0.94/0.94/1.09
    mm made all three disagree at once.

    An unknown (modality, plane) lands on `FALLBACK_GEOMETRY_KEY`, whose value *is* NVIDIA's shipped
    256^3 @ 1 mm -- so the old default survives exactly where it was the right answer.
    """
    from ..data.geometry import GeometryPolicy, dhw_to_xyz

    spec = GeometryPolicy(mode="per_modality_plane").resolve(args.modality, args.plane)
    dim = list(args.dim) if args.dim else [int(v) for v in dhw_to_xyz(spec.target_shape)]
    spacing = list(args.spacing) if args.spacing else [float(v) for v in dhw_to_xyz(spec.target_spacing)]
    source = ("--dim/--spacing" if args.dim and args.spacing
              else f"({args.modality}, {args.plane}) geometry bucket")
    return dim, spacing, source


def build_sampler(args):
    from ..conditioning import ConditioningConfig
    from ..models.adapter import load_adapter_checkpoint, sha256_file
    from ..models.nvidia import (
        DEFAULT_ENV_CONFIG,
        DEFAULT_MODEL_CONFIG,
        DEFAULT_NETWORK_CONFIG,
        define_instance,
        load_autoencoder,
        load_config,
    )
    from ..models.report_conditioned_unet import build_report_conditioned_unet, load_pretrained_maisi_weights
    from ..sampling import ReportToVolumeSampler, SamplerConfig, official_latent_divisor
    from ..text import rebuild_embedder

    device = torch.device(args.device)
    network_config = args.network_config or DEFAULT_NETWORK_CONFIG

    # The adapter checkpoint is read first: it names the text encoder, the context width and the
    # adapter geometry, so nothing has to be re-specified on the command line and a mismatch is
    # impossible rather than merely unlikely.
    payload = torch.load(str(args.adapter), map_location="cpu", weights_only=False)
    stored_text = payload.get("text_encoder") or {}
    stored_config = payload.get("config") or {}
    # `rebuild_embedder` dispatches on the recorded `kind`, so all three conditioning
    # configurations round-trip. This used to be inline and always built RadBERT regardless of what
    # the checkpoint recorded.
    if args.text_encoder:
        stored_text = dict(stored_text, name=args.text_encoder)
    # `--text-checkpoint` overrides where the encoder is loaded from. Every checkpoint written by
    # `cli.train_r2v` already records an absolute path per encoder, and the encoder zoo resolves
    # the rest from MRRATE_PRETRAINED_DIR, so the flag is only for a snapshot that has moved.
    # Passing it out of habit is a live footgun: it is applied to whichever encoder the
    # configuration names, so handing a RadBERT directory to a CXR-BERT arm loads RadBERT weights
    # under the CXR-BERT slot. Warn rather than refuse -- a moved snapshot is a real use case.
    if args.text_checkpoint:
        log.warning(
            "--text-checkpoint %s overrides the encoder path this adapter recorded. It is applied "
            "to the configuration's own encoder, so make sure the snapshot really is that model.",
            args.text_checkpoint,
        )
    embedder = rebuild_embedder(
        stored_text,
        conditioning_name=stored_config.get("conditioning_name"),
        checkpoint=str(args.text_checkpoint) if args.text_checkpoint else None,
        max_length=args.max_report_tokens or None,
    )
    if int(stored_text.get("output_dim", embedder.output_dim)) != embedder.output_dim:
        raise SystemExit(
            f"text encoder width {embedder.output_dim} != the {stored_text.get('output_dim')} the "
            "adapter's projection head was built for"
        )

    unet = build_report_conditioned_unet(
        context_dim=embedder.output_dim,
        network_config=network_config,
        cross_attention_dim=int(stored_config.get("cross_attention_dim", 512)),
        conditioning_levels=stored_config.get("conditioning_levels"),
        condition_mid=bool(stored_config.get("condition_mid", True)),
        use_flash_attention=device.type == "cuda",
    ).to(device)
    base_report = load_pretrained_maisi_weights(unet, args.base_checkpoint)
    log.info("base checkpoint:\n%s", base_report.format())
    load_adapter_checkpoint(
        args.adapter, unet,
        base_checkpoint_sha256=sha256_file(args.base_checkpoint),
        allow_base_mismatch=args.allow_base_mismatch,
        # The live embedder's identity, so `assert_conditioning_compatible` actually runs. Omitting
        # it -- which this call used to do -- skips the check by design, and the check is the only
        # thing that catches an encoder swap between two configurations of equal width: a 768x1
        # `cxr_bert_cls` adapter loads cleanly onto a RadBERT embedder and generates nonsense.
        text_encoder=embedder.identity,
    )
    unet.eval()

    cfg_args = load_config(str(DEFAULT_ENV_CONFIG), str(DEFAULT_MODEL_CONFIG), str(network_config))
    # The output-size / latent-size ratio, computed the official way from the UNet's depth. Not the
    # autoencoder's padding divisor -- see `sampling.official_latent_divisor`.
    divisor = official_latent_divisor(cfg_args.diffusion_unet_def["num_channels"])
    log.info("latent divisor = %d (dim // %d latent voxels per axis)", divisor, divisor)

    autoencoder = None
    if not args.latent_only:
        autoencoder, _cfg, _pad_divisor = load_autoencoder(
            args.vae_checkpoint, DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, network_config, device=str(device)
        )

    noise_scheduler = define_instance(cfg_args, "noise_scheduler")
    scale_factor = payload.get("scale_factor")
    if scale_factor is None:
        from ..training import resolve_scale_factor

        scale_factor = resolve_scale_factor(args.base_checkpoint, "auto")
    log.info("scale_factor = %.6f", scale_factor)

    sampler = ReportToVolumeSampler(
        unet=unet, autoencoder=autoencoder, text_embedder=embedder, noise_scheduler=noise_scheduler,
        scale_factor=scale_factor, divisor=divisor, device=device,
        conditioning=ConditioningConfig(
            report_guidance_scale=args.report_guidance_scale,
            modality_guidance_scale=args.modality_guidance_scale,
        ),
        sampler_config=SamplerConfig(
            num_inference_steps=args.num_inference_steps, random_seed=args.seed,
            batched_guidance=args.batched_guidance,
        ),
    )
    return sampler, embedder, payload


def manifest_for(args, sampler, embedder, payload, extra: dict) -> dict:
    from ..models.adapter import sha256_file

    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "modality": args.modality,
        "plane": args.plane,
        "dim_xyz": list(args.dim),
        "spacing_mm_xyz": list(args.spacing),
        "geometry_source": args.geometry_source,
        "report_formats_trained": list(trained_report_formats(payload)),
        "guidance": {
            "report_guidance_scale": args.report_guidance_scale,
            "modality_guidance_scale": args.modality_guidance_scale,
            "batched": args.batched_guidance,
        },
        "sampler": {"num_inference_steps": args.num_inference_steps,
                    "scheduler": type(sampler.noise_scheduler).__name__,
                    "scale_factor": sampler.scale_factor, "divisor": sampler.divisor},
        "checkpoints": {
            "base": {"path": str(args.base_checkpoint), "sha256": sha256_file(args.base_checkpoint)},
            "adapter": {"path": str(args.adapter), "step": payload.get("step")},
            "vae": str(args.vae_checkpoint),
        },
        "text_encoder": embedder.identity,
        **extra,
    }


def trained_report_formats(payload: dict) -> tuple:
    """Every format name the adapter was trained on. One name normally; several for a run trained
    with a sampled spec (`--report-format a,b`). `()` if the checkpoint records none."""
    stored = (payload.get("config") or {}).get("report_format")
    if not stored:
        return ()
    from ..textenc.formats import parse_format_spec

    return parse_format_spec(stored)


def conditioning_text_for(report_text: str, payload: dict, modality, plane, spacing_mm_xyz):
    """Raw report text -> the string the adapter was trained to be given, plus a note for the log.

    This exists for the free-form path only (`--report` / `--report-file`, i.e. the challenge's
    shape: a report arrives with no volume attached). A metadata format's `[MODALITY]/[PLANE]/
    [SPACING]` prefix is part of what the model learned, so text without it is out of distribution
    -- silently, since generation still succeeds and only the output is worse. Cohort text needs
    nothing here: a cohort is *composed* under its own recorded format, and
    `assert_report_format_matches` refuses it when that is not one of the trained ones.

    Only the prefix is added. The section markers cannot be reconstructed from arbitrary prose, so
    a free-form report is conditioned as-is below the prefix -- which is the honest representation
    of a challenge request whose sectioning is unknown.
    """
    from ..textenc.formats import METADATA_DEPENDENT_FORMATS, meta_prefix_for

    formats = trained_report_formats(payload)
    if not any(name in METADATA_DEPENDENT_FORMATS for name in formats):
        return report_text, None
    prefix = meta_prefix_for(modality, plane, spacing_mm_xyz)
    return f"{prefix}\n{report_text}", prefix


def assert_report_format_matches(payload: dict, cohort, allow_mismatch: bool = False,
                                 embedder=None) -> None:
    """The report text a cohort stores must have been composed the way the adapter was trained.

    A cohort's `reports.json` holds *already-composed* conditioning text, produced at preprocess
    time under that cohort's `report_format`. So this is a metadata comparison, not a re-formatting
    step -- re-composing here is impossible anyway, because the section boundaries are gone by the
    time the text reaches `reports.json`.

    Getting this wrong is silent: an adapter trained on `impression_findings` sampled on
    `findings_impression` text sees the same words in the other order, produces plausible volumes,
    and just scores worse. That is exactly the class of bug this repo's cohort contract exists to
    make impossible, so the default is to refuse.
    """
    # A sectioned-fusion configuration never sees the joined string: it is handed
    # `report_sections.json` and encodes each section on its own tokenizer. So the format that
    # string was composed under is not an input to it, and comparing formats here would refuse
    # every cohort -- configuration D records `report_format=None` by construction, which matches
    # no cohort. What D actually needs is checked separately, by the caller, before any sampling:
    # that the cohort HAS sections at all.
    if embedder is not None and getattr(embedder, "needs_sections", False):
        log.info("conditioning encodes report sections separately; the cohort's report_format is "
                 "not an input to it, so no format check applies")
        return
    trained = (payload.get("config") or {}).get("report_format")
    # `cohort.spec` is the parsed cohort.json, a **dict**. This used to read
    # `cohort.spec.geometry_fingerprint` -- an attribute a dict never has -- so `built` was
    # unconditionally None and the check compared nothing: it passed silently whenever the adapter
    # also recorded no format, and refused *every* cohort whenever it recorded one. The four final
    # adapters all record one, so A, B and C could not have predicted against any cohort at all.
    geometry = cohort.geometry if hasattr(cohort, "geometry") else {}
    if not isinstance(geometry, dict):
        geometry = {}
    built = geometry.get("report_format")
    if trained == built:
        return
    # A run trained on a sampled spec ("a,b") saw both formats, so a cohort composed with either one
    # is in distribution and is accepted. This is the one relaxation the contract allows, and it is
    # not a loophole: the check still refuses any format the model was never trained on.
    trained_names = trained_report_formats(payload)
    if built and built in trained_names:
        return
    message = (
        f"report-format mismatch: the adapter was trained on report_format={trained!r} but this "
        f"cohort's text was composed with report_format={built!r}. The conditioning strings differ, "
        f"which is silent at generation time and only shows up as a worse score.\n"
        f"  Rebuild the cohort with `cli.preprocess --report-format "
        f"{trained_names[0] if trained_names else trained}`, or pass "
        f"--allow-report-format-mismatch for a deliberate cross-format ablation."
    )
    if allow_mismatch:
        log.warning("%s (continuing: --allow-report-format-mismatch)", message)
    else:
        raise SystemExit(message)


def main(argv=None) -> int:
    args = parse_args(argv)
    sampler, embedder, payload = build_sampler(args)

    if args.cohort is not None:
        from ..cohort import Cohort

        cohort = Cohort(args.cohort)
        assert_report_format_matches(payload, cohort,
                                     allow_mismatch=args.allow_report_format_mismatch,
                                     embedder=embedder)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # Same up-front check `cli.predict_r2v` makes: a sectioned-fusion configuration needs the
        # cohort's unjoined per-section text, and finding that out on case 1 of 2,000 is wasteful.
        needs_sections = bool(getattr(embedder, "needs_sections", False))
        if needs_sections and not cohort.has_report_sections:
            raise SystemExit(
                f"this adapter encodes report sections separately, but {args.cohort} has no "
                f"report_sections.json. Rebuild the cohort with `cli.preprocess`."
            )
        written = []
        for case in cohort.cases:
            report_text = cohort.load_report(case.case_id)
            if not report_text:
                raise SystemExit(
                    f"cohort case {case.case_id} carries no report text; regenerate the cohort with "
                    "reports, or use --report/--report-file"
                )
            path = args.out_dir / f"{case.case_id}.nii.gz"
            # The *case's* modality, not `--modality`: a cohort spans four sequences, and
            # conditioning an SWI case on T1w is a wrong class label that still generates fine.
            # `--modality` remains the flag for the free-form path, where there is no case to ask.
            # `with_acquisition_section` is additive: configuration D ignores the extra key, and
            # configuration E needs it because a cohort's stored sections are report text only --
            # the acquisition token is composed from the case's own modality/plane/spacing, exactly
            # as the Dataset composed it during training.
            sections = None
            if needs_sections:
                from ..textenc.formats import with_acquisition_section

                sections = with_acquisition_section(
                    cohort.load_report_sections(case.case_id),
                    case.sequence, case.acquisition_plane, case.spacing_mm,
                )
            volume = sampler.generate(
                report_text, tuple(case.shape), tuple(case.spacing_mm),
                seed=args.seed, modality=case.sequence, report_sections=sections,
            )
            from ..sampling import save_volume

            save_volume(volume, case.spacing_mm, path)
            written.append(str(path))
            log.info("wrote %s %s", path, volume.shape)
        (args.out_dir / "generation_manifest.json").write_text(json.dumps(
            manifest_for(args, sampler, embedder, payload, {"cohort": str(args.cohort), "outputs": written}),
            indent=2, default=str))
        return 0

    report_text = args.report if args.report is not None else args.report_file.read_text()
    log.info("geometry: dim=%s spacing_mm=%s [X,Y,Z] (%s)", args.dim, args.spacing, args.geometry_source)
    report_text, prefix = conditioning_text_for(report_text, payload, args.modality, args.plane,
                                                args.spacing)
    if prefix:
        log.info("adapter trained on a metadata format; prepending %r", prefix)
    log.info("report: %d characters", len(report_text))
    # A free-form report has no section boundaries to recover, so a sectioned configuration gets the
    # whole string as its findings token -- the same routing `SectionedFusionEmbedder.encode` falls
    # back to, done explicitly so the acquisition token can be filled in alongside it. Without it,
    # configuration E would generate from a masked-out metadata token it never trained with one.
    sections = None
    if getattr(embedder, "needs_sections", False):
        from ..textenc.formats import with_acquisition_section

        sections = with_acquisition_section({"findings": report_text}, args.modality, args.plane,
                                            args.spacing)
    if args.latent_only:
        latent = sampler.sample_latent(report_text, args.modality, tuple(args.dim), tuple(args.spacing),
                                        seed=args.seed, report_sections=sections)
        log.info("latent %s finite=%s", tuple(latent.shape), bool(torch.isfinite(latent).all()))
        return 0

    volume = sampler.generate(report_text, tuple(args.dim), tuple(args.spacing), seed=args.seed,
                              modality=args.modality, report_sections=sections)
    from ..sampling import save_volume

    save_volume(volume, args.spacing, args.out)
    log.info("wrote %s %s dtype=%s range=[%d, %d]", args.out, volume.shape, volume.dtype,
             int(volume.min()), int(volume.max()))
    Path(str(args.out) + ".manifest.json").write_text(json.dumps(
        manifest_for(args, sampler, embedder, payload,
                     {"report_characters": len(report_text), "output": str(args.out)}),
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
