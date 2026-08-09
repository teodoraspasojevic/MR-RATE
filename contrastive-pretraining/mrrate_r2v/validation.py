"""Step-based validation for report-adapter training. One evaluation path, one fixed sample set.

This is **conditional** report-to-volume generation: for every validation case,
`generated_i = G(report_i)` with its own `ground_truth_i` held alongside. Every metric below is
computed on report-conditioned generations -- never on unconditional noise -- and
`assert_conditioning_active` refuses to run if the guidance settings would make them report-blind.

    PRIMARY
    val/fvd         Frechet distance, GT volumes vs report-conditioned generations   lower better
    val/fid_2p5d    volume-weighted 2.5D FID, same two sets                          lower better

    DIAGNOSTIC (not headline curves)
    val/ssim                        paired 3D SSIM(generated_i, ground_truth_i)      higher better
    val/sensitivity/*               does the report change the generation at all
    val/*/rank_level                whether N supports the Frechet estimate

**What is measured, and the gap that is not.** `val/fvd` and `val/fid_2p5d` are *marginal*
distribution metrics: they ask whether the **set** of report-conditioned generations resembles the
**set** of real volumes. They do **not** verify that generation *i* matches report *i* -- a model
that produced a well-distributed set of volumes paired to the wrong reports would score identically.
`val/ssim` is paired but *structural*: it asks whether the generation resembles that patient's
anatomy, not whether it is semantically faithful to the report.

**So report-volume semantic fidelity is currently NOT measured here, deliberately.** Doing it
properly needs a frozen, independent, validated cross-modal MRI report-volume model. The one
defensible candidate found is HLIP (`zch0414/clip-vit_base-scan_study-dualdinotxt1568`,
arXiv:2505.21862, MIT) -- notably trained on MR-RATE's *training* split, so this project's `val`
split is unseen -- but it is not adopted here and no substitute is faked in its place.
`AlignmentMetric` is the seam it would plug into. Until then the honest statement is that
report-volume fidelity is unmeasured, and `val/sensitivity/*` is the weaker structural stand-in:
it establishes that the text is *used*, not that it is *honoured*.

Definitions, provenance and limitations: `eval/validation_metrics.py`, `eval/video_features.py`.

**One intensity space, asserted not assumed.** Everything runs in the model-input percentile space
(`video_features.METRIC_INTENSITY_SPACE`), so a generation must be the **decoder's float output**,
not `sampling.postprocess_mr`'s int16 [0, 1000]. `build_validation_runner` in
`cli/train_r2v.py` is what guarantees that; `_check_intensity_space` here is what catches it if
someone changes the sampler.

**On SSIM's interpretation.** A stochastic generator conditioned on a report is not trying to
reproduce that patient's anatomy, and most of a brain is never mentioned in any report -- so this
SSIM is low in absolute terms by construction and is a *training-progress* signal, not a quality
score. `eval/tasks.py` already records the same thing for the `report2volume` task ("expect much
weaker" paired metrics). Reference ceilings (identity, preprocessing round trip, autoencoder
reconstruction) come from `cli.validation_reference` and say what "good" would even look like.

**Distributed behaviour.** Cases are sharded `index % world_size`, so no case is generated twice;
per-case feature vectors and SSIM scores are gathered with `all_gather_object` and every metric is
computed once, from the union, giving results identical to a single process within float tolerance.
Volumes are released as soon as their features are taken, so peak memory is one volume per rank.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch

log = logging.getLogger("mrrate_r2v.validation")

METRIC_NAMES = ("fvd", "fid_2p5d", "ssim")

#: The only validation metrics that become W&B *curves*. Everything else a pass produces -- per
#: plane, per bucket, rank flags, timings, counts, the sensitivity diagnostic -- stays in the
#: returned payload and in `train_summary.json`, and is deliberately kept out of the dashboard: 47
#: curves is not a dashboard. `training.MRRateAdapterTrainer.validate_now` is what applies this.
HEADLINE_METRICS = ("fvd", "fid_2p5d", "ssim")


# --------------------------------------------------------------------------- config


@dataclass
class ValidationConfig:
    """When validation runs, on what, and which metrics.

    `every_steps` counts **optimizer** steps, never micro-steps, so the number means the same thing
    under any `grad_accumulation_steps`.

    `n_quick` is the fixed deterministic subset used at every validation step. It defaults to 64
    rather than something smaller because a Frechet distance below
    `validation_metrics.MIN_FRECHET_SAMPLES` (16) is withheld, and 64 leaves room for per-bucket
    breakdowns. `n_full` is the larger occasional set; both draw from the *same* seeded
    permutation, so the quick set is a prefix of the full set and the two curves measure the same
    population rather than two different ones.
    """

    every_steps: Optional[int] = None
    at_end: bool = True
    n_quick: int = 64
    n_full: Optional[int] = None
    full_every_steps: Optional[int] = None
    seed: int = 0
    num_inference_steps: int = 30
    n_visualize: int = 4
    sequence_frames: int = 16
    enabled_metrics: tuple = METRIC_NAMES
    #: Condition-sensitivity diagnostic (see `ValidationRunner.condition_sensitivity`). Runs every
    #: `sensitivity_every_steps` *optimizer* steps on `n_sensitivity` cases -- deliberately a small
    #: subset, not the whole validation set, because it costs a second generation per case.
    sensitivity_every_steps: Optional[int] = None
    n_sensitivity: int = 8
    #: Where real features are cached between validation steps and between runs. Real volumes never
    #: change, so this is computed once per (case set, extractor) and reused.
    feature_cache_dir: Optional[str] = None
    #: JSON from `cli.validation_reference` -- cached constants (real-vs-real noise floors, SSIM
    #: ceilings) logged as horizontal reference lines, never recomputed during training.
    reference_path: Optional[str] = None

    def __post_init__(self) -> None:
        unknown = set(self.enabled_metrics) - set(METRIC_NAMES)
        if unknown:
            raise ValueError(f"unknown validation metrics {sorted(unknown)}. Choose from: {METRIC_NAMES}")
        if self.n_quick < 2:
            raise ValueError("n_quick must be >= 2")
        if self.n_full is not None and self.n_full < self.n_quick:
            raise ValueError(
                f"n_full ({self.n_full}) must be >= n_quick ({self.n_quick}) -- the quick set is a "
                "prefix of the full set so the two curves stay comparable"
            )
        if self.n_visualize > self.n_quick:
            raise ValueError("n_visualize must be <= n_quick: visualised cases come from the quick set")
        # Only binding when the diagnostic actually runs: `n_sensitivity` keeps its default on a
        # config with a deliberately tiny `n_quick` (a unit test, a smoke run) as long as
        # `sensitivity_every_steps` is unset.
        if self.sensitivity_every_steps:
            if self.n_sensitivity > self.n_quick:
                raise ValueError(
                    f"n_sensitivity ({self.n_sensitivity}) must be <= n_quick ({self.n_quick}): the "
                    "diagnostic reuses quick-set cases"
                )
            if self.n_sensitivity < 2:
                raise ValueError("n_sensitivity must be >= 2: one report has nothing to swap with")


class AlignmentMetric(Protocol):
    """**Unused seam**, kept deliberately.

    None of FVD, 2.5D FID or SSIM measures report-to-volume semantic agreement. If a validated MRI
    image-text model becomes available, implement this and pass it as
    `ValidationRunner(alignment=...)`; it will be logged under its own `name`, which must make its
    status obvious (prefix `proxy_` if it is one). Nothing else in this module pretends to fill the
    gap.
    """

    @property
    def name(self) -> str: ...

    def score(self, real_features: np.ndarray, generated_features: np.ndarray,
              reports: Sequence[str]) -> dict: ...


# --------------------------------------------------------------------------- case selection


@dataclass
class ValidationCase:
    """One fixed validation sample. `case_id` is a hash -- `study_key`/`series_key` never leave
    this object and are never logged or written into results."""

    index: int
    case_id: str
    report_text: str
    report_sections: dict
    modality: str
    plane: str
    shape_xyz: tuple
    spacing_xyz: tuple
    #: Hash of the study key. Needed so the report-shuffle diagnostic can require a report from a
    #: *different study*: MR-RATE reports are study-level and one report is shared by a median of
    #: 6-7 series, so a naive permutation would routinely pair a case with its own report and
    #: measure nothing. Hashed, so no identifier is retained even in memory.
    study_hash: str = ""
    target: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def bucket(self) -> str:
        return f"{self.modality}_{self.plane}"


def case_id_for(study_key: str, series_key: str) -> str:
    return hashlib.sha256(f"{study_key}|{series_key}".encode()).hexdigest()[:16]


def study_hash_for(study_key: str) -> str:
    return hashlib.sha256(str(study_key).encode()).hexdigest()[:16]


def select_validation_cases(dataset, config: ValidationConfig, n: Optional[int] = None) -> list[int]:
    """A fixed, seeded, bucket-stratified list of dataset indices.

    Stratified by (modality, plane) so a 64-case set is not accidentally all axial T1w -- which
    would make every metric track one bucket and call it the model. Round-robin over buckets in
    sorted order, so the result is **prefix-stable**: `select(n=64)` is exactly the first 64 of
    `select(n=256)`.

    Deterministic in `config.seed` alone, so the curve compares like with like across runs, across
    resumes, and across encoder configurations.

    Study-level separation is inherited, not re-derived: this draws from the `val` split, and
    MR-RATE's splits are patient-isolated (0 violations in the release's own
    `patient_split_isolation` check), so no study can appear in both train and val.
    """
    n = n or config.n_quick
    by_bucket: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        key = f"{sample.modality or 'unknown'}_{sample.plane or 'unknown'}"
        by_bucket.setdefault(key, []).append(index)

    rng = random.Random(config.seed)
    for indices in by_bucket.values():
        rng.shuffle(indices)

    ordered: list[int] = []
    buckets = [by_bucket[key] for key in sorted(by_bucket)]
    position = 0
    while len(ordered) < sum(len(b) for b in buckets):
        progressed = False
        for bucket in buckets:
            if position < len(bucket):
                ordered.append(bucket[position])
                progressed = True
        if not progressed:
            break
        position += 1
    if len(ordered) < n:
        log.warning("validation asked for %d cases but the split only has %d", n, len(ordered))
    return ordered[:n]


# --------------------------------------------------------------------------- distributed helpers


def _dist():
    import torch.distributed as dist

    return dist if dist.is_available() and dist.is_initialized() else None


def world_size() -> int:
    d = _dist()
    return d.get_world_size() if d else 1


def rank() -> int:
    d = _dist()
    return d.get_rank() if d else 0


def gather_objects(payload: list) -> list:
    """Union of every rank's list, on every rank. A no-op single-GPU.

    Sorted by case index afterwards by the caller, so the gathered order -- and therefore the
    feature matrix row order and the Frechet distance -- is identical to a single-process run
    regardless of world size.
    """
    d = _dist()
    if d is None:
        return payload
    buckets = [None] * d.get_world_size()
    d.all_gather_object(buckets, payload)
    return [item for bucket in buckets if bucket for item in bucket]


# --------------------------------------------------------------------------- runner


class ValidationRunner:
    """Generates the fixed validation subset and returns the three metrics.

    Constructed once and reused, so the real-feature cache, the case list and the extractors are
    built once. Holds no reference to the trainer: `run(trainer, step, full)` takes it.
    """

    def __init__(
        self,
        dataset,
        sampler_factory,
        sequence_extractor=None,
        inception_extractor=None,
        alignment: Optional[AlignmentMetric] = None,
        config: Optional[ValidationConfig] = None,
        wandb_run=None,
        output_dir: Optional[Path] = None,
    ) -> None:
        from .eval.video_features import PLANE_AXES

        self.dataset = dataset
        self.sampler_factory = sampler_factory
        self.sequence_extractor = sequence_extractor
        self.inception_extractor = inception_extractor
        self.alignment = alignment
        self.config = config or ValidationConfig()
        self.wandb_run = wandb_run
        self.output_dir = Path(output_dir) if output_dir else None
        self.planes = tuple(name for name, _ in PLANE_AXES)
        self._quick_indices = select_validation_cases(dataset, self.config, self.config.n_quick)
        self._full_indices = (
            select_validation_cases(dataset, self.config, self.config.n_full)
            if self.config.n_full else self._quick_indices
        )
        self._cases: dict[int, ValidationCase] = {}
        #: index -> {"fvd": {plane: vec}, "fid": {plane: vec}}. Real volumes never change, so this
        #: is filled on the first pass and reused for the rest of the run.
        self._real_features: dict[int, dict] = {}
        self.reference = self._load_reference()
        #: Whether panels are rendered at all. Must NOT key off `self.wandb_run`: under DDP that is
        #: None on every non-zero rank, and those ranks are exactly the ones that need to render the
        #: panels rank 0 will log. `n_visualize > 0` is identical on every rank, so no broadcast.
        self._wants_panels = self.config.n_visualize > 0
        #: Incremented once per pass, so a panel can say which validation produced it. Distinct
        #: from the optimizer step: two passes can share a step (an interval pass and the
        #: end-of-training pass both fire at the last step).
        self._validation_index = 0
        self._checked_intensity = False
        log.info("validation: %d quick / %d full cases over %d buckets (seed %d), metrics %s",
                 len(self._quick_indices), len(self._full_indices),
                 len({self._bucket_of(i) for i in self._full_indices}), self.config.seed,
                 list(self.config.enabled_metrics))
        self._warn_about_sample_adequacy()

    def _warn_about_sample_adequacy(self) -> None:
        """Say up front whether the configured N can support the requested metrics.

        **Measured, not asserted.** On 512-d features (r3d_18's width) with a known-zero ground
        truth -- two disjoint halves of one population, so the true distance is 0 -- the observed
        Frechet distance is:

            N=16   21576        N=128   3692        N=1024   432
            N=32   14189        N=256   1908
            N=64    6123        N=512    892

        while a genuine, substantial distributional difference (a +0.5 shift of every feature)
        registers **128 at every N**. So at N=64 the sample-size bias is ~48x a real effect, and it
        still exceeds it at N=1024. The conventional FID guidance (N >= 2048) is not conservatism.

        Consequence, and the reason this warning exists rather than a silent number: **SSIM is the
        metric a frequent validation pass can actually support** -- it is paired and per-case, so its
        standard error shrinks like 1/sqrt(N) and it is meaningful at N=32-64. FVD and 2.5D FID need
        hundreds to thousands of volumes, and each validation volume is a full diffusion sampling
        run. Use them on the occasional `--validate-full-every-steps` pass at a large
        `--val-full-samples`, or offline via `cli.evaluate` over a real cohort (which is what this
        repository already computes distribution metrics on, at ~2000 cases).
        """
        distribution_metrics = [m for m in self.config.enabled_metrics if m in ("fvd", "fid_2p5d")]
        if not distribution_metrics:
            return
        dims = []
        if self.sequence_extractor is not None and "fvd" in distribution_metrics:
            dims.append(("fvd", int(getattr(self.sequence_extractor, "feature_dim", 512))))
        if self.inception_extractor is not None and "fid_2p5d" in distribution_metrics:
            dims.append(("fid_2p5d", 2048))
        for name, dim in dims:
            n = self.config.n_quick
            if n < dim:
                # The figure this warning used to quote ("~6100 at N=64 on 512-d features") was
                # wrong by more than two orders of magnitude, and it mattered: it declared a usable
                # curve unusable. `cli.validation_reference` at N=64 seed 0 on the val split
                # measures the real-vs-real floor (true answer 0) at **FVD 30.0 / 2.5D FID 21.1**,
                # against model scores of 43-58 in the same setup -- a real gap, not noise. The
                # covariance is still rank-deficient, so a value is comparable only against others
                # at the same N with the same extractor; that is a narrower claim than "unusable".
                log.warning(
                    "%s is enabled with --val-quick-samples %d against a %d-d feature, so the "
                    "covariance is rank-deficient and the absolute value is not calibrated. "
                    "Compare it only against other values at the same N with the same extractor, "
                    "and against the real-vs-real floor from cli.validation_reference (measured "
                    "FVD 30.0 / FID 21.1 at N=64). For a calibrated number use --val-full-samples "
                    ">= %d or cli.evaluate offline.",
                    name, n, dim, dim,
                )

    # -- setup -------------------------------------------------------------------------

    def _load_reference(self) -> dict:
        path = self.config.reference_path
        if not path or not Path(path).is_file():
            if path:
                log.warning("reference values not found at %s -- run "
                            "`python -m mrrate_r2v.cli.validation_reference` to produce them", path)
            return {}
        reference = json.loads(Path(path).read_text())
        log.info("loaded validation reference values from %s: %s", path,
                 sorted(k for k in reference.get("reference", {})))
        return reference

    def _bucket_of(self, index: int) -> str:
        sample = self.dataset.samples[index]
        return f"{sample.modality or 'unknown'}_{sample.plane or 'unknown'}"

    def _feature_cache_path(self) -> Optional[Path]:
        if not self.config.feature_cache_dir:
            return None
        extractor = getattr(self.sequence_extractor, "name", "none")
        key = hashlib.sha256(json.dumps({
            "indices": self._full_indices, "extractor": extractor,
            "frames": self.config.sequence_frames, "seed": self.config.seed,
        }, sort_keys=True).encode()).hexdigest()[:16]
        directory = Path(self.config.feature_cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"real_features_{extractor}_{key}.npz"

    def _check_intensity_space(self, volume: np.ndarray) -> None:
        """Catch a postprocessed generation being compared against a normalised ground truth.

        `postprocess_mr` scales the decoder's [0, 1] output to [0, 1000] and casts to int16, so the
        symptom is a 1000x intensity offset -- which every metric here would happily consume and
        return a plausible number for. Warned once, loudly, rather than silently producing a
        garbage curve for a whole run.
        """
        if self._checked_intensity:
            return
        self._checked_intensity = True
        peak = float(np.nanmax(np.abs(volume))) if volume.size else 0.0
        if peak > 20.0:
            log.error(
                "validation received a generated volume with max |intensity| = %.1f, but the "
                "metrics expect the model-input percentile space (~[0, 1]). This looks like "
                "sampling.postprocess_mr output (int16 [0, 1000]). Every metric will be "
                "meaningless. Use sampler.decode(...) output, not generate(...).", peak,
            )

    # -- conditioning is active, checked not assumed -----------------------------------

    def assert_conditioning_active(self, trainer) -> dict:
        """Refuse to report a distribution metric on generations that ignored their report.

        FVD and 2.5D FID compare *sets*, so an unconditional generator can score well on both. That
        makes "was conditioning actually on?" unverifiable from the metric value itself, and it has
        to be checked structurally instead. Two live ways it can be off:

        1. **`report_guidance_scale == 0`.** `conditioning.GuidanceBranches.resolve` sets
           `use_modality_report = report_guidance_scale != 0`, so at 0 the report branch is never
           run and `guided_model_output` returns the purely modality-conditioned prediction. The
           run would look normal and every volume would be report-blind.
        2. **`report_dropout_probability >= 1`**, which replaces every report with the learned null
           embedding.

        Raises rather than warns: a whole validation curve computed report-blind is worse than a
        failed run, because it looks like a result.
        """
        config = getattr(trainer, "config", None)
        conditioning = getattr(config, "conditioning", None)
        if conditioning is None:
            return {"checked": False, "reason": "trainer exposes no ConditioningConfig"}
        report_scale = float(getattr(conditioning, "report_guidance_scale", 0.0))
        dropout = float(getattr(conditioning, "report_dropout_probability", 0.0))
        if report_scale == 0.0:
            raise ValueError(
                "validation would generate report-BLIND volumes: report_guidance_scale == 0, so "
                "guided_model_output never evaluates the report branch. FVD and 2.5D FID are "
                "marginal metrics and would not reveal this. Set --report-guidance-scale > 0."
            )
        if dropout >= 1.0:
            raise ValueError(
                "validation would generate report-blind volumes: report_dropout_probability >= 1 "
                "replaces every report with the null embedding."
            )
        return {"checked": True, "report_guidance_scale": report_scale,
                "report_dropout_probability": dropout}

    # -- cases -------------------------------------------------------------------------

    def case(self, index: int) -> ValidationCase:
        if index in self._cases:
            return self._cases[index]
        item = self.dataset[index]
        case = ValidationCase(
            index=index,
            case_id=case_id_for(item["study_key"], item["series_key"]),
            report_text=item["report_text"],
            report_sections=dict(item.get("report_sections_text") or {}),
            modality=item["modality"],
            plane=item["acquisition_plane"],
            shape_xyz=tuple(int(v) for v in item["target_shape"].tolist()),
            spacing_xyz=tuple(float(v) for v in item["target_spacing_mm"].tolist()),
            study_hash=study_hash_for(item["study_key"]),
            # `.float()` before `.numpy()`: R2VDatasetConfig.dtype defaults to bfloat16, which numpy
            # cannot represent ("unsupported ScalarType BFloat16"). Every metric here is float32/64
            # internally anyway, so this cast is required, not merely defensive.
            target=item["image"].squeeze(0).float().numpy().astype(np.float32),
        )
        self._cases[index] = case
        return case

    def _features_for(self, volume: np.ndarray) -> dict:
        """Both feature families for one volume: `{"fvd": {plane: vec}, "fid": {plane: vec}}`."""
        out: dict = {}
        if "fvd" in self.config.enabled_metrics and self.sequence_extractor is not None:
            out["fvd"] = self.sequence_extractor.extract(volume)
        if "fid_2p5d" in self.config.enabled_metrics and self.inception_extractor is not None:
            from .eval.distribution import extract_2p5d_inception_features

            out["fid"] = extract_2p5d_inception_features(volume, self.inception_extractor)
        return out

    # -- the entry point ----------------------------------------------------------------

    def run(self, trainer, step: int, full: bool = False) -> dict:
        """Generate, measure, restore. Returns a flat dict of scalars ready for W&B."""
        from .eval.validation_metrics import ssim_volume, verify_pair

        # Structural, before anything is generated: FVD and 2.5D FID are marginal metrics and would
        # score a report-blind generator just fine, so "conditioning was on" cannot be inferred from
        # them and is checked here instead.
        conditioning_state = self.assert_conditioning_active(trainer)
        self._validation_index += 1
        epoch = int(getattr(trainer, "epoch", 0)) + 1

        indices = self._full_indices if full else self._quick_indices
        mine = [i for n, i in enumerate(indices) if n % world_size() == rank()]
        started = time.time()
        timings = {"generate": 0.0, "features": 0.0, "ssim": 0.0, "sensitivity": 0.0}

        unet = getattr(trainer.unet, "module", trainer.unet)
        was_training = unet.training
        generate = self.sampler_factory(trainer)

        records = []
        try:
            unet.eval()
            with torch.inference_mode():
                for index in mine:
                    case = self.case(index)
                    clock = time.time()
                    generated = generate(case)
                    timings["generate"] += time.time() - clock
                    self._check_intensity_space(generated)

                    record = {"index": index, "case_id": case.case_id, "bucket": case.bucket}

                    clock = time.time()
                    real = self._real_features.get(index)
                    if real is None:
                        real = self._features_for(case.target)
                        self._real_features[index] = real
                    generated_features = self._features_for(generated)
                    timings["features"] += time.time() - clock
                    record["real"] = _to_lists(real)
                    record["generated"] = _to_lists(generated_features)

                    if "ssim" in self.config.enabled_metrics:
                        clock = time.time()
                        verdict = verify_pair(case.target, generated, case.spacing_xyz,
                                              case.spacing_xyz)
                        if verdict.ok:
                            record["ssim"] = ssim_volume(case.target, generated)
                        else:
                            record["ssim_excluded"] = verdict.reason
                        timings["ssim"] += time.time() - clock

                    # The panel is *rendered* on whichever rank generated the case, but only rank 0
                    # holds a W&B run. So the HTML travels through the same gather as the features
                    # and rank 0 logs it below. Rendering here and logging there is what keeps the
                    # visualised case set fixed and independent of world size -- calling
                    # `_visualize` inline would silently drop every panel whose case did not land
                    # on rank 0 (3 of 4 at world_size=4, since the visualised cases are the first
                    # `n_visualize` and sharding is `index % world_size`).
                    if index in set(self._quick_indices[: self.config.n_visualize]):
                        record["panel_html"] = self._render_panel(
                            case, generated, step, epoch=epoch,
                            validation_index=self._validation_index, full=full)
                    records.append(record)
                    del generated       # one volume at a time, never the set
        finally:
            unet.train(was_training)

        gathered = sorted(gather_objects(records), key=lambda r: r["index"])
        metrics = self._metrics(gathered)

        interval = self.config.sensitivity_every_steps
        if interval and (full or step % interval == 0):
            clock = time.time()
            metrics.update(self.condition_sensitivity(trainer, step))
            timings["sensitivity"] = time.time() - clock

        if rank() == 0:
            metrics["val/n_panels"] = self._log_panels(gathered, step)
        metrics["val/conditioning_active"] = int(bool(conditioning_state.get("checked")))
        metrics.update({f"val/time/{k}": round(v, 2) for k, v in timings.items()})
        metrics["val/n_cases"] = len(gathered)
        metrics["val/seconds"] = time.time() - started
        metrics["val/full"] = int(bool(full))
        metrics["val/validation_index"] = self._validation_index
        metrics["val/epoch"] = epoch
        metrics.update(self.reference_scalars())
        if rank() == 0:
            headline = {k: round(v, 5) for k, v in metrics.items()
                        if k in ("val/fvd", "val/fid_2p5d", "val/ssim") and isinstance(v, float)}
            log.info("validation @ optimizer step %d (%s, %d cases, %.1fs): %s", step,
                     "full" if full else "quick", len(gathered), metrics["val/seconds"], headline)
        return metrics

    def _metrics(self, records: list[dict]) -> dict:
        # `panel_html` rides along in the gathered records but is not a feature and not a metric.
        # `_metrics` only reads the keys it names, so nothing here has to strip it.
        from .eval.validation_metrics import PlaneFeatures, aggregate_frechet

        out: dict = {}
        for family, key in (("fvd", "val/fvd"), ("fid", "val/fid_2p5d")):
            usable = [r for r in records
                      if (r.get("real") or {}).get(family) and (r.get("generated") or {}).get(family)]
            if not usable:
                continue
            features = PlaneFeatures(planes=self.planes)
            for record in usable:
                features.add(record["real"][family], record["generated"][family])
            single = family == "fvd" and getattr(self.sequence_extractor, "planes", None) == "n/a"
            out.update(aggregate_frechet(features, key, single_plane=single))

        scores = [r["ssim"]["ssim"] for r in records
                  if r.get("ssim") and r["ssim"].get("ssim") is not None]
        excluded = [r["ssim_excluded"] for r in records if r.get("ssim_excluded")]
        if scores:
            out["val/ssim"] = float(np.mean(scores))
            out["val/ssim/std"] = float(np.std(scores))
            out["val/ssim/whole_volume"] = float(np.mean(
                [r["ssim"]["ssim_whole_volume"] for r in records
                 if r.get("ssim") and r["ssim"].get("ssim_whole_volume") is not None]))
            out["val/ssim/n_valid_pairs"] = len(scores)
        out["val/ssim/n_excluded"] = len(excluded)
        if excluded:
            # Reasons are logged, not just counted -- an excluded pair is a geometry bug, and the
            # count alone does not say which.
            reasons: dict = {}
            for reason in excluded:
                reasons[str(reason).split(":")[0]] = reasons.get(str(reason).split(":")[0], 0) + 1
            log.warning("SSIM excluded %d/%d pairs: %s", len(excluded), len(records), reasons)
            for name, count in reasons.items():
                out[f"val/ssim/excluded/{name}"] = count

        # Per-bucket SSIM as a diagnostic only: one aggregate is the headline, but a single large
        # bucket must not be able to hide a collapsed small one.
        for bucket in sorted({r["bucket"] for r in records}):
            values = [r["ssim"]["ssim"] for r in records
                      if r["bucket"] == bucket and r.get("ssim") and r["ssim"].get("ssim") is not None]
            if values:
                out[f"val/ssim/bucket/{bucket}"] = float(np.mean(values))
        out["val/n_buckets"] = len({r["bucket"] for r in records})
        return out

    # -- condition-sensitivity diagnostic ------------------------------------------------

    def shuffled_report_pairing(self, cases: Sequence[ValidationCase]) -> list:
        """A deterministic derangement that never pairs a case with a report from its own study.

        MR-RATE reports are **study-level** -- one report is shared by a median of 6-7 series -- so
        a plain permutation would routinely hand a case a report that is literally its own, and the
        diagnostic would measure nothing. Pairing is therefore rejected on `study_hash`, not on
        case index.

        A simple rotation by one over study-sorted order, then repaired: deterministic, no RNG, and
        every case gets exactly one donor.
        """
        order = sorted(range(len(cases)), key=lambda i: (cases[i].study_hash, cases[i].case_id))
        pairing = {}
        n = len(order)
        for position, index in enumerate(order):
            for offset in range(1, n):
                donor = order[(position + offset) % n]
                if cases[donor].study_hash != cases[index].study_hash:
                    pairing[index] = donor
                    break
        missing = [i for i in range(len(cases)) if i not in pairing]
        if missing:
            log.warning("condition-sensitivity: %d of %d cases share a study with every other "
                        "case, so no donor report exists for them; excluded",
                        len(missing), len(cases))
        return [(i, pairing[i]) for i in range(len(cases)) if i in pairing]

    def condition_sensitivity(self, trainer, step: int) -> dict:
        """Does the report change the generation, and does the *right* report help?

        A **diagnostic, not a headline metric**, and deliberately built from what already exists --
        no cross-modal model is involved, so nothing here claims to measure report-volume *semantic*
        agreement. Two questions, both answerable with a second generation per case:

        1. **Is text conditioning wired at all?** Hold the diffusion seed, modality and spacing
           fixed, swap in another study's report, and measure how much the output moves
           (`swap_relative_l1`, and `swap_ssim` between the two generations). If swapping the report
           changes nothing -- `swap_ssim` ~ 1.0 -- the model is ignoring the text and no amount of
           FVD improvement means anything.
        2. **Does the correct report help?** Compare paired SSIM against the case's own ground truth
           for the correct report vs the donor report. If conditioning carries real information,
           `ssim_correct > ssim_shuffled`. This is a *structural* comparison, not semantic
           alignment: it asks whether the right report moves the generation toward the right
           patient. Measured, never assumed -- the sign is reported, not enforced.

        Runs on `n_sensitivity` cases from the quick set, at `sensitivity_every_steps`, because it
        doubles generation cost for the cases it touches.
        """
        from .eval.validation_metrics import ssim_volume, verify_pair

        indices = self._quick_indices[: self.config.n_sensitivity]
        cases = [self.case(i) for i in indices]
        pairing = self.shuffled_report_pairing(cases)
        mine = [p for n, p in enumerate(pairing) if n % world_size() == rank()]
        generate = self.sampler_factory(trainer)

        unet = getattr(trainer.unet, "module", trainer.unet)
        was_training = unet.training
        rows = []
        try:
            unet.eval()
            with torch.inference_mode():
                for own, donor in mine:
                    case, other = cases[own], cases[donor]
                    correct = generate(case)
                    # Same case -- same seed, modality, spacing, shape -- with only the report
                    # replaced. Any difference in the output is attributable to the text alone.
                    swapped_case = replace(case, report_text=other.report_text,
                                           report_sections=other.report_sections)
                    shuffled = generate(swapped_case)

                    scale = float(np.abs(correct).mean()) or 1.0
                    row = {
                        "case_id": case.case_id,
                        "swap_relative_l1": float(np.abs(correct - shuffled).mean() / scale),
                    }
                    if verify_pair(correct, shuffled, case.spacing_xyz, case.spacing_xyz).ok:
                        row["swap_ssim"] = ssim_volume(correct, shuffled)["ssim"]
                    if verify_pair(case.target, correct, case.spacing_xyz, case.spacing_xyz).ok:
                        row["ssim_correct"] = ssim_volume(case.target, correct)["ssim"]
                        row["ssim_shuffled"] = ssim_volume(case.target, shuffled)["ssim"]
                    rows.append(row)
                    del correct, shuffled
        finally:
            unet.train(was_training)

        rows = gather_objects(rows)
        out: dict = {"val/sensitivity/n_cases": len(rows)}
        if not rows:
            return out

        def mean_of(key: str):
            values = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(values)) if values else None

        for key in ("swap_relative_l1", "swap_ssim", "ssim_correct", "ssim_shuffled"):
            value = mean_of(key)
            if value is not None:
                out[f"val/sensitivity/{key}"] = value

        swap_ssim = out.get("val/sensitivity/swap_ssim")
        if swap_ssim is not None and swap_ssim > 0.99:
            log.error(
                "CONDITION-SENSITIVITY FAILURE: swapping the report changed the generation almost "
                "not at all (SSIM between correct- and shuffled-report generations = %.5f at a "
                "fixed seed). The model is very likely ignoring its text conditioning -- check "
                "report_guidance_scale, that the adapter is actually training, and that "
                "context_drop_mask is not always True.", swap_ssim,
            )
        correct, shuffled = out.get("val/sensitivity/ssim_correct"), out.get("val/sensitivity/ssim_shuffled")
        if correct is not None and shuffled is not None:
            out["val/sensitivity/ssim_advantage"] = correct - shuffled
            if rank() == 0:
                log.info("condition sensitivity @ step %d over %d cases: swap_ssim=%.4f "
                         "ssim_correct=%.4f ssim_shuffled=%.4f advantage=%+.5f",
                         step, len(rows), swap_ssim if swap_ssim is not None else float("nan"),
                         correct, shuffled, correct - shuffled)
        return out

    def reference_scalars(self) -> dict:
        """The cached reference constants, re-emitted every validation step so W&B can draw them as
        flat lines beside the moving curves. Computed once by `cli.validation_reference`."""
        out = {}
        for name, value in (self.reference.get("reference") or {}).items():
            if isinstance(value, (int, float)):
                out[f"val/reference/{name}"] = float(value)
            elif isinstance(value, dict) and isinstance(value.get("value"), (int, float)):
                out[f"val/reference/{name}"] = float(value["value"])
        return out

    def configuration(self) -> dict:
        """Everything that defines what these numbers are, for the run config and the results."""
        from .eval.validation_metrics import MIN_FRECHET_SAMPLES, ssim_parameters

        config = {
            "metrics": list(self.config.enabled_metrics),
            "n_quick": self.config.n_quick, "n_full": self.config.n_full,
            "seed": self.config.seed, "num_inference_steps": self.config.num_inference_steps,
            "min_frechet_samples": MIN_FRECHET_SAMPLES,
            "ssim": ssim_parameters(),
            "measures_report_alignment": False,
            "note": "FVD and 2.5D FID are distribution-level; SSIM is paired. None measures "
                    "report-to-volume semantic agreement.",
        }
        if self.sequence_extractor is not None:
            config["fvd"] = self.sequence_extractor.configuration()
        if self.inception_extractor is not None:
            from .eval.distribution import feature_configuration

            config["fid_2p5d"] = feature_configuration("inception_v3_imagenet", None, 2048)
            config["fid_2p5d"]["standard"] = False
            config["fid_2p5d"]["adaptation_note"] = (
                "Volume-weighted three-plane variant: one mean-pooled Inception vector per volume "
                "per plane, then per-plane Frechet, then unweighted mean. NOT the challenge's "
                "slice-level FID, and '2.5D FID' is not a challenge or GenerateCT term."
            )
        return config

    # -- visualisation ------------------------------------------------------------------

    def _render_panel(self, case: ValidationCase, generated: np.ndarray, step: int,
                      epoch: int = 0, validation_index: int = 0, full: bool = False):
        """Render the interactive panel on the rank that generated the case. Returns HTML or None.

        Skipped entirely when no rank has a W&B run, so a `--wandb-mode disabled` run pays nothing
        for rendering panels nobody will see. Under DDP `self.wandb_run` is None on non-zero ranks,
        so `_wants_panels` is broadcast-free: it keys off the config, not off this rank's run.
        """
        if not self._wants_panels:
            return None
        try:
            from .eval.figures import validation_panel_html

            return validation_panel_html(case, generated, step, epoch=epoch,
                                         validation_index=validation_index, full=full)
        except Exception as exc:  # noqa: BLE001 -- a plot must never end a training run
            log.warning("validation panel render failed for %s: %s", case.case_id, exc)
            return None

    def _log_panels(self, records: list[dict], step: int) -> int:
        """Rank 0 logs every gathered panel, whichever rank rendered it.

        **The key stays `validation/<case_id>` -- stable across validation steps on purpose.** W&B
        keeps one media panel per key with its own step slider, so a stable key is what lets you drag
        through training and watch *the same case* evolve. Putting the step in the key instead would
        create a fresh panel per (case, step) -- 4 cases x 20 validations = 80 panels -- and destroy
        exactly the comparison the panel exists for.

        The step, epoch, validation index and pass type are instead rendered *inside* the panel (its
        heading and its metadata fields), so an individual panel is still self-describing. W&B also
        embeds the step in the stored filename (`validation/<case>_<step>_<hash>.html`), so the
        provenance is on disk too.
        """
        if self.wandb_run is None or not getattr(self.wandb_run, "enabled", False):
            return 0
        logged = 0
        for record in records:
            html = record.get("panel_html")
            if not html:
                continue
            try:
                self.wandb_run.log_html(f"validation/{record['case_id']}", html, step=step)
                logged += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("validation panel log failed for %s: %s", record["case_id"], exc)
        return logged


def _to_lists(features: dict) -> dict:
    """Feature vectors as plain lists, so `all_gather_object` can pickle them cheaply."""
    return {family: {plane: np.asarray(vector, dtype=np.float64).reshape(-1).tolist()
                     for plane, vector in planes.items()}
            for family, planes in features.items()}


__all__ = [
    "METRIC_NAMES",
    "AlignmentMetric",
    "ValidationCase",
    "ValidationConfig",
    "ValidationRunner",
    "case_id_for",
    "gather_objects",
    "rank",
    "select_validation_cases",
    "world_size",
]
