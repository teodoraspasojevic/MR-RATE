"""Live evaluation: the same Dataset training uses, generation, and metrics -- in one streaming pass.

**This replaces the cohort/prediction-set pipeline.** There is no `cli.preprocess` stage, no frozen
cohort directory, no prediction directory, and no `.npy` on disk. A test run builds the identical
`MRReportToVolumeDataset` + `GeometryBucketBatchSampler` that `cli.train_r2v` builds, from the same
manifest and the same `R2VDatasetConfig`, and diverges only where training would call
`loss.backward()` -- there it calls the sampler instead.

    cli.train_r2v      build_dataset -> DataLoader -> encode -> UNet -> backward
    cli.evaluate       build_dataset -> DataLoader -> encode -> UNet -> sample -> metrics

That shared prefix is the point. The 2026-08-10 posterior-shift divergence (training at 15 mm,
every cohort built at 0, 15.8% of cases displaced, nothing able to see it) was possible only
because the two paths preprocessed independently. They no longer can: there is one
`R2VDatasetConfig`, constructed by one function, and `run_fingerprint` records it.

**Determinism, which is what the cohort contract used to buy.**

- Case selection has **no RNG at all**. Candidates are ordered by `(study_uid, series_id)` within
  each (modality, plane) bucket, then round-robined across buckets in sorted order. Prefix-stable:
  `--n-per-bucket 50` is exactly the first 50 per bucket of a full run, so a cheap run and a full
  run measure nested populations rather than two unrelated draws.
- Sampler noise is `stable_seed(seed, case_id)` -- a function of the case, not of iteration order.
  A rerun, a `--n-per-bucket`, a different world size, or a resumed job reproduces the same volume
  for the same case, bit for bit.
- `run_fingerprint()` hashes the ordered case list together with every preprocessing setting, the
  task, and the model checkpoint. Two runs with the same fingerprint scored the same cases the same
  way; a different one is a different experiment. This is the cohort_id guarantee, computed instead
  of stored.

**Streaming, because the alternative does not fit.** A case is generated, scored, feature-extracted
and released before the next one starts, so peak memory is two volumes per rank rather than the
80 GB a 2,000-case in-memory cohort would need. Volumes are never written; only per-case metric
rows and feature vectors survive a case.

**One intensity space, asserted.** Everything -- ground truth, generations, reconstructions -- lives
in the model-input percentile space (`video_features.METRIC_INTENSITY_SPACE`, ~[0, 1]). A generator
must return `sampler.decode(...)`, never `sampling.postprocess_mr`'s int16 [0, 1000];
`assert_metric_intensity_space` refuses the whole run on the first case rather than producing 2,000
plausible-looking wrong numbers.

Metric selection still belongs to `eval/tasks.py` and nothing here overrides it: `generation` is
`paired=False` and structurally cannot acquire a voxelwise metric.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import aggregate as AGG
from . import geometry_contract as G
from . import summary_csv as SUM
from . import tasks as T
from .runner import EVALUATION_VERSION, RESULT_FILES, compute_paired_metrics, report_image_similarity

log = logging.getLogger("mrrate_r2v.eval.live")

#: Bumped when the live harness changes what a number means. Distinct from `EVALUATION_VERSION`,
#: which versions the metric definitions themselves (shared with the archived cohort path).
LIVE_EVALUATION_VERSION = "mr_rate_live_evaluation_v1"

#: A generated volume peaking above this is not in the percentile space the metrics assume. The
#: same threshold `validation.py:_check_intensity_space` uses, for the same failure: `postprocess_mr`
#: output is 1000x a normalised ground truth and every metric consumes it without complaint.
MAX_PLAUSIBLE_INTENSITY = 20.0

#: Mid-axial slices retained per bucket for the intra-set MS-SSIM diversity measure. Everything else
#: streams, but this metric is inherently pairwise within a population, so some slices must be kept.
#: 256 per bucket is ~32k pairs -- far past where the mean stabilises -- at ~64 MB per bucket. The
#: cap is logged whenever it binds; a silently truncated population reads as a complete one.
DEFAULT_DIVERSITY_SLICES_PER_BUCKET = 256


# --------------------------------------------------------------------------- cases


def case_id_for(study_key: str, series_key: str) -> str:
    """The same hash `cohort.case_id_for` used, so a case keeps its identity across the change."""
    return hashlib.sha256(f"{study_key}|{series_key}".encode("utf-8")).hexdigest()[:16]


def stable_seed(base_seed: int, case_id: str) -> int:
    """Per-case sampler seed. A function of the case, never of iteration order -- so `--n-per-bucket`,
    a rerun, a resume, or a different world size all reproduce the same volume for the same case."""
    return int(hashlib.sha256(f"{base_seed}:{case_id}".encode()).hexdigest()[:8], 16)


@dataclass(frozen=True)
class LiveCase:
    """One evaluation case. Carries everything the metric code needs and no volume.

    `shape`/`spacing_mm` are **(X, Y, Z)**, the package-boundary axis order -- the same convention
    `CohortCase` used, so `geometry_contract` and `summary_csv` are unchanged.
    """

    index: int                  # dataset index -- what `dataset[index]` returns this case
    case_id: str
    study_key: str
    series_key: str
    sequence: str               # modality: T1w / T2w / FLAIR / SWI
    acquisition_plane: str
    shape: tuple
    spacing_mm: tuple

    @property
    def bucket(self) -> str:
        from ..volumes import bucket_name

        return bucket_name(self.sequence, self.acquisition_plane)

    @property
    def study_hash(self) -> str:
        return hashlib.sha256(str(self.study_key).encode()).hexdigest()[:16]


def select_eval_cases(dataset, n_per_bucket: int | None = None) -> list:
    """Dataset indices to evaluate, in a deterministic bucket-interleaved order. **No RNG.**

    Two properties, both load-bearing:

    - **Reproducible without a seed.** Candidates within a bucket are ordered by
      `(study_uid, series_id)`, which is a property of the data rather than of manifest row order,
      dict iteration, or Python version. The same split always yields the same list.
    - **Prefix-stable.** Buckets are round-robined in sorted order, so the first `k` cases of a
      full run are exactly what `--n-per-bucket` returns for the corresponding `k`. A smoke run,
      a 200-per-bucket run and a full-split run are nested populations, not three different
      samples of one.

    `n_per_bucket=None` (the default) evaluates **every** case in the split -- which is what CTFlow
    does on CT-RATE's validation set, and what removes "which subset?" from the list of things a
    reader has to trust.
    """
    by_bucket: dict = {}
    for index, sample in enumerate(dataset.samples):
        key = (sample.modality or "unknown", sample.plane or "unknown")
        by_bucket.setdefault(key, []).append(index)

    for key in by_bucket:
        by_bucket[key].sort(
            key=lambda i: (str(dataset.samples[i].study_uid), str(dataset.samples[i].series_id))
        )
        if n_per_bucket is not None and len(by_bucket[key]) > n_per_bucket:
            by_bucket[key] = by_bucket[key][:n_per_bucket]

    ordered: list = []
    columns = [by_bucket[key] for key in sorted(by_bucket)]
    position = 0
    while True:
        progressed = False
        for column in columns:
            if position < len(column):
                ordered.append(column[position])
                progressed = True
        if not progressed:
            break
        position += 1
    return ordered


def build_cases(dataset, indices) -> list:
    """`LiveCase` per index, read off `dataset.samples` and the dataset's own geometry resolution.

    The grid comes from `dataset.geometry.resolve(modality, plane)` -- the *same* call
    `__getitem__` makes at `dataset.py:367` -- so a case's recorded shape/spacing cannot drift from
    the tensor the dataset will actually hand the model. Reading it from the manifest instead is
    what made the metadata-CSV axis-order trap possible.

    `GeometrySpec` is (D, H, W) internally and `LiveCase` is (X, Y, Z) at the package boundary, so
    the conversion goes through `dhw_to_xyz` and never by hand.
    """
    from ..data.geometry import dhw_to_xyz

    cases = []
    for index in indices:
        sample = dataset.samples[index]
        spec = dataset.geometry.resolve(sample.modality, sample.plane)
        cases.append(LiveCase(
            index=index,
            case_id=case_id_for(sample.study_uid, sample.series_id),
            study_key=sample.study_uid,
            series_key=sample.series_id,
            sequence=sample.modality or "unknown",
            acquisition_plane=sample.plane or "unknown",
            shape=tuple(int(v) for v in dhw_to_xyz(spec.target_shape)),
            spacing_mm=tuple(float(v) for v in dhw_to_xyz(spec.target_spacing)),
        ))
    return cases


# --------------------------------------------------------------------------- the cohort interface


class LiveCohortView:
    """The read-only cohort interface, over an in-memory case list instead of a directory.

    `summary_csv` and the aggregation code are written against a `Cohort`; this satisfies the same
    surface so neither had to be forked. What it deliberately does **not** provide is
    `load_volume` -- there are no stored volumes, and a caller reaching for one is a caller that
    has not been ported to streaming.
    """

    def __init__(self, cases, split: str, geometry: dict, population_bucket_counts: dict,
                 run_id: str, sequences=None) -> None:
        self.cases = list(cases)
        self.root = None
        self._split = split
        self._geometry = dict(geometry)
        self._population = dict(population_bucket_counts or {})
        self._run_id = run_id
        self._sequences = list(sequences) if sequences else sorted({c.sequence for c in self.cases})

    # -- identity ----------------------------------------------------------------------

    @property
    def cohort_id(self) -> str:
        """The run fingerprint. Named `cohort_id` because every consumer -- `summary.json`,
        `run_manifest.json`, `check_run.py`, the W&B config -- already reads that key, and the
        guarantee it carries is unchanged: equal value means the same cases at the same geometry
        under the same preprocessing."""
        return self._run_id

    @property
    def spec(self) -> dict:
        return {"split": self._split, "geometry": self._geometry,
                "population_bucket_counts": self._population}

    @property
    def geometry(self) -> dict:
        return dict(self._geometry)

    @property
    def sequences(self) -> list:
        return list(self._sequences)

    @property
    def population_bucket_counts(self) -> dict:
        return dict(self._population)

    @property
    def bucket_counts(self) -> dict:
        counts: dict = {}
        for case in self.cases:
            counts[case.bucket] = counts.get(case.bucket, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def buckets(self) -> list:
        return sorted(self.bucket_counts)

    def cases_for_bucket(self, bucket: str) -> list:
        return [c for c in self.cases if c.bucket == bucket]

    def bucket_geometry(self, bucket: str) -> dict:
        cases = self.cases_for_bucket(bucket)
        if not cases:
            return {}
        c = cases[0]
        return {"shape_xyz": list(c.shape), "spacing_mm_xyz": list(c.spacing_mm),
                "fov_mm_xyz": [round(s * p, 2) for s, p in zip(c.shape, c.spacing_mm)],
                "n": len(cases)}

    def summary(self) -> dict:
        return {"cohort_id": self.cohort_id, "split": self._split, "n_cases": len(self.cases),
                "counts": {s: sum(1 for c in self.cases if c.sequence == s) for s in self.sequences},
                "bucket_counts": self.bucket_counts, "geometry": self._geometry}

    def load_volume(self, case_id: str):
        raise NotImplementedError(
            "the live harness stores no volumes -- a volume exists only while its case is being "
            "scored. Read it inside the streaming loop, not afterwards."
        )


def run_fingerprint(*, split: str, cases, geometry: dict, task: str, n_per_bucket, seed: int,
                    model_identity: dict) -> str:
    """The comparability hash: same value, same experiment.

    Covers the ordered case list (by `(study_key, series_key)`, so it identifies *which* cases
    without carrying an identifier into any output), every preprocessing setting that changes a
    voxel, the task, the sample cap and the model checkpoint. This is `cohort_id` recomputed from
    the run instead of read from a directory -- and unlike `cohort_id` it cannot go stale, because
    there is no artifact to go stale relative to.
    """
    payload = {
        "live_evaluation_version": LIVE_EVALUATION_VERSION,
        "split": split,
        "geometry": geometry,
        "task": task,
        "n_per_bucket": n_per_bucket,
        "seed": seed,
        "model": model_identity,
        "cases": [[c.study_key, c.series_key] for c in cases],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]


def cases_fingerprint(cases) -> str:
    return hashlib.sha256(
        json.dumps([[c.study_key, c.series_key] for c in cases], separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- guards


def assert_metric_intensity_space(volume: np.ndarray, what: str = "generated volume") -> None:
    """Refuse a whole run whose produced volumes are in the wrong intensity space.

    Checked once, on the first case: the failure is silent by construction (every paired metric and
    both FIDs consume a 1000x-offset pair and return a plausible number), so it has to be caught
    structurally rather than noticed in the output.
    """
    peak = float(np.nanmax(np.abs(volume))) if volume.size else 0.0
    if peak > MAX_PLAUSIBLE_INTENSITY:
        raise SystemExit(
            f"first {what} has max |intensity| = {peak:.1f}, but every metric here assumes the "
            f"model-input percentile space (~[0, 1]) that the Dataset's ground truth lives in. "
            f"This is sampling.postprocess_mr output (int16 [0, 1000]); MAE, PSNR, SSIM and both "
            f"FIDs would all return meaningless numbers. Generate with sampler.decode(...) / "
            f"postprocess=False."
        )


def check_case_geometry(case: LiveCase, produced_shape) -> tuple:
    """`compare_geometry` between a case's grid and what the model produced.

    Kept even though the model is told the grid: a sampler that rounds a shape to a divisor, or a
    reconstruction that fails to undo its padding, silently changes the FOV. Blind resizing is the
    bug this replaced -- a mismatch is excluded with a reason.
    """
    gt = G.GeometryRecord(
        shape=tuple(int(v) for v in case.shape), axis_order=G.DATASET_AXIS_ORDER,
        anatomical_axis_meaning=G.DATASET_ANATOMICAL_AXIS_MEANING,
        spacing_mm=tuple(float(v) for v in case.spacing_mm), orientation=G.DATASET_ORIENTATION,
        affine=None, modality=case.sequence, acquisition_plane=case.acquisition_plane,
        crop_pad=None, valid_bounds=None, preprocessing_version=EVALUATION_VERSION,
        source="ground_truth", study_key=case.study_key, series_key=case.series_key,
    )
    produced = G.GeometryRecord(
        shape=tuple(int(v) for v in produced_shape), axis_order=G.DATASET_AXIS_ORDER,
        anatomical_axis_meaning=G.DATASET_ANATOMICAL_AXIS_MEANING,
        spacing_mm=tuple(float(v) for v in case.spacing_mm), orientation=G.DATASET_ORIENTATION,
        affine=None, modality=case.sequence, acquisition_plane=case.acquisition_plane,
        crop_pad=None, valid_bounds=None, preprocessing_version=EVALUATION_VERSION,
        source="prediction", study_key=case.study_key, series_key=case.series_key,
    )
    comparison = G.compare_geometry(gt, produced)
    return comparison.decision == G.GeometryDecision.STRICT_MATCH, comparison


# --------------------------------------------------------------------------- config


@dataclass
class LiveEvalConfig:
    """Everything that decides what a live evaluation computes and on what."""

    task: T.TaskSpec
    output_dir: Path
    split: str = "test"
    n_per_bucket: int | None = None       # None = the entire split
    seed: int = 42
    device: str = "cpu"
    distribution_metrics: bool = True
    medicalnet_checkpoint: Path | None = None
    fid_bootstrap: int = 30
    min_subgroup_n: int = 10
    diversity_k: int = 5
    diversity_slices_per_bucket: int = DEFAULT_DIVERSITY_SLICES_PER_BUCKET
    skip_metric_groups: tuple = ()
    save_figures: int = 3
    save_nifti_cases: int = 0
    report_classifier: Path | None = None
    report_labels_csv: Path | None = None
    report_image_model: object = None
    #: Interactive ground-truth-vs-produced panels per bucket. Rendered DURING the stream, on
    #: whichever rank generated the case, because the volumes do not exist afterwards. Gated by
    #: `wandb_log_reports` for the same reason `cli.train_r2v` gates it: a panel embeds report text.
    wandb_panels: int = 0
    wandb_log_reports: bool = False
    log_every: int = 25
    extra_run_metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- the harness


class LiveEvaluator:
    """Generate and score one case at a time; aggregate and write once at the end.

    `generate(case, sample) -> np.ndarray` is the only thing that differs between tasks:

        report2volume   sample the adapter-conditioned UNet from `sample["report_text"]`
        reconstruction  encode/decode `sample["image"]` through the frozen autoencoder
        generation      sample the base UNet from the modality label alone, report-blind

    Everything after that call is identical for all three, which is what makes the three numbers
    comparable. `eval/tasks.py` still decides which metric groups run.
    """

    def __init__(self, dataset, cases, config: LiveEvalConfig, cohort_view: LiveCohortView) -> None:
        self.dataset = dataset
        self.cases = list(cases)
        self.config = config
        self.cohort = cohort_view
        self._checked_intensity = False
        self._diversity_kept: dict = {}
        self._diversity_dropped: dict = {}
        self._panel_count: dict = {}
        #: [{case_id, bucket, html}] -- filled by `run`, read by the CLI after it returns. Not part
        #: of `summary.json`: a panel is ~1 MB of base64 and the summary is meant to be readable.
        self.panels: list = []

    def _render_panel(self, case: LiveCase, sample, real, produced):
        """Panel HTML for one case, or None. Rendered inline because the volumes are released as
        soon as the case is scored -- there is no later pass that could go back and read them."""
        wanted = self.config.wandb_panels
        if wanted <= 0 or not self.config.wandb_log_reports or real is None:
            return None
        n = self._panel_count.get(case.bucket, 0)
        if n >= wanted:
            return None
        self._panel_count[case.bucket] = n + 1
        try:
            from .figures import validation_panel_html
            from .wandb_evaluation import _PanelCase

            return validation_panel_html(
                _PanelCase(case, real, sample.get("report_text", ""),
                           dict(sample.get("report_sections_text") or {})),
                generated=produced, step=0, epoch=0, validation_index=0, full=True,
            )
        except Exception as exc:  # noqa: BLE001 -- a panel is never worth an evaluation
            log.warning("could not render panel for %s: %s", case.case_id, exc)
            return None

    # -- per-case work -----------------------------------------------------------------

    def _extract_features(self, case: LiveCase, real, produced, medicalnet, inception):
        """One `CaseFeatures` for this case. Both populations, so `report_consistency` gets the
        real-volume ceiling from the same forward passes the FID already paid for."""
        from . import distribution as DM

        cf = DM.CaseFeatures(case_id=case.case_id, sequence=case.sequence, bucket=case.bucket)
        keep_slices = self._keep_diversity_slice(case.bucket)
        for arr, tag in ((real, "real"), (produced, "gen")):
            if arr is None:
                continue
            if medicalnet is not None:
                setattr(cf, f"medicalnet_{tag}", medicalnet.extract(arr))
            setattr(cf, f"inception_2p5d_{tag}", DM.extract_2p5d_inception_features(arr, inception))
            slice_2d = DM.mid_slice(arr, axis=2)
            _feats, probs = inception.extract_batch(slice_2d[None])
            setattr(cf, f"inception_mid_probs_{tag}", probs[0])
            if keep_slices:
                setattr(cf, f"mid_slice_{tag}", slice_2d)
        return cf

    def _keep_diversity_slice(self, bucket: str) -> bool:
        """Whether this case's mid-slice is retained for intra-set MS-SSIM. See the cap constant."""
        limit = self.config.diversity_slices_per_bucket
        if limit is None or limit <= 0:
            return True
        kept = self._diversity_kept.get(bucket, 0)
        if kept < limit:
            self._diversity_kept[bucket] = kept + 1
            return True
        self._diversity_dropped[bucket] = self._diversity_dropped.get(bucket, 0) + 1
        return False

    def _score_case(self, case: LiveCase, sample, real, produced, groups, medicalnet, inception):
        """Everything one case contributes: a metric row or an exclusion, features, anatomy."""
        from . import anatomy as A

        paired = self.config.task.paired
        outcome = {"case_id": case.case_id, "bucket": case.bucket}

        if paired:
            ok, comparison = check_case_geometry(case, produced.shape)
            if not ok:
                outcome["excluded"] = {
                    "prediction_id": case.case_id, "category": "geometry_incompatible",
                    "reason": "; ".join(comparison.reasons) or comparison.decision.value,
                    "geometry_comparison": comparison.as_dict(),
                }
                return outcome
            row = {"case_id": case.case_id, "sequence": case.sequence,
                   "acquisition_plane": case.acquisition_plane, "bucket": case.bucket,
                   "prediction_id": case.case_id, "shape": list(case.shape)}
            row.update(compute_paired_metrics(real, produced, groups))
            if "report_alignment" in groups:
                sim = report_image_similarity(sample.get("report_text", ""), produced,
                                              self.config.report_image_model)
                row["report_image_similarity_available"] = sim["available"]
                row["report_image_similarity_score"] = sim["score"]
                row["report_image_similarity_unavailable_reason"] = sim["reason"]
            outcome["row"] = row

        if "distribution" in groups or "report_consistency" in groups:
            outcome["features"] = self._extract_features(case, real, produced, medicalnet, inception)
        if "anatomy" in groups:
            # Real measures come from the ground truth even for an unpaired task: the real
            # population is the reference distribution the KS test compares against.
            outcome["anatomy_real"] = A.measure(real) if real is not None else None
            outcome["anatomy_produced"] = A.measure(produced)
        return outcome

    # -- the pass ----------------------------------------------------------------------

    def run(self, generate) -> dict:
        """Stream every case through `generate`, then aggregate and write. Returns the summary."""
        from ..validation import gather_objects, rank, world_size

        config = self.config
        task = config.task
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        groups = task.groups_to_run(distribution_enabled=config.distribution_metrics,
                                    skip=config.skip_metric_groups)
        if not groups:
            raise SystemExit(
                f"--task {task.name} declares {list(task.metric_groups)} but skip="
                f"{list(config.skip_metric_groups)} removes all of them."
            )
        log.info("task=%s paired=%s metric groups=%s", task.name, task.paired, list(groups))
        log.info("run_id=%s  %d cases over %d buckets, split=%s",
                 self.cohort.cohort_id, len(self.cases), len(self.cohort.buckets), config.split)
        if config.skip_metric_groups:
            log.warning("metric groups SKIPPED by request: %s -- recorded in summary.json",
                        list(config.skip_metric_groups))

        medicalnet = inception = None
        if "distribution" in groups or "report_consistency" in groups:
            from . import distribution as DM

            inception = DM.InceptionFeatureExtractor(config.device)
            if config.medicalnet_checkpoint:
                medicalnet = DM.MedicalNetFeatureExtractor(config.medicalnet_checkpoint, config.device)
            else:
                log.warning("no --medicalnet-checkpoint: the 3D FID and the blinded-classifier "
                            "consistency group cannot be computed (2.5D Inception FID still can)")

        # Sharded by position, so every rank generates a disjoint subset and no case twice. The
        # gather below restores a single deterministic order, so results are independent of the
        # world size -- a 4-GPU run and a 1-GPU run produce the same numbers.
        mine = [(n, c) for n, c in enumerate(self.cases) if n % world_size() == rank()]
        started = time.time()
        timings = {"generate": 0.0, "score": 0.0}
        records, figure_payload = [], []

        for position, (_n, case) in enumerate(mine, start=1):
            sample = self.dataset[case.index]
            clock = time.time()
            try:
                produced = generate(case, sample)
            except Exception as exc:  # noqa: BLE001 -- one bad case must not lose the run
                log.warning("case %s failed to generate: %s: %s", case.case_id, type(exc).__name__, exc)
                records.append({"case_id": case.case_id, "bucket": case.bucket,
                                "excluded": {"prediction_id": case.case_id,
                                             "category": "generation_failed",
                                             "reason": f"{type(exc).__name__}: {exc}"}})
                continue
            timings["generate"] += time.time() - clock

            if not self._checked_intensity:
                self._checked_intensity = True
                assert_metric_intensity_space(produced, f"{task.name} output")

            # The ground truth is materialised for UNPAIRED tasks too. `generation` computes no
            # per-case metric against it, but FID and the anatomy KS tests both need the real
            # population as their reference -- and taking it from the very case whose geometry was
            # requested keeps the two populations matched in composition and grid. The old cohort
            # path lined real[i] up with gen[i] arbitrarily within a bucket to achieve the same.
            #
            # `.float()` before `.numpy()`: R2VDatasetConfig.dtype may be bfloat16, which numpy
            # cannot represent. Every metric is float32 internally regardless.
            real = None
            if task.paired or {"distribution", "anatomy", "report_consistency"} & set(groups):
                real = sample["image"].squeeze(0).float().numpy().astype(np.float32)

            clock = time.time()
            outcome = self._score_case(case, sample, real, produced, groups, medicalnet, inception)
            timings["score"] += time.time() - clock

            panel = self._render_panel(case, sample, real, produced)
            if panel:
                outcome["panel_html"] = panel
            if len(figure_payload) < config.save_figures * max(1, len(self.cohort.buckets)):
                figure_payload.append((case, real, produced))
            records.append(outcome)
            del produced, sample, real

            if position % config.log_every == 0 or position == len(mine):
                rate = timings["generate"] / max(position, 1)
                log.info("[%d/%d] %.1fs elapsed, %.2f s/case generate", position, len(mine),
                         time.time() - started, rate)

        for bucket, dropped in sorted(self._diversity_dropped.items()):
            log.info("%s: %d mid-slices beyond the %d-per-bucket diversity cap were not retained "
                     "(intra-set MS-SSIM uses the first %d; every other metric uses all cases)",
                     bucket, dropped, config.diversity_slices_per_bucket,
                     config.diversity_slices_per_bucket)

        gathered = gather_objects(records)
        order = {c.case_id: i for i, c in enumerate(self.cases)}
        gathered.sort(key=lambda r: order.get(r["case_id"], 1 << 30))
        elapsed = time.time() - started

        return self._finish(gathered, groups, out, elapsed, timings, figure_payload)

    # -- aggregation and output ---------------------------------------------------------

    def _finish(self, records, groups, out: Path, elapsed: float, timings: dict,
                figure_payload) -> dict:
        config, task = self.config, self.config.task

        metric_rows = [r["row"] for r in records if r.get("row")]
        excluded = [r["excluded"] for r in records if r.get("excluded")]
        case_features = [r["features"] for r in records if r.get("features") is not None]
        # Panels rode the gather from whichever rank rendered them; the CLI logs them. Under DDP
        # only rank 0 holds a W&B run, and those are exactly the ranks that would otherwise lose
        # every panel they produced.
        self.panels = [{"case_id": r["case_id"], "bucket": r["bucket"], "html": r["panel_html"]}
                       for r in records if r.get("panel_html")]

        distribution_result = None
        if "distribution" in groups and case_features:
            from . import distribution as DM

            distribution_result = DM.compute_distribution_metrics(
                case_features, self.cohort.sequences, min_subgroup_n=config.min_subgroup_n,
                n_bootstrap=config.fid_bootstrap, seed=config.seed,
                k_diversity=config.diversity_k, buckets=self.cohort.buckets,
            )
        elif "distribution" in groups:
            log.warning("no features available -- distribution metrics skipped")

        report_consistency_result = self._report_consistency(groups, records, case_features)
        anatomy_result = self._anatomy(groups, records)

        figures_written = self._save_examples(out, metric_rows, figure_payload)

        paired_names = T.paired_metric_names(groups)
        per_sequence = (AGG.aggregate_metric_rows(metric_rows, lambda r: r["sequence"], paired_names)
                        if metric_rows else {})
        per_bucket = (AGG.aggregate_metric_rows(metric_rows, lambda r: r["bucket"], paired_names)
                      if metric_rows else {})

        summary = {
            "task": task.name,
            "task_summary": task.summary,
            "paired": task.paired,
            "unpaired_reason": task.unpaired_reason,
            "metric_groups_computed": list(groups),
            "metric_groups_skipped": list(config.skip_metric_groups),
            # `cohort_id` is the run fingerprint (see LiveCohortView.cohort_id). The key name is
            # kept so summary.json stays readable by check_run.py and by every earlier result.
            "cohort_id": self.cohort.cohort_id,
            "run_id": self.cohort.cohort_id,
            "split": config.split,
            "n_per_bucket": config.n_per_bucket,
            "evaluated_full_split": config.n_per_bucket is None,
            "n_cohort_cases": len(self.cases),
            "n_predictions": len(self.cases) - len(
                [e for e in excluded if e.get("category") == "generation_failed"]),
            "n_scored": len(metric_rows),
            "n_excluded": len(excluded),
            "paired_metrics": per_sequence,
            "paired_metrics_per_bucket": per_bucket,
            "bucket_geometry": {b: self.cohort.bucket_geometry(b) for b in self.cohort.buckets},
            "distribution_metrics": distribution_result,
            "report_consistency": report_consistency_result,
            "anatomy": anatomy_result,
            "elapsed_sec": round(elapsed, 1),
            "timings_sec": {k: round(v, 1) for k, v in timings.items()},
            "figures": figures_written,
        }

        self._write(out, summary, metric_rows, excluded, paired_names, distribution_result,
                    report_consistency_result, anatomy_result, elapsed)
        log.info("done: %d scored, %d excluded, %.1fs -> %s",
                 len(metric_rows), len(excluded), elapsed, out)
        return summary

    def _anatomy(self, groups, records):
        if "anatomy" not in groups:
            return None
        from . import anatomy as A

        real = [r["anatomy_real"] for r in records if r.get("anatomy_real")]
        produced = [r["anatomy_produced"] for r in records if r.get("anatomy_produced")]
        if not produced:
            return None
        out = {"overall": A.compare_populations(real, produced)}
        by_case = {c.case_id: c for c in self.cases}
        for level, key in (("bucket", lambda c: c.bucket), ("sequence", lambda c: c.sequence)):
            grouped_real, grouped_prod = {}, {}
            for record in records:
                case = by_case.get(record["case_id"])
                if case is None:
                    continue
                name = key(case)
                if record.get("anatomy_real"):
                    grouped_real.setdefault(name, []).append(record["anatomy_real"])
                if record.get("anatomy_produced"):
                    grouped_prod.setdefault(name, []).append(record["anatomy_produced"])
            for name in sorted(set(grouped_real) & set(grouped_prod)):
                out[name] = A.compare_populations(grouped_real[name], grouped_prod[name])
        return out

    def _report_consistency(self, groups, records, case_features) -> dict:
        """The blinded-classifier metric, over the MedicalNet features the distribution pass took.

        Always returns a dict, never None: "this task does not declare the group" and "the file is
        missing" must not look the same on disk.
        """
        task = self.config.task
        if "report_consistency" not in groups:
            return {"available": False,
                    "reason": (f"--task {task.name} does not declare report_consistency"
                               if "report_consistency" not in task.metric_groups
                               else "report_consistency was skipped by --skip-metric-groups")}
        if not case_features:
            return {"available": False,
                    "reason": "no MedicalNet features -- report_consistency reuses the distribution "
                              "pass's features, so it needs --medicalnet-checkpoint"}

        from .report_classifier import (
            auroc, evaluate_consistency, load_classifier_or_none, per_case_consistency,
            prevalence_baseline_auroc,
        )
        from .report_labels import ReportLabels

        classifier, reason = load_classifier_or_none(self.config.report_classifier, self.config.device)
        if classifier is None:
            log.warning("report_consistency unavailable: %s", reason)
            return {"available": False, "reason": reason}
        try:
            labels = ReportLabels(self.config.report_labels_csv)
        except SystemExit as exc:
            return {"available": False, "reason": str(exc)}
        if tuple(labels.labels) != tuple(classifier.labels):
            return {"available": False,
                    "reason": f"label set mismatch: classifier was fitted on {list(classifier.labels)}, "
                              f"{labels.path} provides {list(labels.labels)}"}

        joined = labels.for_cohort(self.cohort)
        by_case = {cf.case_id: cf for cf in case_features}
        by_id = {c.case_id: c for c in self.cases}
        rows = [(by_id[cid], by_case[cid]) for cid in (c.case_id for c in self.cases)
                if cid in by_case and cid in joined
                and by_case[cid].medicalnet_real is not None
                and by_case[cid].medicalnet_gen is not None]
        if not rows:
            return {"available": False,
                    "reason": "no case had both a report label and a MedicalNet feature pair"}

        truth = np.array([joined[case.case_id] for case, _ in rows], dtype=np.int64)
        probabilities_real = classifier.predict_proba(np.stack([cf.medicalnet_real for _, cf in rows]))
        probabilities_gen = classifier.predict_proba(np.stack([cf.medicalnet_gen for _, cf in rows]))
        real_reference = {name: {"auroc": auroc(truth[:, i], probabilities_real[:, i])}
                          for i, name in enumerate(classifier.labels)}
        baseline = prevalence_baseline_auroc(classifier.labels)
        result = evaluate_consistency(probabilities_gen, truth, classifier.labels,
                                      real_reference=real_reference, prevalence_baseline=baseline)
        result["real_reference"] = evaluate_consistency(
            probabilities_real, truth, classifier.labels,
            real_reference=real_reference, prevalence_baseline=baseline)
        per_case = per_case_consistency(probabilities_gen, truth, classifier.labels,
                                        usable_labels=result["labels_usable"])
        per_case_real = per_case_consistency(probabilities_real, truth, classifier.labels,
                                             usable_labels=result["labels_usable"])
        result.update({
            "available": True,
            "n_scored": len(rows),
            "n_cases_without_labels": len(self.cases) - len(rows),
            "label_coverage": labels.cohort_coverage(self.cohort),
            "classifier": {"path": str(self.config.report_classifier),
                           "labels": list(classifier.labels),
                           "provenance": vars(classifier.provenance)},
            "per_case": [
                {"case_id": case.case_id, "bucket": case.bucket,
                 "consistency": None if np.isnan(v) else float(v),
                 "consistency_real": None if np.isnan(r) else float(r)}
                for (case, _), v, r in zip(rows, per_case, per_case_real)
            ],
        })
        finite = per_case[~np.isnan(per_case)]
        result["mean_per_case_consistency"] = float(finite.mean()) if finite.size else None
        finite_real = per_case_real[~np.isnan(per_case_real)]
        result["mean_per_case_consistency_real"] = (float(finite_real.mean())
                                                    if finite_real.size else None)
        log.info("report_consistency: %d cases, macro AUROC %s (real reference %s)",
                 len(rows), result["macro_auroc_usable_labels"],
                 result["real_reference"]["macro_auroc_usable_labels"])
        return result

    def _save_examples(self, out: Path, metric_rows, figure_payload) -> list:
        """Montages (and optional NIfTI triplets) from volumes held back during the stream.

        Never fatal: a plotting failure must not throw away a completed evaluation.
        """
        if self.config.save_figures <= 0 and self.config.save_nifti_cases <= 0:
            return []
        try:
            from . import figures as F
        except ImportError as exc:  # pragma: no cover - PIL missing
            log.warning("cannot import figure support (%s) -- skipping example figures", exc)
            return []

        fig_dir = out / F.FIGURES_DIR
        written, per_bucket_count = [], {}
        try:
            for case, real, produced in figure_payload:
                n = per_bucket_count.get(case.bucket, 0)
                if n >= self.config.save_figures:
                    continue
                per_bucket_count[case.bucket] = n + 1
                name = f"{case.bucket}_ex{n}_{case.case_id}.png"
                if real is not None:
                    F.save_paired_figure(real, produced, fig_dir / name, case_id=case.case_id,
                                         sequence=case.sequence, plane=case.acquisition_plane,
                                         caption="")
                else:
                    F.save_unpaired_figure(produced, None, fig_dir / name,
                                           prediction_id=case.case_id, sequence=case.bucket)
                written.append(f"{F.FIGURES_DIR}/{name}")
                if n < self.config.save_nifti_cases:
                    stem = f"{case.bucket}_ex{n}_{case.case_id}"
                    triplet = [("pred", produced)]
                    if real is not None:
                        triplet = [("gt", real), ("pred", produced),
                                   ("absdiff", np.abs(real - produced))]
                    for tag, arr in triplet:
                        F.save_example_nifti(arr, case.spacing_mm, fig_dir / f"{stem}_{tag}.nii.gz")
                        written.append(f"{F.FIGURES_DIR}/{stem}_{tag}.nii.gz")
        except Exception as exc:  # noqa: BLE001 - figures are a convenience, metrics are the result
            log.warning("example figure generation failed (%s: %s) -- metrics are unaffected",
                        type(exc).__name__, exc)
        if written:
            log.info("wrote %d example file(s) -> %s", len(written), fig_dir)
        return written

    def _write(self, out: Path, summary, metric_rows, excluded, paired_names, distribution_result,
               report_consistency_result, anatomy_result, elapsed) -> None:
        """The canonical result layout -- byte-for-byte the same files the cohort path wrote."""
        from .runner import _write_csv

        _write_csv(out / "per_case_metrics.csv", metric_rows)
        summary["csv_files"] = SUM.write_summary_csv(
            out, self.cohort, metric_rows, paired_names, distribution_result, anatomy_result)
        (out / "excluded_cases.json").write_text(json.dumps(excluded, indent=2))
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        if distribution_result is not None:
            (out / "distribution_metrics.json").write_text(
                json.dumps(distribution_result, indent=2, default=str))
        (out / "report_consistency.json").write_text(
            json.dumps(report_consistency_result, indent=2, default=str))
        _write_csv(out / "report_consistency_per_case.csv",
                   report_consistency_result.get("per_case") or [],
                   fieldnames=("case_id", "bucket", "consistency", "consistency_real"))
        _write_csv(
            out / "report_consistency_per_label.csv",
            [{"label": name, **{k: v for k, v in entry.items() if k != "interpretation"}}
             for name, entry in (report_consistency_result.get("per_label") or {}).items()],
            fieldnames=("label", "auroc", "real_reference_auroc", "prevalence_baseline_auroc",
                        "retention", "average_precision", "prevalence", "n", "n_positive",
                        "mean_predicted_probability", "usable", "low_support"),
        )
        if anatomy_result is not None:
            (out / "anatomy_metrics.json").write_text(
                json.dumps(anatomy_result, indent=2, default=str))

        import os

        (out / "run_manifest.json").write_text(json.dumps({
            "evaluation_version": EVALUATION_VERSION,
            "live_evaluation_version": LIVE_EVALUATION_VERSION,
            "geometry_contract_version": G.GEOMETRY_CONTRACT_VERSION,
            "task": summary["task"],
            "metric_groups_computed": summary["metric_groups_computed"],
            "metric_groups_skipped": summary["metric_groups_skipped"],
            "run_id": self.cohort.cohort_id,
            "split": self.config.split,
            "n_per_bucket": self.config.n_per_bucket,
            "evaluated_full_split": self.config.n_per_bucket is None,
            "n_cases": len(self.cases),
            "cases_sha256": cases_fingerprint(self.cases),
            "geometry": self.cohort.geometry,
            "bucket_counts": self.cohort.bucket_counts,
            "population_bucket_counts": self.cohort.population_bucket_counts,
            "distribution_metrics_enabled": self.config.distribution_metrics,
            "diversity_slices_per_bucket": self.config.diversity_slices_per_bucket,
            "diversity_slices_dropped": dict(sorted(self._diversity_dropped.items())),
            "seed": self.config.seed,
            "device": self.config.device,
            "elapsed_sec": round(elapsed, 1),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            **self.config.extra_run_metadata,
        }, indent=2, sort_keys=True, default=str))


__all__ = [
    "DEFAULT_DIVERSITY_SLICES_PER_BUCKET",
    "LIVE_EVALUATION_VERSION",
    "LiveCase",
    "LiveCohortView",
    "LiveEvalConfig",
    "LiveEvaluator",
    "RESULT_FILES",
    "assert_metric_intensity_space",
    "build_cases",
    "case_id_for",
    "cases_fingerprint",
    "check_case_geometry",
    "run_fingerprint",
    "select_eval_cases",
    "stable_seed",
]
