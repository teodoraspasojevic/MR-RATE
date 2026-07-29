# `mrrate_r2v.eval` — scoring a generated volume

One pipeline, three tasks, one result layout.

Full pipeline context: [`docs/R2V.md`](../../../docs/R2V.md). This file is about the evaluation
layer itself — use it when you want to know what a metric means, why a case was excluded, or how to
add a task.

---

## The shape of it

```
cohort (frozen GT)  +  prediction set  +  --task   ──►  run_evaluation()  ──►  results/
```

`run_evaluation` in [`runner.py`](runner.py) is the only path. Both the CLI and the tests call it;
there is no second implementation to drift.

**It reads `.npy` files and nothing else.** No manifest, no archive, no Dataset, no model. That is
deliberate: the evaluator cannot preprocess differently from the cohort, because it does not
preprocess at all.

### What it does, in order

1. **Refuses to run unless the prediction set's `cohort_id` matches the `--gt` cohort's.** Same
   cases, same FOV, same count — or no numbers at all.
2. **Refuses to run on an incomplete cohort or prediction set.** A missing volume file must never be
   silently treated as a smaller sample.
3. **For paired tasks: matches by `case_id`, then checks geometry** before any voxelwise metric. A
   case that fails is excluded with a reason, never resized to fit.
4. **Computes exactly the metric groups [`tasks.py`](tasks.py) declares** for the task.
5. **Writes one canonical layout** — the same five files for every task.

---

## Tasks

| `--task` | Ground truth per prediction? | Metric groups |
|---|---|---|
| `reconstruction` | yes, that exact input volume | fidelity, perceptual, distribution |
| `report2volume` | yes, the series the report describes | fidelity, perceptual, distribution, report_alignment |
| `generation` | **no** | distribution |

`generation` gets no voxelwise metrics because no real patient corresponds to a generated volume —
comparing pixel by pixel against an arbitrary real scan measures "how different are two random
brains." This lives in `tasks.py` as a property of the task, so it cannot be bypassed by forgetting
a flag. `test_eval_tasks_and_runner.py` asserts it.

### Adding a task

Add a `TaskSpec` to `TASKS` in `tasks.py`. If it needs a new metric family, add a group to
`METRIC_GROUPS` (with `needs_pair` set correctly) and its metric names to `GROUP_METRIC_NAMES`, then
compute them in `runner.compute_paired_metrics`. The runner and CLI pick it up with no changes.

---

## The modules

| Module | Owns | Needs |
|---|---|---|
| `tasks.py` | which metrics are valid for which task | stdlib |
| `runner.py` | the pipeline | numpy |
| `geometry_contract.py` | may these two volumes be compared? | numpy, nibabel |
| `paired.py` | voxelwise + detail metrics on one pair | numpy, scipy (skimage for SSIM, lazily) |
| `distribution.py` | FID, Inception Score, precision/recall/density/coverage | torch, torchvision |
| `features.py` | fingerprint-gated feature cache | numpy |
| `aggregate.py` | per-sequence and overall means | numpy |
| `pairing.py` | identifier matching, for importing external NIfTIs | stdlib |
| `wandb_logging.py` | optional W&B, degrades to a no-op | — |

`__init__.py` re-exports **nothing**, so a heavy dependency in one module never blocks another.
Import what you need:

```python
from mrrate_r2v.eval import paired as M
M.psnr(gt_array, pred_array)          # no cohort, no model, no torch
```

---

## Why shape equality is not enough

The single rule `geometry_contract.py` exists to enforce: **equal `.shape` is never treated as proof
that two volumes occupy the same physical space.**

- **Same shape, different anatomy.** Crop one volume from two different corners and both crops are
  the same size while showing different brain regions. Only the affine catches this.
- **Different shape, same anatomy.** Resample one scan at two spacings and you get two shapes that
  are still legitimately the same scan — just not comparable voxel-by-voxel yet.

So `compare_geometry` checks shape, spacing, orientation, affine rotation, and origin/FOV overlap as
*separate* quantities and returns one of four verdicts:

| Verdict | Meaning | What happens |
|---|---|---|
| `STRICT_MATCH` | same grid | metrics computed |
| `DECODER_BOUNDARY_CORRECTABLE` | shape differs, everything else agrees | corrected only if the caller can *prove* the padding it applied (`CropPadRecord`); otherwise excluded |
| `WORLD_ALIGNED_ELIGIBLE` | different grid, same anatomy | resampled only on explicit opt-in, and reported separately |
| `INCOMPATIBLE` | anything else | excluded, reason recorded |

The implementation this replaced resized two volumes whenever their shapes merely differed, with no
check for *why*. A score computed that way looks precise and proves nothing.

Where padding is legitimately needed — the NVIDIA VAE requires axes divisible by a model-derived
divisor — `predict_vae.py` pads at the end of each axis before encoding and crops back the *exact*
recorded amount after decoding, so the reconstruction returns on the cohort's own grid.

---

## The metrics

### Fidelity — how close, voxel by voxel

`mae`, `mse`, `psnr`, `ncc`, `ssim_3d`, `relative_intensity_error`, each with a `_fg`
foreground-restricted variant.

The foreground mask is derived from the **ground truth only**. Deriving it from the prediction would
let a mostly-empty prediction choose its own easier evaluation region;
`test_foreground_mask_comes_from_ground_truth_only` pins this.

Two things to know before quoting a number:

- **Intensities are not clipped to [0, 1].** The default percentile normalizer leaves values above
  the 99.5th percentile above 1.0. `data_range=1.0` is a fixed reference scale for cross-sample
  comparability, not a real maximum — so PSNR here is relative, not absolute.
- **Geometry is not checked inside these functions.** They only ever see arrays. The caller must
  have gotten `STRICT_MATCH` first; `runner.py` does.

### Perceptual — is detail surviving

`edge_preservation_ratio`, `laplacian_variance_ratio`, `high_frequency_energy_ratio`, `ssim_2d_mean`
per plane. Around 1.0 means preserved; below means blurred. 0.6-0.9 is common for a compressive VAE.

Per-plane SSIM excludes background-only slices — an empty slice's SSIM is uninformative and would
bias the mean toward whatever a near-constant comparison scores. Check `n_slices_used` alongside the
mean.

### Distribution — do the populations match

| Metric | Reads as |
|---|---|
| MedicalNet FID (3D) | lower is better. Generation will be much higher than reconstruction — expected, not a bug. MedicalNet was trained on many organs, not brain MRI specifically. |
| 2.5D Inception FID | lower is better, computed on slices across all three planes. Not comparable in scale to the MedicalNet number — different feature space. Inception-v3 has never seen a medical image. |
| Inception Score | higher is better, but not very informative for brain MRI. Prefer the next row. |
| Precision / Recall / Density / Coverage | precision = do outputs look real; recall/coverage = do they span the real range. High precision + low recall is classic mode collapse. Needs 50+ per group to be stable. |

The Fréchet distance is computed in `distribution.frechet_distance` — float64 throughout, with
epsilon regularization retried on a near-singular covariance product (routine when the sample is
smaller than the feature dimension). Every FID comes with a rank-deficiency flag and a bootstrap CI
so one number is never read as more precise than the sample supports.

It is computed here rather than via `monai.metrics.FIDMetric`: that path passes a `disp=` argument
scipy removed in 1.17, so it raises on any current scipy. Keeping it in-package also means an
evaluation run needs no monai.

For `generation` there is no pairing, so the real population is the whole cohort and the produced
population is every prediction item, truncated to `min(n_real, n_gen)` per sequence — Fréchet
distance needs two populations of comparable size, not a correspondence. The truncation is logged.

### Report alignment

Currently **unavailable**: no validated MRI image-text model exists in this project. Every case
records `report_image_similarity_available=False` with a concrete reason. It is never silently
substituted with a different model — a number attributed to report alignment would then be measuring
something else. To wire one in, pass an object with `.score(report_text, volume) -> float` as
`EvaluationInputs.report_image_model`.

---

## Making it fast

**It is CPU-compute-bound, not I/O-bound.** Measured on real 256³ volumes off Lustre:

| | Time per case | Share |
|---|---|---|
| reading both `.npy` volumes (134 MB) | 39 ms @ 3.4 GB/s | **0.5%** |
| metric compute | 6,750 ms | 99.5% |

So **staging the cohort to node-local `$TMPDIR` is not worth doing** — it would optimize 0.5% of
the work, a ceiling of 1.005×. Lustre already reads these volumes faster than the metrics can
consume them.

What does help is `--workers`, because per-case scoring is embarrassingly parallel. Measured on 8
real case pairs:

| `--workers` | s/case | speedup |
|---|---|---|
| 1 | 6.75 | 1.00× |
| 2 | 3.37 | 2.00× |
| 4 | 1.72 | 3.93× |
| 8 | 0.92 | 7.32× |

Results are byte-identical at every worker count (`test_parallel_scoring_gives_identical_results_to_serial`
asserts it) — `--workers` is purely a wall-clock knob. `slurm/05_evaluate.sbatch` passes
`$SLURM_CPUS_PER_TASK` and asks for 32 CPUs.

**Two caveats.** `--workers` only parallelizes *paired* scoring, so it does nothing for
`--task generation`, which has no paired metrics. And distribution metrics run their feature
extractors serially on the GPU in the parent process — that cost is unaffected by `--workers`.

Where the per-case time goes, if you want to cut it further:

| Metric | Cost | Share |
|---|---|---|
| `high_frequency_energy_ratio` (2× 3D FFT) | 1,590 ms | 24% |
| `ssim_3d` | 2,005 ms | 30% |
| `ssim_2d_mean` × 3 planes | 1,664 ms | 25% |
| `laplacian_variance_ratio` | 753 ms | 11% |
| `edge_preservation_ratio` | 432 ms | 6% |
| mae/mse/psnr/ncc (whole + fg) | 367 ms | 5% |

Dropping the `perceptual` group would roughly halve it, at the cost of the detail-preservation
diagnostics. The FFT metric's frequency-radius mask is already cached by shape (it was being
rebuilt per volume — three 134 MB float64 arrays, 239 ms each time).

## Reading a result directory

| File | What it is |
|---|---|
| `summary.json` | **start here.** Per-sequence and overall means, which metric groups ran, how many cases were scored and excluded. |
| `per_case_metrics.csv` | one row per scored case |
| `distribution_metrics.json` | population metrics, when computed |
| `excluded_cases.json` | every unscored case, with a specific reason |
| `run_manifest.json` | `cohort_id`, task, checkpoint hashes, contract versions |

Always compare `n_scored` with `n_cohort_cases`. Exclusion categories you may see:

| Category | Meaning |
|---|---|
| `missing_prediction` | a cohort case the model produced nothing for |
| `geometry_incompatible` | prediction grid does not match the cohort's; the reasons list says which check failed |
| `no_matching_case` | prediction names a `case_id` not in this cohort |
| `duplicate_case` | two predictions for the same case |
| `unpaired_item` | a paired task got an item with no `case_id` |

A good FID does not prove clinical correctness, and a good reconstruction does not predict that a
report-conditioned model will score well — each stage measures something different.

---

## Privacy

`study_uid`/`series_id` are anonymized but still identifiers. They are used for matching and never
written into `cohort.json`, `summary.json`, or any results file — those carry `case_id` (a hash) and
`cohort_id`. Quote the `cohort_id` in a paper, not a case list.

---

## Testing

```bash
python -m pytest tests/test_eval_tasks_and_runner.py tests/test_cohort_contract.py \
                 tests/test_eval_geometry_contract.py tests/test_eval_paired_metrics.py \
                 tests/test_eval_distribution.py tests/test_eval_features.py \
                 tests/test_eval_pairing.py -v --no-cov
```

CPU, synthetic data, seconds. `test_eval_tasks_and_runner.py` runs the whole pipeline end to end.
