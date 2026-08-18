# `mrrate_r2v.eval` — scoring a generated volume

One metric set, two callers, one place it's defined.

Full pipeline context: [`../README.md`](../README.md) and
[`docs/R2V.md`](../../../docs/R2V.md). History of the metric set this replaced is in
[`../DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md).

---

## The shape of it

```
manifest ──► Dataset (the one training uses) ──► generate ──► ChallengeAccumulator ──► metrics.json
                                     one case at a time, nothing stored
```

`LiveEvaluator.run` in [`live.py`](live.py) is `cli.evaluate`'s path. `ValidationRunner.run` in
[`../validation.py`](../validation.py) is the periodic during-training curve. Both score against
[`challenge_metrics.ChallengeAccumulator`](challenge_metrics.py), so a training curve and a final
score are the same numbers, computed the same way, just at different sample sizes.

**There is no cohort and no prediction set.** Evaluation builds the same `MRReportToVolumeDataset`
`cli.train_r2v` builds, from the same manifest and the same `R2VDatasetConfig`, and streams
generate-then-score one case at a time. Volumes are never written to disk.

## The metrics -- identical to the official evaluation container

[`challenge/`](challenge/) is a vendored port of the VLM3D `mr-volume-generation` challenge's own
evaluation code
(`github.com/forithmus/VLM3D-Dockers/tree/main/mr_challenges/mrgen_evaluation`), so a locally
computed number means the same thing the real leaderboard's does:

| key | what |
|---|---|
| `MSE_mean`, `PSNR_mean`, `SSIM_mean` | per-pair, percentile-normalised `[0,1]`, shape mismatch resolved by resampling the prediction (never exclusion) |
| `FID_2p5D_XY/XZ/YZ/Avg` | SqueezeNet1.1 features pooled per plane over every scored pair, one Fréchet distance per plane, `Avg` = their mean |
| `dice` | a literal copy of `SSIM_mean` -- the platform's own primary-metric shim, not real Dice |
| `n_total_files` / `n_scored_files` / `n_missing_outputs` / `n_excluded_out_of_scope_modality` | bookkeeping. Only T1w/T2w/FLAIR/SWI are scored; a case whose generation is missing is excluded from the means, not penalised with a worst-case value -- both are the official code's own behaviour, reproduced exactly |

`challenge_metrics.ChallengeAccumulator` is the only thing that computes these: `.add(...)` per
scored pair, `.add_missing(...)` for an out-of-scope or failed case, `.finalize()` (or the
module-level `combine()`, for pooling several ranks' `.state()`) for the metrics dict. See its
docstring for the two behavioural quirks reproduced deliberately from the official code.

**Nothing else is computed.** The older evaluation here (fidelity/perceptual/distribution/anatomy/
report_alignment/report_consistency, ~30 metrics defined independently of the challenge) is gone.

## `cli.evaluate`

Three tasks -- `report2volume` (trained adapter), `reconstruction` (frozen autoencoder round-trip),
`generation` (frozen base UNet, report-blind) -- differ only in how a volume is produced
(`cli/evaluate.py`'s three `build_*` functions); every task is scored the same way. Case selection
has no RNG: candidates are ordered by `(study_uid, series_id)` within each (modality, plane) bucket
and round-robined, so `--n-per-bucket N` is always the first N of a full run. Output is one
`<out>/metrics.json`, shaped `{"metrics": {...}, "per_case": [...], ...}`.

## Validation during training

`validation.py`'s `ValidationRunner` runs the same `ChallengeAccumulator` on a small, fixed, seeded
sample of the `val` split every `--validate-every-steps` optimizer steps, logging
`val/MSE_mean`, `val/PSNR_mean`, `val/SSIM_mean`, `val/FID_2p5D_XY/XZ/YZ/Avg`, `val/dice` -- the same
metrics, computed the same way, as `cli.evaluate`. There is exactly one regime (one fixed sample
size, one schedule); no separate quick/full passes, no other metrics, no reference-ceiling
computation.

## W&B

One table (`challenge_metrics`, columns `metric`/`value`) plus the same values as run summary/scalars
so W&B's normal run-comparison views work. Example ground-truth-vs-generated panels are rendered by
[`figures.py`](figures.py)'s `validation_panel_html` -- the same renderer for both `cli.evaluate` and
training-time validation, gated by `--wandb-log-reports` because a panel embeds report text.

## Other modules

| Module | Owns |
|---|---|
| `padding.py` | VAE divisor padding (`pad_to_divisible`/`crop_using_record`), used by `cli/evaluate.py:reconstruct` and `training.py`. Unrelated to metric comparison -- that goes through the official code's own resize-on-mismatch, not exclusion. |
| `figures.py` | the example-panel renderer, shared with training-time validation |
| `wandb_evaluation.py` | `_PanelCase`, the shim `figures.py` needs to render a panel from a `LiveCase` |
| `wandb_logging.py` | generic, no-op-safe W&B wrapper, no metric-specific logic |

## Privacy

`study_uid`/`series_id` are anonymized but still identifiers; they are used only for matching and
never written into results -- everything on disk uses `case_id` (a hash).
