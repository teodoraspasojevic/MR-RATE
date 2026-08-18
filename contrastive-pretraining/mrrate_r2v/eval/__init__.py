"""Evaluation: generate a volume, score it against the official VLM3D challenge metrics.

See `mrrate_r2v/eval/README.md` for what each metric means. `cli.evaluate` (final scoring) and
`validation.py` (the periodic during-training curve) are the two callers.

    challenge/             vendored port of the official evaluation container -- MSE/PSNR/SSIM,
                           2.5D FID, modality scope. The source of truth for what a metric means.
    challenge_metrics.py   ChallengeAccumulator: the one place that runs challenge/ and reduces
                           it to the metrics dict both callers log
    live.py                the streaming per-case harness cli.evaluate uses
    padding.py             VAE divisor padding (pad_to_divisible / crop_using_record)
    figures.py             the ground-truth-vs-generated example panel renderer
    wandb_evaluation.py    the shim `figures.py` needs to render a panel from a `LiveCase`
    wandb_logging.py       optional W&B wrapper, degrades to a no-op

**This package deliberately re-exports nothing.** Import the module you need directly:

    from mrrate_r2v.eval.challenge_metrics import ChallengeAccumulator
"""
