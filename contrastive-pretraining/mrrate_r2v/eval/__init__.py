"""Evaluation: frozen ground-truth cohort + predictions + task -> metrics.

See `mrrate_r2v/eval/README.md` for what each metric means and how to read a result directory.

    tasks.py              which metrics are valid for which task (the registry)
    runner.py             the single pipeline -- everything goes through run_evaluation()
    geometry_contract.py  may these two volumes be compared voxelwise?
    paired.py             voxelwise + detail metrics on one pair
    distribution.py       FID / Inception Score / precision-recall-density-coverage
    features.py           fingerprint-gated feature cache
    aggregate.py          per-sequence and overall means
    pairing.py            identifier matching, for importing an external checkpoint's files
    wandb_logging.py      optional W&B wrapper, degrades to a no-op

**This package deliberately re-exports nothing.** Import the module you need directly, so a
heavy dependency in one module never blocks another -- `geometry_contract` needs only numpy and
nibabel, `distribution` needs torch, and neither should make the other unimportable:

    from mrrate_r2v.eval import paired as M
    from mrrate_r2v.eval.runner import run_evaluation, EvaluationInputs
    from mrrate_r2v.eval.tasks import get_task

Every module works on plain numpy arrays, so you can use one metric without loading a cohort,
a model, or the runner:

    M.psnr(gt_array, pred_array)
"""
