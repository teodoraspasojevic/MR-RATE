"""Command-line entry points. Run from `contrastive-pretraining/`:

    python -m mrrate_r2v.cli.build_manifest           stage 0: index the storage (once per location)
    python -m mrrate_r2v.cli.train_r2v                train the report-conditioning adapter
    python -m mrrate_r2v.cli.evaluate --task ...      generate AND score, in one pass
    python -m mrrate_r2v.cli.train_report_classifier  fit the blinded classifier evaluate.py uses
    python -m mrrate_r2v.cli.generate_r2v             one report -> one .nii.gz, for eyeballing

**There is no preprocess stage and no predict stage.** `cli.evaluate` builds the same Dataset
`cli.train_r2v` builds, from the same manifest and the same `R2VDatasetConfig`, and streams
generate-then-score one case at a time. Nothing is frozen to disk in between, so nothing can go
stale relative to anything else, and the two paths cannot preprocess differently.

The removed scripts (`preprocess`, `predict_vae`, `predict_generation`, `predict_r2v`,
`import_predictions`) wrote and read cohort/prediction directories; `eval/live.py` replaces all of
them. `cohort.py`, `predictions.py`, `volumes.py` and `eval/runner.py` remain as the library for
reading results produced before the change.

Every script accepts `--help`. Only `evaluate` computes metrics, and `eval/tasks.py` -- not any
CLI flag -- decides which metrics a task gets.
"""
