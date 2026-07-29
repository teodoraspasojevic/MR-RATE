"""Command-line entry points. Run from `contrastive-pretraining/`:

    python -m mrrate_r2v.cli.build_manifest       stage 0: index the storage (once per location)
    python -m mrrate_r2v.cli.preprocess           stage 1: freeze a ground-truth cohort
    python -m mrrate_r2v.cli.predict_vae          stage 2: VAE reconstruction inference
    python -m mrrate_r2v.cli.predict_generation   stage 2: unconditional generation inference
    python -m mrrate_r2v.cli.predict_r2v          stage 2: report-to-volume inference
    python -m mrrate_r2v.cli.import_predictions   stage 2: adopt an external checkpoint's NIfTIs
    python -m mrrate_r2v.cli.evaluate             stage 3: metrics

Every script accepts `--help`. No script computes metrics except `evaluate`, and no script
decides which cases or what FOV except `preprocess`.
"""
