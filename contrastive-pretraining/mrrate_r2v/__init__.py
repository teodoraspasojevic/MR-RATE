"""MR-RATE report-to-volume: dataset, preprocessing, model inference, evaluation.

Full guide: `docs/R2V.md`. The pipeline is three stages, each a CLI under `mrrate_r2v.cli`:

    1. build_manifest  ->  once per storage location
    2. train_r2v       ->  --split train
    3. evaluate        ->  --split test, generates and scores in one streaming pass

Train and test are the same program up to the point where one backprops and the other samples --
both build the dataset straight from the manifest, so training and evaluation preprocessing can
never drift apart undetected.
"""

__version__ = "2.0"
