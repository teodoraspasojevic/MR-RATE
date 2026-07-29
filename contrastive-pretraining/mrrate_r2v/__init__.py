"""MR-RATE report-to-volume: dataset, preprocessing, model inference, evaluation.

Full guide: `docs/R2V.md`. The pipeline is three stages, each a CLI under `mrrate_r2v.cli`:

    1. preprocess   ->  a frozen ground-truth cohort   (cohort.py)
    2. predict_*    ->  a prediction set               (predictions.py)
    3. evaluate     ->  metrics                        (eval/runner.py)

`cohort.py` and `predictions.py` define the two on-disk contracts that hold it together: a
prediction set records which cohort it was produced against, and the evaluator refuses to
score a mismatch. That is what makes two experiments comparable.
"""

__version__ = "2.0"
