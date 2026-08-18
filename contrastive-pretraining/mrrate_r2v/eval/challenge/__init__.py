"""Vendored port of the official VLM3D `mr-volume-generation` scoring container.

Source: github.com/forithmus/VLM3D-Dockers/tree/main/mr_challenges/mrgen_evaluation, read in full
and ported here so our own evaluation computes the identical numbers the real leaderboard does --
same normalization, same feature extractor, same Frechet formula. Comments translated from Turkish
to English; the math and control flow are unchanged. `score.py` itself is not ported: its job is
scanning two on-disk directories (`predictions/`, `ground_truth/`), which does not apply here since
we generate in-memory -- that orchestration is reproduced in `eval/challenge_metrics.py` instead.
"""
