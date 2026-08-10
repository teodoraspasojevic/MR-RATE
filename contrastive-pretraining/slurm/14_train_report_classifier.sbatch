#!/bin/bash -l
#SBATCH --job-name=r2v_report_clf
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_report_clf_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_report_clf_%j.err
#
# Fit the blinded pathology classifier that `--task report2volume` scores report consistency with.
# Run this ONCE; every arm's evaluation then reuses the same instrument, which is what makes the
# four consistency numbers comparable.
#
#   sbatch slurm/14_train_report_classifier.sbatch [out_name]
#
#   sbatch slurm/14_train_report_classifier.sbatch report_classifier_v2
#
# **No cohorts.** Like `cli.evaluate`, this reads the Dataset straight from the manifest and builds
# the same `R2VDatasetConfig` from the same flags -- so the classifier is fitted on volumes
# preprocessed exactly the way the volumes it will later score are. Under the old pipeline those
# were two separate cohort directories that could drift apart with nothing able to see it.
#
# It fits on the TRAIN split and selects the epoch on VAL, both capped per (modality, plane) by a
# deterministic no-RNG prefix (R2V_CLF_N / R2V_CLF_VAL_N). Fitting on `test` would make every
# consistency number circular -- the generations are scored against that split -- and the CLI
# refuses it before any feature extraction unless --allow-test-split is passed.
#
# GPU is for the MedicalNet feature extraction (one forward pass per volume); the head itself is
# ~140k parameters and fits in seconds. Features are cached under $WORKSPACE/cache/r2v/clf_features
# keyed by the run fingerprint, so re-running with different head hyperparameters costs seconds
# rather than the extraction again -- and a cache from a different split or normalizer cannot be
# silently reused.

unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup

OUT_NAME="${1:-report_classifier_v2}"
OUT="$WORKSPACE/models/${OUT_NAME}.pt"
FEATURE_CACHE="$WORKSPACE/cache/r2v/clf_features"

preflight "$MANIFEST_CSV" "$REPORT_INDEX_CSV" "$MEDICALNET_CHECKPOINT"
mkdir -p "$FEATURE_CACHE" "$WORKSPACE/models"

# run_py, not run_py_text: no text encoder is involved. The classifier reads volumes and the
# report-derived labels, never the report text -- that is what makes it "blinded".
#
# The preprocessing flags must match what `05_evaluate.sbatch` passes, or the head is fitted in a
# different intensity/geometry space from the one it scores in and measures a domain gap rather
# than a pathology.
run_py mrrate_r2v.cli.train_report_classifier \
    --manifest "$MANIFEST_CSV" \
    --report-index "$REPORT_INDEX_CSV" \
    --split "${R2V_CLF_SPLIT:-train}" \
    --val-split "${R2V_CLF_VAL_SPLIT:-val}" \
    --n-per-bucket "${R2V_CLF_N:-500}" \
    --val-n-per-bucket "${R2V_CLF_VAL_N:-100}" \
    --posterior-shift-mm "${R2V_POSTERIOR_SHIFT:-15}" \
    --normalizer "${R2V_NORMALIZER:-percentile}" \
    --geometry-mode per_modality_plane \
    --medicalnet-checkpoint "$MEDICALNET_CHECKPOINT" \
    --feature-cache-dir "$FEATURE_CACHE" \
    --epochs "${R2V_CLF_EPOCHS:-60}" \
    --hidden "${R2V_CLF_HIDDEN:-256}" \
    --lr "${R2V_CLF_LR:-1e-3}" \
    --seed "${R2V_SEED:-0}" \
    --device cuda \
    --out "$OUT"

echo
echo "=== classifier ==="
ls -la "$OUT" "$OUT.report.json"
echo
echo "Use it:  sbatch slurm/05_evaluate.sbatch report2volume <tag> <adapter.pt>"
echo "         (05 picks up \$MEDICALNET_CHECKPOINT and this file automatically)"
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS 2>&1 || true
