#!/bin/bash -l
#SBATCH --job-name=r2v_evaluate
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_evaluate_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_evaluate_%j.err
#
# The single evaluator. One job script for every task.
#
#   sbatch slurm/05_evaluate.sbatch <task> <cohort_name> <pred_tag>
#
#   sbatch slurm/05_evaluate.sbatch reconstruction test_v1 vae_20260729
#   sbatch slurm/05_evaluate.sbatch generation     test_v1 generation_20260729
#   sbatch slurm/05_evaluate.sbatch report2volume  test_v1 r2v_ckpt3
#
# A GPU is requested because the distribution metrics run feature extractors on it. Fidelity-only
# runs (drop --distribution-metrics below) are fine on a CPU partition.
#
# --cpus-per-task=32 and --workers are load-bearing: per-case scoring is CPU-compute-bound
# (measured 1.6 s/case fidelity + 1.3 s/volume anatomy against 0.11 s to read one) and it
# parallelizes near-linearly. Staging the cohort to node-local $TMPDIR would NOT help.
#
# Every metric group runs. `perceptual` (edge preservation, Laplacian variance, HF energy,
# per-plane SSIM) used to be skipped here because it was ~40% of a 6.75 s/case cost at 256^3. At
# the per-bucket FOVs that cost is 1.6 s/case, so the whole group adds ~40 s across 2000 cases at
# 32 workers -- and it measures blur, which is a compressive VAE's primary failure mode. To skip
# it again add `--skip-metric-groups perceptual`; summary.json records the omission either way, so
# a partial run is never mistaken for a full one.

unset SLURM_EXPORT_ENV
set -euo pipefail

# Absolute path, not $(dirname "$BASH_SOURCE"): Slurm runs a COPY of this script from
# /var/tmp/slurmd_spool/job<id>/, where _common.sh does not exist. Override with R2V_REPO=/path.
R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup
TASK="${1:?usage: sbatch 05_evaluate.sbatch <task> <cohort_name> <pred_tag>}"
NAME="${2:?missing cohort_name}"
TAG="${3:?missing pred_tag}"
COHORT="$COHORT_ROOT/$NAME"
PRED="$PRED_ROOT/$TAG"
RESULTS="$RESULT_ROOT/${TASK}_${NAME}_${TAG}"
preflight "$COHORT/cohort.json" "$PRED/predictions.json"

MEDICALNET_ARG=()
[[ -f "$MEDICALNET_CHECKPOINT" ]] && MEDICALNET_ARG=(--medicalnet-checkpoint "$MEDICALNET_CHECKPOINT")

run_py mrrate_r2v.cli.evaluate \
    --task "$TASK" --gt "$COHORT" --pred "$PRED" --out "$RESULTS" \
    --distribution-metrics "${MEDICALNET_ARG[@]}" --device cuda \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --save-figures 3 --save-nifti-cases 1 \
    --overwrite

echo
echo "=== metrics_per_bucket.csv (geometry + every metric, one row per bucket) ==="
column -s, -t < "$RESULTS/metrics_per_bucket.csv" 2>/dev/null || cat "$RESULTS/metrics_per_bucket.csv"
echo
echo "=== metrics_summary.csv (per modality, then overall_macro / overall_weighted) ==="
column -s, -t < "$RESULTS/metrics_summary.csv" 2>/dev/null || cat "$RESULTS/metrics_summary.csv"
echo
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS 2>&1 || true
