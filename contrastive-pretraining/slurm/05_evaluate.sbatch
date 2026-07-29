#!/bin/bash -l
#SBATCH --job-name=r2v_evaluate
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
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
# (measured 6.75 s/case at 256^3, of which I/O is 0.5%), and it parallelizes near-linearly.
# Staging the cohort to node-local $TMPDIR would NOT help -- Lustre already reads these volumes
# at ~3.4 GB/s, i.e. 39 ms against 6.75 s of compute.
#
# --skip-metric-groups perceptual drops edge-preservation / Laplacian-variance / HF-energy /
# per-plane SSIM: ~40% of per-case time. Fidelity (MAE/MSE/PSNR/NCC/SSIM-3D) and the distribution
# metrics are unaffected. summary.json records this as metric_groups_skipped, so a run with it is
# never mistaken for a full one -- and two runs are only comparable on the groups both computed.
# Remove the flag if you want the detail-preservation diagnostics back.

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
    --skip-metric-groups perceptual \
    --overwrite

echo
cat "$RESULTS/summary.json"
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS 2>&1 || true
