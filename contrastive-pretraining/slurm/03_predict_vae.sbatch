#!/bin/bash -l
#SBATCH --job-name=r2v_predict_vae
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_predict_vae_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_predict_vae_%j.err
#
# VAE reconstruction inference over a cohort. Set --time from the smoke test's per-case rate.
#
#   sbatch slurm/03_predict_vae.sbatch <cohort_name> [run_tag]

unset SLURM_EXPORT_ENV
set -euo pipefail

# Absolute path, not $(dirname "$BASH_SOURCE"): Slurm runs a COPY of this script from
# /var/tmp/slurmd_spool/job<id>/, where _common.sh does not exist. Override with R2V_REPO=/path.
R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup
NAME="${1:?usage: sbatch 03_predict_vae.sbatch <cohort_name> [run_tag]}"
TAG="${2:-vae_$(date +%Y%m%d)}"
COHORT="$COHORT_ROOT/$NAME"
preflight "$COHORT/cohort.json" "$VAE_CHECKPOINT"

run_py mrrate_r2v.cli.predict_vae \
    --cohort "$COHORT" --checkpoint "$VAE_CHECKPOINT" --device cuda \
    --out "$PRED_ROOT/$TAG" --overwrite

echo "Now: sbatch slurm/05_evaluate.sbatch reconstruction $NAME $TAG"
