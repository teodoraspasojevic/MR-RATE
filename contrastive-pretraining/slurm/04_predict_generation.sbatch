#!/bin/bash -l
#SBATCH --job-name=r2v_predict_gen
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_predict_gen_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_predict_gen_%j.err
#
# Unconditional generation. Diffusion sampling is slow -- time a small run first and scale
# --time from its measured rate rather than guessing.
#
#   sbatch slurm/04_predict_generation.sbatch <cohort_name> [n_per_bucket] [run_tag]
#
# Each bucket is generated at ITS OWN shape and spacing -- the cohort's, i.e. NVIDIA's published
# recommended FOV for that (modality, plane). Omit n_per_bucket (or pass 0) to match the cohort's
# own per-bucket counts exactly, which is what makes a per-bucket FID compare distributions
# rather than sample sizes.

unset SLURM_EXPORT_ENV
set -euo pipefail

# Absolute path, not $(dirname "$BASH_SOURCE"): Slurm runs a COPY of this script from
# /var/tmp/slurmd_spool/job<id>/, where _common.sh does not exist. Override with R2V_REPO=/path.
R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup
NAME="${1:?usage: sbatch 04_predict_generation.sbatch <cohort_name> [n_per_bucket] [run_tag]}"
N="${2:-0}"
TAG="${3:-generation_$(date +%Y%m%d)}"
COHORT="$COHORT_ROOT/$NAME"
preflight "$COHORT/cohort.json" "$VAE_CHECKPOINT" "$UNET_CHECKPOINT"

N_ARG=(--n-per-bucket "$N")
[[ "$N" == "0" ]] && N_ARG=()   # omitting the flag means "match the cohort's own counts"

run_py mrrate_r2v.cli.predict_generation \
    --cohort "$COHORT" --checkpoint "$VAE_CHECKPOINT" \
    --unet-checkpoint "$UNET_CHECKPOINT" --device cuda \
    "${N_ARG[@]}" --seed 42 \
    --out "$PRED_ROOT/$TAG" --overwrite

echo "Now: sbatch slurm/05_evaluate.sbatch generation $NAME $TAG"
