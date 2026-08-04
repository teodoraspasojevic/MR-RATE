#!/bin/bash -l
#SBATCH --job-name=r2v_train
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_train_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_train_%j.err
#
# Trains the report-conditioning adapter on MR-RATE. The frozen NVIDIA denoiser and RadBERT are
# never updated -- the trainer asserts that before the first step and logs the two parameter counts.
#
# Smoke run (default: 4 steps, ~minutes) -- the acceptance gate before any long run:
#   sbatch slurm/06_train_r2v.sbatch
#
# Real run: raise --time, --epochs and R2V_MAX_STEPS, and consider --save-every-steps. Note that
# `#SBATCH --export=NONE` means a plain `VAR=x sbatch ...` does NOT reach the job -- pass overrides
# through --export:
#   sbatch --export=ALL,R2V_MAX_STEPS=0,R2V_EPOCHS=1 --time=24:00:00 slurm/06_train_r2v.sbatch
#
# Multi-GPU (the official launcher shape -- diff_model_train.py sets DDP up the same way):
#   #SBATCH --gres=gpu:h200:4 ... and replace `run_py_text` below with
#   `apptainer exec ... torchrun --nproc_per_node=4 -m mrrate_r2v.cli.train_r2v --num-gpus 4 ...`

unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup
preflight "$MANIFEST_CSV" "$REPORT_INDEX_CSV" "$UNET_CHECKPOINT" "$VAE_CHECKPOINT" \
          "$RADBERT_CHECKPOINT" "$SIF_IMAGE_TEXT"

TAG="${R2V_TAG:-r2v_adapter_${SLURM_JOB_ID:-local}}"
RUN_DIR="$WORKSPACE/runs/$TAG"
MAX_STEPS="${R2V_MAX_STEPS:-4}"
EPOCHS="${R2V_EPOCHS:-1}"
mkdir -p "$RUN_DIR"

STEP_ARG=()
[[ "$MAX_STEPS" != "0" ]] && STEP_ARG=(--max-steps "$MAX_STEPS")

run_py_text mrrate_r2v.cli.train_r2v \
    --manifest "$MANIFEST_CSV" \
    --report-index "$REPORT_INDEX_CSV" \
    --base-checkpoint "$UNET_CHECKPOINT" \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --text-encoder radbert \
    --text-checkpoint "$RADBERT_CHECKPOINT" \
    --max-report-tokens 512 \
    --report-sections findings impression \
    --geometry-mode per_modality_plane \
    --split train \
    --out "$RUN_DIR" \
    --epochs "$EPOCHS" "${STEP_ARG[@]}" \
    --batch-size 1 \
    --lr 1e-5 \
    --report-dropout-probability 0.10 \
    --modality-dropout-probability 0.10 \
    --scale-factor auto \
    --log-every 1 \
    --save-format adapter \
    --num-workers 8 \
    --device cuda

echo "=== adapter checkpoint ==="
ls -la "$RUN_DIR"
cat "$RUN_DIR/train_summary.json"
