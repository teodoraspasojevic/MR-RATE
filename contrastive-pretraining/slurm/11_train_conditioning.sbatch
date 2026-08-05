#!/bin/bash -l
#SBATCH --job-name=r2v_cond
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_cond_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_cond_%j.err
#
# Trains the report-conditioning adapter under one of the three supported configurations
# (slurm/configs/). The frozen NVIDIA denoiser and the text encoders are never updated -- the trainer
# asserts that before the first step and logs both parameter counts.
#
# Smoke run (4 steps, no validation -- the acceptance gate before any long run):
#   sbatch --export=ALL,R2V_CONFIG=A slurm/11_train_conditioning.sbatch
#   sbatch --export=ALL,R2V_CONFIG=B slurm/11_train_conditioning.sbatch
#   sbatch --export=ALL,R2V_CONFIG=C slurm/11_train_conditioning.sbatch
#
# Real single-GPU run with validation and W&B:
#   sbatch --export=ALL,R2V_CONFIG=A,R2V_MAX_STEPS=0,R2V_VALIDATE_EVERY=500,R2V_WANDB=online \
#          --time=24:00:00 slurm/11_train_conditioning.sbatch
#
# Real 4-GPU run on one node (DDP via torchrun; set R2V_NGPU to match --gres):
#   sbatch --export=ALL,R2V_CONFIG=C,R2V_MAX_STEPS=0,R2V_NGPU=4,R2V_VALIDATE_EVERY=500 \
#          --gres=gpu:h200:4 --cpus-per-task=32 --time=24:00:00 slurm/11_train_conditioning.sbatch
#
# `#SBATCH --export=NONE` means a plain `VAR=x sbatch ...` does NOT reach the job -- always pass
# overrides through --export=ALL,... as above. R2V_MAX_STEPS=0 removes the step cap.

unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup

# ---------------------------------------------------------------- configuration
CONFIG="${R2V_CONFIG:?set R2V_CONFIG to A, B or C (see slurm/configs/README.md)}"
case "$CONFIG" in
    A|a) CONFIG_FILE="A_cxr_bert_cls.sh" ;;
    B|b) CONFIG_FILE="B_radbert_mean.sh" ;;
    C|c) CONFIG_FILE="C_report2ct_style.sh" ;;
    *)   CONFIG_FILE="${CONFIG}.sh" ;;
esac
CONFIG_PATH="$CP_ROOT/slurm/configs/$CONFIG_FILE"
[[ -f "$CONFIG_PATH" ]] || { echo "no such configuration: $CONFIG_PATH" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONFIG_PATH"
echo "=== configuration $CONFIG ($CONFIG_FILE): $R2V_CONDITIONING ==="

preflight "$MANIFEST_CSV" "$REPORT_INDEX_CSV" "$UNET_CHECKPOINT" "$VAE_CHECKPOINT" \
          "$PRETRAINED_DIR" "$SIF_IMAGE_TEXT"

TAG="${R2V_TAG:-r2v_${R2V_CONDITIONING}_${SLURM_JOB_ID:-local}}"
RUN_DIR="$WORKSPACE/runs/$TAG"
mkdir -p "$RUN_DIR"

MAX_STEPS="${R2V_MAX_STEPS:-4}"
EPOCHS="${R2V_EPOCHS:-1}"
NGPU="${R2V_NGPU:-1}"

ARGS=(
    --manifest "$MANIFEST_CSV"
    --report-index "$REPORT_INDEX_CSV"
    --base-checkpoint "$UNET_CHECKPOINT"
    --vae-checkpoint "$VAE_CHECKPOINT"
    --conditioning "$R2V_CONDITIONING"
    --max-report-tokens "$R2V_MAX_REPORT_TOKENS"
    --geometry-mode per_modality_plane
    --split train --val-split val
    --out "$RUN_DIR"
    --epochs "$EPOCHS"
    --batch-size "$R2V_BATCH_SIZE"
    --grad-accumulation-steps "$R2V_GRAD_ACCUM"
    --lr "${R2V_LR:-1e-5}"
    --report-dropout-probability 0.10
    --modality-dropout-probability 0.10
    --scale-factor auto
    --log-every "${R2V_LOG_EVERY:-10}"
    --save-format adapter
    --keep-last-n "${R2V_KEEP_LAST_N:-3}"
    --num-workers "${R2V_NUM_WORKERS:-8}"
    --seed "${R2V_SEED:-0}"
    --num-gpus "$NGPU"
    --device cuda
)
# Empty for configuration C, which encodes sections separately rather than joining them.
[[ -n "${R2V_REPORT_FORMAT:-}" ]] && ARGS+=(--report-format "$R2V_REPORT_FORMAT")
[[ "$MAX_STEPS" != "0" ]] && ARGS+=(--max-steps "$MAX_STEPS")
[[ -n "${R2V_SAVE_EVERY:-}" ]] && ARGS+=(--save-every-steps "$R2V_SAVE_EVERY")
[[ -n "${R2V_TEXT_POOLING:-}" ]] && ARGS+=(--text-pooling "$R2V_TEXT_POOLING")

# Validation is off by default so a smoke run stays minutes, not hours: each validation step runs a
# full diffusion sampler over --val-quick-samples cases.
if [[ -n "${R2V_VALIDATE_EVERY:-}" ]]; then
    ARGS+=(
        --validate-every-steps "$R2V_VALIDATE_EVERY"
        --val-quick-samples "${R2V_VAL_QUICK:-32}"
        --val-inference-steps "${R2V_VAL_STEPS:-30}"
        --val-visualize "${R2V_VAL_VISUALIZE:-4}"
        --val-seed "${R2V_VAL_SEED:-0}"
        --medicalnet-checkpoint "$MEDICALNET_CHECKPOINT"
    )
    [[ -n "${R2V_VALIDATE_FULL_EVERY:-}" ]] && ARGS+=(--validate-full-every-steps "$R2V_VALIDATE_FULL_EVERY")
    [[ -n "${R2V_VAL_FULL:-}" ]] && ARGS+=(--val-full-samples "$R2V_VAL_FULL")
else
    ARGS+=(--no-validate-at-end)
fi

# W&B off unless asked. `--wandb-log-reports` is separate and off by default: the validation panel
# embeds report text, which is patient data even after anonymisation.
WANDB_MODE_ARG="${R2V_WANDB:-disabled}"
ARGS+=(--wandb-mode "$WANDB_MODE_ARG")
if [[ "$WANDB_MODE_ARG" != "disabled" ]]; then
    ARGS+=(--wandb-project "${R2V_WANDB_PROJECT:-mr-rate-r2v}" --wandb-group "$R2V_CONDITIONING")
    [[ -n "${R2V_WANDB_ENTITY:-}" ]] && ARGS+=(--wandb-entity "$R2V_WANDB_ENTITY")
    [[ -n "${R2V_WANDB_LOG_REPORTS:-}" ]] && ARGS+=(--wandb-log-reports)
    # Added to APPTAINER_ARGS directly, not just exported: `setup` already built that array, so an
    # export here would stay on the host and the run inside the container would write its offline
    # data to the image's cwd instead of the run directory.
    export WANDB_DIR="$RUN_DIR"
    APPTAINER_ARGS+=(--env "WANDB_DIR=$RUN_DIR")
    [[ -n "${WANDB_API_KEY:-}" ]] && APPTAINER_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
fi

if [[ "$NGPU" -gt 1 ]]; then
    echo "=== DDP: torchrun --nproc_per_node=$NGPU ==="
    apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE_TEXT" \
        torchrun --nproc_per_node="$NGPU" --standalone \
        -m mrrate_r2v.cli.train_r2v "${ARGS[@]}"
else
    run_py_text mrrate_r2v.cli.train_r2v "${ARGS[@]}"
fi

echo "=== run directory ==="
ls -la "$RUN_DIR"
cat "$RUN_DIR/train_summary.json"
