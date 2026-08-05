#!/bin/bash -l
#SBATCH --job-name=r2v_valref
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_valref_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_valref_%j.err
#
# Computes the fixed reference values the validation curves are read against, ONCE, and caches them
# to JSON. Not a model score: SSIM identity/perturbation sanity checks, the autoencoder
# reconstruction reference, and the real-vs-real FVD / 2.5D FID noise floors.
#
#   sbatch --export=ALL,R2V_VALREF_N=64 slurm/12_validation_reference.sbatch
#
# Pass the resulting JSON to training with --validation-reference. Use the SAME --n-samples and
# --seed as --val-quick-samples / --val-seed, or the noise floor describes a different sample than
# the curve.

unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup
preflight "$MANIFEST_CSV" "$REPORT_INDEX_CSV" "$VAE_CHECKPOINT" "$MEDICALNET_CHECKPOINT" \
          "$SIF_IMAGE_TEXT"

N="${R2V_VALREF_N:-64}"
SEED="${R2V_VALREF_SEED:-0}"
OUT="$WORKSPACE/cache/r2v/validation_reference_n${N}_seed${SEED}.json"
mkdir -p "$(dirname "$OUT")"

run_py_text mrrate_r2v.cli.validation_reference \
    --manifest "$MANIFEST_CSV" \
    --report-index "$REPORT_INDEX_CSV" \
    --split "${R2V_VALREF_SPLIT:-val}" \
    --n-samples "$N" \
    --seed "$SEED" \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --medicalnet-checkpoint "$MEDICALNET_CHECKPOINT" \
    --torch-home "$TORCH_HOME_DIR" \
    --device cuda \
    --out "$OUT"

echo "=== pass this to training ==="
echo "  --validation-reference $OUT"
