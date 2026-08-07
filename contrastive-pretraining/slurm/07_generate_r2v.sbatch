#!/bin/bash -l
#SBATCH --job-name=r2v_generate
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_generate_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_generate_%j.err
#
# Report-to-volume inference with a trained adapter, through NVIDIA's own sampler and decoder.
#
#   sbatch slurm/07_generate_r2v.sbatch <ws>/runs/<tag>/adapter_last.pt
#
# The adapter path is positional because `#SBATCH --export=NONE` (FAU's recommended setting, kept
# from the other scripts here) means a plain `VAR=x sbatch ...` does NOT reach the job. Everything
# else below is an environment variable, so override those with
# `sbatch --export=ALL,R2V_STEPS=10 slurm/07_generate_r2v.sbatch <adapter>`.
#
# Variants:
#   R2V_REPORT_GUIDANCE=0        modality-only generation, i.e. NVIDIA's original behaviour
#   R2V_REPORT_GUIDANCE=1        the plain report-conditioned prediction
#   R2V_MODALITY_GUIDANCE=0      report guidance only, modality left conditioned
#   R2V_LATENT_ONLY=1            skip the VAE decode (cheap check of the diffusion loop)
#   R2V_COHORT=<cohort dir>      one volume per cohort case from its paired MR-RATE report,
#                                writing a prediction set that cli.evaluate can score

unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup

ADAPTER="${1:-${R2V_ADAPTER:-}}"
[[ -n "$ADAPTER" ]] || { echo "usage: sbatch $0 <adapter_checkpoint.pt>" >&2; exit 1; }
preflight "$ADAPTER" "$UNET_CHECKPOINT" "$VAE_CHECKPOINT" "$RADBERT_CHECKPOINT" "$SIF_IMAGE_TEXT"

OUT_DIR="$WORKSPACE/samples/${R2V_TAG:-r2v_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT_DIR"
REPORT="${R2V_REPORT:-Findings: A 12 mm peripherally enhancing lesion is seen in the right frontal lobe with surrounding vasogenic oedema. No midline shift. Impression: Solitary enhancing mass, most consistent with metastasis.}"

COMMON=(--base-checkpoint "$UNET_CHECKPOINT"
        --vae-checkpoint "$VAE_CHECKPOINT"
        --adapter "$ADAPTER"
        --text-checkpoint "$RADBERT_CHECKPOINT"
        --modality "${R2V_MODALITY:-T1w}"
        --plane "${R2V_PLANE:-AXIAL}"
        --num-inference-steps "${R2V_STEPS:-30}"
        --report-guidance-scale "${R2V_REPORT_GUIDANCE:-4.0}"
        --modality-guidance-scale "${R2V_MODALITY_GUIDANCE:-10.0}"
        --seed "${R2V_SEED:-1234}"
        --device cuda)
[[ "${R2V_LATENT_ONLY:-0}" == "1" ]] && COMMON+=(--latent-only)

# --dim/--spacing are passed only when explicitly set. Their default is the (modality, plane)
# bucket's own trained grid, and hardcoding 256^3 @ 1 mm here would override that for every bucket
# -- conditioning the model on a spacing it never trained on for that pair, and putting that same
# wrong spacing into the [SPACING] marker under a *_meta format.
GEOMETRY=()
[[ -n "${R2V_DIM_X:-}" ]] && GEOMETRY+=(--dim "$R2V_DIM_X" "${R2V_DIM_Y:?set all three}" "${R2V_DIM_Z:?set all three}")
[[ -n "${R2V_SPACING_X:-}" ]] && GEOMETRY+=(--spacing "$R2V_SPACING_X" "${R2V_SPACING_Y:?set all three}" "${R2V_SPACING_Z:?set all three}")

if [[ -n "${R2V_COHORT:-}" ]]; then
    run_py_text mrrate_r2v.cli.generate_r2v "${COMMON[@]}" --cohort "$R2V_COHORT" --out-dir "$OUT_DIR"
else
    run_py_text mrrate_r2v.cli.generate_r2v "${COMMON[@]}" ${GEOMETRY+"${GEOMETRY[@]}"} \
        --report "$REPORT" \
        --out "$OUT_DIR/sample.nii.gz"
fi

echo "=== output ==="
ls -la "$OUT_DIR"
