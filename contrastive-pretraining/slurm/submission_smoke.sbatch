#!/bin/bash -l
#SBATCH --job-name=r2v_sub_smoke
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_sub_smoke_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_sub_smoke_%j.err
#
# Runs `submission/predict.py` -- the challenge container's entry point -- end to end on real GPU
# hardware, against `submission/mock_input/prompts.json`.
#
#   sbatch slurm/submission_smoke.sbatch <ws>/runs/<tag>/adapter_step0004200.pt
#
# This does NOT build or run the Docker image (Helma has no Docker, only Apptainer). What it does
# validate is everything inside the image that can fail: the prompt parser, the case_id parser, the
# per-bucket geometry, the metadata prefix, the section splitter, checkpoint/resume, and the actual
# sampler and decoder call. A Docker build that fails after this passes fails on packaging, which is
# the cheap half to debug.
#
# It exercises the container's own environment contract -- FORITHMUS_INPUT/OUTPUT/CHECKPOINT and the
# R2V_* knobs -- rather than a CLI flag path, so the thing under test is the thing that ships.
#
#   R2V_STEPS=10                           fewer diffusion steps for a faster check
#   R2V_SMOKE_RESUME=1                     second pass, to prove /checkpoint resume skips the done
#
# Override with `sbatch --export=ALL,R2V_STEPS=10 slurm/submission_smoke.sbatch <adapter>`,
# because `--export=NONE` means a plain `VAR=x sbatch ...` never reaches the job.

unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup

ADAPTER="${1:-$WORKSPACE/runs/r2v_final_C_radbert_tokens/adapter_step0004200.pt}"
SUBMISSION_DIR="$REPO_ROOT/submission"
OUT_DIR="${R2V_SMOKE_OUT:-$WORKSPACE/cache/r2v/submission_smoke/out}"
CKPT_DIR="${R2V_SMOKE_CKPT:-$WORKSPACE/cache/r2v/submission_smoke/checkpoint}"

# `routing.py` was folded into predict.py by the declutter -- the modality/plane for a case now
# comes from parsing its case_id, not from a distribution-matching router, so there is no second
# module to check for. Leaving it here aborted every smoke in pre-flight.
preflight "$ADAPTER" "$VAE_CHECKPOINT" "$UNET_CHECKPOINT" \
          "$SUBMISSION_DIR/predict.py" \
          "$SUBMISSION_DIR/mock_input/prompts.json"

# A resume run must find the previous run's /checkpoint; a fresh run must not, or it would report
# success without generating anything.
if [[ -z "${R2V_SMOKE_RESUME:-}" ]]; then
    rm -rf "$OUT_DIR" "$CKPT_DIR"
fi
mkdir -p "$OUT_DIR" "$CKPT_DIR"

echo "adapter : $ADAPTER"
echo "output  : $OUT_DIR"
echo "resume  : ${R2V_SMOKE_RESUME:-0}"

# The container sets these in its Dockerfile/entrypoint; here they are passed explicitly so the
# code path is identical. R2V_MODELS_DIR stands in for the /weights -> /opt/app/models bridge.
apptainer exec "${APPTAINER_ARGS[@]}" \
    --env "FORITHMUS_INPUT=$SUBMISSION_DIR/mock_input" \
    --env "FORITHMUS_OUTPUT=$OUT_DIR" \
    --env "FORITHMUS_CHECKPOINT=$CKPT_DIR" \
    --env "FORITHMUS_WEIGHTS=$WORKSPACE/models" \
    --env "R2V_MODELS_DIR=$WORKSPACE/models" \
    --env "R2V_ADAPTER=$ADAPTER" \
    --env "R2V_BASE_CHECKPOINT=$UNET_CHECKPOINT" \
    --env "R2V_VAE_CHECKPOINT=$VAE_CHECKPOINT" \
    --env "R2V_NUM_INFERENCE_STEPS=${R2V_STEPS:-30}" \
    --env "R2V_CHECKPOINT_EVERY=1" \
    "$SIF_IMAGE_TEXT" python3 "$SUBMISSION_DIR/predict.py"

echo
echo "=== outputs ==="
ls -lh "$OUT_DIR"
echo
echo "=== headers (shape / dtype / spacing must match the routed bucket) ==="
apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE_TEXT" python3 - "$OUT_DIR" <<'PY'
import sys, pathlib, nibabel as nib, numpy as np
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.nii.gz")):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    spacing = tuple(round(float(v), 3) for v in img.header.get_zooms()[:3])
    finite = bool(np.isfinite(data).all())
    print(f"{path.name:26s} {str(data.shape):>18s} {str(data.dtype):>8s} "
          f"spacing={spacing} range=[{data.min():.1f}, {data.max():.1f}] finite={finite}")
    assert finite, f"{path.name} contains non-finite voxels"
    assert data.max() > data.min(), f"{path.name} is constant"
print("all volumes finite and non-constant")
PY

echo
echo "=== resume state ==="
# `run_manifest.json` is gone with the old predict.py. The resume contract is now `done.json` (the
# filenames already generated) plus `outputs/` (their bytes), which is what `load_done` reads back
# on a restart -- so that pair is what a smoke has to show.
cat "$CKPT_DIR/done.json"
echo
echo "backed-up volumes: $(ls "$CKPT_DIR/outputs" | wc -l)"
