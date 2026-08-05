#!/bin/bash
# Shared setup for every mrrate_r2v Slurm job. Sourced, never run directly.
#
# Defines the paths, the apptainer invocation, and a `run_py` helper so no job script repeats
# them. Change a path here and every job follows.
#
# Every .sbatch script sources this by ABSOLUTE path, via $R2V_REPO. That is not laziness:
# Slurm copies the batch script to /var/tmp/slurmd_spool/job<id>/slurm_script before running it,
# so `dirname "${BASH_SOURCE[0]}"` inside an sbatch script points at the spool directory and can
# never find this file. Override the repo location with `R2V_REPO=/path sbatch ...` if you move it.

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
REPO_ROOT="$R2V_REPO"
CP_ROOT="$REPO_ROOT/contrastive-pretraining"
WORKSPACE="${R2V_WORKSPACE:-/hnvme/workspace/y100dc19-nvidia-mri-brain}"
DATA_WORKSPACE="${R2V_DATA_WORKSPACE:-/hnvme/workspace/y100dc19-MR-Rate-raw}"
SIF_IMAGE="$WORKSPACE/containers/nvidia_mri_vae_dmvae.sif"
# Same image plus transformers/safetensors, for the report-conditioning jobs (the base image has no
# text-encoder stack). Its torch is >= 2.6, so RadBERT's pytorch_model.bin loads directly.
SIF_IMAGE_TEXT="$WORKSPACE/containers/nvidia+redbert.sif"
RADBERT_CHECKPOINT="$WORKSPACE/pretrained/RadBERT-RoBERTa-4m"
PRETRAINED_DIR="$WORKSPACE/pretrained"
# The text-encoder benchmark's CPU scoring stage needs scikit-learn, which neither container has.
# It runs on the host module-system python instead (see run_py_host); the GPU embedding stage
# still runs in SIF_IMAGE_TEXT like every other model job.
HOST_PYTHON="${R2V_HOST_PYTHON:-/apps/python/3.12-conda/envs/pytorch2.5.1/bin/python3}"
TEXTBENCH_ROOT="$WORKSPACE/cache/r2v/textbench"
REPORT_CORPUS="$WORKSPACE/cache/r2v/report_analysis/reports_all.jsonl"

# Persistent artifacts. Cohorts and predictions are large -- workspace only, never git or $HOME.
MANIFEST_CSV="$DATA_WORKSPACE/r2v_manifest/manifest_shards_native.csv"
REPORT_INDEX_CSV="$DATA_WORKSPACE/r2v_manifest/report_index_shards_native.csv"
COHORT_ROOT="$WORKSPACE/cache/r2v/cohorts"
PRED_ROOT="$WORKSPACE/cache/r2v/predictions"
RESULT_ROOT="$WORKSPACE/cache/r2v/results"
VAE_CHECKPOINT="$WORKSPACE/models/autoencoder_v1.pt"
# NVIDIA stores this path relative to cwd in its env config, so it must be given absolutely.
UNET_CHECKPOINT="$WORKSPACE/models/diff_unet_3d_rflow-mr-brain_v0.pt"
MEDICALNET_CHECKPOINT="$WORKSPACE/pretrained/medicalnet/resnet_10_23dataset_statedict.pth"

preflight() {
    local fail=0 p
    for p in "$@"; do
        [[ -e "$p" ]] || { echo "MISSING: $p"; fail=1; }
    done
    [[ "$fail" -eq 0 ]] || { echo "Pre-flight failed -- aborting before any GPU work." >&2; exit 1; }
}

setup() {
    echo "=== job ${SLURM_JOB_ID:-local} on $(hostname -f) ==="
    echo "R2V_REPO=$R2V_REPO"
    preflight "$REPO_ROOT" "$CP_ROOT" "$WORKSPACE" "$SIF_IMAGE" "$REPO_ROOT/NV-Generate-CTMR"
    mkdir -p "$WORKSPACE/slurm_logs" "$COHORT_ROOT" "$PRED_ROOT" "$RESULT_ROOT"
    APPTAINER_ARGS=(--nv
        --bind "$REPO_ROOT:$REPO_ROOT"
        --bind "$WORKSPACE:$WORKSPACE"
        --bind "$DATA_WORKSPACE:$DATA_WORKSPACE"
        --bind "$HOME:$HOME"
        # --pwd + PYTHONPATH make `python3 -m mrrate_r2v.cli.*` resolvable regardless of where
        # the job was submitted from, instead of relying on apptainer inheriting the host cwd.
        --pwd "$CP_ROOT"
        --env "PYTHONPATH=$CP_ROOT")
    cd "$CP_ROOT"
}

# run_py <module> [args...] -- runs a mrrate_r2v CLI inside the container from $CP_ROOT.
run_py() {
    [[ -n "${APPTAINER_ARGS+x}" ]] || { echo "run_py called before setup()" >&2; exit 1; }
    apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE" python3 -m "$@"
}

# run_sh <command...> -- same container and environment, for anything that is not `python3 -m`.
run_sh() {
    [[ -n "${APPTAINER_ARGS+x}" ]] || { echo "run_sh called before setup()" >&2; exit 1; }
    apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE" "$@"
}

# run_py_text <module> [args...] -- like run_py, in the image that also has the text encoder stack.
run_py_text() {
    [[ -n "${APPTAINER_ARGS+x}" ]] || { echo "run_py_text called before setup()" >&2; exit 1; }
    apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE_TEXT" python3 -m "$@"
}

# run_py_host <module> [args...] -- no container, host python, for the CPU-only scoring stages
# that need scikit-learn. Never use this for anything touching a model checkpoint or a GPU:
# the containers exist precisely so those runs are reproducible.
run_py_host() {
    [[ -x "$HOST_PYTHON" ]] || { echo "HOST_PYTHON not executable: $HOST_PYTHON" >&2; exit 1; }
    PYTHONPATH="$CP_ROOT" "$HOST_PYTHON" -m "$@"
}
