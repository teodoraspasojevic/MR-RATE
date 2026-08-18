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

# Persistent artifacts.
#
# **There is no COHORT_ROOT and no PRED_ROOT any more.** `cli.evaluate` builds the Dataset from
# MANIFEST_CSV, generates and scores in one streaming pass, and writes only RESULT_ROOT -- a few
# MB of CSV and JSON per run instead of ~39 GB of frozen volumes plus ~18 GB of predictions. The
# case list is reproduced from the manifest on demand rather than stored, so nothing can go stale.
MANIFEST_CSV="$DATA_WORKSPACE/r2v_manifest/manifest_shards_native.csv"
REPORT_INDEX_CSV="$DATA_WORKSPACE/r2v_manifest/report_index_shards_native.csv"
RESULT_ROOT="$WORKSPACE/cache/r2v/results"
VAE_CHECKPOINT="$WORKSPACE/models/autoencoder_v1.pt"
# NVIDIA stores this path relative to cwd in its env config, so it must be given absolutely.
UNET_CHECKPOINT="$WORKSPACE/models/diff_unet_3d_rflow-mr-brain_v0.pt"
# torchvision's squeezenet1_1 weights (the challenge FID_2p5D feature extractor). Staged here
# rather than in $HOME/.cache so the containers see it and no job depends on a download.
TORCH_HOME_DIR="$WORKSPACE/pretrained/torchhub"

# RRZE's outbound HTTP proxy. Compute nodes have no direct route off-site, so anything that talks
# to the internet -- in practice only W&B in `online` mode -- needs this. The login node does have a
# direct route, which is why an interactive `curl api.wandb.ai` works and the same job on h200 does
# not: without the proxy `wandb.init(mode="online")` hangs, then degrades to a no-op with a warning
# (`eval/wandb_logging.py` catches it), so the failure looks like "W&B is just not logging".
#
# Set R2V_NO_PROXY=1 to skip this entirely.
#
# `no_proxy` covers loopback and the cluster's own domain. NCCL and torchrun's c10d rendezvous use
# raw sockets and are not proxied at all, so they are unaffected either way -- the entries are there
# so any *HTTP* call to a node (a local service, a health check) keeps working. The allocation's own
# nodes are appended by name because Python's `urllib`/`requests` match `no_proxy` by hostname
# suffix and do not understand CIDR blocks.
PROXY_URL="${R2V_PROXY_URL:-http://proxy.rrze.uni-erlangen.de:80}"

setup_proxy() {
    [[ -n "${R2V_NO_PROXY:-}" ]] && { echo "proxy: disabled (R2V_NO_PROXY set)"; return 0; }
    local no_proxy_list="localhost,127.0.0.1,::1,.nhr.fau.de,.fau.de,.rrze.uni-erlangen.de"
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        local host
        for host in $(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null); do
            no_proxy_list+=",${host},${host}.nhr.fau.de"
        done
    fi
    export http_proxy="$PROXY_URL"
    export https_proxy="$PROXY_URL"
    export HTTP_PROXY="$PROXY_URL"
    export HTTPS_PROXY="$PROXY_URL"
    export no_proxy="$no_proxy_list"
    export NO_PROXY="$no_proxy_list"
    echo "proxy: $PROXY_URL (no_proxy=$no_proxy_list)"
}

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
    setup_proxy
    preflight "$REPO_ROOT" "$CP_ROOT" "$WORKSPACE" "$SIF_IMAGE"
    mkdir -p "$WORKSPACE/slurm_logs" "$RESULT_ROOT"
    APPTAINER_ARGS=(--nv
        --bind "$REPO_ROOT:$REPO_ROOT"
        --bind "$WORKSPACE:$WORKSPACE"
        --bind "$DATA_WORKSPACE:$DATA_WORKSPACE"
        --bind "$HOME:$HOME"
        # --pwd + PYTHONPATH make `python3 -m mrrate_r2v.cli.*` resolvable regardless of where
        # the job was submitted from, instead of relying on apptainer inheriting the host cwd.
        --pwd "$CP_ROOT"
        --env "PYTHONPATH=$CP_ROOT"
        # The encoder zoo's default checkpoint root. Passed explicitly so it follows $WORKSPACE
        # rather than the hardcoded fallback in textenc/encoders.py -- otherwise overriding
        # R2V_WORKSPACE would move every path in this file except where encoders are loaded from.
        --env "MRRATE_PRETRAINED_DIR=$PRETRAINED_DIR"
        --env "TORCH_HOME=$TORCH_HOME_DIR")
    # The proxy has to be inside the container too: apptainer does not inherit the host environment
    # for these, so exporting them in `setup_proxy` alone leaves the *host* able to reach W&B and the
    # process that actually calls `wandb.init` unable to.
    local var
    for var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY; do
        [[ -n "${!var:-}" ]] && APPTAINER_ARGS+=(--env "$var=${!var}")
    done
    # W&B, only when the job asked for it. Credentials are never set here: wandb reads
    # WANDB_API_KEY or ~/.netrc (which is bind-mounted with $HOME), so nothing is hardcoded and a
    # job without credentials degrades to a no-op rather than failing.
    for var in WANDB_API_KEY WANDB_MODE WANDB_DIR WANDB_CACHE_DIR; do
        [[ -n "${!var:-}" ]] && APPTAINER_ARGS+=(--env "$var=${!var}")
    done
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
