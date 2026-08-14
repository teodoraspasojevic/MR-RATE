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
# Trains the report-conditioning adapter under one of the supported configurations
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
# Real 4-GPU run on one node (DDP via torchrun; R2V_NGPU is the TOTAL rank count):
#   sbatch --export=ALL,R2V_CONFIG=C,R2V_MAX_STEPS=0,R2V_NGPU=4,R2V_VALIDATE_EVERY=500 \
#          --gres=gpu:h200:4 --cpus-per-task=32 --time=24:00:00 slurm/11_train_conditioning.sbatch
#
# Multi-node (2 nodes x 4 GPUs = 8 ranks). --ntasks-per-node=1 is REQUIRED: one torchrun per node,
# which then forks one rank per GPU. R2V_NGPU must equal nodes x GPUs-per-node:
#   sbatch --export=ALL,R2V_CONFIG=A,R2V_MAX_STEPS=0,R2V_NGPU=8,R2V_VALIDATE_EVERY=500 \
#          --nodes=2 --ntasks-per-node=1 --gres=gpu:h200:4 --cpus-per-task=32 \
#          --time=24:00:00 slurm/11_train_conditioning.sbatch
#
# `#SBATCH --export=NONE` means a plain `VAR=x sbatch ...` does NOT reach the job -- always pass
# overrides through --export=ALL,... as above. R2V_MAX_STEPS=0 removes the step cap.

unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup

# ---------------------------------------------------------------- configuration
CONFIG="${R2V_CONFIG:?set R2V_CONFIG to A, B, C, D or E (see slurm/configs/README.md)}"
case "$CONFIG" in
    # The lineup. A is the pooled-conditioning baseline (one token, degenerate attention);
    # B and C keep the token axis; D is the sectioned multi-encoder fusion, and E is D plus a third
    # token carrying [MODALITY]/[PLANE]/[SPACING] -- the metadata A/B/C get from their format prefix
    # and D structurally cannot.
    #
    # **The letters changed on 2026-08-07.** C used to mean report2ct_style and B used to mean
    # radbert_mean. If you have an older submission script, re-read it before reusing it -- an
    # unchanged `R2V_CONFIG=C` now trains a different model, and nothing about that is loud.
    A|a) CONFIG_FILE="A_cxr_bert_cls.sh" ;;
    B|b) CONFIG_FILE="B_cxr_bert_tokens.sh" ;;
    C|c) CONFIG_FILE="C_radbert_tokens.sh" ;;
    D|d) CONFIG_FILE="D_report2ct_style.sh" ;;
    E|e) CONFIG_FILE="E_report2ct_style_meta.sh" ;;
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
# Empty for configurations D and E, which encode sections separately rather than joining them.
[[ -n "${R2V_REPORT_FORMAT:-}" ]] && ARGS+=(--report-format "$R2V_REPORT_FORMAT")
[[ "$MAX_STEPS" != "0" ]] && ARGS+=(--max-steps "$MAX_STEPS")
[[ -n "${R2V_SAVE_EVERY:-}" ]] && ARGS+=(--save-every-steps "$R2V_SAVE_EVERY")
[[ -n "${R2V_SAVE_EVERY_EPOCHS:-}" ]] && ARGS+=(--save-every-epochs "$R2V_SAVE_EVERY_EPOCHS")
[[ -n "${R2V_TEXT_POOLING:-}" ]] && ARGS+=(--text-pooling "$R2V_TEXT_POOLING")

# Resume. `R2V_EPOCHS` is how many epochs THIS job runs, not the total -- the trainer continues the
# absolute epoch numbering from the checkpoint. `R2V_RESUME_LR_SCHEDULE=restart` is what you want
# when extending a run that finished: its PolynomialLR is at exactly 0 and continuing it trains
# nothing. The trainer refuses that case rather than running it, so this cannot fail quietly.
if [[ -n "${R2V_RESUME:-}" ]]; then
    [[ -f "$R2V_RESUME" ]] || { echo "R2V_RESUME does not exist: $R2V_RESUME" >&2; exit 1; }
    ARGS+=(--resume "$R2V_RESUME"
           --resume-lr-schedule "${R2V_RESUME_LR_SCHEDULE:-continue}")
    echo "=== resuming from $R2V_RESUME (lr schedule: ${R2V_RESUME_LR_SCHEDULE:-continue}) ==="
fi

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
        --val-metrics ${R2V_VAL_METRICS:-fvd fid_2p5d ssim}
        --val-fvd-extractor "${R2V_VAL_FVD_EXTRACTOR:-r3d18}"
        --val-feature-cache "$WORKSPACE/cache/r2v/val_features"
        --torch-home "$TORCH_HOME_DIR"
    )
    # Condition-sensitivity diagnostic: swaps in another study's report at a fixed seed to prove the
    # model uses its text at all. Costs a second generation per case, so it runs on a small subset.
    [[ -n "${R2V_SENSITIVITY_EVERY:-}" ]] && ARGS+=(
        --val-sensitivity-every-steps "$R2V_SENSITIVITY_EVERY"
        --val-sensitivity-samples "${R2V_SENSITIVITY_N:-8}")
    [[ -n "${R2V_VALIDATION_REFERENCE:-}" ]] && ARGS+=(
        --validation-reference "$R2V_VALIDATION_REFERENCE")
    [[ -n "${R2V_VALIDATE_FULL_EVERY:-}" ]] && ARGS+=(--validate-full-every-steps "$R2V_VALIDATE_FULL_EVERY")
    [[ -n "${R2V_VAL_FULL:-}" ]] && ARGS+=(--val-full-samples "$R2V_VAL_FULL")
    [[ -n "${R2V_VALIDATE_FULL_AT_END:-}" ]] && ARGS+=(--validate-full-at-end)
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
    # wandb's own service handshake, over a shared filesystem inside a container, occasionally
    # exceeds the 30 s default and then `wandb.init` raises -- which `WandbRun` catches, so the run
    # continues with no logging at all. 300 s costs nothing when things are fine.
    APPTAINER_ARGS+=(--env "WANDB__SERVICE_WAIT=300")
fi

if [[ "$WANDB_MODE_ARG" == "online" ]]; then
    # Fail before the GPU work rather than after it. `online` mode is the only thing in this job that
    # needs the internet, and an unreachable W&B degrades to silence (see `setup_proxy`), so the run
    # would burn its whole walltime and produce no curves. 401/404 both mean "reached the server".
    # Probed with `requests` inside the container, not host `curl`: requests is the library wandb
    # itself uses, so this tests the exact proxy handling the real call will get.
    echo "=== W&B online: probing api.wandb.ai through the proxy ==="
    code="$(apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE_TEXT" python3 -c \
        'import requests,sys
try: sys.stdout.write(str(requests.get("https://api.wandb.ai/", timeout=30).status_code))
except Exception as e: sys.stdout.write("000"); sys.stderr.write(f"{type(e).__name__}: {e}\n")' || echo 000)"
    echo "    api.wandb.ai -> HTTP $code"
    if [[ "$code" == "000" ]]; then
        echo "W&B is unreachable from inside the container (proxy=$PROXY_URL). Either fix the" >&2
        echo "proxy, or submit with R2V_WANDB=offline and 'wandb sync $RUN_DIR/wandb' afterwards." >&2
        exit 1
    fi
    [[ -n "${WANDB_API_KEY:-}" || -f "$HOME/.netrc" ]] || {
        echo "W&B online needs credentials: export WANDB_API_KEY, or 'wandb login' once to write" >&2
        echo "~/.netrc (it is bind-mounted into the container with \$HOME)." >&2
        exit 1; }
fi

# ---------------------------------------------------------------- launch
#
# Three launch shapes. `--num-gpus` is the **total world size** (ranks), not GPUs per node, because
# that is what the Python side cross-checks against torchrun's WORLD_SIZE.
#
#   1 rank              plain python, no process group
#   1 node, N ranks     torchrun --standalone   (its own rendezvous, no network setup)
#   M nodes, N per node srun + torchrun c10d rendezvous on the first node
#
# The multi-node form follows the NHR recipe in docs/nhr_official_docs/apps_pytorch.md verbatim:
# one *task* per node (`--ntasks-per-node=1`), each task running torchrun, which then forks
# `--nproc-per-node` ranks locally. Requesting one task per GPU instead would start N torchruns per
# node, each spawning N ranks -- N^2 processes fighting over N GPUs.
NNODES="${SLURM_JOB_NUM_NODES:-1}"

if [[ "$NNODES" -gt 1 ]]; then
    GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-$(( NGPU / NNODES ))}"
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
    # Derived from the job id so two concurrent multi-node jobs cannot collide on a port.
    MASTER_PORT=$(( 29400 + SLURM_JOB_ID % 1000 ))
    echo "=== DDP multi-node: $NNODES nodes x $GPUS_PER_NODE GPUs = $((NNODES * GPUS_PER_NODE)) ranks ==="
    echo "    rendezvous ${MASTER_ADDR}:${MASTER_PORT} (c10d, id $SLURM_JOB_ID)"
    [[ "$NGPU" -eq $(( NNODES * GPUS_PER_NODE )) ]] || {
        echo "R2V_NGPU=$NGPU but the allocation gives $((NNODES * GPUS_PER_NODE)) ranks" >&2; exit 1; }
    # NCCL over InfiniBand. Left to NCCL's own detection; set NCCL_DEBUG=INFO to see what it chose.
    srun --cpu-bind=verbose --ntasks-per-node=1 \
        apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE_TEXT" \
        torchrun \
            --nnodes="$NNODES" \
            --nproc-per-node="$GPUS_PER_NODE" \
            --rdzv-id="$SLURM_JOB_ID" \
            --rdzv-backend=c10d \
            --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            -m mrrate_r2v.cli.train_r2v "${ARGS[@]}"
elif [[ "$NGPU" -gt 1 ]]; then
    echo "=== DDP single-node: torchrun --standalone --nproc_per_node=$NGPU ==="
    apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE_TEXT" \
        torchrun --nproc_per_node="$NGPU" --standalone \
        -m mrrate_r2v.cli.train_r2v "${ARGS[@]}"
else
    run_py_text mrrate_r2v.cli.train_r2v "${ARGS[@]}"
fi

echo "=== run directory ==="
ls -la "$RUN_DIR"
cat "$RUN_DIR/train_summary.json"
