#!/bin/sh
# Bridge the platform's /weights mount into the paths predict.py and the text-encoder zoo expect,
# then hand over to predict.py as PID 1.
#
# Splitting on file-vs-directory is the whole trick, and it means weights.zip needs no manifest:
#
#   /weights/<name>.pt   (a file)       -> /opt/app/models/<name>.pt        checkpoints
#   /weights/<Name>/     (a directory)  -> /opt/app/pretrained/<Name>/      HF text encoders
#
# `MRRATE_PRETRAINED_DIR=/opt/app/pretrained` (set in the Dockerfile) is what makes the second one
# work for every adapter arm without naming encoders here: `rebuild_embedder` re-resolves each
# encoder directory from that variable rather than from the absolute cluster path recorded in the
# checkpoint, so a three-encoder arm needs no extra wiring.
set -eu

WEIGHTS_DIR="${FORITHMUS_WEIGHTS:-/weights}"
MODELS_DIR="${R2V_MODELS_DIR:-/opt/app/models}"
PRETRAINED_DIR="${MRRATE_PRETRAINED_DIR:-/opt/app/pretrained}"
OUTPUT_DIR="${FORITHMUS_OUTPUT:-/output}"
CHECKPOINT_DIR="${FORITHMUS_CHECKPOINT:-/checkpoint}"

# The platform guarantees the /output symlink, not the leaf directory (organizers' README §9).
mkdir -p "$MODELS_DIR" "$PRETRAINED_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"

if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "[entrypoint] FATAL: $WEIGHTS_DIR is not mounted. Submit with --weights weights.zip." >&2
    exit 1
fi

for path in "$WEIGHTS_DIR"/*; do
    [ -e "$path" ] || continue
    name=$(basename "$path")
    if [ -d "$path" ]; then
        # A `pretrained/` wrapper directory is unwrapped one level so both zip layouts work:
        # encoder directories at the zip root, or collected under `pretrained/`.
        if [ "$name" = "pretrained" ]; then
            for inner in "$path"/*; do
                [ -e "$inner" ] || continue
                ln -sfn "$inner" "$PRETRAINED_DIR/$(basename "$inner")"
            done
        else
            ln -sfn "$path" "$PRETRAINED_DIR/$name"
        fi
    else
        ln -sfn "$path" "$MODELS_DIR/$name"
    fi
done

echo "[entrypoint] checkpoints in $MODELS_DIR:"
ls -lL "$MODELS_DIR" 2>&1 | sed 's/^/[entrypoint]   /'
echo "[entrypoint] text encoders in $PRETRAINED_DIR:"
ls -L "$PRETRAINED_DIR" 2>&1 | sed 's/^/[entrypoint]   /'
echo "[entrypoint] input mount ${FORITHMUS_INPUT:-/input}:"
ls -l "${FORITHMUS_INPUT:-/input}" 2>&1 | head -20 | sed 's/^/[entrypoint]   /'

# Sanity check + GPU-count detection in one Python startup, so a broken image says so immediately
# and legibly instead of failing deep inside model loading. `torch.cuda.is_available()` silently
# returning False (no libcudart, CPU base image, --gpus not wired) is the organizers' own
# documented first troubleshooting entry -- it would otherwise burn the whole time budget on CPU
# rather than fail. Detecting GPU count here (rather than hardcoding 1) means today's confirmed
# single-accelerator tiers (Docker_Submission.md lists none with >1 GPU) run exactly as before, and
# predict.py's DDP support (see its ddp_setup()) is exercised automatically if a multi-GPU tier
# ever exists -- mirrors the CT track's own "detect instead of hardcode" pattern in its entrypoint.sh.
echo "[entrypoint] sanity check: critical imports + GPU visibility"
NPROC=$(python - <<'PYEOF' || exit 1
import sys
try:
    import numpy, torch
except ImportError as exc:
    print(f"[entrypoint] FATAL: critical import failed: {exc}", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("[entrypoint] FATAL: torch.cuda.is_available() is False -- no GPU visible to this "
          "container. Generation needs a GPU; running on CPU would silently burn the whole time "
          "budget instead of failing loudly. Check --gpus / nvidia-container-toolkit wiring.",
          file=sys.stderr)
    sys.exit(1)
n = torch.cuda.device_count()
print(f"[entrypoint] sanity check OK: numpy {numpy.__version__}, torch {torch.__version__}, "
      f"{n} GPU(s) visible", file=sys.stderr)
print(n)
PYEOF
)
case "$NPROC" in ''|*[!0-9]*) echo "[entrypoint] FATAL: sanity check produced no GPU count" >&2; exit 1 ;; esac

# `exec` matters beyond tidiness: predict.py (as PID 1, directly or as torchrun's child) must
# receive the platform's SIGTERM so its handler runs, and so /output is flushed before the
# container is torn down.
if [ "$NPROC" -gt 1 ]; then
    echo "[entrypoint] $NPROC GPU(s) visible -- launching under torchrun"
    exec torchrun --standalone --nproc_per_node="$NPROC" /opt/app/predict.py
else
    exec python /opt/app/predict.py
fi
