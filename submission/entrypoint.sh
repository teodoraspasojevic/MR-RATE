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

# `exec` matters beyond tidiness: predict.py must be PID 1 so the platform's SIGTERM reaches its
# handler directly, and so gcsfuse flushes /output before Vertex tears the container down.
exec python /opt/app/predict.py
