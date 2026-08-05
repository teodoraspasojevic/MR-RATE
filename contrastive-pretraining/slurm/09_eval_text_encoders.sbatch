#!/bin/bash -l
# Score the cached embeddings into the encoder x report-format x metric matrix. CPU stage.
#
#   sbatch slurm/09_eval_text_encoders.sbatch [RUN_NAME] [TRAIN_LIMIT] [FUSION ...]
#   sbatch slurm/09_eval_text_encoders.sbatch budget512 20000 \
#          bioclinical_mbert+radbert medembed_large+radbert
#
# TRAIN_LIMIT must match what 08_embed_reports.sbatch ran with, or the runner fails on the
# study-id check rather than silently scoring a different sample.
#
# Runs on the host python, not in a container: the scoring stage needs scikit-learn, which neither
# SIF has. Nothing here loads a model or touches a GPU. See run_py_host in _common.sh.
#SBATCH --job-name=r2v_eval_textenc
#SBATCH --partition=cpu
# Helma rejects single-node CPU jobs that are not a multiple of 48 (one NUMA domain),
# so 48 is the floor even though BLAS is deliberately capped far below it -- see below.
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --account=y100dc
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_eval_textenc_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_eval_textenc_%j.err
unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup

RUN_NAME="${1:-budget512}"
TRAIN_LIMIT="${2:-20000}"
shift 2 || true

CACHE="$TEXTBENCH_ROOT/$RUN_NAME/embeddings"
OUT="$TEXTBENCH_ROOT/$RUN_NAME/results"
preflight "$REPORT_CORPUS" "$MANIFEST_CSV" "$CACHE"

FUSION_ARGS=()
for pair in "$@"; do FUSION_ARGS+=(--fusion "$pair"); done

# Deliberately a small thread count, not the whole allocation: this stage is hundreds of small
# logistic fits, where BLAS thread setup dominates. Measured 18 s per row at 8 threads and ~6
# minutes at 48. The CLI applies the same cap itself, before numpy is imported; this is here so
# the value is visible in the job log. See MRRATE_BLAS_THREADS in cli/eval_text_encoders.py.
export MRRATE_BLAS_THREADS="${MRRATE_BLAS_THREADS:-8}"

echo "run=$RUN_NAME train_limit=$TRAIN_LIMIT fusion=${*:-none} blas_threads=$MRRATE_BLAS_THREADS"
# Extra flags for one-off variants, e.g. the dimension-matched fusion control:
#   R2V_EVAL_ARGS="--mean-only" R2V_RESULT_SUFFIX=_meanonly sbatch slurm/09_... 
EXTRA_ARGS=()
[[ -n "${R2V_EVAL_ARGS:-}" ]] && read -r -a EXTRA_ARGS <<< "$R2V_EVAL_ARGS"
OUT="$OUT${R2V_RESULT_SUFFIX:-}"

MRRATE_BLAS_THREADS="$MRRATE_BLAS_THREADS" run_py_host mrrate_r2v.cli.eval_text_encoders \
    --corpus       "$REPORT_CORPUS" \
    --manifest-csv "$MANIFEST_CSV" \
    --cache        "$CACHE" \
    --out          "$OUT" \
    --train-limit  "$TRAIN_LIMIT" \
    "${EXTRA_ARGS[@]}" \
    "${FUSION_ARGS[@]}"

echo "=== metrics_matrix.csv ==="
# `| head` closes the pipe early, which under `set -o pipefail` makes a fully successful run
# report as FAILED in sacct. Truncate inside the pipeline instead of closing it.
column -t -s, "$OUT/metrics_matrix.csv" | awk "NR <= 40"
