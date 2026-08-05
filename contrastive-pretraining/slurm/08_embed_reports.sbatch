#!/bin/bash -l
# Cache text-encoder embeddings for the selection benchmark. GPU stage.
#
#   sbatch slurm/08_embed_reports.sbatch [MAX_LENGTH] [TRAIN_LIMIT] [RUN_NAME] [ENCODER ...]
#   sbatch slurm/08_embed_reports.sbatch 512 20000 budget512
#   sbatch slurm/08_embed_reports.sbatch 512 20000 budget512 radbert bioclinical_mbert
#
# With no ENCODER given, every staged encoder is embedded in one job. Naming a subset lets you
# fan the work out over several jobs writing into the same cache directory -- they touch disjoint
# files, so there is no coordination to get wrong. When the GPU partitions are backed up, the same
# script runs on CPU with the directives overridden on the command line:
#
#   for E in radbert bioclinical_mbert medembed_large medembed_small; do
#     sbatch --partition=cpu --gres=NONE --cpus-per-task=32 --time=08:00:00 \
#            slurm/08_embed_reports.sbatch 512 20000 budget512 $E
#   done
#
# MAX_LENGTH is applied to every encoder so the matrix compares them at one token budget.
# Pass "native" to let each encoder use its own default instead (8192-context models then read
# every report in full -- run that as a *separate* RUN_NAME, never mixed into the same cache).
#SBATCH --job-name=r2v_embed_reports
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_embed_reports_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_embed_reports_%j.err
unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup

MAX_LENGTH="${1:-512}"
TRAIN_LIMIT="${2:-20000}"
RUN_NAME="${3:-budget${MAX_LENGTH}}"
shift 3 2>/dev/null || true
ENCODER_ARGS=()
[[ $# -gt 0 ]] && ENCODER_ARGS=(--encoders "$@")
OUT="$TEXTBENCH_ROOT/$RUN_NAME/embeddings"

preflight "$REPORT_CORPUS" "$MANIFEST_CSV" "$PRETRAINED_DIR"

LENGTH_ARGS=(--max-length "$MAX_LENGTH")
[[ "$MAX_LENGTH" == "native" ]] && LENGTH_ARGS=()

# Seven of the eight formats. `findings_impression_meta` is deliberately absent: its prefix is a
# (modality, plane) pair, which is a property of a *series*, while this benchmark's unit is a
# *study* -- one report covers a median of 6-7 series across several modalities, so there is no
# single correct prefix to give it here. Without a prefix it is byte-identical to
# findings_impression, so embedding it would spend an eighth of the GPU budget duplicating a row.
# Evaluate that format end-to-end through cli.train_r2v instead, where the unit really is a series.
FORMATS=(raw findings impression findings_impression impression_findings
         clinical_findings_impression full_structured)
# Override to split one slow encoder's formats across two jobs writing into the same cache
# directory: R2V_FORMATS="full_structured clinical_findings_impression" sbatch ...
[[ -n "${R2V_FORMATS:-}" ]] && read -r -a FORMATS <<< "$R2V_FORMATS"
# Extra flags for one-off variants, e.g. the CLS-only study: R2V_EMBED_ARGS="--pooling cls"
EMBED_EXTRA=()
[[ -n "${R2V_EMBED_ARGS:-}" ]] && read -r -a EMBED_EXTRA <<< "$R2V_EMBED_ARGS"

echo "run=$RUN_NAME max_length=$MAX_LENGTH train_limit=$TRAIN_LIMIT encoders=${*:-all} -> $OUT"
# HF_HUB_OFFLINE: every checkpoint is staged locally; a hub call here would be a bug, not a fallback.
# Pin the thread pool to this job's allocation. Without it, several jobs sharing a CPU node each
# size their pool from the node's total core count and oversubscribe it by their own multiplicity.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
APPTAINER_ARGS+=(--env "HF_HUB_OFFLINE=1" --env "MRRATE_PRETRAINED_DIR=$PRETRAINED_DIR"
                 --env "OMP_NUM_THREADS=$OMP_NUM_THREADS"
                 --env "MKL_NUM_THREADS=$OMP_NUM_THREADS"
                 --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}")

run_py_text mrrate_r2v.cli.embed_reports \
    --corpus       "$REPORT_CORPUS" \
    --manifest-csv "$MANIFEST_CSV" \
    --out          "$OUT" \
    --train-limit  "$TRAIN_LIMIT" \
    --batch-size   64 \
    --negation-pairs 4000 \
    --pretrained-dir "$PRETRAINED_DIR" \
    --formats "${FORMATS[@]}" \
    "${ENCODER_ARGS[@]}" \
    "${EMBED_EXTRA[@]}" \
    "${LENGTH_ARGS[@]}"

echo "=== caches ==="
ls -1 "$OUT" | head -20
echo "next: sbatch slurm/09_eval_text_encoders.sbatch $RUN_NAME $TRAIN_LIMIT"
