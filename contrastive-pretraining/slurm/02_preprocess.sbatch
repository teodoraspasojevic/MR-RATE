#!/bin/bash -l
#SBATCH --job-name=r2v_preprocess
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --account=y100dc
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_preprocess_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_preprocess_%j.err
#
# Build ONE frozen ground-truth cohort. CPU only -- no GPU needed or requested.
# Run this once per experiment set; every predict/evaluate job then reuses the result.
#
#   sbatch slurm/02_preprocess.sbatch <cohort_name> [n_per_sequence]
#
#   sbatch slurm/02_preprocess.sbatch test_v1 200     # 200 per sequence
#   sbatch slurm/02_preprocess.sbatch test_full 0     # 0 = every eligible case
#
# Disk: ~67 MB per case at 256^3 float32. 4 sequences x 200 = ~54 GB. Check your quota first.

unset SLURM_EXPORT_ENV
set -euo pipefail

# Absolute path, not $(dirname "$BASH_SOURCE"): Slurm runs a COPY of this script from
# /var/tmp/slurmd_spool/job<id>/, where _common.sh does not exist. Override with R2V_REPO=/path.
R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup
preflight "$MANIFEST_CSV" "$REPORT_INDEX_CSV"

NAME="${1:?usage: sbatch 02_preprocess.sbatch <cohort_name> [n_per_sequence]}"
N="${2:-200}"
COHORT="$COHORT_ROOT/$NAME"

N_ARG=(--n-per-sequence "$N")
[[ "$N" == "0" ]] && N_ARG=()   # omitting the flag means "every eligible case"

echo "cohort=$COHORT  n_per_sequence=${N:-all}"
df -h "$COHORT_ROOT" | tail -1

run_py mrrate_r2v.cli.preprocess \
    --manifest-csv "$MANIFEST_CSV" --report-index-csv "$REPORT_INDEX_CSV" \
    --split test --sequences T1w T2w FLAIR SWI "${N_ARG[@]}" \
    --series-selection one_per_study_per_sequence \
    --geometry-mode fixed --seed 42 \
    --out "$COHORT" --overwrite

echo
echo "=== cohort contract ==="
cat "$COHORT/cohort.json"
du -sh "$COHORT"
echo
echo "Pass --cohort $COHORT to every predict job for this experiment set."
