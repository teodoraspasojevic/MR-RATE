#!/bin/bash -l
#SBATCH --job-name=r2v_preprocess
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
# 48, not fewer: Helma's `cpu` partition rejects anything that is not a multiple of 48
# (full NUMA domains). Preprocessing is a serial per-case loop, so most of these cores sit
# idle -- the constraint is the scheduler's, not this workload's.
#SBATCH --time=08:00:00
#SBATCH --account=y100dc
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_preprocess_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_preprocess_%j.err
#
# Build ONE frozen ground-truth cohort. CPU only -- no GPU needed or requested.
# Run this once per experiment set; every predict/evaluate job then reuses the result.
#
#   sbatch slurm/02_preprocess.sbatch <cohort_name> [n_per_bucket]
#
#   sbatch slurm/02_preprocess.sbatch test_v1 200     # 200 per (modality, plane) bucket
#   sbatch slurm/02_preprocess.sbatch test_full 0     # 0 = every eligible case
#
# Per BUCKET, not per sequence: the real plane mix is ~81% axial, so a per-sequence draw leaves
# the coronal/sagittal buckets far too small for a stable per-bucket FID. 10 buckets x 200 = 2000.
#
# Disk: ~14 MB per case (compressed, at the per-bucket FOVs) -> ~28 GB for 2000. Volumes are
# bundled one archive per bucket, so this is ~10 files, not 2000: /hnvme has a FILE-COUNT quota.

unset SLURM_EXPORT_ENV
set -euo pipefail

# Absolute path, not $(dirname "$BASH_SOURCE"): Slurm runs a COPY of this script from
# /var/tmp/slurmd_spool/job<id>/, where _common.sh does not exist. Override with R2V_REPO=/path.
R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup
preflight "$MANIFEST_CSV" "$REPORT_INDEX_CSV"

NAME="${1:?usage: sbatch 02_preprocess.sbatch <cohort_name> [n_per_bucket]}"
N="${2:-200}"
COHORT="$COHORT_ROOT/$NAME"

echo "cohort=$COHORT  n_per_bucket=$N   (0 = every eligible case)"
df -h "$COHORT_ROOT" | tail -1

# Every flag that changes a voxel is spelled out here rather than left to a default, because it
# is hashed into cohort_id and a silent default change would silently make two cohorts
# incomparable. --posterior-shift-mm 0: the 15 mm defacing shift belongs to the contrastive
# pipeline; measured on this data it won 8 of 10 buckets to leave it off.
run_py mrrate_r2v.cli.preprocess \
    --manifest-csv "$MANIFEST_CSV" --report-index-csv "$REPORT_INDEX_CSV" \
    --split test --sequences T1w T2w FLAIR SWI --n-per-bucket "$N" \
    --series-selection one_per_study_per_bucket \
    --geometry-mode per_modality_plane \
    --normalizer percentile --posterior-shift-mm 0 --seed 42 \
    --out "$COHORT" --overwrite

echo
echo "=== cohort contract ==="
cat "$COHORT/cohort.json"
du -sh "$COHORT"
echo
echo "Pass --cohort $COHORT to every predict job for this experiment set."
