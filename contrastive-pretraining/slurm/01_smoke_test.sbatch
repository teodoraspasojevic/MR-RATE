#!/bin/bash -l
#SBATCH --job-name=r2v_smoke
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_smoke_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_smoke_%j.err
#
# The whole pipeline on 2 real cases: preprocess -> predict_vae -> evaluate.
# Must pass before any pilot or full run. Takes a few minutes.
#
#   sbatch slurm/01_smoke_test.sbatch

unset SLURM_EXPORT_ENV
set -euo pipefail

# Absolute path, not $(dirname "$BASH_SOURCE"): Slurm runs a COPY of this script from
# /var/tmp/slurmd_spool/job<id>/, where _common.sh does not exist. Override with R2V_REPO=/path.
R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup
preflight "$MANIFEST_CSV" "$REPORT_INDEX_CSV" "$VAE_CHECKPOINT"

TAG="smoke_${SLURM_JOB_ID}"
COHORT="$COHORT_ROOT/$TAG"
PRED="$PRED_ROOT/${TAG}_vae"
RESULTS="$RESULT_ROOT/${TAG}_vae"

echo "=== 0. environment ==="
run_sh python3 -c "
import torch, monai, nibabel, mrrate_r2v
print('mrrate_r2v', mrrate_r2v.__version__, '-- importable from', mrrate_r2v.__file__)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('monai', monai.__version__, 'nibabel', nibabel.__version__)
assert torch.cuda.is_available(), 'no GPU visible'
print('GPU:', torch.cuda.get_device_name(0))
"

echo
echo "=== 1. preprocess: 2 T1w cases -> frozen cohort ==="
run_py mrrate_r2v.cli.preprocess \
    --manifest-csv "$MANIFEST_CSV" --report-index-csv "$REPORT_INDEX_CSV" \
    --split test --sequences T1w --n-per-bucket 2 \
    --out "$COHORT" --overwrite

echo
echo "=== 2. predict: VAE reconstruction ==="
run_py mrrate_r2v.cli.predict_vae \
    --cohort "$COHORT" --checkpoint "$VAE_CHECKPOINT" --device cuda \
    --out "$PRED" --overwrite

echo
echo "=== 3. evaluate ==="
run_py mrrate_r2v.cli.evaluate \
    --task reconstruction --gt "$COHORT" --pred "$PRED" --out "$RESULTS" \
    --skip-metric-groups perceptual --overwrite

echo
echo "=== output check ==="
for f in summary.json per_case_metrics.csv excluded_cases.json run_manifest.json \
         metrics_per_bucket.csv metrics_summary.csv; do
    [[ -f "$RESULTS/$f" ]] && echo "OK: $f" || { echo "MISSING: $f"; exit 1; }
done
N=$(python3 -c "import json;print(json.load(open('$RESULTS/summary.json'))['n_scored'])")
echo "scored cases: $N"
column -s, -t < "$RESULTS/metrics_per_bucket.csv" || cat "$RESULTS/metrics_per_bucket.csv"

sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS,AllocTRES 2>&1 || true

[[ "$N" -ge 1 ]] || { echo "SMOKE TEST FAILED: 0 cases scored." >&2; exit 1; }
echo
echo "SMOKE TEST PASSED -- $N cases. Cohort kept at $COHORT for reuse."
