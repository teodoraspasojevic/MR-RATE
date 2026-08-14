#!/bin/bash -l
#SBATCH --job-name=r2v_bench
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_bench_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_bench_%j.err
#
# Measures the H200 environment and per-configuration step time / memory / throughput. Runs a few
# dozen synthetic steps per (configuration, batch size) -- NOT a training run: no manifest is read
# and no checkpoint is written.
#
#   sbatch slurm/10_benchmark_h200.sbatch                     # all three configs, batches 1-8
#   sbatch --export=ALL,R2V_BENCH_ENV_ONLY=1 slurm/10_benchmark_h200.sbatch    # environment only
#
# `#SBATCH --export=NONE` means a plain `VAR=x sbatch ...` does not reach the job; pass overrides
# through --export as above.

unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup
preflight "$UNET_CHECKPOINT" "$VAE_CHECKPOINT" "$PRETRAINED_DIR" "$SIF_IMAGE_TEXT"

OUT_DIR="$WORKSPACE/cache/r2v/benchmarks"
mkdir -p "$OUT_DIR"
OUT_JSON="$OUT_DIR/h200_${SLURM_JOB_ID:-local}.json"

if [[ -n "${R2V_BENCH_ENV_ONLY:-}" ]]; then
    run_py_text mrrate_r2v.cli.benchmark_h200 --env-only --out "$OUT_JSON"
    cat "$OUT_JSON"
    exit 0
fi

run_py_text mrrate_r2v.cli.benchmark_h200 \
    --configs ${R2V_BENCH_CONFIGS:-cxr_bert_cls radbert_mean report2ct_style report2ct_style_meta} \
    --batch-sizes ${R2V_BENCH_BATCHES:-1 2 4 8} \
    --shape ${R2V_BENCH_SHAPE:-256 256 256} \
    --steps "${R2V_BENCH_STEPS:-12}" \
    --base-checkpoint "$UNET_CHECKPOINT" \
    --base-vae "$VAE_CHECKPOINT" \
    --max-report-tokens 512 \
    --scratch "$OUT_DIR/scratch" \
    --out "$OUT_JSON"

echo "=== recommendations ==="
"$HOST_PYTHON" -c "
import json, sys
d = json.load(open('$OUT_JSON'))
for r in d.get('recommendations', []):
    print(json.dumps(r))
"
