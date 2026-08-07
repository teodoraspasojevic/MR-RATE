#!/bin/bash -l
#SBATCH --job-name=wb_proxy
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=00:12:00
#SBATCH --account=y100dc
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/wb_proxy_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/wb_proxy_%j.err
#
# Does W&B `online` logging work from a COMPUTE node, and is `setup_proxy` what makes it work?
#
# Runs a real wandb.init/log/finish twice through the same WandbRun wrapper the trainer uses: once
# with the proxy, once with R2V_NO_PROXY=1. The second run is the control -- without it, a success
# with the proxy proves nothing, because the login node has a direct route and a compute node might
# too. Logs synthetic numbers only; no report text, no identifiers.
#
#   sbatch --export=ALL slurm/wandb_proxy_test.sbatch
#
# Throwaway; delete this file and the `mrrate-proxy-selftest` W&B project once you trust the setup.

unset SLURM_EXPORT_ENV
set -uo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
SCRATCH="/hnvme/workspace/y100dc19-nvidia-mri-brain/cache/wb_proxy_test"
mkdir -p "$SCRATCH"

probe() {
    local label="$1"
    echo "############ $label ############"
    ( source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
      setup >/dev/null
      echo "http_proxy on host = ${http_proxy:-<unset>}"
      export WANDB_DIR="$SCRATCH/$label"; mkdir -p "$WANDB_DIR"
      APPTAINER_ARGS+=(--env "WANDB_DIR=$WANDB_DIR" --env "WANDB__SERVICE_WAIT=120")
      timeout 240 apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE_TEXT" python3 - <<PY
import os
from mrrate_r2v.eval.wandb_logging import WandbRun
print("proxy inside container =", os.environ.get("http_proxy", "<unset>"))
r = WandbRun(mode="online", entity=None, project="mrrate-proxy-selftest",
             run_name="computenode-$label", group=None, tags=["throwaway"], config={})
print("RESULT enabled =", r.enabled)
for s in range(3):
    r.log({"train/loss": 1.0 / (s + 1)}, step=s)
print("RESULT summary =", r.finish())
PY
      echo "probe exit = $?" )
    echo
}

echo "=== node: $(hostname -f) ==="
probe with_proxy
R2V_NO_PROXY=1 probe no_proxy
echo "=== done ==="
