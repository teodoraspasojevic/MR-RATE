# PyTorch -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/apps/pytorch/

## Installation via pip/conda

### Preparation

1. Start an interactive job on a cluster node
2. Load the Python module:
   ```bash
   module add python
   ```
3. Optional: Create and activate a virtual environment using conda or venv

### Installation

Visit https://pytorch.org/get-started/locally/ and configure:
- PyTorch Build: **Stable**
- OS: **Linux**
- Package: **Conda** or **Pip**
- Language: **Python**
- Compute Platform: Select appropriate CUDA version

Execute the generated command line.

### Test Installation

Run on a compute node:
```bash
python3 -c 'import torch; print(torch.cuda.is_available())'
# Output when GPUs are usable: True
```

## Using Containers

### Build Custom Container

Use this `pytorch.def` file structure:

```
Bootstrap: docker
From: dockerhub-mirror.rrze.uni-erlangen.de/pytorch/pytorch:latest

%files
    requirements.txt /

%post
    apt-get update -y
    apt-get clean
    pip install -r requirements.txt

%runscript
    exec "$@"
```

Build and run:
```bash
apptainer build pytorch.sif pytorch.def
apptainer exec pytorch.sif python training.py
```

### Using Existing Containers

```bash
cd $WORK
export APPTAINER_CACHEDIR=$(mktemp -d)
apptainer pull pytorch-latest.sif docker://pytorch/pytorch:latest
apptainer pull pytorch-ngc.sif docker://nvcr.io/nvidia/pytorch:25.09-py3
rm -r "$APPTAINER_CACHEDIR"
```

## Increasing Performance

PyTorch >= 2.0 includes `torch.compile` for optimization:

```python
import torch

device = "cuda"
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
model = torch.compile(model, backend="inductor").to(device)
model(torch.randn(1,3,64,64).to(device))
```

## Distributed Data Parallel (DDP) Training

DDP enables data-parallel training across multiple GPUs. Requirements:
- Each process calls `torch.distributed.init_group()` (explicit or implicit)
- One GPU per process using `torch.cuda.set_device()`
- Data distribution defined via `DistributedSampler`

### Backends

- **nccl**: GPUs only (preferred for GPU work, supports InfiniBand)
- **gloo**: CPUs and GPUs (lower performance for GPU work)
- **MPI**: Requires custom PyTorch compilation

### Launch via torchrun

```bash
#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --gres=gpu:h100:4
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT=29400

srun --cpu-bind=verbose torchrun \
     --nnodes=$SLURM_JOB_NUM_NODES \
     --nproc-per-node=$SLURM_GPUS_ON_NODE \
     --rdzv-id=$SLURM_JOB_ID \
     --rdzv-backend=c10d \
     --rdzv-endpoint="${MASTER_ADDR}":"${MASTER_PORT}" \
     script.py
```

PyTorch Lightning is recommended for users new to multi-GPU execution.

## NCCL Debugging

```bash
export NCCL_DEBUG=INFO   # extensive output
export NCCL_DEBUG=WARN   # warnings only
```

For A100 nodes, enforce specific InfiniBand devices:
```bash
export NCCL_IB_HCA="=mlx5_0:1,mlx5_3:1"
```
