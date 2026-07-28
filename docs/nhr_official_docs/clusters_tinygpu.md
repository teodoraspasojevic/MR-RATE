# TinyGPU Cluster -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/clusters/tinygpu/

## Overview

TinyGPU is a GPU cluster featuring various consumer and data center NVIDIA GPUs. The system runs Ubuntu 20.04 LTS and serves users with "Tier3 Grundversorgung" accounts only -- not NHR project accounts.

## Hardware Configuration

| Hostname | Nodes (GPUs) | GPU Type | CPU Configuration | RAM | SSD | Partition |
|----------|--------------|----------|-------------------|-----|-----|-----------|
| tg06x | 8 (32) | 4x RTX 2080 Ti (11GB) | 2x Xeon Gold 6134, 32 cores/64 threads | 96GB | 1.8TB | work |
| tg07x | 4 (16) | 4x Tesla V100 (32GB) | 2x Xeon Gold 6134, 32 cores/64 threads | 96GB | 2.9TB | v100 |
| tg08x | 7 (56) | 8x RTX 3080 (10GB) | 2x Xeon Gold 6226R, 64 cores/128 threads | 384GB | 3.8TB | work, rtx3080 |
| tg09x | 8 (32) | 4x A100 (40GB) | 2x EPYC 7662, 128 cores | 512GB | 5.8TB | a100 |

## Access

Connect via SSH to the shared frontend: `ssh tinyx.nhr.fau.de`

## Software and Environment

- Software provided through environment modules
- Conda available via the `python` module
- Containers supported through Apptainer
- cuDNN pre-installed on all nodes

## Compiler Notes

**Important:** The `a100` partition uses AMD processors; Intel-specific compiled software may fail there. Use `-mavx2 -mfma` for compatibility across all partitions.

## Batch Processing

Slurm commands require the `.tinygpu` suffix:
- `sbatch.tinygpu`
- `squeue.tinygpu`
- `salloc.tinygpu`

**GPU Requirements:** All jobs must request at least one GPU using `--gres=gpu:#`

### Available Partitions

| Partition | Walltime | GPU Type | GPU Count | CPU Cores/GPU | Memory/GPU |
|-----------|----------|----------|-----------|---------------|-----------|
| work (default) | 0-24:00:00 | RTX 2080 Ti or RTX 3080 | 1-4/1-8 | 8 (16 threads) | 22GB |
| rtx3080 | 0-24:00:00 | RTX 3080 | 1-8 | 8 (16 threads) | 46GB |
| a100 | 0-24:00:00 | A100 | 1-4 | 32 (32 threads) | 117GB |
| v100 | 0-24:00:00 | V100 | 1-4 | 8 (16 threads) | 22GB |

## Interactive Jobs

```bash
salloc.tinygpu --gres=gpu:1 --time=01:00:00
```

Maximum interactive job duration is 4 hours.

## Sample Batch Scripts

**Python (Single GPU A100):**
```bash
#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
conda activate environment-for-script
python3 train.py
```

## File Systems

Available on all nodes: `$HOME`, `$HPCVAULT`, `$WORK`, and `$TMPDIR` (node-local SSD, deleted after job completion). Minimum SSD capacity: 1.8TB (shared among concurrent jobs on a node).
