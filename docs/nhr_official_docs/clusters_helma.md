# Helma GPU Cluster -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/clusters/helma/

## Overview

Helma is a GPU cluster featuring NVIDIA H100 and H200 processors paired with AMD Genoa (Zen4) host CPUs, plus a CPU partition with AMD Turin (Zen5c).

| GPU Type (Memory) | # Nodes (# GPUs) | CPUs per Node | Main Memory | Node-Local SSD | Partition |
|---|---|---|---|---|---|
| 4 x Nvidia H100 (94 GB HBM2e) | 96 (384) | 2 x AMD EPYC 9554, 128 cores @3.1 GHz | 768 GB | 15 TB | h100 |
| 4 x Nvidia H200 (141 GB HBM3e) | 96 (384) | 2 x AMD EPYC 9554, 128 cores @3.1 GHz | 768 GB | 15 TB | h200 |

## Accessing Helma

Access requires application approval; regular HPC accounts don't have default access. Connect via SSH using `ssh helma.nhr.fau.de`.

## Software Environment

Helma runs AlmaLinux 9 (RHEL 9 compatible). Software is managed through environment modules, with most packages installed via Spack. Two environment branches exist: `gpu-env/2025` (default) and `cpu-env/2026`.

### Key Features
- Python/Conda available through module system
- Apptainer containers supported
- MKL may underperform on AMD processors; workarounds exist but are not officially promoted

## Filesystems

Only `$HOME`, `hnvme` workspaces, and `$TMPDIR` mount on compute nodes. Each node has a 15 TB local NVMe SSD at `$TMPDIR` that deletes when jobs end.

## Job Submission

### Partitions and Resources

| Partition | Walltime | GPU Type | GPU Count | CPU Cores/GPU | Host Memory/GPU |
|---|---|---|---|---|---|
| preempt | 0-24:00:00 | H100 (94 GB) | 1-4 | 32 | 192 GB |
| h100 | 0-24:00:00 | H100 (94 GB) | 1-4 | 32 | 192 GB |
| h200 | 0-24:00:00 | H200 (141 GB) | 1-4 | 32 | 192 GB |

### Interactive Job Example

```bash
salloc --gres=gpu:h100:1 --time=1:00:00
```

### Python Single-GPU Batch Job

```bash
#!/bin/bash -l
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate environment-for-script

python3 train.py
```

### MPI-Parallel Job (Single-Node)

```bash
#!/bin/bash -l
#SBATCH --ntasks=32
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

srun ./application
```

### Hybrid MPI/OpenMP Job

```bash
#!/bin/bash -l

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun ./hybrid_application
```

### Multi-Node Job

```bash
#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --time=1:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

srun ./application
```

## Hardware Details

### Node Configuration

| Component | h100/h200 | CPU Partition |
|---|---|---|
| Processor | AMD EPYC 9554 (Zen4) | AMD EPYC 9955 (Zen5c) |
| Cores | 128 total (2x64) | 384 cores (2x192) |
| Memory | 768 GB DDR5 | -- |
| Cache | 512 MB L3 | -- |
| NVMe | 15 TB | -- |

### Processor Specifications

- **Microarchitecture**: Zen4
- **SMT**: Disabled (security)
- **Base Frequency**: 3.1 GHz
- **Max Boost**: 3.75 GHz
- **Memory**: DDR5 @ 4,800 MT/s

## Performance

A LINPACK benchmark measured 16.94 PFlop/s on 384 H100/94GB GPGPUs in October 2024.

## History

Named after Wilhelmine, Margravine of Brandenburg-Bayreuth (1709-1758), who founded the University of Erlangen in 1743 with her husband Friedrich.

## Financing

Funded by Bavaria's High-Tech Agenda, NHR federal/state funding, FAU, University of Technology Nurnberg, and associated universities.
