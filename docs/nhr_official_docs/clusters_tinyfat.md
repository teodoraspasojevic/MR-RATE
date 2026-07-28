# TinyFat Cluster -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/clusters/tinyfat/

## Overview

TinyFat is designed for serial or moderately parallel (OpenMP) applications that require large amounts of memory in one machine.

### Hardware Configuration

| Hostname | Nodes | CPUs and Cores | Memory | Local SSD | Partition |
|----------|-------|----------------|--------|-----------|-----------|
| memoryhog | 1 | 2x Intel Xeon Platinum 8360Y, 72 cores/144 threads @2.4GHz | 2 TB | n/a | Interactive only |
| tf04x | 3 | 2x Intel Xeon E5-2680 v4, 28 cores/56 threads @2.4 GHz | 512 GB | 1 TB | broadwell512 |
| tf05x | 8 | 2x Intel Xeon E5-2643 v4, 12 cores/24 threads @3.4 GHz | 256 GB | 1 TB | broadwell256, long256 |
| tf06x-tf09x | 36 | 2x AMD EPYC 7502, 64 cores/128 threads | 512 GB | 3.5 TB | work |

## Access

Connect via SSH: `ssh tinyx.nhr.fau.de`

TinyFat is restricted to accounts with "Tier3 Grundversorgung" status, not NHR project accounts.

## Operating System and Software

The cluster runs **Ubuntu 20.04 LTS**. Software is provided through environment modules. Containers are supported via Apptainer, and Python/Conda is available through the `python` module.

## Compilation Flags

| Partition | Architecture | GCC/LLVM | Intel |
|-----------|--------------|----------|-------|
| all | Zen2, Broadwell | `-mavx2 -mfma` or `-march=x86-64-v3` | `-mavx2 -mfma` |
| work | Zen2 | `-march=znver2` | `-mavx2 -mfma` |
| broadwell*, long256 | Broadwell | `-march=broadwell` | `-march=broadwell` |

## Filesystems

Available across all nodes: `$HOME`, `$HPCVAULT`, `$WORK`, and `$TMPDIR` (node-local SSD, deleted after job completion).

## Batch Processing with Slurm

### Key Differences

Slurm commands include the `.tinyfat` suffix: `sbatch.tinyfat`, `salloc.tinyfat`, `srun.tinyfat`, `sinfo.tinyfat`

### Partitions

| Name | Walltime | Cores | Exclusivity | Memory |
|------|----------|-------|------------|--------|
| work | 0-2:00:00 (default) or 0-24:00:00 | 1-64 | Shared | 512 GB |
| broadwell256 | 0-24:00:00 | 12 | Exclusive | 256 GB |
| broadwell512 | 0-24:00:00 | 28 | Exclusive | 512 GB |
| long256 | 0-60:00:00 | 12 | Exclusive | 256 GB |

### SMT/Hyperthreading

SMT is enabled by default. Use `--hint=multithread` to enable it explicitly or `--hint=nomultithread` to disable for physical cores only.

## Interactive Jobs

**Single-core (1 hour):**
```bash
salloc.tinyfat -n 1 --time=01:00:00
```

**Multiple cores (10 cores, 1 hour):**
```bash
salloc.tinyfat --cpus-per-task=10 --time=01:00:00
```

## Batch Job Examples

### Serial Job
```bash
#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

./application
```

### OpenMP Job (6 threads)
```bash
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=4:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./application
```

For thread pinning efficiency, use: `OMP_PLACES=cores` and `OMP_PROC_BIND=true`
