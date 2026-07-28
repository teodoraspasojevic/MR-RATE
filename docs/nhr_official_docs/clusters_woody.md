# Woody Cluster -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/clusters/woody/

## Overview

The Woody cluster is designed for throughput computing and consists of nodes with different generations of Intel CPUs.

## Node Configuration

| Hostnames | Nodes | CPU Type | Cores | Memory | SSD | Partition | Constraint |
|-----------|-------|----------|-------|--------|-----|-----------|-----------|
| w12xx, w13xx | 64 | Intel Xeon E3-1240 v5 (Skylake) | 4 @ 3.5 GHz | 32 GB | 1 TB | work | sl |
| w14xx, w15xx | 112 | Intel Xeon E3-1240 v6 (Kaby Lake) | 4 @ 3.7 GHz | 32 GB | 900 GB | work | kl |
| w22xx-w25xx | 110 | 2x Intel Xeon Gold 6326 (Ice Lake) | 2x 16 @ 2.9 GHz | 256 GB | 1.8 TB | work | icx |

## Access

Connect via: `ssh woody.nhr.fau.de`

Note: NHR accounts are not enabled by default on this Tier3 resource.

## Software Environment

Woody runs AlmaLinux 8 (Red Hat Enterprise Linux 8 compatible). Software is managed through environment modules. Python and Conda are available, and containers are supported via Apptainer.

## Compilation Guidelines

Target specific CPU architectures with appropriate flags:

- **Skylake/Kaby Lake/Ice Lake**: `-march=skylake` or `-march=x86-64-v3`
- **Ice Lake only**: `-march=icelake-server`

**AVX512 Warning**: Only Ice Lake nodes support AVX512. Code compiled with AVX512 on frontend nodes will fail on other architectures with "Illegal Instruction" errors.

## Job Submission

- **Only single-node jobs** are permitted
- Resources allocated at core granularity (7.75 GB memory per core)
- Max walltime: 24 hours (shorter partition available for quick jobs)
- Core limits: 1-32 per job (4 for Skylake/Kaby Lake nodes)

### Example Batch Scripts

**Serial Job:**
```bash
#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

./application
```

**OpenMP Job:**
```bash
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./application
```

## Performance Notes

Ice Lake nodes show superior performance only with AVX512 instructions. Memory bandwidth per core differs significantly across architectures, affecting throughput workloads.
