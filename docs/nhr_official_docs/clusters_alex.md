# Alex GPU Cluster -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/clusters/alex/

## Overview

Alex is a GPU cluster with Nvidia A40 and A100 GPGPUs paired with AMD Milan ("Zen3") processors. The system is accessible to NHR users and available upon request for Tier3 users.

### Hardware Configuration

| GPU Type | # Nodes | # GPUs | CPUs | Memory | Local SSD | Partition |
|----------|---------|--------|------|--------|-----------|-----------|
| 8x Nvidia A40 (40 GB) | 44 | 352 | 2x AMD EPYC 7713, 128 cores @2.0 GHz | 512 GB | 7 TB | `a40` |
| 8x Nvidia A100 (40 GB) | 20 | 160 | 2x AMD EPYC 7713, 128 cores @2.0 GHz | 1 TB | 14 TB | `a100` |
| 8x Nvidia A100 (80 GB) | 18 | 144 | 2x AMD EPYC 7713, 128 cores @2.0 GHz | 2 TB | 14 TB | `a100` |

Login nodes alex1 and alex2 have no GPUs.

## Accessing Alex

FAU HPC accounts require explicit access. Users must complete a request form at:
https://hpc.fau.de/tier3-access-to-alex/

Access via SSH:
```bash
ssh alex.nhr.fau.de
```

## Software Environment

Alex runs AlmaLinux 8 (RHEL 8 binary compatible). Software is managed through environment modules.

### Key Software Resources
- Applications and development tools available via modules
- Conda installation accessible through the `python` module
- Container support via Apptainer
- Most software centrally installed using Spack

### Known Issues

**Intel MKL on AMD Processors:** Intel MKL may underperform on Alex's AMD CPUs. Modern MKL versions no longer support previous workarounds reliably.

## Compiler Information

### CPU Targeting

For AMD Zen3 processors on Alex:

| Target | GCC/LLVM | Intel oneAPI/Classic | NVHPC |
|--------|----------|----------------------|-------|
| Auto-detect | `-march=native` | not recommended | `-tp=native` |
| Zen3 | `-march=znver3` | `-mavx2 -mfma` | `-tp=zen3` |

> Intel oneAPI/Classic compilers should use `-mavx2 -mfma` instead of `-march=native` or `-xHost`, as the latter options might generate non-optimal code for AMD CPUs.

### GPU Targeting (NVCC)

| GPU | Compute Capability | NVCC Flags |
|-----|-------------------|-----------|
| A100 | 8.0 | `-gencode arch=compute_80,code=sm_80` |
| A40 | 8.6 | `-gencode arch=compute_86,code=sm_86` |

## Multi-Process Service (MPS)

The MPS daemon enables cooperative multi-process CUDA applications, typically MPI jobs, benefiting performance when GPU capacity is underutilized.

## Storage Systems

Three standard filesystems available on all nodes:
- `$HOME` -- user home directory
- `$HPCVAULT` -- long-term vault storage
- `$WORK` -- working directory

### Node-Local Storage

Each node includes local NVMe SSD accessible via `$TMPDIR`:
- **a40 partition:** 7 TB
- **a100 partition:** 14 TB
- Data deleted when job ends
- Shared among concurrent jobs on same node

### Fast NVMe Storage

Alex connects to the Lustre NVMe (anvme) storage system. Access via workspaces.

## Batch Processing

Resources controlled through Slurm batch system. Compile code on login nodes only.

### Partition Details

| Partition | Walltime | GPU Type | GPUs | Cores/GPU | Memory/GPU | Slurm Option |
|-----------|----------|----------|------|-----------|-----------|-------------|
| `a40` | 0-24:00:00 | A40 (40 GB) | 1-8 | 16 | 60 GB | `--gres=gpu:a40:#` |
| `a100` | 0-24:00:00 | A100 (40 GB) | 1-8 | 16 | 120 GB | `--gres=gpu:a100:#` |
| `a100` | 0-24:00:00 | A100 (80 GB) | 1-8 | 16 | 240 GB | `--gres=gpu:a100:# -C a100_80` |

**Multi-node jobs** available on-demand for NHR projects only. Require separate account enablement via hpc-support@fau.de.

## Interactive Jobs

### Single GPU Interactive Session

```bash
salloc --gres=gpu:a40:1 --time=1:00:00
```

### MIG Partition (Debugging/CPU-Only)

Requires `-p a100mig` specification. Options:
- **CPU only:** `-p a100mig -n 4` (4 cores, no GPU; max 8 cores per job)
- **Small A100 fraction:** `--gres=gpu:a100small:1` (10 GB VRAM, 4 cores)
- **Medium A100 fraction:** `--gres=gpu:a100med:1` (20 GB VRAM, 4 cores)

## Batch Job Script Examples

### Python (Single GPU)

```bash
#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate environment-for-script

python3 train.py
```

### MPI Parallel Job (Single-Node)

```bash
#!/bin/bash -l
#SBATCH --ntasks=16
#SBATCH --gres=gpu:a40:1
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

srun ./application
```

### Hybrid MPI/OpenMP Job (Single-Node)

```bash
#!/bin/bash -l

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun ./hybrid_application
```

> In recent Slurm versions, `--cpus-per-task` values must be manually set for `srun` via the `SRUN_CPUS_PER_TASK` variable.

### Multi-Node Job (NHR Projects)

```bash
#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:a100:8
#SBATCH --qos=a100multi
#SBATCH --time=1:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

srun ./application
```

## GPU Architecture Details

### Nvidia A40

- **Architecture:** Ampere
- **Compute Capability:** 8.6
- **Memory:** 48 GB GDDR6
- **Memory Bandwidth:** 696 GB/s
- **CUDA Cores:** 10,752 (84 SMs)
- **Peak FP32:** 37.4 TFLOPS
- **Interconnect:** PCIe Gen4 (31.5 GB/s bidirectional)
- **Max Power:** 300 W

Suitable for single-precision workloads (e.g., molecular dynamics). More cost-effective than A100 with higher FP32 performance.

### Nvidia A100

- **Architecture:** Ampere
- **Compute Capability:** 8.0
- **Memory:** 40 GB or 80 GB HBM2
- **Memory Bandwidth:** 1,555 GB/s (40GB) / 2,039 GB/s (80GB)
- **CUDA Cores:** 6,912 (108 SMs)
- **Peak FP64:** 9.7 TFLOPS
- **Peak FP32 Tensor:** 156 TFLOPS
- **Interconnect:** NVLink (600 GB/s GPU-to-GPU via NVSwitch)
- **Max Power:** 400 W

## Processor Details

### AMD EPYC 7713 "Milan" (Zen3)

| Specification | Value |
|---------------|-------|
| Cores | 64 per socket (128 per node with 2 sockets) |
| SMT Threads | Disabled (security) |
| Max Boost | 3.675 GHz |
| Base Frequency | 2.0 GHz |
| L3 Cache | 256 MB per socket |
| Memory Type | DDR4 @ 3,200 MHz |
| Memory Channels | 8 per socket |
| NPS Setting | 4 |
| Theoretical Memory Bandwidth | 204.8 GB/s per socket |
| TDP | 225 W |

## Performance Metrics

- **160 A100/40GB GPUs** (Jan 2022): 1.73 PFlop/s LINPACK
- **160 A100/40GB + 96 A100/80GB** (May 2022): 2.938 PFlop/s (Rank 184, Top500 June 2022; Rank 17, Green500)
- **160 A100/40GB + 120 A100/80GB** (Oct 2022): 3.24 PFlop/s (Rank 174, Top500 Nov 2022; Rank 33, Green500)

## Naming and Financing

**Name:** "Alex" references Alexander, Margrave of Brandenburg-Ansbach (1736-1806), an early FAU benefactor.

**Funding Sources:**
- German Research Foundation (DFG) - INST 90/1171-1
- NHR federal/state funding (BMBF, Bavarian Ministry of Science and Arts)
- BMBF "HPC4AAI" proposal (7 A100 nodes to HS Coburg)
- External group funding (1 A100 node from Erlangen)
- FAU institutional support for HPC activities
