# Fritz CPU Cluster -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/clusters/fritz/

## Overview

Fritz is a parallel CPU cluster featuring Intel Ice Lake and Sapphire Rapids processors with InfiniBand networking and Lustre-based parallel filesystem accessible via `$FASTTMP`. Access is available to NHR users and Tier3 users upon request.

**Cluster Specifications:**

| Nodes | CPU Configuration | Memory | Partition |
|-------|------------------|--------|-----------|
| 992 | 2x Intel Xeon Platinum 8360Y (72 cores) @ 2.4 GHz | 256 GB | singlenode, multinode |
| 48 | 2x Intel Xeon Platinum 8470 (104 cores) @ 2.0 GHz | 1 TB | spr1tb |
| 16 | 2x Intel Xeon Platinum 8470 (104 cores) @ 2.0 GHz | 2 TB | spr2tb |

Login nodes (`fritz[1-4]`) feature 2x Intel Xeon Platinum 8360Y processors with 512 GB memory.

## Accessing Fritz

FAU HPC accounts do not have access to Fritz by default. Users must request access via https://hpc.fau.de/tier3-access-to-fritz/

Connect using: `ssh fritz.nhr.fau.de`

## Software Environment

Fritz runs AlmaLinux 8 (RHEL 8 compatible). Software is provisioned through environment modules. Most packages are installed via Spack. Load `000-all-spack-pkgs` to see all available packages.

Containers are supported through Apptainer.

## Compiler Configuration

Fritz has two CPU microarchitectures:
- **Ice Lake Server** (frontend nodes, 992 nodes)
- **Sapphire Rapids** (48 + 16 specialized nodes)

Code compiled exclusively for Sapphire Rapids cannot run on Ice Lake systems.

**Compilation Flags:**

| Target | GCC/LLVM/Intel |
|--------|---|
| All nodes | `-march=icelake-server` |
| Ice Lake only | `-march=icelake-server` |
| Sapphire Rapids | `-march=sapphirerapids` |

## Filesystems

All frontends and nodes mount `$HOME`, `$HPCVAULT`, and `$WORK`.

**Node-Local Storage:** `$TMPDIR` provides job-specific RAM disk storage deleted upon job completion.

**Parallel Filesystem:** `$FASTTMP` mounted on all Fritz frontends and nodes.

| Property | Details |
|----------|---------|
| Mount point | `/lustre/$GROUP/$USER/` |
| Capacity | 3.5 PB |
| Technology | Lustre-based parallel filesystem |
| Backup | None |
| Deletion policy | High-watermark (80% threshold) |

`$FASTTMP` supports parallel I/O with >20 GB/s aggregate bandwidth and handles large files optimally (minimum 1 MB block size). Files smaller than 1 MB will reside only on one server, creating overhead inefficiency.

## Batch Processing

Slurm controls resource allocation in node-granularity exclusive allocations.

**Available Partitions:**

| Partition | Walltime | Nodes | Cores/Node | Memory | Notes |
|-----------|----------|-------|-----------|--------|-------|
| singlenode (default) | 0-24:00:00 | 1 | 72 | 256 GB | - |
| multinode | 0-24:00:00 | 2-64 | 72 | 256 GB | - |
| spr1tb | 0-24:00:00 | 1-8 | 104 | 1 TB | `-p spr1tb` |
| spr2tb | 0-24:00:00 | 1-2 | 104 | 2 TB | `-p spr2tb` |
| big | 0-24:00:00 | 65-256 | 72 | 256 GB | Request only |

### Interactive Jobs

**Single-node (Ice Lake):**
```bash
salloc -N 1 --time=01:00:00
```

**Single-node (Sapphire Rapids):**
```bash
salloc -N 1 --partition=spr1tb --time=01:00:00
```

**Multi-node (Ice Lake):**
```bash
salloc -N 4 --time=01:00:00
```

### Batch Script Examples

**MPI Single-Node:**
```bash
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --time=2:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
srun ./application
```

**OpenMP Single-Node:**
```bash
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=2:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./application
```

**Hybrid OpenMP/MPI Single-Node:**
```bash
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=1:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun ./hybrid_application
```

> **Warning:** Recent Slurm versions require manual `SRUN_CPUS_PER_TASK` setting to avoid propagation errors.

**MPI Multi-Node:**
```bash
#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=72
#SBATCH --time=2:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
srun ./application
```

## Remote Visualization

Node `fviz1` supports remote visualization with VirtualGL, featuring:
- 2x Intel Xeon Platinum 8360Y processors
- 1 TB memory
- Nvidia A16 GPU (partitioned into 4 virtual GPUs, 16 GB RAM each)
- 30 TB NVMe SSD storage

GPUs on fviz1 are unsuitable for machine learning and prohibited for such use.

**Requesting Visualization:**
```bash
/apps/virtualgl/submitvirtualgljob.sh --time=hours:minutes:seconds
```

## Technical Details

**Performance Metrics:**
- 1.84 PFlop/s on 512 nodes (April 2022)
- 2.233 PFlop/s on 612 nodes (May 2022, Top500 #323)
- 3.578 PFlop/s on 986 nodes (November 2022, Top500 #151)

**Processor Details:**

| Property | Ice Lake 8360Y | Sapphire Rapids 8470 |
|----------|---|---|
| Cores (SMT threads) | 36 (72) | 52 (104) |
| Base frequency | 2.4 GHz | 2.0 GHz |
| Max turbo | 3.5 GHz | 3.8 GHz |
| L3 cache | 54 MB | 105 MB |
| TDP | 250 W | 350 W |
| Memory channels | 16x DDR4-3200 | 16x DDR5-4800 |

All nodes have SMT disabled and sub-NUMA clustering enabled.

**Network:** Fritz uses HDR100 Infiniband (100 Gbit/s) organized in islands of 64 nodes with 1:4 blocking factor between islands.

## Administrative Information

**Name Origin:** The cluster name references Friedrich, Margrave of Brandenburg-Bayreuth (1711-1763), founder of Friedrich-Alexander-Universitat.

**Financing:** Fritz was funded by DFG (INST 90/1171-1), NHR federal/state support, BMBF "HPC4AAI" program for HS Coburg nodes, and FAU HPC strengthening initiatives.
