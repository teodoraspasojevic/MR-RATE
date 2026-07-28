# Quick Start -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/getting_started/

## Getting started with HPC

This guide provides an overview of running applications on HPC systems. For deeper details, refer to linked documentation sections.

**Monthly HPC beginner's introduction**: Online hands-on introduction offered every second Wednesday of the month at the HPC cafe.

## Getting an HPC account

Account creation varies by user status. Visit the "Getting an account" section for details. Accounts are managed through the HPC portal, where you can upload SSH keys.

## HPC Clusters

All clusters run Linux with text-mode interfaces, requiring basic Linux knowledge. Different clusters serve specific purposes:

- **Multi-node MPI-parallel jobs**: Fritz
- **GPU jobs**: TinyGPU, Alex
- **Single-core/single-node throughput**: Woody
- **Single-core/single-node with high memory**: TinyFat

## Connecting to HPC systems

OpenSSH connection is used across platforms. Setup guides available for:

- Command-line (Linux, Mac, Windows PowerShell)
- MobaXterm (Windows GUI)
- VS Code
- JupyterHub for interactive work

## Working with data

| Mount Point | Purpose | Technology | Backup | Quota |
|---|---|---|---|---|
| `/home/hpc` | Source/input/results | NFS | YES | 50 GB |
| `/home/vault` | Mid/long-term storage | NFS | YES | 500 GB |
| `/home/{woody,saturn...}` | General-purpose | NFS | NO | 500 GB-project |
| `/lustre` | High-performance I/O | Lustre | NO | Inodes only |

Environment variables `$HOME`, `$HPCVAULT`, and `$WORK` are automatically set and accessible across all systems.

### File system quota

Nearly all filesystems enforce quotas on data volume and file counts. Soft quotas can be temporarily exceeded; hard quotas are absolute limits. Check usage with `quota -s` or `shownicerquota.pl`.

### Data transfer

Recommended tools include `scp` and `rsync` for efficiency, especially with large files. Windows users can utilize the Linux subsystem, PowerShell `scp`, or WinSCP. Remote filesystems can also be mounted locally.

## Available Software

Standard Linux packages are pre-installed. Most software is provided via environment modules, allowing easy version switching. Key module commands:

| Command | Function |
|---|---|
| `module avail` | Lists available modules |
| `module list` | Shows loaded modules |
| `module load <name>` | Loads specified module |
| `module unload <name>` | Removes module |

### Compiling applications

Load necessary compiler modules before compilation. Same modules must be loaded in Slurm scripts. Open MPI and Intel MPI are available.

### Python

Use provided Python modules with Conda installations rather than system Python. Load versions with `module load python/<version>`. Install additional packages via pip or conda in personal directories.

## Running Jobs

Front-end nodes support editing and compilation only. Large computational jobs require the batch system.

### Interactive jobs

Use interactive jobs for testing and debugging, which open shells on compute nodes for real-time program execution.

### Batch jobs

Create job scripts specifying commands and resources (nodes, runtime). Jobs run when resources become available. Output writes to files in submission directories. Slurm is the batch system across all clusters.

### Job status

- `sinfo`: View cluster node status (idle, mixed, allocated)
- `squeue`: Check job status with delay reasons
- MOTD announces configuration changes and maintenance

## Good practices

- Regularly review job results to prevent resource waste
- Examine performance and resource usage via ClusterCockpit
- Run scaling experiments to optimize parallelism
- Use appropriate filesystems for data types
- Open support tickets when assistance is needed
