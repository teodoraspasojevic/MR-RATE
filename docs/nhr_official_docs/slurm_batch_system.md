# Batch System Slurm -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/batch-processing/batch_system_slurm/

## Overview

All NHR@FAU clusters utilize Slurm for resource management and job scheduling. Compute nodes are inaccessible directly; the batch system manages job queuing and priority based on specified computational resources.

When logging into an HPC system, you are placed on a login node. Login nodes should not be used for computational work -- they serve only for data management, workflow setup, and job preparation.

## Batch Job Submission with `sbatch`

Batch jobs encapsulate computational work with specifications including:
- Resource requirements (nodes, cores, GPUs)
- Runtime (typically max 24 hours)
- Runtime environment setup
- Application execution commands

Submit jobs using:
```bash
sbatch [options] <job_script>
```

For TinyFat and TinyGPU clusters, use wrapper commands `sbatch.tinyfat` or `sbatch.tinygpu`.

## Interactive Jobs with `salloc`

Interactive shells on compute nodes enable debugging and testing:
```bash
salloc [options for number of nodes, walltime, etc.]
```

Settings from the calling shell (e.g. loaded module paths) will automatically be inherited. Run `module purge` before `salloc` to avoid conflicts.

## Running Parallel Applications with `srun`

Use `srun` instead of `mpirun` for MPI-parallel applications within job allocations.

## Options for `sbatch`/`salloc`/`srun`

| Option | Description |
|--------|-------------|
| `-N`, `--nodes=<N>` | Number of requested nodes (default: 1) |
| `-n`, `--ntasks=<N>` | Total number of tasks/MPI processes |
| `--ntasks-per-node=<N>` | Tasks per node |
| `-c`, `--cpus-per-task=<N>` | Threads/logical cores per task |
| `-t`, `--time=HH:MM:SS` | Wall clock runtime |
| `-p`, `--partition=<name>` | Target partition |
| `--job-name=<name>` | Job identifier for `squeue` |
| `--mail-user=<email>` | Notification email address |
| `--mail-type=<type>` | Notification triggers (BEGIN, END, FAIL, TIME_LIMIT, ALL) |
| `--exclusive` | Exclusive node usage |
| `-a`, `--array=<arg>` | Array job submission |
| `--constraint=hwperf` | Hardware performance counter access |
| `--export=none` | Clean environment (sbatch only) |
| `--gres=gpu:<type>:<count>` | GPU resources |

**Note:** Beginning with Slurm 22.05, `srun` must explicitly request `--cpus-per-task` or set the `SRUN_CPUS_PER_TASK` environment variable.

## Job Script General Structure

```bash
#!/bin/bash -l                     # Interpreter directive; -l initializes modules

#SBATCH --nodes=X                  # Resource requirements
#SBATCH --ntasks=X                 # All #SBATCH lines must be uninterrupted
#SBATCH --time=hh:mm:ss
#SBATCH --job-name=job123
#SBATCH --export=NONE              # Clean environment

unset SLURM_EXPORT_ENV             # Enable script environment export to srun

module load <modules>              # Setup environment

srun ./application [options]       # Execute parallel application
```

## Managing and Controlling Jobs

### Job and Cluster Status

| Command | Purpose |
|---------|---------|
| `squeue <options>` | Display user job status |
| `scontrol show job <jobID>` | Show detailed job information |
| `sinfo` | Cluster status overview |

### Editing Jobs

Modify pending job resources:
```bash
scontrol update TimeLimit=4:00:00 JobId=<jobID>
```

### Canceling Jobs

```bash
scancel <jobID>                    # Cancel specific job
scancel -u <your_username>         # Cancel all your jobs
```

### Attaching to Running Jobs

```bash
srun --jobid=<jobID> --overlap --pty /bin/bash -l
```

Useful for monitoring GPU utilization via `nvidia-smi`.

## Slurm Environment Variables

| Variable | Description |
|----------|-------------|
| `$SLURM_JOB_ID` | Job identifier |
| `$SLURM_SUBMIT_DIR` | Submission directory |
| `$SLURM_JOB_NODELIST` | Job node list |
| `$SLURM_JOB_NUM_NODES` | Allocated node count |
| `$SLURM_CPUS_PER_TASK` | Cores per task; set `$OMP_NUM_THREADS` to this value |
| `$SLURM_ARRAY_JOB_ID` | First job ID in an array |
| `$SLURM_ARRAY_TASK_ID` | Individual array element ID |
| `$SLURM_GPUS_ON_NODE` | GPUs available on current node |

### Environment Export

SLURM automatically propagates environment variables that are set in the shell at the time of submission. Use `#SBATCH --export=NONE` and `unset SLURM_EXPORT_ENV` for clean job environments. Unsetting `SLURM_EXPORT_ENV` ensures proper `srun` propagation.

## Advanced Topics

### Array Jobs

Submit multiple jobs with identical parameters as a unified unit:
```bash
#SBATCH --array=0-15
#SBATCH --array=0-19%5    # max 5 concurrent
```

### Chain Jobs

Automatically submit follow-up jobs by calling `sbatch` within your script. Check runtime to avoid infinite resubmission:
```bash
if [ "$SECONDS" -gt "3600" ]; then
  sbatch job_script
fi
```

### Job Priorities

| Reason in squeue | Description |
|--------|-------------|
| `Priority` | Higher priority jobs are queued |
| `Dependency` | Waiting for dependent job completion |
| `Resources` | Awaiting resource availability |
| `AssociationGroupResourceLimit` | Association/group resource exhausted |
| `QOSGrpResourceLimit` | QoS resource limit reached |
| `ReqNodeNotAvail` | Required node unavailable |

### Exclusive Jobs for Benchmarking

Use `--exclusive` to prevent other jobs on your nodes. Exclusive jobs are billed for all available node resources regardless of actual usage.
