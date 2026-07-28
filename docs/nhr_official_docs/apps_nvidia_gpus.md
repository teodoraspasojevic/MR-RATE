# Working with NVIDIA GPUs -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/apps/nvidia-gpus/

## CUDA Compilers and Libraries

The `cuda` module provides CUDA compilers and runtime libraries. Loading a module like `module load cuda/12.1.0` configures environment variables and sets `CUDA_HOME` or `CUDA_INSTALL_PATH`. The NVIDIA HPC compilers (formerly PGI) are available through `nvhpc` modules.

## GPU Statistics in Job Output

Slurm automatically appends GPU utilization statistics to job outputs. For each CUDA binary executed, the report includes GPU name, bus ID, process ID, GPU utilization percentage, memory utilization, maximum memory usage, and execution time.

## NVIDIA System Management Interface

`nvidia-smi` is a command-line utility for monitoring GPU utilization and processes.

Continuous monitoring:
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -l 1
```

Attach to a running job to check GPU utilization:
```bash
srun --jobid=<jobID> --overlap --pty /bin/bash -l
nvidia-smi
```

## nvtop GPU Status Viewer

`nvtop` functions as a task monitor for NVIDIA GPUs, displaying memory, utilization, temperature, and process information. Available as a module on Alex and TinyGPU clusters.

## NVIDIA Multi-Process Service (MPS)

MPS enables cooperative multi-process CUDA applications, benefiting performance when single-process GPU compute capacity remains underutilized.

### Single-GPU Jobs

Start the MPS daemon with environment variables, run applications in parallel, then stop the daemon:
```bash
echo quit | nvidia-cuda-mps-control
```

### Multi-GPU Jobs

For multiple GPUs, initialize separate MPS servers for each GPU using UUIDs, ensuring correct `CUDA_MPS_PIPE_DIRECTORY` configuration per process.

## GPU-Profiling with NVIDIA Tools

Two primary profiling tools are available:

- **nsys (Nsight Systems)**: Profiles whole application behavior
- **ncu (Nsight Compute)**: Analyzes specific kernel performance

### Nsight Systems

```bash
nsys profile ./a.out
nsys profile --stats=true -o filename --force-overwrite=true ./a.out
```

### Nsight Compute

```bash
ncu ./a.out
ncu --launch-skip 10 --launch-count 5 ./a.out
```

## LIKWID

LIKWID 5.0 supports NVIDIA GPUs with an API mirroring CPU functionality. Provides GPU-specific performance counter access through command-line applications.

## Compute Capabilities Summary

| GPU | Cluster | Compute Capability | NVCC Flag |
|-----|---------|-------------------|-----------|
| A100 | Alex, TinyGPU | 8.0 | `-gencode arch=compute_80,code=sm_80` |
| A40 | Alex | 8.6 | `-gencode arch=compute_86,code=sm_86` |
| H100 | Helma | 9.0 | `-gencode arch=compute_90,code=sm_90` |
| H200 | Helma | 9.0 | `-gencode arch=compute_90,code=sm_90` |
| RTX 3080 | TinyGPU | 8.6 | `-gencode arch=compute_86,code=sm_86` |
| V100 | TinyGPU | 7.0 | `-gencode arch=compute_70,code=sm_70` |
