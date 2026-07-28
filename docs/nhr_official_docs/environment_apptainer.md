# Container (Apptainer) -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/environment/apptainer/

## Overview

Containerization enables packaging applications with all dependencies into a single portable unit, enhancing reproducibility and shareability across computing platforms. NHR@FAU uses Apptainer (formerly Singularity) as its standard container solution, specifically designed for HPC systems without performance penalties.

**Key Properties:**
- Available on all NHR@FAU systems
- All filesystems automatically mounted inside containers
- GPU-dependent application support
- Not suitable for multi-node MPI applications

## Basic Commands

| Command | Purpose |
|---------|---------|
| `apptainer build <name> <def-file>` | Build an Apptainer image |
| `apptainer pull <URI>` | Pull image from a URI |
| `apptainer run <name>` | Run default command in container |
| `apptainer exec <name> <command>` | Execute specific command |
| `apptainer shell <name>` | Enter container with shell |
| `apptainer inspect <name>` | Check container metadata |

## Using Existing Containers

Download containers from repositories like DockerHub and automatically convert them:
```bash
apptainer pull docker://<repository>
```

## Building Containers

### Interactive Build

1. Create writable sandbox: `apptainer build --sandbox <name> docker://<repository>`
2. Enter and modify: `apptainer shell --writable <sandbox_name>`
3. Convert to image: `apptainer build <name>.sif <sandbox_name>`

### Definition File Build

Definition files specify reproducible build configurations using sections like:

- **%post** -- Build-time installation commands
- **%files** -- Copy files into container
- **%runscript** -- Default execution command
- **%environment** -- Set environment variables

Example basic definition:
```
Bootstrap: docker
From: ubuntu:latest

%post
apt-get update
apt-get install -y python

%runscript
exec python "$@"
```

## GPU Support

Apptainer natively supports GPU applications. On GPU clusters (Alex, Helma, TinyGPU), device libraries automatically bind-mount without additional options needed.

Requirements:
- Host GPU driver and CUDA libraries installed
- Container CUDA version compatible with host

Use `--nv` flag if GPU support issues arise:
```bash
apptainer run --nv <container>
```

## Important Configuration Notes

- **Cache location**: Set `$APPTAINER_CACHEDIR` to `$WORK` to preserve `$HOME` space
- **MPI**: Not recommended due to version compatibility requirements
- **RDMA/Infiniband**: Include libraries like `rdma-cora` and `libibverbs1` for efficient communication
- **Default mounts**: All filesystems mount by default; use `--contain` to prevent this

## Known Issues and Solutions

**Permission error during build**: Replace `apptainer build` with `/apps/singularity/apptainer-wrapper.sh build`

**Package version conflicts**: Use `--contain` flag or bind `/home` to empty path:
```bash
apptainer run --bind /tmp:/home
```

## Pulling PyTorch / TensorFlow Containers

```bash
cd $WORK
export APPTAINER_CACHEDIR=$(mktemp -d)
apptainer pull pytorch-latest.sif docker://pytorch/pytorch:latest
apptainer pull tensorflow-latest.sif docker://tensorflow/tensorflow:latest-gpu
rm -r "$APPTAINER_CACHEDIR"
```

Use in job scripts:
```bash
apptainer exec pytorch-latest.sif python3 train.py
```
