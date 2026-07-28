# TensorFlow -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/apps/tensorflow/

## Overview

TensorFlow is a machine learning framework available on the NHR@FAU HPC cluster systems.

**Security Warning:** Do not run TensorBoard on a multi-user system like cluster frontends or GPU nodes. Anyone with access could attach to your port and execute code under your account. The documentation recommends running TensorBoard locally with mounted filesystems via SSHFS instead.

## Installation via pip/conda

### Preparation

1. Start an interactive job on a cluster node
2. Load Python: `module add python`
3. Optionally create a virtual environment (conda or venv)

### Installation Methods

**Using pip:**
```bash
pip install tensorflow
# With CUDA Toolkit included (TensorFlow > 2.13.1):
pip install tensorflow[and-cuda]
```

**Using conda (conda-forge):**
```bash
conda install tensorflow-gpu -c conda-forge
# With CUDA override:
CONDA_OVERRIDE_CUDA="12.2" conda install tensorflow cudatoolkit>=12.2 -c conda-forge
```

Note: Do not use the Anaconda channel, as GPU support only extends to version 2.4.1 (2021).

### Testing Installation

Run on a compute node:
```python
python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Determining Required CUDA Version

Check which CUDA version your TensorFlow installation requires:

```python
python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"
```

Reference the official [TensorFlow GPU compatibility table](https://www.tensorflow.org/install/source#gpu).

## Troubleshooting

### No GPUs Detected Despite CUDA Module Loaded

If the message states "Could not find cuda drivers," the loaded CUDA module version is likely incompatible with TensorFlow. Load the correct CUDA version matching your TensorFlow installation.

### libdevice Directory Warnings

```bash
export "XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT"
```

### TensorRT Not Found Warning

```bash
module avail tensorrt
module add tensorrt/8.5.3.1-cuda11.8-cudnn8.6
```

Match the tensorrt version to your loaded CUDA module.

## Using Docker Images via Apptainer

### From DockerHub

```bash
cd $WORK
export APPTAINER_CACHEDIR=$(mktemp -d)
apptainer pull tensorflow-latest.sif docker://tensorflow/tensorflow:latest-gpu
rm -r "$APPTAINER_CACHEDIR"
```

### From Nvidia NGC

```bash
cd $WORK
export APPTAINER_CACHEDIR=$(mktemp -d)
singularity pull tensorflow-ngc-23.11-tf2-py3.sif docker://nvcr.io/nvidia/tensorflow:23.11-tf2-py3
rm -r "$APPTAINER_CACHEDIR"
```

### Using the Container

```bash
apptainer exec tensorflow-latest.sif ./script.py
```
