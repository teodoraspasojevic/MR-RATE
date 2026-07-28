# FAQs -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/faq/

## General Information

**Acknowledging resource usage:** Guidelines are available in the Acknowledgment section for how to credit NHR@FAU systems in publications.

**HPC export controls:** FAU members should contact `exportkontrolle@fau.de` for guidance on export regulations.

## Accessing HPC

**Account creation:** Access methods vary by user status. Details are provided in the Getting an Account section.

**Alex and Fritz cluster access:** FAU staff and students with existing HPC accounts can request access through designated links. External scientists must submit NHR proposals.

## SSH Access

**No password available:** Accounts created through the HPC portal use SSH key authentication exclusively. Users must generate an SSH key pair and upload the public key to the portal.

**Password prompts at csnhr:** The dialog server does not retain SSH keys for subsequent connections. Solutions include using SSH proxy jump, creating additional key pairs on csnhr, or configuring SSH agent forwarding.

**XRDP connection issues:** Kill stuck sessions:
```bash
ssh <USERNAME>@csnhr.nhr.fau.de
pkill -9 -u <USERNAME>
```

## HPC Portal

**SSH key propagation:** Updated keys take 2-4 hours to synchronize across all systems.

**New account access:** Newly created accounts become usable the following morning after file systems and Slurm databases are updated.

**Mailing list management:** Users can opt in/out of nhr-users and nhr-sysannounce lists through the Profile section after logging in.

## Batch System (Slurm)

**Job priority:** The system automatically assigns priorities based on waiting time, partition, user group, and fairshare (recent CPU/GPU usage).

**Running computational work:** All intensive tasks must be submitted via Slurm batch system.

**Command line argument syntax:** Options must precede the batch script:
```bash
sbatch [OPTIONS] script [args ...]
```
Arguments following the script are passed to the script, not sbatch.

**Interactive jobs:** Use `salloc` for testing or debugging. Run `module purge` first to avoid inheriting conflicting module paths.

**Alex GPU-only resources:** Request `-p a100mig` partition; `-n 4` provides CPU cores without GPUs (32 cores maximum, 8 per job).

**Alex debugging with partial GPUs:**
- `-p a100mig --gres=gpu:a100small:1` provides 10GB VRAM + 4 CPU cores
- `-p a100mig --gres=gpu:a100med:1` provides 20GB VRAM + 4 CPU cores

**A100 GPU selection:**
- Use `-C a100_80` for 80GB models
- Use `-C a100_40` for 40GB models

**Module command not found:** Ensure job scripts start with `#!/bin/bash -l` (login shell).

**Attaching to running jobs:** See batch system documentation for attachment procedures.

## Software

**Container permissions errors:** Use `/apps/singularity/apptainer-wrapper.sh build <options>` instead of direct `apptainer build` commands on Ubuntu systems.

**sudo access:** Users cannot obtain administrative privileges. Install software into personal directories using environment modules, Spack user installation, or containers.

**Available software:** Check the Applications and Software Development sections. Load `000-all-spack-pkgs` module to view all Spack packages.

**Internet access on compute nodes:** Configure proxy servers in job scripts:
```bash
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
```

Some applications require uppercase variable names (HTTP_PROXY, HTTPS_PROXY).

**GPU not used by applications:** Verify PyTorch and TensorFlow installations support GPUs. Ensure required CUDA modules (cuda, cudnn, tensorrt) are loaded.

**Conda configuration errors:** After loading the python module, execute:
```bash
conda config --add envs_dirs $WORK/software/private/conda/envs
conda config --add pkgs_dirs $WORK/software/private/conda/pkgs
```

## File Systems and Data Storage

**Doubled quota usage:** Data on `/home/hpc` and `/home/vault` is replicated across two arrays, temporarily doubling quota consumption. All quotas have been increased accordingly.

**$WORK access via JupyterHub:** Create a symbolic link in your home directory:
```bash
ln -s $WORK $HOME/work
```

**Node-local storage ($TMPDIR):** Each node provides at least 1.8TB of local SSD space. Data is automatically deleted when jobs end; preserve important files by copying to network filesystems.

**Large file collections:** Use containerized formats (HDF5), file-based databases, or archive files with tar compression. Store on $TMPDIR for faster access.

**HuggingFace cache location:** Redirect from `$HOME` (limited space) to larger directories:
```bash
export HF_HOME=$WORK/.cache/huggingface
```

**Line ending conversions:**
```bash
dos2unix    # Windows to Linux
mac2unix    # MacOS to Linux
unix2dos    # Linux to Windows
unix2mac    # Linux to MacOS
```

## Hardware

**SMT/Hyperthreading:** SMT is disabled on most NHR@FAU systems for security and performance consistency.

**Infiniband/RDMA in containers:** Include `rdma-core` and `libibverbs1` libraries.

**NCCL debugging:** Set `NCCL_DEBUG=INFO` for extensive output or `NCCL_DEBUG=WARN` for warnings only.

**A100 node networking:** Infiniband devices appear as `mlx5_0:1/IB` and `mlx5_3:1/IB`. Enforce specific devices:
```bash
export NCCL_IB_HCA="=mlx5_0:1,mlx5_3:1"
```
