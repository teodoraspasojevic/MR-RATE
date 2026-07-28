# NHR@FAU Documentation -- Scraped Index

Source root: https://doc.nhr.fau.de/
Scraped: 2026-06-13

This index lists all scraped pages. Each file contains the full content of the corresponding documentation page as clean markdown.

---

## Getting Started and Accounts

| File | Source URL | Summary |
|------|-----------|---------|
| `getting_started.md` | https://doc.nhr.fau.de/getting_started/ | Overview of HPC basics: clusters, data layout, modules, Slurm, and good practices for new users |
| `account.md` | https://doc.nhr.fau.de/account/ | How to obtain an HPC account depending on affiliation (BayernKI/NHR, FAU, lectures, external courses) |
| `hpc_portal.md` | https://doc.nhr.fau.de/hpc-portal/ | HPC Portal SSO login, SSH key upload, user/manager/advisor roles, project management tab |
| `nhr_application.md` | https://doc.nhr.fau.de/nhr-application/ | How to apply for NHR compute projects, project types (Starter/Normal/Large), resource quotas, and JARDS submission |
| `acknowledgment.md` | https://doc.nhr.fau.de/acknowledgment/ | Required acknowledgment text for publications using Tier3, NHR, or BayernKI resources |
| `faq.md` | https://doc.nhr.fau.de/faq/ | Frequently asked questions covering SSH, Portal, Slurm, software, filesystems, and hardware |

---

## Cluster Hardware

| File | Source URL | Summary |
|------|-----------|---------|
| `clusters_helma.md` | https://doc.nhr.fau.de/clusters/helma/ | Helma GPU cluster: H100/H200 GPUs (94-141 GB HBM), AMD Zen4 CPUs, 768 GB RAM, 15 TB NVMe per node, hnvme workspaces; 16.94 PFlop/s LINPACK |
| `clusters_alex.md` | https://doc.nhr.fau.de/clusters/alex/ | Alex GPU cluster: A40 and A100 GPUs (40/80 GB), AMD Zen3 CPUs, anvme workspaces, MIG partition for debugging |
| `clusters_fritz.md` | https://doc.nhr.fau.de/clusters/fritz/ | Fritz CPU cluster: Intel Ice Lake/Sapphire Rapids, InfiniBand, Lustre $FASTTMP (3.5 PB), multi-node MPI workloads |
| `clusters_woody.md` | https://doc.nhr.fau.de/clusters/woody/ | Woody throughput cluster: Skylake/Kaby Lake/Ice Lake nodes, single-node jobs only, 32 GB-256 GB RAM per node |
| `clusters_tinygpu.md` | https://doc.nhr.fau.de/clusters/tinygpu/ | TinyGPU cluster: consumer and data-center GPUs (RTX 2080 Ti, V100, RTX 3080, A100), Tier3-only, `.tinygpu` Slurm suffix |
| `clusters_tinyfat.md` | https://doc.nhr.fau.de/clusters/tinyfat/ | TinyFat cluster: high-memory nodes (256 GB to 2 TB), AMD EPYC and Intel Broadwell, Tier3-only, `.tinyfat` Slurm suffix |
| `clusters_meggie.md` | https://doc.nhr.fau.de/clusters/meggie/ | Meggie cluster (DECOMMISSIONED): formerly 728-node Broadwell MPI cluster; migrate to Woody or Fritz |

---

## Access and Connectivity

| File | Source URL | Summary |
|------|-----------|---------|
| `access_ssh.md` | https://doc.nhr.fau.de/access/ssh-command-line/ | SSH key generation, `~/.ssh/config` template for all clusters, port forwarding for Jupyter, troubleshooting |
| `access_jupyterhub.md` | https://doc.nhr.fau.de/access/jupyterhub/ | JupyterHub access via HPC Portal, available resources by account type, registering conda/venv kernels |

---

## Data and Storage

| File | Source URL | Summary |
|------|-----------|---------|
| `data_filesystems.md` | https://doc.nhr.fau.de/data/filesystems/ | Complete filesystem table ($HOME 50 GB, $HPCVAULT 500 GB, $WORK, $FASTTMP Lustre, $TMPDIR), snapshots, quotas, ACLs |
| `data_workspaces.md` | https://doc.nhr.fau.de/data/workspaces/ | Temporary NVMe Lustre workspaces (anvme/hnvme) via ws_allocate/ws_find/ws_extend; up to 90-day lifetime |
| `data_copying.md` | https://doc.nhr.fau.de/data/copying/ | Data transfer using scp, rsync, and WinSCP via the csnhr.nhr.fau.de dialog server |
| `data_staging.md` | https://doc.nhr.fau.de/data/staging/ | Node-local SSD staging in/out patterns via $TMPDIR, including shared staging for concurrent jobs on same node |

---

## Software Environment

| File | Source URL | Summary |
|------|-----------|---------|
| `environment_modules.md` | https://doc.nhr.fau.de/environment/modules/ | Environment modules overview, key commands (load/unload/avail/list), custom module trees in $HOME/.modulefiles |
| `environment_python.md` | https://doc.nhr.fau.de/environment/python-env/ | Python conda and venv setup on HPC, one-time conda init, configuring envs/pkgs dirs to $WORK, Jupyter kernel registration |
| `environment_apptainer.md` | https://doc.nhr.fau.de/environment/apptainer/ | Apptainer (Singularity) container usage: building, pulling from DockerHub/NGC, GPU support, definition file syntax |

---

## Slurm Job Scheduling

| File | Source URL | Summary |
|------|-----------|---------|
| `slurm_batch_system.md` | https://doc.nhr.fau.de/batch-processing/batch_system_slurm/ | Slurm reference: sbatch/salloc/srun options, job script structure, managing/canceling jobs, array jobs, environment variables |
| `job_monitoring_clustercockpit.md` | https://doc.nhr.fau.de/job-monitoring-with-clustercockpit/ | ClusterCockpit web interface for monitoring flop rates, memory bandwidth, GPU utilization, accessed via HPC Portal |

---

## Applications

| File | Source URL | Summary |
|------|-----------|---------|
| `apps_pytorch.md` | https://doc.nhr.fau.de/apps/pytorch/ | PyTorch installation (pip/conda/container), torch.compile optimization, DDP multi-GPU training, torchrun job scripts |
| `apps_tensorflow.md` | https://doc.nhr.fau.de/apps/tensorflow/ | TensorFlow installation (pip/conda/container), CUDA version determination, TensorBoard security warning, troubleshooting |
| `apps_nvidia_gpus.md` | https://doc.nhr.fau.de/apps/nvidia-gpus/ | CUDA modules, nvidia-smi, nvtop, MPS daemon, Nsight Systems/Compute profiling, LIKWID GPU support |
| `apps_spack.md` | https://doc.nhr.fau.de/apps/spack/ | Spack user package manager: installing to $WORK, spec/install syntax, using Spack-built packages as modules |

---

## Cluster Quick-Reference Table

| Cluster | Type | GPUs | CPUs | Access | Slurm Suffix |
|---------|------|------|------|--------|--------------|
| Helma | GPU | H100 (94 GB), H200 (141 GB) | AMD Zen4, 128c | NHR (apply required) | none |
| Alex | GPU | A40 (48 GB), A100 (40/80 GB) | AMD Zen3, 128c | NHR + Tier3 (request) | none |
| Fritz | CPU | none | Intel Ice Lake (72c), Sapphire Rapids (104c) | NHR + Tier3 (request) | none |
| Woody | CPU | none | Intel Skylake/Kaby Lake/Ice Lake | Tier3 default | none |
| TinyGPU | GPU | RTX 2080 Ti, V100, RTX 3080, A100 | Intel/AMD | Tier3 default | `.tinygpu` |
| TinyFat | CPU | none | Intel Broadwell / AMD Zen2, up to 2 TB RAM | Tier3 default | `.tinyfat` |

## Key Environment Variables

| Variable | Filesystem | Notes |
|----------|-----------|-------|
| `$HOME` | `/home/hpc/GROUP/USER` | 50 GB, backed up, snapshotted |
| `$HPCVAULT` | `/home/vault/GROUP/USER` | 500 GB, backed up, snapshotted |
| `$WORK` | `/home/woody/GROUP/USER` (default) | 1 TB+, no backup |
| `$FASTTMP` | `/lustre/GROUP/USER` | Fritz/Alex parallel I/O, 3.5 PB, no backup |
| `$TMPDIR` | Node-local SSD | Deleted after job; 1.8-15 TB depending on cluster |
| `HF_HOME` | Set to `$WORK/.cache/huggingface` | Avoids filling $HOME with model weights |

## Support Contact

- Email: hpc-support@fau.de
- HPC Portal: https://portal.hpc.fau.de
- Publications list: nhr-redaktion@lists.fau.de
