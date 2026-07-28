# JupyterHub -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/access/jupyterhub/

## Overview

Access to NHR@FAU systems is available through JupyterHub. Login procedures depend on your HPC account type.

## JupyterHub for NHR Project Accounts

- Login via HPC Portal
- Globally accessible
- Available resources:
  - 2 cores/4GB on a shared node
  - One A40 or A100 GPU in Alex cluster
  - One Fritz cluster node

## JupyterHub for Tier3 HPC Accounts

- Login via HPC Portal
- Accessible only within FAU network or via VPN
- Available resources:
  - 2 cores/4GB on a shared node
  - 1-4 dedicated GTX1080Ti GPUs
  - 1-4 cores and 8-32 GB on TinyFat cluster

Tier3 accounts can also access R-Studio and Whisper through JupyterHub.

## Login Procedure via HPC Portal

1. Log in at the [HPC Portal](https://portal.hpc.fau.de)
2. Navigate to the **User** page
3. Under **Your accounts**, select the account for JupyterHub
4. Click **Go to JupyterHub**
5. Accept the Terms of Service in the new window and proceed to JupyterHub

## Using Custom Environments in JupyterHub

### Register a Conda Environment as a Kernel

```bash
conda install -n <env_name> ipykernel
conda run -n <env_name> python3 -m ipykernel install --user --name <env_name>
```

### Register a venv as a Kernel

```bash
source <path_to_env>/bin/activate
pip install ipykernel
python3 -m ipykernel install --user --name=<env_name>
```

### Access $WORK from JupyterHub

Create a symbolic link in your home directory:
```bash
ln -s $WORK $HOME/work
```

## Running Jupyter Notebook Manually via SSH Port Forwarding

1. Log into the cluster frontend or start a job on your desired compute node

2. Get the FQDN of the node:
   ```bash
   hostname
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook --no-browser --port=<remote port>
   ```

4. Set up port forwarding on your local machine:
   ```bash
   ssh -L <local port>:localhost:<remote port> <remote server>
   ```

5. Open your browser at: `http://localhost:<local port>/tree?token=<token>`
