# Python Virtual Environments -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/environment/python-env/

## Overview

NHR@FAU HPC systems support Python environments via two approaches: conda and venv. These isolated environments maintain independent package sets. Always use the system Python module rather than the OS Python.

## Conda Environments

### Initial Setup (One-Time)

Before creating conda environments, users must initialize the system once:

1. **Create `~/.bash_profile`** if it does not exist:
   ```bash
   if [ -f ~/.bashrc ]; then . ~/.bashrc; fi
   ```

2. **Load Python module**:
   ```bash
   module add python
   ```

3. **Configure conda storage** to use `$WORK` directory instead of `$HOME`:
   ```bash
   conda config --add pkgs_dirs $WORK/software/private/conda/pkgs
   conda config --add envs_dirs $WORK/software/private/conda/envs
   ```

### Creating Environments

Create and activate a conda environment:
```bash
conda create -n <env_name> python=<version>
conda activate <env_name>
```

Alternatively, use an environment file for reproducibility:
```bash
conda env export --from-history --file environment.yml
conda env create -f environment.yml
```

## Virtual Environments with venv

For lightweight environments, use Python's built-in venv:

```bash
python3 -m venv <path_to_env>
source <path_to_env>/bin/activate
```

**Important**: Conda does not work inside activated venv environments.

## Jupyter Integration

Make environments available in Jupyter by installing `ipykernel`:

**Conda environments**:
```bash
conda install -n <env_name> ipykernel
conda run -n <env_name> python3 -m ipykernel install --user --name <env_name>
```

**venv environments**:
```bash
pip install ipykernel
python3 -m ipykernel install --user --name=<env_name>
```

## Key Recommendations

- Use interactive jobs on target clusters for package installation to ensure proper GPU support
- Configure proxy settings for systems without direct internet access:
  ```bash
  export http_proxy=http://proxy.nhr.fau.de:80
  export https_proxy=http://proxy.nhr.fau.de:80
  ```
- Install packages inside an activated environment (no `--user` flag needed)
- Redirect HuggingFace cache to `$WORK` to avoid filling `$HOME`:
  ```bash
  export HF_HOME=$WORK/.cache/huggingface
  ```
