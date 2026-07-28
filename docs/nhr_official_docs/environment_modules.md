# Environment Modules -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/environment/modules/

## Overview

Environment modules allow having different versions of compilers, applications, and libraries installed, but only make one specific version available to your shell at a time. Module management adjusts shell environment variables when loading and unloading software packages. Available modules vary across clusters.

## Module Commands

| Command | Purpose |
|---------|---------|
| `module avail` | Lists all available modules |
| `module whatis` | Shows verbose listing of modules |
| `module list` | Displays currently loaded modules |
| `module load <pkg>` | Loads a module package |
| `module load <pkg>/<version>` | Loads specific package version |
| `module unload <pkg>` | Removes a module |
| `module help <pkg>` | Shows detailed module description |
| `module show <pkg>` | Displays environment variable changes |

**Important note:** Module changes only affect the current shell session. New shells require reloading modules. Always load the same modules in Slurm job scripts.

## Important Modules

| Module | Function |
|--------|----------|
| `gcc` | GCC compiler suite |
| `intel` | Intel Classic/OneAPI compilers |
| `intelmpi` | Intel MPI library |
| `openmpi` | Open MPI library |
| `python` | Conda Python environment |
| `cuda` | CUDA compilers and runtime |
| `000-all-spack-pkgs` | View all Spack-installed packages |

## Custom Module Trees

Users can create personal module directories at `$HOME/.modulefiles` for software not provided by the system. Each module requires:

1. A folder structure: `$HOME/.modulefiles/mymod/version`
2. A module file containing configuration directives
3. Registration via `module use -a $HOME/.modulefiles`

Module files typically define paths, environment variables, and dependencies using standard directives like `prepend-path` and `setenv`.

## Tips

- Run `module purge` before `salloc` to avoid inheriting conflicting module paths from your login shell
- Job scripts must begin with `#!/bin/bash -l` (login shell) for `module` to be available
- Use `module avail <partial-name>` to search for a specific package
