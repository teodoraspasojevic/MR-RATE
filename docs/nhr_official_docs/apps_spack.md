# Spack Package Manager -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/apps/spack/

## Overview

Spack is a package manager available on NHR@FAU systems as the `user-spack` module. It enables users to install specialized software not provided by the computing center.

**Key Benefits:**
- Easy management of multiple package versions
- Separate installations for different dependency combinations

**Trade-off:** Users must specify exact package details, dependencies, and compiler versions.

## Basic Operations

### Loading Spack

```bash
module load user-spack
```

All software installs to `$WORK/USER-SPACK`.

### Common Commands

```bash
spack compilers             # List available compilers
spack list netcdf           # Search packages
spack find openmpi          # Find installed packages
spack info numactl          # Get package information
spack versions kokkos       # Check available versions
spack spec <pkgspec>        # Preview dependencies before installation
```

### Installation Syntax

```bash
spack install hdf5@1.10.1 %gcc@4.7.3 +debug ^openmpi+cuda fabrics=auto ^hwloc+gl
```

This installs hdf5 version 1.10.1 with gcc 4.7.3, enables debug mode, and specifies OpenMPI and hwloc configurations.

### Network Configuration

For systems requiring proxy servers:

```bash
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
```

## Using Installed Packages

Installed packages automatically become available as modules:

```bash
module load netcdf-fortran/4.5.4-oneapi2022.1.0-openmpi-3bmi7ym
```

Unload with:
```bash
module unload netcdf-fortran
```

## Limitations

Spack's built-in `spack load`, `spack unload`, and `spack env` commands are currently unavailable at NHR@FAU systems. Use the module system instead.
