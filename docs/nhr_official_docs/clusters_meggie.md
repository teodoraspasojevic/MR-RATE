# Meggie Cluster (Decommissioned) -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/clusters/meggie/

## Status Notice

**Meggie has been decommissioned. Users should migrate to Woody or Fritz instead.**

## Overview

Meggie was FAU's high-performance compute cluster manufactured by Megware, designed for distributed-memory (MPI) or hybrid parallel programs requiring medium to high communication capabilities.

### Hardware Specifications

- **728 compute nodes**: Each equipped with two Intel Xeon E5-2630v4 "Broadwell" processors (10 cores per chip at 2.2 GHz, 25 MB shared cache, 64 GB RAM)
- **2 frontend nodes**: Same CPU configuration with 128 GB RAM
- **Storage**: Lustre-based parallel filesystem (~1 PB capacity, >9000 MB/s aggregated I/O bandwidth)
- **Network**: Intel Omni-Path interconnect (100 GBit/s per link)
- **Peak Performance**: ~481 TFlop/s (measured LINPACK)

## Access

SSH connection: `ssh meggie.rrze.fau.de`

NHR accounts were not enabled by default for this system.
