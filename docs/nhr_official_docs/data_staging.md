# Using Node-Local SSDs (Data Staging) -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/data/staging/

## Overview

Node-local SSDs are available on the Alex, TinyFat, TinyGPU, and Woody clusters, providing scratch space and caching capabilities. These drives deliver superior performance compared to network-based storage options.

The SSDs are accessed through the `$TMPDIR` environment variable, offering higher bandwidth and reduced latency relative to `$HOME`, `$HPCVAULT`, and `$WORK` directories.

## Staging Data In and Out

**Staging in** involves transferring files to `$TMPDIR` from slower filesystems (such as `$WORK`) at job startup.

**Staging out** means copying newly created or modified data from `$TMPDIR` to persistent storage before the temporary data is deleted upon job completion.

### Basic Example Script

```bash
#!/bin/bash -l
#SBATCH --gres=gpu:<GPU>:<NGPUS>
#SBATCH --time=<TIME>
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module add python
conda activate YOUR_ENVIRONMENT

# Stage data to $TMPDIR
# tar xf "$WORK/dataset.tar" "$TMPDIR"
# cp -r "$WORK/your-datasets" "$TMPDIR"

python3 train.py --dataset-path "$TMPDIR" --workdir "$TMPDIR" ...

# Copy results back to persistent storage
cp -r "$TMPDIR/results" "$WORK"
```

## Sharing Data Across Concurrent Jobs

Multiple jobs running simultaneously on the same node can share staged data, reducing redundant copying. This approach requires:

- Data staged to `$TMPDIR`
- Read-only data access
- Jobs belonging to the same data class

### Implementation Template

```bash
#!/bin/bash -l
#SBATCH --gres=gpu:<GPU>:<NGPUS>
#SBATCH --time=<TIME>
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Assign a job class identifier
readonly JOB_CLASS="TODO"

# Define shared staging directory
readonly STAGING_DIR="/tmp/$USER-$JOB_CLASS"

# Create staging directory with restricted permissions
(umask 0077; mkdir -p "$STAGING_DIR") || { echo "ERROR: creating $STAGING_DIR failed"; exit 1; }

# Implement file locking to prevent race conditions
(
  exec {FD}>"$STAGING_DIR/.lock"
  flock "$FD"

  # Stage data only if not already present
  if [ ! -f "$STAGING_DIR/.complete" ]; then
    # TODO: Insert data staging commands here

    : > "$STAGING_DIR/.complete"
  fi
)

# Application can now use data from $STAGING_DIR
```

**Important considerations:**

- Slurm does not guarantee concurrent job placement on the same node
- This approach provides benefits only if jobs happen to run together
- Do not attempt to force specific node assignments, as this causes scheduling delays
