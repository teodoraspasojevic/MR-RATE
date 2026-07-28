# Workspaces -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/data/workspaces/

## Overview

Workspaces provide temporary storage on NVMe Lustre systems (anvme and hnvme) with configurable lifespans. They are managed through the `hpc-workspace` system and automatically delete when their duration expires.

**Important note:** NVMe Lustre storage is currently in beta mode with no availability guarantees.

### System Availability
- **anvme**: Accessible on Alex and Fritz clusters
- **hnvme**: Accessible exclusively on Helma cluster

## Prerequisites

Execute workspace commands on the frontends or compute nodes of Alex, Fritz, or Helma clusters.

## Key Commands

### Creating a Workspace
```bash
ws_allocate <name> [<days>]
```
- Duration range: 1-90 days
- Default: 1 day if omitted
- Output includes the workspace path

### Finding a Workspace Path
```bash
ws_find <name>
```

Store the path in a variable:
```bash
STORAGE_DIR="$(ws_find <name>)"
```

### Listing Workspaces
```bash
ws_list [<pattern>]
```

Shows all workspaces with remaining duration and creation details.

### Extending Duration
```bash
ws_extend <name> [<days>]
```

**Note:** This *sets* duration to the specified days, not extends it beyond the current duration.

### Deleting Early
```bash
ws_release <name>
```

### Sharing Workspaces
```bash
ws_share (un)share <name> <account>
```

### Restoring Expired Workspaces
```bash
ws_restore -l              # List restorable workspaces
ws_restore -n <name> -t <target name>  # Restore to new workspace
```

## Quota Management

Check usage and limits with:
```bash
lfs quota /anvme
```

Current quotas include inode limits (files/folders) and volume limits.

## Additional Resources

- [HLRS Workspace Documentation](https://kb.hlrs.de/platforms/index.php/Workspace_mechanism)
- [GitHub User Guide](https://github.com/holgerBerger/hpc-workspace/blob/master/user-guide.md)
