# Copying Data -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/data/copying/

## Overview

This page describes tools for transferring data between local and remote locations. The dialog server `csnhr.nhr.fau.de` is used in examples because:

- The default SSH configuration tunnels all connections through this server, avoiding unnecessary load
- All filesystems (`$HOME`, `$HPCVAULT`, `$WORK`) are mounted there

## scp

`scp` is a command-line tool for copying data between local or remote locations.

**General syntax:**
```bash
scp [<options>] <source> [<source> [...]] <destination>
```

**Common options:**
- `-r`: recursively copy directories (mandatory for directories)
- `-p`: preserve modification times and access rights

**Prerequisites:**
- SSH connection must be configured with SSH keys
- Successful connection to `csnhr.nhr.fau.de`

**Copy from local to remote:**
```bash
scp -r data_dir1 ~/other_data_dir2 /mnt/data3 csnhr.nhr.fau.de:data
```
Copies three directories into `$HOME/data` on the remote server.

**Copy from remote to local:**
```bash
scp -r csnhr.nhr.fau.de:results/2023 /mnt/backup/
```

**Using wildcards with remote locations** (escape with backslash):
```bash
scp csnhr.nhr.fau.de:results/2023/\*.dat .
```

## rsync

`rsync` is similar to `scp` but offers additional features like resuming interrupted transfers.

**General syntax:**
```bash
rsync [<options>] <source> [<source> [...]] <destination>
```

**Common options:**
- `-a`: archive mode (preserves attributes, permissions, symlinks)
- `--append-verify`: resume partially transferred files
- `-p`: preserve modification times and access rights
- `-v`: verbose output
- `-z`: compress data during transfer

**Note:** Using compression (`-z`) when transferring between NHR@FAU systems might increase duration.

**Copy local directory to remote:**
```bash
rsync -avz samples/2023 csnhr.nhr.fau.de:samples
```

**Copy directory contents (with trailing slash):**
```bash
rsync -avz samples/2023/ csnhr.nhr.fau.de:samples
```

**Copy remote directory to local:**
```bash
rsync -avz csnhr.nhr.fau.de:/home/atuin/<group>/<user>/results ~/simulation
```

## WinSCP (Windows GUI)

WinSCP is a graphical SFTP client for Windows enabling drag-and-drop file transfers.

### Connecting

1. Enter `csnhr.nhr.fau.de` as the hostname
2. Enter your HPC username
3. Click **Advanced...** to open Advanced Site Settings
4. Select **Authentication** in the left tree
5. Specify the path to your private SSH key under **Private key file**
6. Click **OK** and then **Login**

### Accessing a Cluster Frontend via WinSCP

Configure `csnhr.nhr.fau.de` as a tunnel host in Advanced Site Settings -> Connection -> Tunnel, then configure the target cluster node (fritz, alex, etc.) in the Login dialog.
