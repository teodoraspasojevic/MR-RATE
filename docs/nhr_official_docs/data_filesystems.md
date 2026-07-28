# File Systems -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/data/filesystems/

## Available Filesystems

| Mount Point | Environment Variable | Purpose | Technology | Backup | Snapshots | Data Lifetime | Quota |
|---|---|---|---|---|---|---|---|
| `/home/hpc` | `$HOME` | Source, input, important results | NFS | YES | YES | Account lifetime | 50 GB |
| `/home/vault` | `$HPCVAULT` | Mid-/long-term storage | NFS | YES | YES | Account lifetime | 500 GB |
| `/home/{woody,saturn,titan,janus,atuin}` | `$WORK` | General-purpose, log files | NFS | NO | NO | Account lifetime | Tier3: 1000 GB, NHR: project quota |
| `/lustre` | `$FASTTMP` (Fritz/Alex) | High performance parallel I/O | Lustre via InfiniBand/Ethernet | NO | NO | High watermark | Inodes only |
| `/anvme`, `/hnvme` | Workspace (Alex/Helma) | High performance IOPs | Lustre via InfiniBand/Ethernet | NO | NO | Workspace lifetime | Inodes only |
| Node-local | `$TMPDIR` | Node-local job-specific directory | SSD/RAM disk | NO | NO | Job runtime | NO |

## Home Directory ($HOME)

Home directories are located at `/home/hpc/GROUPNAME/USERNAME`. This is your default login location. The system performs regular snapshots and backups, making it suitable for job scripts, source code, or unrecoverable input files. Due to these protections, capacity is limited to 50 GB with no extension options available.

## Vault ($HPCVAULT)

Located at `/home/vault/GROUPNAME/USERNAME`, this filesystem offers mid and long-term storage of files with snapshots and backups, though less frequently than `$HOME`. It provides 500 GB quota per user.

## Work Directory ($WORK)

Use `$WORK` as your general work directory. It has neither snapshots nor backup, so keep critical data in `$HOME` or `$HPCVAULT`. Actual location depends on your account type:

- `/home/woody`: Mounted on all systems (default)
- `/home/saturn`, `/home/titan`, `/home/janus`: For shareholder groups with group quotas
- `/home/atuin`: For NHR projects with quotas set per group proposal

## Parallel Filesystems ($FASTTMP)

Available on Fritz frontends and nodes, `$FASTTMP` is a Lustre-based parallel filesystem mounted at `/lustre/$GROUP/$USER/`. It offers 3.5 PB capacity with no data volume limits, but restricts file count.

**Key characteristics:**
- High-watermark deletion triggers at ~80% capacity, removing oldest/largest files first
- Supports aggregate bandwidth of > 20 GB/s
- Designed for large files written simultaneously via MPI-I/O
- Not made for handling large amounts of small files
- Files under 1 MB stored on single servers with performance overhead

**Note on tar:** Use `tar -mx` or `touch` in combination with `find` to update the modification time since unpacking preserves original timestamps.

## Node-Local Job-Specific Directory ($TMPDIR)

Within SLURM jobs, `$TMPDIR` points to node-local fast scratch space automatically created and removed.

**Location by cluster:**
- **Alex**: Node-local NVMe SSD
- **Fritz**: Node-local RAM disk (reduces available application RAM)
- **TinyFat, TinyGPU, Woody**: Node-local SSD

Use `$TMPDIR` for training data during a job to reduce pressure from `$WORK`, increase the I/O bandwidth, and reduce the I/O latency.

## Quotas

Nearly all filesystems impose quotas on data volume and/or file counts, set per user or group. Soft quotas can be exceeded temporarily; hard quotas are absolute limits.

**Check quotas with:**
- `shownicerquota.pl` (user-friendly, NHR@FAU systems only)
- `quota -s` (standard Unix command)

## Snapshots for $HOME and $HPCVAULT

Snapshots record periodic filesystem states in hidden `.snapshots` directories. They protect against accidental deletion but do not provide protection against a loss of the filesystem like backups do.

**Recovery example:**

List snapshots:
```bash
ls -l /home/hpc/exam/example1/.snapshots/
```

Restore file:
```bash
cp '/home/hpc/exam/example1/.snapshots/@GMT-2019.03.05-03.00.00/important.txt' '/home/hpc/exam/example1/important.txt'
```

**$HOME snapshot intervals:**
- Every 30 minutes: 6 copies (3 hours coverage)
- Every odd hour: 12 copies (1 day coverage)
- Daily at 03:00 UTC: 7 copies (1 week coverage)
- Weekly Sundays at 03:00 UTC: 4 copies (4 weeks coverage)

**$HPCVAULT snapshot intervals:**
- Daily at 03:00 UTC: 7 copies (1 week coverage)
- Weekly Sundays at 03:00 UTC: 4 copies (4 weeks coverage)

Times are in GMT/UTC (04:00-05:00 German time depending on daylight saving).

## Advanced Topics

### Limitations on File Numbers

Having a large number of small files is bad for filesystem performance. Limits are set higher for `$HOME` than `$HPCVAULT`. Archive small unused files using `tar`, `zip`, etc.

### Access Control Lists (ACLs)

Beyond standard Unix `chmod` permissions, NFS v4 ACLs are supported via `nfs4_setfacl` and `nfs4_getfacl` commands. These are compatible with Windows Explorer interfaces.

### Storage Infrastructure

The system uses Lenovo hardware and IBM Spectrum Scale/GPFS software (operational since September 2020).

**Technical specifications:**
- 5 file servers (SR650, 128 GB RAM, 100 GB Ethernet)
- 1 archive frontend (SR650, 128 GB RAM)
- 1 TSM server (SR650, 512 GB RAM)
- IBM TS4500 tape library with 8 LTO8 drives, 3,370 LTO8 slots, >700 LTO7M tapes
- 4 Lenovo DE6000H storage arrays with 8 DE600S expansion units
- Usable capacity: 5 PB (vault), 1 PB (FauDataCloud), 40 TB (homes)
