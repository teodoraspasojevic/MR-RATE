# FAU / NHR@FAU HPC Execution Profile — 3D Latent Diffusion Training for MR-RATE

**Scope**: what is operationally feasible for training/fine-tuning a high-resolution 3D latent diffusion (report-to-volume) model on NHR@FAU infrastructure, given (a) the locally scraped official documentation at `docs/nhr_official_docs/` and (b) live, read-only queries run from inside this session.

**Method note / classification legend** (applied to every substantive claim below):
- **VERIFIED** — directly supported by a local doc file or a live command run in this session.
- **INFERRED** — strongly implied by local evidence but not stated verbatim.
- **ASSUMED** — a provisional design choice made by this report because the diffusion model's architecture/resolution is not specified anywhere in this repo (see §Grounding below).
- **UNKNOWN** — could not be determined from local docs or the cheap queries permitted for this task.

**Live session context** (VERIFIED): this conversation is running with a real shell on the login node `helma2.nhr.fau.de` (`hostname -f`, this session), under HPC account `<user>` / group `<group>` (redacted per task instructions — the real values are the account this session is authenticated as). All `sinfo`/`scontrol`/`sacctmgr`/`module`/quota commands below were run directly, without a `.tinygpu`/`.tinyfat` wrapper, confirming this is a Helma frontend (Helma uses plain Slurm commands — `docs/nhr_official_docs/slurm_batch_system.md:24` only requires the wrapper suffix for TinyFat/TinyGPU). No jobs were submitted; no network access was attempted; no dataset files were opened.

**Redaction note**: real NHR account/group codes surfaced by live queries (this account's own code, the code of the account that owns the `MR-RATE` raw-data workspace, and a ~40-entry allow-list of other accounts on the `h200` partition) have been replaced with placeholders (`<user>`, `<group>`, `<data-account>`) throughout this document, per task instructions.

---

## Grounding: what "the model" is, and why several numbers below are ASSUMED

This repo does not yet contain a diffusion model. Per the existing audit (`docs/design/recommended_next_steps.md:3,29`, `docs/design/audit_progress.md:120`), the intended architecture is **3D latent diffusion** (a VAE compressing volumes to a latent grid + a diffusion/flow-matching model with text cross-attention, analogous to NVIDIA's `NV-Generate-MR-Brain`), explicitly **not** the contrastive VJEPA2 model already implemented in `contrastive-pretraining/`. A search of `docs/challange_docs/` (the forithmus challenge-platform docs) found no resolution, latent-shape, or architecture specification — those docs describe submission mechanics (Docker upload, schema auto-detection, evaluation pipeline), not model requirements (checked `Data_Schemas_Mock_Data.md`, `Evaluation_Pipeline.md`, `Docker_Submission.md`, `Quick_Start.md`).

Consequently, every VRAM/batch-size/storage number below that depends on an actual model shape is **ASSUMED**, anchored to the one concrete resolution that does exist in this codebase: the `contrastive-pretraining` loader's default crop/pad target of **(256, 384, 384)** voxels at 1.0×0.5×0.5 mm spacing (cited in `docs/design/audit_progress.md:39`, itself sourced from `contrastive-pretraining/scripts/data.py:171-241,335-337` per `CLAUDE.md`). This is a reasonable starting anchor (INFERRED — it's the only validated geometry target in the pipeline) but **not** a confirmed target for the diffusion model, which may need a different resolution, FOV, or per-modality strategy. Treat all derived VRAM/storage figures as planning inputs to validate empirically once an architecture is chosen, not as commitments.

Dataset scale used for storage arithmetic below (all **VERIFIED**, cited from the existing audit, not re-derived in this session): 705,254 series / 98,334 studies / 83,425 patients across 28 batches; native 8.1 TB / coreg 17.6 TB (absent locally) / atlas 12.3 TB / metadata+reports 415 GB (`docs/design/audit_progress.md:51`); sum of compressed per-series image size for the 633,511 present series ≈ 7.4 TB (`docs/design/audit_progress.md:81`).

---

## 1. Scheduler and job submission system

**VERIFIED**. All NHR@FAU clusters use **Slurm** (`docs/nhr_official_docs/slurm_batch_system.md:5-7`). Submission is via `sbatch` (batch), `salloc` (interactive), `srun` (parallel launch within an allocation) — same doc, lines 11-37. Two clusters require a Slurm command suffix instead of plain commands: TinyGPU (`sbatch.tinygpu`/`squeue.tinygpu`/`salloc.tinygpu`, `clusters_tinygpu.md:34-38`) and TinyFat (`sbatch.tinyfat`/`salloc.tinyfat`/`srun.tinyfat`/`sinfo.tinyfat`, `clusters_tinyfat.md:44`). Helma, Alex, Fritz, and Woody use plain `sbatch`/`salloc`/`srun` — confirmed live on Helma (`sinfo`, `scontrol`, `sacctmgr` all worked unsuffixed this session). Scheduler internals (live, this session): `SchedulerType=sched/backfill`, `SelectType=select/cons_tres` (core+memory consumable-resource scheduling), `PreemptType=preempt/partition_prio`, cluster-wide `PreemptMode=REQUEUE` (though the `preempt` partition itself overrides this to `CANCEL` — see §7).

## 2. Available GPU partitions

**VERIFIED** (docs + live). Only Alex, Helma, and TinyGPU have GPUs; Fritz, Woody, TinyFat, and the decommissioned Meggie are CPU-only.

| Cluster | GPU partitions | Notes |
|---|---|---|
| **Helma** | `h100`, `h200`, `preempt` (spans both GPU pools) | Live `sinfo` also shows a `cpu` partition (312 AMD Zen5c nodes) and its preemptible twin `preempt_cpu` — these exist but are **not named** in `clusters_helma.md`'s job-submission partition table (which lists only `preempt`/`h100`/`h200`); the doc's hardware table does mention a generic "CPU Partition" column (`clusters_helma.md:9,116`). This is a doc/live-state gap this report fills (INFERRED-completion). |
| **Alex** | `a40`, `a100`, `a100mig` (debug/MIG fractions) | `clusters_alex.md:90-113` |
| **TinyGPU** | `work` (RTX 2080 Ti / RTX 3080), `rtx3080`, `a100`, `v100` | `clusters_tinygpu.md:42-49`; Tier3-only, not NHR-project accounts |
| Fritz | none (CPU: `singlenode`, `multinode`, `spr1tb`, `spr2tb`, `big`) | `clusters_fritz.md:69-77` |
| Woody | none (CPU: `work`) | `clusters_woody.md:11-15` |
| TinyFat | none (CPU: `work`, `broadwell256`, `broadwell512`, `long256`) | `clusters_tinyfat.md:46-53` |
| Meggie | decommissioned, migrate to Woody/Fritz | `clusters_meggie.md:5-7` |

Given the project's data already lives on Helma-exclusive `hnvme` workspaces (`DATA_PATH`/`SHARDS_PATH` per prior audit), **Helma is the natural home cluster** for this workload; Alex would require either physically moving data to `anvme`/`$WORK` or cross-cluster access (see §17).

**Live account-access finding**: this session's account association shows QOS entitlement `mq_health,preempt` (`sacctmgr show assoc`, this session). Partition ACLs (`scontrol show partition`, this session): `h100` is `AllowAccounts=ALL`; `h200` restricts to an explicit allow-list of ~40 NHR accounts that **includes this account** (redacted here per instructions); `preempt` and `cpu`/`preempt_cpu` are open to all accounts. Net effect: this account can already target `h100`, `h200`, `cpu`, and `preempt` directly — no additional access request appears necessary for GPU partitions on Helma (contrast with Alex/Fritz, which require an explicit Tier3-access request even for FAU accounts, `clusters_alex.md:21-23`, `clusters_fritz.md:21`).

## 3. GPU models and VRAM

**VERIFIED** (docs, cross-checked live).

| Cluster/partition | GPU | VRAM | Architecture / compute capability |
|---|---|---|---|
| Helma `h100` | NVIDIA H100 | 94 GB HBM2e | Hopper, cc 9.0 (`apps_nvidia_gpus.md:78`) |
| Helma `h200` | NVIDIA H200 | 141 GB HBM3e | Hopper, cc 9.0 (`apps_nvidia_gpus.md:79`) |
| Alex `a40` | NVIDIA A40 | 40/48 GB GDDR6 (table says 40, GPU-arch section says 48) | Ampere, cc 8.6 (`clusters_alex.md:13,190`) |
| Alex `a100` | NVIDIA A100 | 40 GB or 80 GB HBM2 | Ampere, cc 8.0 (`clusters_alex.md:14-15,203`) |
| TinyGPU `work`/`rtx3080` | RTX 2080 Ti (11 GB) / RTX 3080 (10 GB) | consumer-grade | `clusters_tinygpu.md:13,15` |
| TinyGPU `v100` | Tesla V100 | 32 GB | cc 7.0 (`apps_nvidia_gpus.md:81`) |
| TinyGPU `a100` | A100 | 40 GB | (`clusters_tinygpu.md:16`) |

For a **high-resolution 3D** model, H200 (141 GB) is the clear first choice on this account — nearly 1.5x the per-GPU memory of H100 and roughly 3-3.5x the debug-partition A100 fractions on Alex. Live-confirmed node hardware (`scontrol show node h11-01`, this session): `RealMemory=750000` (≈732 GiB usable, close to the doc's 768 GB, `clusters_helma.md:11`), 128 cores/node (`CoresPerSocket=64` × 2 sockets), 4× H100 GPUs (`Gres=gpu:h100:4`). GPU interconnect (NVLink vs. PCIe) for Helma's H100/H200 is **not stated** in `clusters_helma.md` (unlike Alex's A100 doc, which explicitly states NVLink 600 GB/s, `clusters_alex.md:208`) — **UNKNOWN**, worth confirming with `nvidia-smi topo -m` inside an actual job before relying on it for tensor/pipeline parallelism.

## 4. GPUs per node

**VERIFIED**. Helma: 4 GPUs/node (both `h100` and `h200`, `clusters_helma.md:11-12`, live-confirmed `Gres=gpu:h100:4`). Alex: 8 GPUs/node for both `a40` and `a100` (`clusters_alex.md:13-15`). TinyGPU varies by sub-cluster: 4/node (`tg06x`, `tg07x`, `tg09x`) or 8/node (`tg08x`) (`clusters_tinygpu.md:13-16`).

## 5. CPU and memory allocation rules

**VERIFIED** (docs + live `scontrol show partition`, this session).

| Partition | CPU cores/GPU | Mem/GPU | Notes |
|---|---|---|---|
| Helma `h100`/`h200`/`preempt` | 32 (`DefCpuPerGPU=32`, live) | ~179 GB effective (`DefMemPerCPU=5600`×32; doc rounds to 192 GB, `clusters_helma.md:37-39`) | `MaxMemPerCPU` == `DefMemPerCPU` (hard cap = default; cannot request more mem/core than the default split) |
| Helma `cpu`/`preempt_cpu` | n/a (CPU-only) | `DefMemPerCPU=1950` MB/core, 384 cores/node | Exclusive-user allocation (`ExclusiveUser=YES`, live) |
| Alex `a40` | 16 | 60 GB | `clusters_alex.md:94` |
| Alex `a100` (40 GB card) | 16 | 120 GB | `clusters_alex.md:95` |
| Alex `a100` (80 GB card) | 16 | 240 GB (`-C a100_80`) | `clusters_alex.md:96`, `faq.md:57-59` |
| Fritz | node-exclusive, 72 or 104 cores/node, 256 GB–2 TB/node | Whole-node granularity | `clusters_fritz.md:71-77` |
| Woody | core granularity, 7.75 GB/core, 1-32 cores/job | `clusters_woody.md:39-41` |
| TinyFat | core granularity on `work` (shared); exclusive on `broadwell*`/`long256` | `clusters_tinyfat.md:46-53` |

## 6. Job time limits

**VERIFIED** (docs + live).

| Partition | Max walltime | Default walltime if unset |
|---|---|---|
| Helma `h100`, `h200`, `cpu` | 24 h (`1-00:00:00`) | **10 min** (`DefaultTime=00:10:00`, live) |
| Helma `preempt`, `preempt_cpu` | 48 h (`2-00:00:00`) | 10 min |
| Alex `a40`/`a100` | 24 h | — |
| Fritz all partitions | 24 h | — |
| Woody | 24 h | — |
| TinyGPU | 24 h (interactive capped at **4 h**, `clusters_tinygpu.md:57`) | — |
| TinyFat `work` | 2 h default, up to 24 h | — |
| TinyFat `long256` | up to 60 h | — |

**Operational gotcha (VERIFIED live)**: every Helma partition silently defaults to a 10-minute walltime if `--time`/`-t` is omitted — every template in §"SLURM templates" below sets `--time` explicitly for this reason.

## 7. QoS and account requirements

**VERIFIED** (docs + live). Helma and Alex require **explicit application/approval**; Fritz requires a Tier3 access request; Woody/TinyGPU/TinyFat are default Tier3 ("Grundversorgung") resources (`clusters_helma.md:16`, `clusters_alex.md:21-23`, `clusters_fritz.md:21`, `clusters_woody.md:21`, `clusters_tinygpu.md:7`, `clusters_tinyfat.md:22`). This session's account already has Helma access (it is executing on `helma2.nhr.fau.de`).

Live QoS detail (`sacctmgr show qos`, this session) — no per-QoS `MaxWall`/`MaxTRESPerUser`/`MaxTRESPerAccount` fields were set for `normal`, `preempt`, or this project's own QoS (`mq_health`); resource limits are therefore governed entirely by the **partition** definitions in §6/§5, not by an additional QoS-level cap (**UNKNOWN** whether a group-level fairshare/TRES cap exists elsewhere in the association tree — not queried further, to keep this a cheap/read-only pass). One QoS-level fact **is** load-bearing: the `preempt` partition sets `PreemptMode=CANCEL` (live) — jobs there are **killed outright**, not automatically requeued, when a higher-priority `h100`/`h200` job needs the node. Any job run under `preempt` must checkpoint proactively; it will not resume on its own.

Cluster-wide QoS names visible live also include several `mq_*` entries tied to other groups/projects (redacted) — this confirms Helma's QoS model is per-project ("mq_" = presumably "managed queue"), consistent with `nhr_application.md`'s statement that NHR project resources are allocated per approved project (GPU-hour bands in §"NHR project types" below).

For scaling beyond the current allocation, `nhr_application.md:20-29` documents the NHR project tiers relevant to a "full challenge training" ask:

| Type | Annual GPU-hours | Review |
|---|---|---|
| Test/Porting (3-4mo) | up to 3,000 | Technical only |
| Starter (≤12mo) | up to 10,000 | Technical only |
| Normal | 6,000–60,000 | Technical + scientific |
| Large | 60,000–180,000 | Technical + 2 external reviews |

## 8. Interactive job support

**VERIFIED**. `salloc` is supported on every cluster (`slurm_batch_system.md:26-33`); run `module purge` first to avoid inherited module-path conflicts (same doc line 33, repeated in `faq.md:49`). Cluster-specific examples: Helma `salloc --gres=gpu:h100:1 --time=1:00:00` (`clusters_helma.md:43-45`); Alex `salloc --gres=gpu:a40:1 --time=1:00:00` (`clusters_alex.md:104-105`), plus MIG-fraction debug options (`--gres=gpu:a100small:1` = 10 GB/4 cores, `--gres=gpu:a100med:1` = 20 GB/4 cores, `clusters_alex.md:108-113`); TinyGPU `salloc.tinygpu --gres=gpu:1 --time=01:00:00`, **capped at 4 h** (`clusters_tinygpu.md:53-57`, the only cluster with a documented interactive-duration ceiling). To attach to and inspect an already-running job (e.g. to watch `nvidia-smi`): `srun --jobid=<jobID> --overlap --pty /bin/bash -l` (`slurm_batch_system.md:104-108`).

## 9. Job-array support and limits

**VERIFIED**. Syntax: `#SBATCH --array=0-15` or `--array=0-19%5` (cap 5 concurrent) (`slurm_batch_system.md:129-135`). Live cluster-wide Slurm config (`scontrol show config`, this session): `MaxArraySize=10000` (max index span per array), `MaxJobCount=50000` (total jobs the scheduler will track cluster-wide). No account/QoS-specific array limit was surfaced (see §7 caveat). This exact mechanism has already been used successfully by this project: `SHARDS_PATH`'s `_work/index.sqlite` + `slurm_logs/` record a prior SLURM array job (one task per WebDataset shard, job IDs recorded in `docs/design/audit_progress.md:61`) — i.e., the array-job pattern for per-study/per-shard parallel processing is independently **VERIFIED as already working** on this account, not merely documented.

## 10. Home, project, work, and scratch storage

**VERIFIED** (docs + live). Four relevant tiers reach Helma compute nodes: `$HOME`, `hnvme` workspaces, and `$TMPDIR` mount on Helma compute nodes; `$WORK` mounts on the frontend but the doc explicitly states **only** `$HOME`, `hnvme`, and `$TMPDIR` mount on Helma *compute* nodes (`clusters_helma.md:27-29`) — `$WORK`/`$HPCVAULT` are frontend/login-node-only on this cluster, a real constraint for job scripts that assume `$WORK` is readable from inside a Helma batch job (**INFERRED risk**: verify with a trivial `ls $WORK` inside an actual allocation before depending on it).

Live env vars this session: `HOME=/home/hpc/<group>/<user>`, `HPCVAULT=/home/vault/<group>/<user>`, `WORK=/home/woody/<group>/<user>` (this account uses the default Tier3 `woody` mount for `$WORK`, not an NHR-project `atuin` mount — consistent with `data_filesystems.md:26-30`'s statement that `/home/atuin` is reserved for NHR projects with a group quota).

| Filesystem | Purpose | Backup/Snapshots | Compute-node reach on Helma |
|---|---|---|---|
| `$HOME` | source, scripts, small important results | YES/YES | yes |
| `$HPCVAULT` | mid/long-term archival | YES/YES (less frequent) | **no** (frontend only) |
| `$WORK` | general working dir, logs | NO/NO | **no** (frontend only, per `clusters_helma.md:27-29`) |
| `hnvme` (workspace) | high-IOPS scratch, NVMe-backed Lustre | NO/NO, **beta, no availability guarantee** (`data_workspaces.md:9`) | yes |
| `$TMPDIR` | node-local job scratch | NO/NO | yes (15 TB NVMe SSD/node, `clusters_helma.md:11,29`) |

## 11. Storage quotas and inode limits

**VERIFIED, live-measured this session** (`shownicerquota.pl`, `quota -s`, `lfs quota -u <user> /hnvme`):

| Filesystem | Soft quota | Hard quota | Files (soft/hard) | Doc-stated value | Reconciliation |
|---|---|---|---|---|---|
| `$HOME` | 100 GB | 200 GB | 500K / 1,000K | 50 GB (`data_filesystems.md:9,18`) | **Discrepancy** — resolved by `faq.md:91`: "Data on `/home/hpc` and `/home/vault` is replicated across two arrays, temporarily doubling quota consumption. All quotas have been increased accordingly" — the live 100/200 GB is the current, already-doubled figure; the static doc page still shows the pre-increase 50 GB. |
| `$HPCVAULT` | 1,000 GB | 2,000 GB | 200K / 400K | 500 GB (`data_filesystems.md:10,22`) | Same doubling explanation applies. |
| `hnvme` (this account, aggregate across its own workspaces) | **0 (no byte quota configured)** | 0 | 81,920 / 102,400 | "Inodes only" (`data_filesystems.md:13`) | Matches docs — no byte cap, but a real inode cap exists and is already live-measured at **59,939 / 81,920 (≈73%) of the soft inode limit**, consumed by 4 pre-existing workspaces (`nvidia-mri-brain`, `aibay_main`, `MR-Rate-raw`, `merlin-project`) before any diffusion-project files are written. |

**This inode ceiling is a genuine near-term risk**, not a theoretical one: any preprocessing/caching approach that writes one file per series (up to 636,218 valid series) will blow through the 81,920/102,400 inode budget almost immediately. Favor few-large-files layouts (the project's own existing WebDataset `.tar` shards, or a single HDF5/`.npz`-per-shard container) over one-file-per-series caching — this is also explicitly the advice in `faq.md:100` ("Large file collections: use containerized formats (HDF5), file-based databases, or archive files with tar compression"). `$WORK` quota was **not** independently queried this session (the `shownicerquota.pl`/`quota -s` output only reports `$HOME`/`$HPCVAULT`/`hnvme`) — **UNKNOWN** exact live number; doc default is "Tier3: 1000 GB" (`data_filesystems.md:11`).

## 12. Scratch retention/deletion policy

**VERIFIED**. `hnvme` workspaces: managed by `ws_allocate <name> [days]`, 1-90 day lifetime, auto-deleted on expiry, extendable via `ws_extend` which **sets** (not adds to) the remaining duration, and restorable for a grace period via `ws_restore` (`data_workspaces.md:16-67`). Live-confirmed on this project's own workspace: `MR-Rate-raw` created 2026-07-27, 88 days 21 hours remaining at query time (`ws_list`, this session) — consistent with a 90-day allocation. **Explicit beta caveat in the docs**: "NVMe Lustre storage is currently in beta mode with no availability guarantees" (`data_workspaces.md:9`) — combined with no backup/snapshots (§10-11), this means the ~20+ TB of already-downloaded MR-RATE derivatives sitting on `hnvme` (`DATA_PATH`, owned by `<data-account>`, and this account's own `SHARDS_PATH`) have **no safety net** beyond the ability to re-download from Hugging Face. `$TMPDIR`: deleted automatically at job end, no retention beyond job runtime (`data_filesystems.md:45-54`). `$FASTTMP` (Fritz/Alex only, not mounted on Helma): high-watermark deletion at ~80% capacity, oldest/largest files first (`data_filesystems.md:34-41`) — not directly relevant unless compute moves off Helma.

**Recommendation**: calendar-remind `ws_extend MR-Rate-raw 90` (and any diffusion-specific workspace) well before the 90-day mark, since `ws_extend` resets to a fixed duration rather than adding time — a late extension after expiry requires the separate `ws_restore` recovery path.

## 13. Recommended storage locations

| Artifact | Recommended location | Rationale |
|---|---|---|
| Source code / training scripts | `$HOME` | Backed up + snapshotted (`data_filesystems.md:18`), small footprint, matches doc's stated purpose ("Source, input, important results") |
| Raw/HF-downloaded datasets (native/coreg/atlas tars) | `hnvme` workspace | Already the case (`DATA_PATH`); high-IOPS Lustre needed for hundreds of thousands of small compressed NIfTI members; accept the beta/no-backup risk since HF hub remains the durable source |
| Preprocessing caches (`.npz`, resampled volumes) | `hnvme` workspace, packed into few-large-files shards (WebDataset `.tar`/HDF5), **not** one file per series (§11 inode limit) | Large volume, high-IOPS read pattern during training |
| Precomputed VAE latents | `hnvme` workspace (persistent across epochs/jobs) or `$TMPDIR` (if regenerated per-job) | Much smaller than raw caches once a VAE exists (§Feasibility (c)) — cheap enough to keep on `hnvme` long-term |
| Downloaded model weights (pretrained VAE/text-encoder) | `$WORK`, with `HF_HOME=$WORK/.cache/huggingface` | Documented pattern (`environment_python.md:81-84`, `faq.md:102-105`, `INDEX.md:107`) to avoid filling the small `$HOME` quota; **caveat**: `$WORK` does not mount on Helma compute nodes (§10) — weights must be staged to `$TMPDIR` or `hnvme` at job start if the training job itself needs to read them, or pre-copied to `hnvme`/`$HOME` |
| Training checkpoints (frequent/rolling) | `$WORK` (or `hnvme` if too large for `$WORK`'s quota) staged via `$TMPDIR` during the job, copied out at job end (`data_staging.md:29-37` pattern) | No backup needed for short-lived rolling checkpoints |
| Training checkpoints (milestone/final) | `$HPCVAULT` | Backed up + snapshotted — appropriate for "important results" that must survive accidental deletion |
| Logs (SLURM stdout/err, W&B offline, TensorBoard) | `$HOME` (small) or `$WORK` (if verbose) | `$HOME` backup protects run provenance; avoid running TensorBoard live on shared nodes (`apps_tensorflow.md:9` security warning) — sync logs and view via SSHFS/local mount instead |

## 14. Internet access restrictions on login and compute nodes

**VERIFIED via documentation only** — per the task's top-level constraint, no live network probe was attempted this session. Multiple docs document a required HTTP(S) proxy for outbound access from jobs/interactive sessions: `export http_proxy=http://proxy.nhr.fau.de:80` / `https_proxy=...` (`faq.md:73-77`, `environment_python.md:75-78`, `apps_spack.md:44-51`), with a note that some tools additionally need the uppercase `HTTP_PROXY`/`HTTPS_PROXY` variants (`faq.md:79`). This is **INFERRED** to mean compute (and likely login) nodes sit behind an outbound proxy/firewall by default rather than having open internet access — the docs describe how to configure around this restriction but do not explicitly state "compute nodes have no direct internet access" as a standalone policy sentence. The consistent guidance to redirect `HF_HOME` to `$WORK` (§13) presumes Hugging Face Hub reachability *through* the proxy is expected to work for interactive/batch package and model downloads. Exact behavior (which hosts/ports are proxied vs. blocked, whether login nodes differ from compute nodes) is **UNKNOWN** without an actual network test, which was intentionally not run.

## 15. Supported containers (Apptainer/Singularity)

**VERIFIED, docs + live**. Apptainer (formerly Singularity) is the standard and only documented container runtime across all NHR@FAU systems (`environment_apptainer.md:7-13`). Live on Helma this session: `apptainer version 1.5.2-1.el9` and a `singularity` compatibility symlink are present as **system binaries directly on `PATH`** — no `module load` needed (unlike the earlier finding that `module avail apptainer`/`singularity` returned nothing; the binary is simply pre-installed system-wide, not delivered as a module). GPU support is automatic on GPU clusters including Helma — device libraries bind-mount without extra flags, `--nv` only needed if issues arise (`environment_apptainer.md:63-74`). Explicit caveat: **"Not suitable for multi-node MPI applications"** (`environment_apptainer.md:13`) — relevant for multi-node distributed training (§18): prefer running PyTorch/NCCL directly from a module-provided or user conda environment for multi-node jobs, reserving Apptainer for single-node/single-process reproducibility, or ensure the container properly bind-mounts host InfiniBand libraries (`rdma-core`, `libibverbs1`, same doc lines 79-80) if containerized multi-node is attempted anyway. Pre-built PyTorch/TensorFlow containers can be pulled from DockerHub or NGC directly (`apps_pytorch.md:63-71`, `apps_tensorflow.md:80-94`), with `$APPTAINER_CACHEDIR` redirected to `$WORK` to protect the `$HOME` quota.

## 16. Module and Conda policies

**VERIFIED, docs + live**. Environment-modules system (Lmod-style: `module avail/list/load/unload/show/whatis/help`, `environment_modules.md:9-21`). Helma has two software-environment branches, `gpu-env/2025` (default) and `cpu-env/2026` (`clusters_helma.md:20`) — confirmed live (`module avail` shows `/apps/modules/helma` tree with both `cpu-env/2026` and `gpu-env/2025<L:S>` present, the latter tagged as the loaded/default one this session). Live module inventory relevant to this project: `python/3.12-conda` (default), plus convenience bundles `python/pytorch2.5.1` and `python/tensorflow2.17`; `cuda/{11.8.0, 12.6.2, 12.8.0, 12.9.0, 13.0.2}`; `cudnn` for both CUDA 11 and 12 (multiple versions); `nccl/2.28.7-gcc11.5.0-cuda`; `openmpi`/`hpcx`/`intelmpi` (CUDA-aware builds); Spack packages via `000-all-spack-pkgs`/`user-spack`. No `apex`/`transformer-engine`/`deepspeed` module was found — anything needing FP8 Transformer Engine or DeepSpeed/ZeRO would have to be pip-installed into a user conda env (feasible, just not centrally provided — **UNKNOWN** whether it installs cleanly against the provided CUDA/cuDNN/NCCL stack without testing).

Conda policy: one-time init (`module add python` + `conda config --add pkgs_dirs $WORK/software/private/conda/pkgs` / `--add envs_dirs $WORK/...`) to keep packages off the small `$HOME` quota (`environment_python.md:9-29`, repeated in `faq.md:83-87`). No `sudo`/root access anywhere; use modules, `user-spack` (installs to `$WORK/USER-SPACK`, `apps_spack.md:23`), or Apptainer containers for anything not centrally provided (`faq.md:69`).

## 17. Recommended data transfer mechanism

**VERIFIED**. `scp`/`rsync` through the dialog server `csnhr.nhr.fau.de` (all default SSH configs tunnel through it; `$HOME`/`$HPCVAULT`/`$WORK` are all mounted there, `data_copying.md:7-11`). `rsync -avz` recommended for resumability, though compression (`-z`) may slow transfers *between* NHR@FAU systems (`data_copying.md:60-61`) — relevant if moving data between Helma's `hnvme` and, say, Alex's `anvme` for a future cross-cluster run. WinSCP is the documented Windows GUI option, tunneled the same way (`data_copying.md:78-93`). This project's own prior practice (custom `rsync` scripts moving data `anvme → hnvme`, per `docs/design/audit_progress.md:48`) already follows this documented mechanism.

## 18. Multi-node distributed-training support

**VERIFIED for Helma, PARTIALLY VERIFIED for Alex, UNKNOWN for this account's entitlement**. `apps_pytorch.md:99-121` gives a full Helma multi-node `torchrun` template (2 nodes × 4 H100, `srun torchrun --rdzv-backend=c10d`) — this is the canonical launch pattern for DDP across nodes. `clusters_helma.md:96-109` repeats an equivalent multi-node `srun` example without a special QoS flag. By contrast, `clusters_alex.md:98,168-182` is explicit that **multi-node jobs on Alex are "available on-demand for NHR projects only" and "require separate account enablement via hpc-support@fau.de"**, gated behind `--qos=a100multi`. Helma's docs do **not** state an equivalent restriction, but this account's only confirmed QoS entries (`mq_health`, `preempt`) showed no multi-node-specific QoS name, and the `preempt` partition itself is capped at `MaxNodes=1` (live) — meaning multi-node runs on Helma would need to go through the `h100`/`h200` partitions directly (`MaxNodes=64` there, live) under a QoS this account already has, or may require the same kind of explicit enablement as Alex. **This is a genuine open question** — recommend confirming with `hpc-support@fau.de` before assuming multi-node H100/H200 access is unrestricted for this project, by analogy with Alex's documented policy.

NCCL guidance: `nccl` backend preferred for GPU-GPU (`apps_pytorch.md:95`); debug via `NCCL_DEBUG=INFO/WARN` (`apps_pytorch.md:127-129`, `faq.md:121`); InfiniBand HCA pinning example (`NCCL_IB_HCA="=mlx5_0:1,mlx5_3:1"`) is documented specifically for **Alex's A100 nodes** (`apps_pytorch.md:132-134`, `faq.md:123-125`) — the equivalent device names for Helma's H100/H200 nodes are **UNKNOWN** from these docs and would need an in-job `ibstat`/`ibv_devinfo` check. Apptainer's stated unsuitability for multi-node MPI (§15) is the other practical constraint on the containerization approach for this axis.

## 19. SLURM environment variables and launch conventions

**VERIFIED**. Standard variables: `$SLURM_JOB_ID`, `$SLURM_SUBMIT_DIR`, `$SLURM_JOB_NODELIST`, `$SLURM_JOB_NUM_NODES`, `$SLURM_CPUS_PER_TASK`, `$SLURM_ARRAY_JOB_ID`, `$SLURM_ARRAY_TASK_ID`, `$SLURM_GPUS_ON_NODE` (`slurm_batch_system.md:110-121`). Since Slurm 22.05, `srun` requires an explicit `--cpus-per-task` or `SRUN_CPUS_PER_TASK` env var (same doc line 58, repeated as a warning in `clusters_fritz.md:139`, `clusters_alex.md:166`). **Canonical NHR@FAU job-script idiom**, identical across every cluster's sample scripts in these docs: `#!/bin/bash -l` shebang (login shell, required for `module` to work, `environment_modules.md:49`) + `#SBATCH --export=NONE` + `unset SLURM_EXPORT_ENV` immediately after the `#SBATCH` block, to get a clean, reproducible environment while still letting `srun` propagate the script's own exports (`slurm_batch_system.md:69-76`, repeated verbatim in every `clusters_*.md` batch example read for this report). All templates in this report follow that idiom.

## 20. Restrictions relevant to medical data

**NOT FOUND / UNKNOWN**. None of the 29 scraped NHR@FAU docs contain HIPAA/GDPR/medical-data-specific governance language, a "sensitive data" enclave policy, or a special-handling requirement for patient-derived datasets — this was checked across `account.md`, `data_filesystems.md`, `data_workspaces.md`, `faq.md`, `nhr_application.md`, and the rest. The only privacy/compliance-adjacent item found is generic: an export-control contact (`exportkontrolle@fau.de`, `faq.md:9`) unrelated to medical data specifically. This is a **genuine documentation gap**, not a confirmation that no such policy exists — recommend the user separately confirm data-processing-agreement/IRB-compliance requirements for hosting MR-RATE (CC BY-NC-SA licensed, Istanbul Medipol University IRB scope per the existing audit, `docs/design/recommended_next_steps.md:52`) on NHR@FAU storage, likely via FAU's data protection office, independent of what these HPC-operations docs cover. Operationally (**INFERRED**, not stated anywhere): the combination of "no backup on `$WORK`/`hnvme`" + "`hnvme` is beta/no-SLA" (§10-12) means this project is already relying on the *upstream* HF-hosted, de-identified dataset as the durable copy-of-record, not the HPC storage — consistent with, but not a substitute for, whatever formal data-governance terms apply.

---

## Feasibility assessment

| Workload | Feasibility | Basis |
|---|---|---|
| Preprocessing MR-RATE NIfTI volumes (reorient/resample/crop/normalize) | **HIGH** | CPU-bound, embarrassingly parallel per-series/per-study. Already empirically timed at small scale: the prior audit's Phase-3 pilot parsed 37 files in 195.5s (≈5.3 s/file) via `nibabel` (`docs/design/audit_progress.md:110,114`). Full preprocessing (resample+crop+normalize+write, not just parse) will be slower per file — **ASSUMED** 2-5x that pilot rate — but trivially absorbed by a SLURM array job on Helma's 312-node `cpu` partition (§2/§9). |
| Building `.npz` caches | **HIGH**, contingent on inode planning | Same array-job mechanism as above. Storage is the real constraint, not compute — see arithmetic below; **must** use few-large-files shard layout (§11/§13), not one `.npz` per series. |
| Precomputing VAE latents | **MEDIUM** — gated on a trained VAE existing first | Needs GPU (VAE forward pass) but is otherwise embarrassingly parallel across an array job on `h100`/`h200`. Chicken-and-egg with training the VAE itself (see Profile roadmap below). Storage footprint drops by roughly the VAE's spatial-compression factor cubed (ASSUMED example below). |
| VAE reconstruction tests | **HIGH** | Small-scale, single-GPU, short iterations — ideal for `salloc`/`preempt`. |
| Single-GPU inference | **HIGH** | Trivial once a model exists; single H100/H200 comfortably fits any plausible inference batch. |
| Multi-GPU inference (single node) | **HIGH** | Straightforward data-parallel replication across the 4 GPUs/node. |
| Multi-GPU inference (multi-node) | **MEDIUM** | Same open multi-node-entitlement question as §18. |
| Fine-tuning the diffusion U-Net | **MEDIUM-HIGH** | Feasible on one Helma node (4× H100/H200) with mixed precision + activation checkpointing, **ASSUMED** batch sizes below since no architecture is fixed yet. |
| Adding/training text-conditioning (cross-attention) layers | **MEDIUM-HIGH** | Same tier as U-Net fine-tuning, modest extra VRAM for text encoder + cross-attention KV. |
| Full training from scratch (VAE + diffusion, full dataset) | **MEDIUM**, gated on project-tier and open questions | The heaviest tier — see Profile 3. Gated on (1) an NHR project allocation sized for it (§7 table: Normal/Large tier, 6,000-180,000 GPU-hours/yr, vs. this account's current unspecified-but-likely-smaller allocation), (2) resolving multi-node entitlement (§18), and (3) the still-open modeling questions already flagged in `docs/design/recommended_next_steps.md:45-56` (unit of training, geometry strategy, report-to-series attribution) — those are modeling decisions, not something this HPC-focused audit can resolve. |

### Storage arithmetic behind the table above (ASSUMED anchor resolution — see Grounding)

Anchor: single-channel float32 volume at the contrastive loader's default (256, 384, 384) target ⇒ 256×384×384×4 bytes ≈ **151 MB/volume**.

- Per-study cache, one volume/study (98,334 studies): **≈14.8 TB** (float32) / ≈7.4 TB (float16).
- Per-series cache, all present series (636,218): **≈96 TB** (float32) / ≈48 TB (float16) — impractical; reinforces the existing audit's recommendation (`docs/design/recommended_next_steps.md:11,31`) to curriculum-restrict to one modality/plane stratum first (e.g. T1w axial, ~231,800 series).
- VAE latents, **ASSUMED** 4x spatial downsampling/axis (64x volume reduction) at 4 latent channels vs. 1 input channel (net ≈6.8x size reduction per volume ⇒ ≈9.4 MB/volume float32): per-study ≈**0.9 TB**, per-series (all 636,218) ≈**6 TB**. This ~15x storage reduction vs. raw `.npz` caching is the main practical argument for the "cache latents, not raw volumes, once the VAE is validated" sequencing used in the profiles below.

These multipliers are illustrative planning inputs (**ASSUMED**), not a commitment to any specific VAE design — recompute once an architecture is fixed.

---

## Three compute profiles

### Profile 1 — Minimal development experiment

Goal: prove out training-unit design and a tiny VAE+diffusion pilot on ~500-1,000 studies, one modality/plane (per `docs/design/recommended_next_steps.md:7-15`).

- **GPU type/count**: 1× H100 (94 GB) or H200 (141 GB), single GPU. Use `salloc`/short `sbatch` under the `preempt` QoS (already available to this account) for cheap, fast iteration.
- **Expected VRAM pressure**: LOW-MEDIUM — small VAE + small UNet at reduced resolution (e.g. sub-128³ crops), comfortably under 40 GB even with headroom (**ASSUMED**, no fixed architecture).
- **Batch size**: 2-8 volumes (**ASSUMED**).
- **Gradient accumulation**: 1-4 steps, to emulate a larger effective batch given the small pilot slice.
- **Mixed precision**: bf16 (native Tensor Core support on Hopper; safer numerics than fp16, standard practice — **ASSUMED** as the default choice, not doc-specified).
- **Activation checkpointing**: off/optional at this scale.
- **Data-loader workers**: 4-8 (single-GPU `salloc` on Helma grants `DefCpuPerGPU=32` cores by default, per §5 — ample headroom).
- **Expected storage**: a few hundred GB (500-1,000 study `.npz`/latent subset) — fits in a small dedicated `hnvme` workspace or even `$WORK`.
- **Checkpoint frequency**: every 30-60 min given `preempt`'s `CANCEL`-on-preemption behavior (§7) — cannot rely on auto-requeue.
- **Job duration strategy**: short interactive sessions (1-4h) or short `preempt`-QoS batch jobs with frequent checkpointing; use the documented chain-job self-resubmission pattern (`slurm_batch_system.md:137-144`) if iteration needs to span a preemption.

### Profile 2 — Single-node pilot

Goal: production-scale preprocessing validation + a meaningful-stratum VAE/diffusion pilot (e.g. full T1w-axial stratum, ~231,800 series, or a curated curriculum slice), per `docs/design/recommended_next_steps.md:17-24`.

- **GPU type/count**: 1 full Helma node, 4× H100 or 4× H200, single-node DDP via `torchrun`.
- **Expected VRAM pressure**: MEDIUM-HIGH; H200's 141 GB gives materially more headroom for resolution/batch than H100's 94 GB (**ASSUMED** magnitude, architecture-dependent).
- **Batch size**: **ASSUMED** 4-16/GPU, ⇒ ~16-64 global batch across 4 GPUs, resolution-dependent.
- **Gradient accumulation**: 2-8 steps to reach a target effective batch (e.g. 128-256) without exceeding VRAM.
- **Mixed precision**: bf16, plus `torch.compile(backend="inductor")` for throughput (documented and supported, `apps_pytorch.md:73-84`).
- **Activation checkpointing**: recommended **ON** for the diffusion U-Net's deep 3D conv stack, to buy back batch size/resolution headroom.
- **Data-loader workers**: up to 128 cores/node ÷ 4 processes ⇒ ~16-24 workers/GPU-process, leaving cores for the main process + NCCL threads.
- **Expected storage**: prefer caching **VAE latents only** at this stage (≈0.9-6 TB per the arithmetic above), not raw `.npz` volumes (14.8-96 TB) — stage per-epoch shards to node-local `$TMPDIR` (15 TB/node) to reduce load on the shared, beta, no-SLA `hnvme` filesystem (§10/§12), mirroring the pattern this project already uses in `SHARDS_PATH`'s WebDataset shards.
- **Checkpoint frequency**: every 1-2h, given the 24h `MaxTime` on `h100`/`h200` (§6, live-verified) — the job **must** self-checkpoint and chain-resubmit to train longer than 24h in one Slurm job.
- **Job duration strategy**: chain-job pattern (`slurm_batch_system.md:137-144`) with resume every ~20-22h to leave safety margin under the 24h cap; run under the `h100`/`h200` partition directly (not `preempt`) for a guaranteed non-preemptible single-node run — this account already has explicit `h200` access (§2).

### Profile 3 — Full challenge training

Goal: full-scale fine-tuning or from-scratch training across the curated/curriculum dataset.

- **GPU type/count**: multi-node H100/H200 (e.g. 4-16 nodes × 4 GPUs = 16-64 GPUs), **contingent on resolving the open multi-node-entitlement question (§18)** and likely requiring an upgraded NHR project tier (Normal/Large, §7 table) given the GPU-hour scale this implies.
- **Expected VRAM pressure**: HIGH — largest resolution/model configuration under consideration; prefer H200 (141 GB) over H100 (94 GB) for headroom.
- **Batch size**: small per-GPU (**ASSUMED** 1-4) at high resolution, relying on data-parallel replica count + gradient accumulation for effective batch size.
- **Gradient accumulation**: as needed to reach a target effective batch (e.g. 256-1,024); less accumulation needed than Profile 2 given more data-parallel replicas.
- **Mixed precision**: bf16 by default; FP8 (Hopper Transformer Engine) is hardware-capable but **not confirmed available** in the provided module stack (no `transformer-engine`/`apex` module found, §16) — would require a self-managed install, UNKNOWN compatibility without testing.
- **Activation checkpointing**: **ON**, likely combined with FSDP-style parameter/optimizer sharding (PyTorch-native `FSDP`, no DeepSpeed module found — self-managed if needed, §16).
- **Data-loader workers**: as Profile 2, scaled per node; additionally plan for aggregate `hnvme` I/O contention across many concurrent nodes reading the same beta/no-SLA filesystem (§12) — stage per-epoch/per-node shard subsets to `$TMPDIR` rather than reading directly from `hnvme` on every step.
- **Expected storage**: up to the full ~38 TB across all present derivatives (native+atlas; coreg would add another 17.6 TB if downloaded) plus checkpoint storage — recommend a rolling-window checkpoint policy (keep N most recent + periodic milestones) rather than every-step retention, and revisit the inode budget (§11) for whatever shard format is chosen at this scale.
- **Checkpoint frequency**: milestone-based (every few hours / N-thousand steps) plus a distinct short-interval "resume" checkpoint, given the 24h job cap and (if `preempt` is used for any part of this) `CANCEL`-mode preemption risk.
- **Job duration strategy**: long chain-of-jobs across the 24h/node cap, likely spanning days-to-weeks of wall-clock; multi-node `torchrun`/`c10d` rendezvous re-established fresh at each chain link (checkpoint/resume, not live migration) — see `apps_pytorch.md:99-121` template, extended below.

---

## Draft SLURM templates (NOT submitted — for review only)

All templates follow the canonical idiom from §19 (`#!/bin/bash -l`, `--export=NONE` + `unset SLURM_EXPORT_ENV`) and set `--time` explicitly (§6 gotcha). Adjust partition/account/paths before use; none of these were run.

### T1 — Interactive single-GPU dev session (Profile 1)

```bash
# Interactive salloc — not a script, run directly on the Helma frontend
module purge
salloc --partition=preempt --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=02:00:00
```

### T2 — Single-GPU VAE/diffusion pilot batch job (Profile 1)

```bash
#!/bin/bash -l
#SBATCH --job-name=mrrate-diffusion-pilot
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out

unset SLURM_EXPORT_ENV
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load python cuda/12.6.2
conda activate mrrate-diffusion

# preempt QoS can CANCEL this job without requeue — checkpoint frequently inside train.py
python3 train_vae_diffusion_pilot.py \
    --manifest "$WORK/mrrate_pilot_manifest.parquet" \
    --data-dir "$(ws_find MR-Rate-raw)" \
    --output-dir "$WORK/pilot_runs/$SLURM_JOB_ID" \
    --checkpoint-every-min 30 \
    --batch-size 4 --grad-accum 2 --precision bf16
```

### T3 — Single-node 4-GPU pilot with chain-job resubmission (Profile 2)

```bash
#!/bin/bash -l
#SBATCH --job-name=mrrate-diffusion-node
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out

unset SLURM_EXPORT_ENV
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python cuda/12.6.2
conda activate mrrate-diffusion

# Stage this epoch's latent shard subset onto node-local NVMe
mkdir -p "$TMPDIR/latents"
cp "$(ws_find latents-cache)"/shard_*.tar "$TMPDIR/latents/"

MASTER_ADDR="$(hostname)"
MASTER_PORT=29400

srun --cpu-bind=verbose torchrun \
     --nnodes=1 --nproc-per-node=$SLURM_GPUS_ON_NODE \
     --rdzv-id="$SLURM_JOB_ID" --rdzv-backend=c10d \
     --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
     train_diffusion.py \
       --latents-dir "$TMPDIR/latents" \
       --resume-from "$WORK/checkpoints/latest.pt" \
       --checkpoint-dir "$WORK/checkpoints" \
       --batch-size 8 --grad-accum 4 --precision bf16 \
       --activation-checkpointing --max-hours 19.5

# Chain: resubmit self if training isn't finished and < 19.5h were used
if [ -f "$WORK/checkpoints/CONTINUE" ]; then
    sbatch "$0"
fi
```

### T4 — Preprocessing / cache-building array job (CPU partition)

```bash
#!/bin/bash -l
#SBATCH --job-name=mrrate-preprocess
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-499%100
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%A_%a.out

unset SLURM_EXPORT_ENV
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load python
conda activate mrrate-preprocess

python3 preprocess_shard.py \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --num-shards 500 \
    --manifest "$WORK/series_manifest.parquet" \
    --data-dir "$(ws_find MR-Rate-raw)" \
    --output-dir "$(ws_find preproc-cache)" \
    --output-format webdataset-tar   # few-large-files, not one .npz/series (inode budget, §11)
```

### T5 — Multi-node full-scale training (Profile 3, draft only — confirm multi-node entitlement first, §18)

```bash
#!/bin/bash -l
#SBATCH --job-name=mrrate-diffusion-fullscale
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out

unset SLURM_EXPORT_ENV
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN

module load python cuda/12.6.2 nccl/2.28.7-gcc11.5.0-cuda
conda activate mrrate-diffusion

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT=29400

srun --cpu-bind=verbose torchrun \
     --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_ON_NODE \
     --rdzv-id="$SLURM_JOB_ID" --rdzv-backend=c10d \
     --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
     train_diffusion.py \
       --latents-dir "$(ws_find latents-cache)" \
       --resume-from "$HPCVAULT/checkpoints/latest.pt" \
       --checkpoint-dir "$WORK/checkpoints" \
       --milestone-dir "$HPCVAULT/checkpoints" \
       --fsdp --activation-checkpointing --precision bf16 \
       --max-hours 19.5

if [ -f "$WORK/checkpoints/CONTINUE" ]; then
    sbatch "$0"
fi
```

---

## Open questions this report cannot resolve

1. Whether multi-node GPU jobs on Helma require the same explicit `hpc-support@fau.de` enablement that Alex documents for its `a100multi` QoS (§18) — recommend confirming before Profile 2/3 planning assumes it's unrestricted.
2. This account's actual NHR project tier / annual GPU-hour allocation (not surfaced by the read-only `sacctmgr` queries run here) — needed to size how much of Profile 3 is affordable under the current project vs. requiring a tier upgrade (§7).
3. Whether Helma's H100/H200 GPUs are NVLink- or PCIe-connected within a node (§3) — affects tensor/pipeline-parallelism design for the largest model configurations.
4. `$WORK`'s live quota for this account (only `$HOME`/`$HPCVAULT`/`hnvme` were reported by `shownicerquota.pl`, §11).
5. Any FAU/NHR medical-data-handling policy not captured in these 29 operations-focused docs (§20) — a separate compliance conversation, not an HPC-capacity one.
