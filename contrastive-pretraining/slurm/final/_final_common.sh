# Every hyperparameter of the four final runs, in one file.
#
# The four run_*.sh scripts set R2V_CONFIG and nothing else, so "the only difference between the
# four jobs is the conditioning mechanism" is enforced by construction rather than by four copies
# staying in sync. Change a number here and all four change together; that is the point.
#
# Usage (from contrastive-pretraining/):
#   R2V_LR=3e-4 slurm/final/run_A_cxr_bert_cls.sh
#   SMOKE=1 slurm/final/run_B_cxr_bert_tokens.sh      # 25-min memory check, no LR needed
#
# `--export=NONE` in the sbatch means the environment does NOT reach the job on its own; these
# scripts pass everything through `--export=ALL,...` explicitly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${WORKSPACE:-/hnvme/workspace/y100dc19-nvidia-mri-brain}"

# ---------------------------------------------------------------- learning rate
#
# **1e-3**, from the five-arm sweep (jobs 711945, 713011-14; configuration B, 600 optimizer steps at
# effective batch 256, 0 skipped steps in every arm). Best over each run:
#
#     lr      best ssim   best fvd   best fid      trajectory
#     1e-3      0.3454      27.99      30.49       peaks at step 300
#     3e-3      0.3370      26.14      27.77       declines from step 150 -- peaked before measurement
#     1e-5      0.3354      31.28      35.99
#     1e-4      0.3278      36.58      31.75
#     3e-4      0.3225      33.33      30.84
#
# The robust finding is the *pair*: 1e-3 and 3e-3 beat the low-LR trio on every metric, and the
# training loss separates nothing (0.9204-0.9212 across a 300x LR range), exactly as expected for a
# random-timestep diffusion objective. Between the two the result is split -- 1e-3 wins SSIM, 3e-3
# wins FVD/FID by ~7% -- and the margin is inside the within-arm wobble either way, so this is a
# judgement call, not a measurement:
#
#   * SSIM is the least ambiguous metric at this N (paired, per-case, fixed seed), and 1e-3 wins it.
#   * 3e-3's SSIM declines monotonically from its *first* validation, i.e. it peaked before step 150.
#     That is a stability warning, and it matters more here than in the sweep: the final run is ~7.5x
#     longer, so PolynomialLR holds the rate high for far more steps than 600.
#   * Both sweeps now agree the optimum is at the high end; nothing below 1e-3 distinguished itself,
#     and 1e-5 (NVIDIA's default, for training the whole 188M-parameter UNet) is not right for an
#     8M adapter starting from a zero-init projection.
#
# **Caveat worth knowing when reading the final curves.** SSIM falls while report-sensitivity rises
# (1e-3: ssim 0.3454 -> 0.3214 while ssim_advantage 0.0004 -> 0.0183). The model appears to start by
# emitting generic average anatomy -- which scores well on a paired metric -- and to use the report
# progressively more. For a report-to-volume task that trade is the desired direction, so do not read
# a falling SSIM as failure on its own. Sensitivity is measured on 8 cases and is noisy.
SMOKE="${SMOKE:-0}"
R2V_LR="${R2V_LR:-1e-3}"

# ---------------------------------------------------------------- scale
# 2 nodes x 4 H200. R2V_NGPU is the TOTAL rank count and the sbatch cross-checks it against
# torchrun's WORLD_SIZE, so it must equal nodes x GPUs-per-node.
#
# Overridable so a SMOKE run can ask for one node: a 2-node 25-minute job backfills into far fewer
# gaps than a 1-node one (Slurm projected 06:32 next morning for the 2-node smokes), and the
# question a smoke actually answers -- does batch 8 fit, is the conditioning wired right -- is
# per-GPU and does not need the second node. Leave it at 2 for the real runs.
NODES="${NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NGPU=$(( NODES * GPUS_PER_NODE ))

# Effective batch = batch x accum x world_size = 8 x 4 x 8 = 256.
# **The sbatch does not rescale accum by node count**, so if you change NODES you must change
# GRAD_ACCUM too or the effective batch silently moves and the swept LR no longer applies.
#
# batch 8, not the configs' default 4: at batch 4 the real per-bucket geometries leave the GPU idle
# and throughput halves (9.98 vol/s measured in job 711945 vs 20.2 in job 710049), which is 16 h per
# epoch instead of 7.9. Measured peaks: 35.6 GB at batch 4 (job 711503), 85-87 GB at batch 8 on the
# 256^3 worst-case bucket for configurations A and D. **Batch 8 is not measured for B and C**, whose
# cross-attention carries up to 512 keys instead of 1 -- run `SMOKE=1` once per configuration before
# committing a long job. If it OOMs, set BATCH_SIZE=4 and GRAD_ACCUM=8 to hold the effective batch.
BATCH_SIZE=8
GRAD_ACCUM=4

# ---------------------------------------------------------------- budget
# Whole training split (575,187 samples), twice. No --max-steps cap.
EPOCHS=2
# ~11 h at 2 nodes x batch 8 (10 h of steps + ~1.3 h of validation), so this is >2x margin.
#
# 24 h is the **hard ceiling** on h200 (`sinfo`: TIMELIMIT 1-00:00:00); the mq_health QOS adds no
# further MaxWall. Asking for more than the estimate is not free -- a 2-node 24 h job backfills into
# far fewer gaps than a 2-node 16 h one, and queue wait, not walltime, is what is currently costing
# days. Lower this to ~16 h if the jobs sit pending.
#
# Past 24 h there is only the `preempt` partition (48 h, and the account has the QOS), where a job
# can be killed at any moment by higher-priority work. That is survivable only with checkpoint
# resume, and `--resume` is NOT plumbed into 11_train_conditioning.sbatch today -- so treat 24 h as
# the real limit until it is.
WALLTIME=24:00:00

# ---------------------------------------------------------------- validation
#
# **Budgeted from measurement, not from the earlier estimate, which was ~3x too optimistic.**
# `val/seconds` recorded in the sweep's checkpoints: **572-765 s per pass at N=64** (jobs 711945,
# 713012). The recorded components -- generate 26-28 s, features 9-73 s, SSIM 70-239 s, sensitivity
# 192-282 s -- account for only ~20% of that; the rest is unattributed, and it is *not* the Frechet
# `sqrtm` (timed locally at 1.3 s for 512-d and 25 s for 2048-d across three planes), so it is
# almost certainly re-reading and preprocessing the validation volumes on every pass. That means the
# cost grows with N rather than being fixed, so N=128 is budgeted at ~15-20 min per pass.
#
# Hence 600, not 400: ~7 points across a 2-epoch (~4,500 optimizer step) run is still plenty to see
# a val curve turn over, and 7 x ~17 min is ~2 h against ~11 h of training rather than ~3 h.
VALIDATE_EVERY=600
VAL_QUICK=128
# One full pass late in training, not one per epoch: at N=512 a pass is expensive and the calibrated
# distribution numbers come from `cli.evaluate` over a real cohort anyway.
VALIDATE_FULL_EVERY=4000
VAL_FULL=512
# The condition-sensitivity swap costs a second generation per case and measured 192-282 s per pass,
# so it gets its own rarer schedule. It is the diagnostic that says whether the report is used at all.
SENSITIVITY_EVERY=1800
SENSITIVITY_N=8

# Real-vs-real noise floor from cli.validation_reference (job 713833).
#
# **n256, not n64, and the factor of two is the point**: `real_vs_real_baseline` splits its sample in
# half, so an n=256 reference describes a 128-per-side floor -- which is exactly what VAL_QUICK=128
# compares. Measured FVD 10.22 / 2.5D FID 7.83 there, against 30.01 / 21.14 at n=64 (32 per side):
# the floor falls by ~3x for a 4x larger sample, so a reference taken at the wrong N understates the
# model's true margin badly. ssim_autoencoder is 0.915 +/- 0.036 (n=256), confirming 0.910 at n=64.
VALREF_JSON="$WORKSPACE/cache/r2v/validation_reference_n256_seed0.json"

# ---------------------------------------------------------------- reporting
# The order-agnostic spec: [MODALITY]/[PLANE]/[SPACING] prefix, then findings/impression in one of
# two orders sampled per training sample (textenc/formats.py ORDER_AGNOSTIC_META_SPEC). Training on
# one fixed order teaches the model that order, and nothing at submission time would detect that the
# challenge's reports use the other.
#
# Configuration D ignores this and must: it encodes findings and impression with three separate
# tokenizers and never joins them into one string, so there is no section order to be robust to --
# and, unavoidably, nowhere to put the metadata prefix. That is the one respect in which the four
# jobs are not identical apart from the conditioning, and it is structural, not a choice.
REPORT_FORMAT="findings_impression_meta,impression_findings_meta"

SEED=0
NUM_WORKERS=8
WANDB_MODE=online
WANDB_PROJECT=mr-rate-r2v

# h22-05 ran 5.2x slower than an identical allocation on 2026-08-07 (job 711946: 132 s/optimizer
# step against 25.6 s on h22-18) and never recovered. Excluded until someone shows it is healthy.
EXCLUDE_NODES=h22-05

# ---------------------------------------------------------------- submit
submit_final_run() {
    local config="$1" tag="$2"

    local exports="ALL,R2V_CONFIG=${config},R2V_NGPU=${NGPU}"
    exports+=",R2V_BATCH_SIZE=${BATCH_SIZE},R2V_GRAD_ACCUM=${GRAD_ACCUM}"
    exports+=",R2V_SEED=${SEED},R2V_NUM_WORKERS=${NUM_WORKERS},R2V_LOG_EVERY=25"
    # **Not passed inside --export.** Its value contains a comma and --export is comma-separated, so
    # listing it here splits it into `R2V_REPORT_FORMAT=findings_impression_meta` plus a junk entry
    # -- which trains on one fixed section order while looking like it did the right thing. Exported
    # into the environment instead and carried by the `ALL` above.
    # D's own config file sets R2V_REPORT_FORMAT empty; overriding it would break it.
    if [[ "$config" != "D" ]]; then
        export R2V_REPORT_FORMAT="$REPORT_FORMAT"
    else
        unset R2V_REPORT_FORMAT
    fi

    local walltime="$WALLTIME"
    if [[ "$SMOKE" == "1" ]]; then
        # Eight optimizer steps, no validation, no W&B: this exists to answer "does batch 8 fit and
        # how fast is a step", and nothing else. Peak memory is in the job's GPU utilisation block.
        tag="smoke_${tag}"
        walltime=00:25:00
        exports+=",R2V_MAX_STEPS=$(( 8 * GRAD_ACCUM )),R2V_EPOCHS=1,R2V_LR=1e-4,R2V_WANDB=disabled"
    else
        exports+=",R2V_MAX_STEPS=0,R2V_EPOCHS=${EPOCHS},R2V_LR=${R2V_LR}"
        exports+=",R2V_VALIDATE_EVERY=${VALIDATE_EVERY},R2V_VAL_QUICK=${VAL_QUICK}"
        exports+=",R2V_VALIDATE_FULL_EVERY=${VALIDATE_FULL_EVERY},R2V_VAL_FULL=${VAL_FULL}"
        exports+=",R2V_SENSITIVITY_EVERY=${SENSITIVITY_EVERY},R2V_SENSITIVITY_N=${SENSITIVITY_N}"
        exports+=",R2V_WANDB=${WANDB_MODE},R2V_WANDB_PROJECT=${WANDB_PROJECT}"
        exports+=",R2V_SAVE_EVERY=${VALIDATE_EVERY},R2V_KEEP_LAST_N=3"
        if [[ -f "$VALREF_JSON" ]]; then
            exports+=",R2V_VALIDATION_REFERENCE=${VALREF_JSON}"
        else
            echo "note: $VALREF_JSON does not exist yet (job 713010), so the curves will have no" >&2
            echo "      real-vs-real reference line. Not fatal; rerun 12_validation_reference.sbatch." >&2
        fi
        # W&B online needs credentials inside the container; the sbatch fails fast without them.
        if [[ -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
            echo "W&B online needs WANDB_API_KEY exported, or 'wandb login' run once." >&2
            exit 1
        fi
    fi
    exports+=",R2V_TAG=${tag}"

    set -x
    sbatch --job-name="$tag" \
           --nodes="$NODES" --ntasks-per-node=1 \
           --gres="gpu:h200:${GPUS_PER_NODE}" --cpus-per-task=32 \
           --time="$walltime" --exclude="$EXCLUDE_NODES" \
           --export="$exports" \
           "$REPO_ROOT/slurm/11_train_conditioning.sbatch"
}
