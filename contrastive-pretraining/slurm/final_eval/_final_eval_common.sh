# Every parameter of the four final EVALUATIONS, in one file.
#
# The mirror of `slurm/final/_final_common.sh`, and for the same reason: the four `run_*.sh`
# scripts set R2V_CONFIG and nothing else, so "the only difference between the four evaluations is
# the conditioning mechanism" is enforced by construction rather than by four command lines staying
# in sync. Same split, same case list, same seed, same guidance, same sampler steps, same
# checkpoint rule.
#
# Usage (from contrastive-pretraining/):
#   slurm/final_eval/run_A_cxr_bert_cls.sh
#   SMOKE=1 slurm/final_eval/run_D_report2ct_style.sh          # 8 cases/bucket end to end, ~20 min
#   N_PER_BUCKET=200 slurm/final_eval/run_B_cxr_bert_tokens.sh # the old cohort scale
#   FULL_SPLIT=1 slurm/final_eval/run_C_radbert_tokens.sh      # every test-split case
#
# **One job per arm, not two.** There is no predict stage any more: `cli.evaluate` builds the
# dataset, generates and scores in a single pass, so there is no intermediate prediction set to
# hand off, to go stale, or to be scored against the wrong cohort. `CLASSIFIER_JOB=<id>` still
# holds the evaluation until the blinded classifier exists, with **afterany**: a missing classifier
# makes one metric group unavailable-with-a-reason, which is not a reason to sit in the queue.
#
# `--export=NONE` in 05_evaluate.sbatch means the environment does NOT reach the job on its own;
# everything below is passed through `--export=ALL,...` explicitly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${WORKSPACE:-/hnvme/workspace/y100dc19-nvidia-mri-brain}"
RUNS="$WORKSPACE/runs"

# ---------------------------------------------------------------- what is evaluated
#
# **The test split, deterministically ordered, with no random sampling anywhere.** Case selection
# orders candidates by `(study_uid, series_id)` within each (modality, plane) bucket and
# round-robins across buckets, so the list is reproducible without a seed and every prefix is
# bucket-balanced. `--seed` fixes only the sampler noise, per case, via `stable_seed(seed, case_id)`.
#
# Three scales, and they are NESTED -- a smoke run is the first 8 per bucket of the 200 run, which
# is the first 200 per bucket of the full run. So a cheap result is an early look at the same
# population, never a different draw.
#
#   SMOKE=1        8/bucket      80 cases     ~10 min   wiring check; no metric means anything
#   (default)      200/bucket    2,000 cases  ~4 h      the scale every earlier result was produced at
#   FULL_SPLIT=1   entire split  34,453 cases ~64 h     what CTFlow does on CT-RATE's validation set
#
# 200/bucket is the default because it is what the four arms have always been compared at, and
# because per-bucket FID at N=200 is already stable enough to rank them. Quote the full-split run
# for the paper if the GPU budget allows -- and note it needs `--time` raised well past 8 h.
N_PER_BUCKET="${N_PER_BUCKET:-200}"
SPLIT="${SPLIT:-test}"

# ---------------------------------------------------------------- which checkpoint
#
# **`adapter_last.pt` where it exists, `adapter_step0004200.pt` for C.**
#
# A, B and D COMPLETED at optimizer step 4,493 (jobs 714497/714498/714500). C (job 714499) was
# killed by a host-RAM OOM during its N=512 validation pass at step 4200 -- after that pass and its
# checkpoint were written, so its step-4200 weights are intact, but it has no `adapter_last.pt`
# and no `train_summary.json`.
#
# The cost of this rule is that C is compared at 4,200 steps against three arms at 4,493, i.e. 6.5%
# fewer optimizer steps. That is a real caveat and belongs in any write-up of the four-way result.
# `adapter_step0004200.pt` exists for all four if you want the step-matched comparison instead:
# set CHECKPOINT_KIND=step4200.
CHECKPOINT_KIND="${CHECKPOINT_KIND:-last}"

resolve_checkpoint() {
    local run_dir="$1"
    case "$CHECKPOINT_KIND" in
        last)
            if [[ -f "$run_dir/adapter_last.pt" ]]; then
                echo "$run_dir/adapter_last.pt"
            else
                echo "$run_dir/adapter_step0004200.pt"
            fi ;;
        step4200) echo "$run_dir/adapter_step0004200.pt" ;;
        best_ssim) echo "$run_dir/adapter_best_ssim.pt" ;;
        best_fvd) echo "$run_dir/adapter_best_fvd.pt" ;;
        best_fid) echo "$run_dir/adapter_best_fid_2p5d.pt" ;;
        *) echo "unknown CHECKPOINT_KIND=$CHECKPOINT_KIND" >&2; exit 1 ;;
    esac
}

# ---------------------------------------------------------------- preprocessing
#
# **15 mm, to match training.** `cli.train_r2v` had no `--posterior-shift-mm` flag until
# 2026-08-10, so the final four trained at `R2VDatasetConfig`'s default of 15 while the old
# `02_preprocess.sbatch` hardcoded 0. Measured: 15.8% of test cases differ between the two
# (correlation 0.63-0.85 on those), concentrated in the coronal/sagittal buckets -- so scoring at 0
# penalised exactly those buckets for a displacement the model did not cause.
#
# This is now also *checked*, not just set: `cli.evaluate` compares these flags against what the
# adapter recorded and refuses the run on a mismatch, before any GPU work.
POSTERIOR_SHIFT="${POSTERIOR_SHIFT:-15}"
NORMALIZER="${NORMALIZER:-percentile}"

# `report_format`: the order-agnostic *_meta spec A, B and C were trained on. Omitting it is silent
# -- the text would carry no [MODALITY]/[PLANE]/[SPACING] prefix and be out of distribution for
# three of the four arms. D ignores it and reads the sections instead.
REPORT_FORMAT="${REPORT_FORMAT:-findings_impression_meta}"

# ---------------------------------------------------------------- sampling
#
# NVIDIA's own inference defaults, unchanged, so the report term is the only departure from
# `diff_model_infer.py`. These are also what the training loop's validation sampled at, which is
# what makes the offline numbers readable next to the training curves.
STEPS="${R2V_STEPS:-30}"
REPORT_GUIDANCE="${R2V_REPORT_GUIDANCE:-4.0}"
MODALITY_GUIDANCE="${R2V_MODALITY_GUIDANCE:-10.0}"

# Per-case seed is `stable_seed(SEED, case_id)`, so it depends on the case rather than on iteration
# order. Shared across all four arms, so the noise draw is paired between them.
SEED="${R2V_SEED:-42}"

# ---------------------------------------------------------------- W&B
#
# One project and one group for the four arms, with the run named after the arm, so the W&B run
# table IS the four-way comparison: `metrics/all` is sortable per run and the headline scalars
# (psnr_fg, ssim3d, both FIDs, report-consistency macro AUROC, train_samples_seen) become columns.
#
# **`R2V_WANDB_REPORTS=1` means the panels carry patient report text**, so this project must be
# private. Set it to 0 for a shared project; the metrics table is unaffected either way.
WANDB_MODE="${R2V_WANDB:-online}"
WANDB_PROJECT="${R2V_WANDB_PROJECT:-mr-rate-r2v-eval}"
WANDB_GROUP="${R2V_WANDB_GROUP:-final_four_${SPLIT}}"
WANDB_PANELS="${R2V_WANDB_PANELS:-6}"
WANDB_REPORTS="${R2V_WANDB_REPORTS:-1}"

# ---------------------------------------------------------------- training provenance
#
# Configuration C's training job died before writing `train_summary.json`, which is where
# `world_size` (and hence the effective batch, and hence the training-sample count) is recorded.
# All four arms were launched from `slurm/final/_final_common.sh` at 2 nodes x 4 GPUs, so 8 is a
# recorded fact of the run rather than a guess; `world_size_source` in the results says which arms
# took it from the summary and which were told. Only C needs it.
TRAIN_WORLD_SIZE="${R2V_TRAIN_WORLD_SIZE:-8}"

# ---------------------------------------------------------------- run map
run_dir_for() {
    case "$1" in
        A) echo "$RUNS/r2v_final_A_cxr_bert_cls" ;;
        B) echo "$RUNS/r2v_final_B_cxr_bert_tokens" ;;
        C) echo "$RUNS/r2v_final_C_radbert_tokens" ;;
        D) echo "$RUNS/r2v_final_D_report2ct_style" ;;
        *) echo "unknown config '$1' (expected A, B, C or D)" >&2; exit 1 ;;
    esac
}

# ---------------------------------------------------------------- submit
submit_final_eval() {
    local config="$1" tag="$2"
    local run_dir adapter run_tag walltime n_per_bucket

    run_dir="$(run_dir_for "$config")"
    adapter="$(resolve_checkpoint "$run_dir")"
    [[ -f "$adapter" ]] || { echo "missing checkpoint: $adapter" >&2; exit 1; }

    run_tag="r2v_final_${tag}"
    n_per_bucket="$N_PER_BUCKET"

    # **Walltime from the measured rate, not from the training loop's timer.** Jobs 718272-8
    # produced 8 volumes in 53.5-55.2 s on one H200 = 6.7 s/case at 30 inference steps. The obvious
    # earlier estimate was 4x too optimistic: `val/time/generate` in the training checkpoints works
    # out to 1.62 GPU-s/case, but that timer excludes the VAE decode -- a sliding-window inferer at
    # 80^3 with 0.4 overlap, which dominates. Scoring adds ~1 s/case on top.
    walltime=08:00:00
    if [[ "${SMOKE:-0}" == "1" ]]; then
        run_tag="smoke_${run_tag}"
        n_per_bucket=8
        walltime=00:40:00
    elif [[ "${FULL_SPLIT:-0}" == "1" ]]; then
        run_tag="full_${run_tag}"
        n_per_bucket=""              # empty => the entire split
        walltime=72:00:00
    fi

    echo "=== config $config"
    echo "    run dir    : $run_dir"
    echo "    checkpoint : $(basename "$adapter")  (CHECKPOINT_KIND=$CHECKPOINT_KIND)"
    echo "    split      : $SPLIT"
    echo "    scale      : ${n_per_bucket:-<entire split>} per bucket"
    echo "    run tag    : $run_tag"

    local exports="ALL,R2V_STEPS=${STEPS},R2V_REPORT_GUIDANCE=${REPORT_GUIDANCE}"
    exports+=",R2V_MODALITY_GUIDANCE=${MODALITY_GUIDANCE},R2V_SEED=${SEED}"
    exports+=",R2V_SPLIT=${SPLIT},R2V_POSTERIOR_SHIFT=${POSTERIOR_SHIFT}"
    exports+=",R2V_NORMALIZER=${NORMALIZER},R2V_REPORT_FORMAT=${REPORT_FORMAT}"
    exports+=",R2V_TRAIN_WORLD_SIZE=${TRAIN_WORLD_SIZE}"
    exports+=",R2V_WANDB=${WANDB_MODE},R2V_WANDB_PROJECT=${WANDB_PROJECT}"
    exports+=",R2V_WANDB_GROUP=${WANDB_GROUP},R2V_WANDB_NAME=${config}_${tag}"
    exports+=",R2V_WANDB_PANELS=${WANDB_PANELS},R2V_WANDB_REPORTS=${WANDB_REPORTS}"
    [[ -n "$n_per_bucket" ]] && exports+=",R2V_N_PER_BUCKET=${n_per_bucket}"
    [[ -n "${R2V_WANDB_ENTITY:-}" ]] && exports+=",R2V_WANDB_ENTITY=${R2V_WANDB_ENTITY}"

    # afterany, not afterok: a missing classifier costs one metric group (recorded unavailable with
    # a reason), which is not a reason to leave the evaluation queued forever.
    local dependency=()
    [[ -n "${CLASSIFIER_JOB:-}" ]] && dependency=(--dependency="afterany:${CLASSIFIER_JOB}")

    local job_id
    job_id=$(sbatch --parsable --job-name="evaluate_${run_tag}" --time="$walltime" \
        --export="$exports" ${dependency+"${dependency[@]}"} \
        "$REPO_ROOT/slurm/05_evaluate.sbatch" report2volume "$run_tag" "$adapter")
    echo "    job        : $job_id"
    echo "    results    : $WORKSPACE/cache/r2v/results/report2volume_${run_tag}"
}
