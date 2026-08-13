#!/bin/bash -l
# The three-stage study: which conditioning, at which guidance scale, how robust to report format.
#
#   slurm/final_eval/run_sweep.sh cfg       # stage 1: best guidance scale per arm      (20 jobs)
#   slurm/final_eval/run_sweep.sh format    # stage 2: report-format robustness         (12 jobs)
#   slurm/final_eval/run_sweep.sh headline  # stage 3: the four-way result               (4 jobs)
#
# **This is deliberately NOT the full 4 x 5 x 4 = 80-run grid.** Guidance scale and report format
# are close to independent, so the grid spends 16 runs re-measuring each format at a guidance scale
# already known to be wrong for that arm. Factorising into three stages answers the same three
# questions in 36 jobs:
#
#     which guidance scale is best for each arm?   stage 1, all arms x all scales
#     how much does the report format cost?        stage 2, each arm at ITS best scale
#     which conditioning wins?                     stage 3, each arm at its best
#
# Stage 2 and 3 read stage 1's answer, so run them in that order and pass BEST_CFG_* explicitly.
# There is no `all`: see the case below for why chaining cannot replace reading stage 1.
#
# **Every job in every stage runs on the SAME 2,000 cases** (200/bucket x 10 buckets), and so do
# the reconstruction and generation baselines. Verified rather than assumed: all four dataset
# configurations -- with and without --report-format, in either section order -- select an
# identical ordered case list (sha 19e94f024c80bc2e), matching the 2,000 cases the completed
# reconstruction baseline actually scored, set and order. --report-format changes how the report
# text is composed, never which samples exist.
#
# Cost at that N, from the two completed baselines (5:04 and 6:31 for 2,000 cases):
#     stage 1  20 jobs  ~120 GPU-h        stage 2  12 jobs  ~72 GPU-h
#     stage 3   4 jobs   ~24 GPU-h        total    36 jobs  ~216 GPU-h
# CFG_VALUES="1.0 4.0 7.0" samples the guidance curve more coarsely for ~168 GPU-h total; it trims
# the axis rather than the population, which is the only trim that keeps the runs comparable.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_final_eval_common.sh"

# Not `${1:?...}` with braces in the message: bash ends the expansion at the FIRST `}`, so the
# stray one lands in the value and every stage name arrives as e.g. "cfg}".
STAGE="${1:-}"
[[ -n "$STAGE" ]] || { echo "usage: run_sweep.sh <cfg|format|headline|all>" >&2; exit 1; }

# ---------------------------------------------------------------- the axes
#
# **Guidance scales.** `report_guidance_scale` is classifier-free guidance on the report branch,
# which works at all only because training used `--report-dropout-probability 0.10` (the null
# embedding CFG interpolates away from). NVIDIA's own default, and what every validation curve was
# sampled at, is 4.0 -- so it is in the list as the incumbent, not as the assumed winner.
#
#     1.0   conditional only, no amplification -- the honest baseline
#     2.0   typical diffusion sweet spot, lower end
#     3.0
#     4.0   the incumbent; what the training curves and every earlier number used
#     7.0   strong -- included to find where fidelity starts degrading, not because it will win
#
# 0.0 is deliberately absent: it makes `guided_model_output` skip the report branch entirely, so
# the volumes are report-BLIND. That is a useful control but it is not a guidance setting, and
# `--task generation` already measures it properly.
CFG_VALUES="${CFG_VALUES:-1.0 2.0 3.0 4.0 7.0}"

# **Report formats.** Two questions, not one:
#
#   in-distribution   findings_impression_meta / impression_findings_meta
#                     A, B and C trained on the SAMPLED spec of both, so both are in distribution.
#                     Comparing them measures ORDER ROBUSTNESS honestly.
#   out-of-distribution  findings_impression / impression_findings
#                     Same section order, but no [MODALITY]/[PLANE]/[SPACING] prefix. The arms
#                     never saw this, so it measures DEGRADATION WITHOUT METADATA -- a different
#                     question, and one `assert_report_format_matches` refuses by default.
#                     R2V_ALLOW_FORMAT_MISMATCH=1 is what makes it a deliberate ablation.
FORMATS_IN_DIST="${FORMATS_IN_DIST:-findings_impression_meta impression_findings_meta}"
FORMATS_OOD="${FORMATS_OOD:-findings_impression impression_findings}"

# **Configuration D is excluded from the format axis, on purpose.** `report2ct_style` is sectioned
# fusion: it encodes findings and impression as separate cross-attention tokens read from
# `report_sections_text`, and never sees the joined string `--report-format` composes. Running it
# across four formats would produce four identical results and read as evidence of robustness.
FORMAT_ARMS="${FORMAT_ARMS:-A B C}"
ALL_ARMS="${ALL_ARMS:-A B C D}"

arm_tag() {
    case "$1" in
        A) echo "A_cxr_bert_cls" ;;   B) echo "B_cxr_bert_tokens" ;;
        C) echo "C_radbert_tokens" ;; D) echo "D_report2ct_style" ;;
    esac
}

# Filled in from stage 1's `metrics_summary.csv`. Until then every arm uses the incumbent, and the
# script says so rather than pretending a sweep already happened.
declare -A BEST_CFG=( [A]="${BEST_CFG_A:-4.0}" [B]="${BEST_CFG_B:-4.0}"
                      [C]="${BEST_CFG_C:-4.0}" [D]="${BEST_CFG_D:-4.0}" )

submit() {   # submit <arm> <tag suffix> <n_per_bucket> <cfg> <format> <walltime> [allow_mismatch]
    local arm="$1" suffix="$2" n="$3" cfg="$4" fmt="$5" walltime="$6" allow="${7:-0}"
    local run_dir adapter tag
    run_dir="$(run_dir_for "$arm")"
    adapter="$(resolve_checkpoint "$run_dir")"
    [[ -f "$adapter" ]] || { echo "missing checkpoint: $adapter" >&2; exit 1; }
    tag="$(arm_tag "$arm")_${suffix}"

    [[ -n "$n" ]] || { echo "every stage runs at a fixed N; got an empty one for $arm" >&2; exit 1; }
    local exports="ALL,R2V_STEPS=${STEPS},R2V_REPORT_GUIDANCE=${cfg}"
    exports+=",R2V_MODALITY_GUIDANCE=${MODALITY_GUIDANCE},R2V_SEED=${SEED}"
    exports+=",R2V_SPLIT=${SPLIT},R2V_POSTERIOR_SHIFT=${POSTERIOR_SHIFT}"
    exports+=",R2V_NORMALIZER=${NORMALIZER},R2V_REPORT_FORMAT=${fmt}"
    exports+=",R2V_TRAIN_WORLD_SIZE=${TRAIN_WORLD_SIZE},R2V_ALLOW_FORMAT_MISMATCH=${allow}"
    exports+=",R2V_WANDB=${WANDB_MODE},R2V_WANDB_PROJECT=${WANDB_PROJECT}"
    exports+=",R2V_WANDB_GROUP=sweep_${STAGE},R2V_WANDB_NAME=${tag}"
    exports+=",R2V_WANDB_PANELS=${WANDB_PANELS},R2V_WANDB_REPORTS=${WANDB_REPORTS}"
    exports+=",R2V_N_PER_BUCKET=${n}"
    # No `--frechet-batch-size` override any more. It existed because the old 50/bucket stage could
    # not fill a 512-case batch; every stage now runs at N_PER_BUCKET so the default 512 applies
    # everywhere, which is what keeps `overall_batched_n512` comparable across every job.
    # Keep the volumes for the headline runs only -- ~19 GB each at float16, and the cheap sweep
    # stages exist to be thrown away. SAVE_VOLUMES=1 forces it on for every stage.
    [[ "${SAVE_VOLUMES:-0}" == "1" || "$STAGE" == "headline" ]] && exports+=",R2V_SAVE_VOLUMES=1"
    [[ -n "${R2V_WANDB_ENTITY:-}" ]] && exports+=",R2V_WANDB_ENTITY=${R2V_WANDB_ENTITY}"

    local dependency=()
    [[ -n "${CLASSIFIER_JOB:-}" ]] && dependency=(--dependency="afterany:${CLASSIFIER_JOB}")

    local job
    job=$(sbatch --parsable --job-name="ev_${tag}" --time="$walltime" --export="$exports" \
        ${dependency+"${dependency[@]}"} \
        "$REPO_ROOT/slurm/05_evaluate.sbatch" report2volume "$tag" "$adapter")
    # The adapter is printed because it is SUBSTITUTED for configuration C: that run died before
    # writing adapter_last.pt, so it falls back to adapter_step0004200.pt. A silent substitution in
    # a 12-job sweep is exactly how a step-mismatched comparison gets read as a like-for-like one.
    printf '    %-46s cfg=%-4s fmt=%-26s n=%-6s ckpt=%-26s job=%s\n' \
        "$tag" "$cfg" "$fmt" "${n:-full}" "$(basename "$adapter")" "$job"
}

TRAINED_FORMAT="${TRAINED_FORMAT:-findings_impression_meta}"

case "$STAGE" in
cfg)
    # 4 arms x 5 scales at N_PER_BUCKET (2,000 cases, ~6 h each) = 20 jobs, ~120 GPU-h.
    #
    # **This stage used to run at 50/bucket and could not work.** Per-bucket FID compares two
    # 512-d MedicalNet covariances; 50 samples estimates one of rank <= 49, `frechet_distance`
    # rejects the resulting matrix square root as untrustworthy, and on 2026-08-12 that took out
    # all 20 jobs -- 18 to walltime grinding through failing bootstrap resamples, 2 to an uncaught
    # raise that discarded 85 minutes of finished sampling. The cheap-N discount was never real.
    echo "=== stage 1: guidance-scale sweep (${CFG_VALUES}) at ${N_PER_BUCKET}/bucket"
    for arm in $ALL_ARMS; do
        for cfg in $CFG_VALUES; do
            submit "$arm" "cfg${cfg}" "$N_PER_BUCKET" "$cfg" "$TRAINED_FORMAT" "$WALLTIME"
        done
    done
    echo
    echo "When these land, pick the best scale per arm from metrics_summary.csv (overall_pooled"
    echo "and batched_fixed_n on FVD/FID, plus report_consistency), then:"
    echo "    BEST_CFG_A=... BEST_CFG_B=... BEST_CFG_C=... BEST_CFG_D=... run_sweep.sh format"
    ;;
format)
    # 3 arms x 4 formats at 200/bucket = 12 jobs, ~50 GPU-h. D is excluded (format-invariant).
    echo "=== stage 2: report-format robustness at ${N_PER_BUCKET}/bucket"
    echo "    arms: $FORMAT_ARMS   (D excluded: sectioned fusion never reads the joined string)"
    for arm in $FORMAT_ARMS; do
        for fmt in $FORMATS_IN_DIST; do
            submit "$arm" "fmt_${fmt}" "$N_PER_BUCKET" "${BEST_CFG[$arm]}" "$fmt" "$WALLTIME" 0
        done
        for fmt in $FORMATS_OOD; do
            submit "$arm" "fmt_${fmt}_ood" "$N_PER_BUCKET" "${BEST_CFG[$arm]}" "$fmt" "$WALLTIME" 1
        done
    done
    ;;
headline)
    # The result table: 4 arms, each at its own best scale, one fixed format.
    echo "=== stage 3: headline four-way at ${N_PER_BUCKET}/bucket"
    for arm in $ALL_ARMS; do
        submit "$arm" "headline" "$N_PER_BUCKET" \
            "${BEST_CFG[$arm]}" "$TRAINED_FORMAT" "$WALLTIME"
    done
    ;;
all)
    # **Removed, not fixed.** It submitted all three stages at once and its own header claimed
    # they were chained with --dependency=afterany. They were not, and could not be: stages 2 and
    # 3 read BEST_CFG, which does not exist until a human has read stage 1's metrics_summary.csv.
    # `all` therefore silently pinned 16 of the 36 jobs to the incumbent 4.0 and answered a
    # question nobody asked. A dependency cannot substitute for the judgement in between.
    echo "'all' is gone: stages 2 and 3 need BEST_CFG_* from stage 1, which is a human decision." >&2
    echo "Run:  run_sweep.sh cfg   ->  read metrics_summary.csv  ->  BEST_CFG_A=... run_sweep.sh format" >&2
    exit 1
    ;;
*)
    echo "unknown stage '$STAGE' (cfg | format | headline | all)" >&2; exit 1 ;;
esac
