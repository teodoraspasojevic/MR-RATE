#!/bin/bash -l
# The three-stage study: which conditioning, at which guidance scale, how robust to report format.
#
#   slurm/final_eval/run_sweep.sh cfg       # stage 1: best guidance scale per arm      (20 jobs)
#   slurm/final_eval/run_sweep.sh format    # stage 2: report-format robustness         (12 jobs)
#   slurm/final_eval/run_sweep.sh headline  # stage 3: the four-way result               (4 jobs)
#   slurm/final_eval/run_sweep.sh all       # all three, chained by dependency
#
# **This is deliberately NOT the full 4 x 5 x 4 = 80-run grid.** That costs ~340 GPU-hours and most
# of it answers nothing: guidance scale and report format are close to independent, so the grid
# spends 16 runs re-measuring each format at a guidance scale that is already known to be wrong for
# that arm. Factorising into three stages answers the same three questions for ~87 GPU-hours:
#
#     which guidance scale is best for each arm?   stage 1, cheap N, all arms x all scales
#     how much does the report format cost?        stage 2, each arm at ITS best scale
#     which conditioning wins?                     stage 3, each arm at its best, full scale
#
# Stage 2 and 3 read stage 1's answer, so run them in order (or use `all`, which chains them with
# `--dependency=afterany` and expects you to fill in BEST_CFG below once stage 1 lands).
#
# Every stage keeps the four arms on the SAME cases: selection is deterministic and prefix-stable,
# so a stage-1 run at 50/bucket is literally the first 50 per bucket of the stage-3 run. Nothing has
# to be held fixed by hand.

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

    local exports="ALL,R2V_STEPS=${STEPS},R2V_REPORT_GUIDANCE=${cfg}"
    exports+=",R2V_MODALITY_GUIDANCE=${MODALITY_GUIDANCE},R2V_SEED=${SEED}"
    exports+=",R2V_SPLIT=${SPLIT},R2V_POSTERIOR_SHIFT=${POSTERIOR_SHIFT}"
    exports+=",R2V_NORMALIZER=${NORMALIZER},R2V_REPORT_FORMAT=${fmt}"
    exports+=",R2V_TRAIN_WORLD_SIZE=${TRAIN_WORLD_SIZE},R2V_ALLOW_FORMAT_MISMATCH=${allow}"
    exports+=",R2V_WANDB=${WANDB_MODE},R2V_WANDB_PROJECT=${WANDB_PROJECT}"
    exports+=",R2V_WANDB_GROUP=sweep_${STAGE},R2V_WANDB_NAME=${tag}"
    exports+=",R2V_WANDB_PANELS=${WANDB_PANELS},R2V_WANDB_REPORTS=${WANDB_REPORTS}"
    [[ -n "$n" ]] && exports+=",R2V_N_PER_BUCKET=${n}"
    # A small run cannot fill a 512-case batch, so the fixed-N column would be unavailable. 256
    # keeps two batches at 50/bucket, which is what gives the sweep an error bar to rank on.
    [[ -n "$n" && "$n" -le 100 ]] && exports+=",R2V_FRECHET_BATCH=256"
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
    # 4 arms x 5 scales at 50/bucket (500 cases, ~1 h each) = 20 jobs, ~20 GPU-h.
    # Cheap N is defensible here because every arm is compared at the SAME N, and what is being
    # ranked is one arm against itself across scales. Do not quote these FIDs as absolute numbers.
    echo "=== stage 1: guidance-scale sweep (${CFG_VALUES}) at ${CFG_N_PER_BUCKET:-50}/bucket"
    for arm in $ALL_ARMS; do
        for cfg in $CFG_VALUES; do
            submit "$arm" "cfg${cfg}" "${CFG_N_PER_BUCKET:-50}" "$cfg" "$TRAINED_FORMAT" 02:00:00
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
            submit "$arm" "fmt_${fmt}" "$N_PER_BUCKET" "${BEST_CFG[$arm]}" "$fmt" 08:00:00 0
        done
        for fmt in $FORMATS_OOD; do
            submit "$arm" "fmt_${fmt}_ood" "$N_PER_BUCKET" "${BEST_CFG[$arm]}" "$fmt" 08:00:00 1
        done
    done
    ;;
headline)
    # The result table: 4 arms, each at its own best scale, one fixed format, full scale.
    echo "=== stage 3: headline four-way at ${HEADLINE_N_PER_BUCKET:-$N_PER_BUCKET}/bucket"
    for arm in $ALL_ARMS; do
        submit "$arm" "headline" "${HEADLINE_N_PER_BUCKET:-$N_PER_BUCKET}" \
            "${BEST_CFG[$arm]}" "$TRAINED_FORMAT" "${HEADLINE_WALLTIME:-08:00:00}"
    done
    echo
    echo "FULL_SPLIT: re-run with HEADLINE_N_PER_BUCKET= and HEADLINE_WALLTIME=72:00:00"
    echo "            (~29,027 cases per arm at one_per_study_per_bucket, ~60 GPU-h each)"
    ;;
all)
    "$0" cfg; echo; "$0" format; echo; "$0" headline
    ;;
*)
    echo "unknown stage '$STAGE' (cfg | format | headline | all)" >&2; exit 1 ;;
esac
