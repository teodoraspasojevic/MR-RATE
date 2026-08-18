#!/bin/bash -l
# The three-stage study: which conditioning, at which guidance scale, how robust to report format.
#
#   slurm/final_eval/run_sweep.sh cfg       # stage 1: best guidance scale per arm      (25 jobs)
#   slurm/final_eval/run_sweep.sh format    # stage 2: report-format robustness         (12 jobs)
#   slurm/final_eval/run_sweep.sh headline  # stage 3: the arm-vs-arm result             (5 jobs)
#
# **This is deliberately NOT the full 5 x 5 x 4 = 100-run grid.** Guidance scale and report format
# are close to independent, so the grid spends most of its runs re-measuring each format at a
# guidance scale already known to be wrong for that arm. Factorising into three stages answers the
# same three questions in 42 jobs:
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
#     stage 1  25 jobs  ~150 GPU-h        stage 2  12 jobs  ~72 GPU-h
#     stage 3   5 jobs   ~30 GPU-h        total    42 jobs  ~252 GPU-h
# CFG_VALUES="1.0 4.0 7.0" samples the guidance curve more coarsely for ~168 GPU-h total; it trims
# the axis rather than the population, which is the only trim that keeps the runs comparable.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_final_eval_common.sh"

# Not `${1:?...}` with braces in the message: bash ends the expansion at the FIRST `}`, so the
# stray one lands in the value and every stage name arrives as e.g. "cfg}".
STAGE="${1:-}"
[[ -n "$STAGE" ]] || { echo "usage: run_sweep.sh <cfg|format|headline|representative|unbalanced|all>" >&2; exit 1; }

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

# **Configurations D and E are excluded from the format axis, on purpose.** Both are sectioned
# fusion: they encode findings and impression as separate cross-attention tokens read from
# `report_sections_text`, and never see the joined string `--report-format` composes. Running them
# across four formats would produce four identical results and read as evidence of robustness.
# E does condition on the metadata -- through its own `acquisition` token, which is composed from
# the case's modality/plane/spacing and is therefore unaffected by `--report-format` too.
FORMAT_ARMS="${FORMAT_ARMS:-A B C}"
ALL_ARMS="${ALL_ARMS:-A B C D E}"

arm_tag() {
    case "$1" in
        A) echo "A_cxr_bert_cls" ;;   B) echo "B_cxr_bert_tokens" ;;
        C) echo "C_radbert_tokens" ;; D) echo "D_report2ct_style" ;;
        E) echo "E_report2ct_style_meta" ;;
    esac
}

# Filled in from stage 1's `metrics_summary.csv`. Until then every arm uses the incumbent, and the
# script says so rather than pretending a sweep already happened.
declare -A BEST_CFG=( [A]="${BEST_CFG_A:-4.0}" [B]="${BEST_CFG_B:-4.0}"
                      [C]="${BEST_CFG_C:-4.0}" [D]="${BEST_CFG_D:-4.0}"
                      [E]="${BEST_CFG_E:-4.0}" )

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
    exports+=",R2V_ALLOW_FORMAT_MISMATCH=${allow}"
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
        "$REPO_ROOT/slurm/evaluate.sbatch" report2volume "$tag" "$adapter")
    # The adapter is printed because it is SUBSTITUTED for configuration C: that run died before
    # writing adapter_last.pt, so it falls back to adapter_step0004200.pt. A silent substitution in
    # a 12-job sweep is exactly how a step-mismatched comparison gets read as a like-for-like one.
    printf '    %-46s cfg=%-4s fmt=%-26s n=%-6s ckpt=%-26s job=%s\n' \
        "$tag" "$cfg" "$fmt" "${n:-full}" "$(basename "$adapter")" "$job"
}

TRAINED_FORMAT="${TRAINED_FORMAT:-findings_impression_meta}"

case "$STAGE" in
cfg)
    # 5 arms x 5 scales at N_PER_BUCKET (2,000 cases, ~6 h each) = 25 jobs, ~150 GPU-h.
    #
    # **This stage used to run at 50/bucket and could not work.** Per-bucket FID compares two
    # 512-d MedicalNet covariances; 50 samples estimates one of rank <= 49, `frechet_distance`
    # rejects the resulting matrix square root as untrustworthy, and on 2026-08-12 that took out
    # all 20 jobs -- 18 to walltime grinding through failing bootstrap resamples, 2 to an uncaught
    # raise that discarded 85 minutes of finished sampling. The cheap-N discount was never real.
    echo "=== stage 1: guidance-scale sweep (${CFG_VALUES}) at ${N_PER_BUCKET}/bucket over ${ALL_ARMS}"
    for arm in $ALL_ARMS; do
        for cfg in $CFG_VALUES; do
            submit "$arm" "cfg${cfg}" "$N_PER_BUCKET" "$cfg" "$TRAINED_FORMAT" "$WALLTIME"
        done
    done
    echo
    echo "When these land, pick the best scale per arm from metrics_summary.csv (overall_pooled"
    echo "and batched_fixed_n on FVD/FID, plus report_consistency), then:"
    echo "    BEST_CFG_A=... BEST_CFG_B=... BEST_CFG_C=... BEST_CFG_D=... BEST_CFG_E=... run_sweep.sh format"
    ;;
format)
    # 3 arms x 4 formats at 200/bucket = 12 jobs, ~50 GPU-h. D and E are excluded (format-invariant).
    echo "=== stage 2: report-format robustness at ${N_PER_BUCKET}/bucket"
    echo "    arms: $FORMAT_ARMS   (D, E excluded: sectioned fusion never reads the joined string)"
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
    # The result table: every arm at its own best scale, one fixed format. D vs E is the pair to
    # read for whether the acquisition token earned its place; they are identical otherwise.
    echo "=== stage 3: headline (${ALL_ARMS}) at ${N_PER_BUCKET}/bucket"
    for arm in $ALL_ARMS; do
        submit "$arm" "headline" "$N_PER_BUCKET" \
            "${BEST_CFG[$arm]}" "$TRAINED_FORMAT" "$WALLTIME"
    done
    ;;
representative)
    # **Why this stage exists.** Every other stage takes the FIRST N per bucket of the
    # (study_uid, series_id) order. That is a convenience sample, and on the MR-RATE test split it
    # is measurably skewed against the 27,027 cases it leaves out (measured 2026-08-14, one row per
    # study per bucket, joined to splits_merged_majority):
    #
    #     PP_Cerebrovascular       prefix 14.43%  rest 11.41%   two-proportion z = +4.06
    #     BP_Hemorrhagic_lesions   prefix 10.92%  rest  8.38%                   z = +3.91
    #     PP_Spinal                prefix  0.75%  rest  1.79%                   z = -3.45
    #
    # all three surviving Bonferroni over 14 labels. Summed absolute deviation from population
    # prevalence is 0.146 for the prefix; **all 20 seeds tested (0-19) score lower**, median 0.070.
    # So the skew is a property of the prefix, not an unlucky draw: a prefix run answers "how does
    # the model do on these 2,000 cases", and only a seeded draw answers "on this split".
    #
    # This bites `report_consistency` hardest: PP_Cerebrovascular is configuration D's single best
    # label (AUROC 0.563 at cfg 4.0) and the prefix over-represents it by 26% relative.
    #
    # Seed 42 is the repo's standard, chosen a priori rather than by taking the most favourable
    # draw -- it scores 0.092 (0.63x the prefix), mid-pack rather than best.
    #
    # The tag carries `_rand<seed>`, and `run_id`/`cases_sha256` differ, so these can neither
    # overwrite nor be confused with the prefix runs. **They are NOT case-by-case comparable with
    # the 20 cfg jobs** -- different case list, so no paired per-case test across the two sets.
    # They ARE comparable to each other: one case list shared across the cfg values.
    #
    # **Disabled, not fixed.** This stage needs `cli.evaluate` to support a seeded random draw
    # (`--case-selection-seed` or equivalent); no such flag exists today, so `evaluate.sbatch`
    # silently drops the `R2V_CASE_SELECTION_SEED` export this stage used to set and every job ran
    # the ordinary first-N-per-bucket prefix instead -- the exact bias this stage exists to avoid,
    # with no error to say so. Re-enable once that CLI support exists; the analysis above is why
    # it's worth adding rather than a reason to just delete this stage.
    echo "'representative' needs seeded random case selection in cli.evaluate, which doesn't exist yet." >&2
    echo "It used to silently fall back to the ordinary first-N prefix -- the exact bias this stage" >&2
    echo "exists to avoid. Disabled until cli.evaluate grows that flag." >&2
    exit 1
    ;;
unbalanced)
    # **The `representative` stage fixed WHICH cases; this one fixes the MIXTURE.**
    #
    # Every other stage forces all ten (modality, plane) buckets to the same size, so `overall_*`
    # describes a 10%-each population that does not exist in the data. Measured shares, one row per
    # study per bucket over the 29,027-case test split -- and the all-series shares training
    # actually saw, which track them within ~2pp:
    #
    #     bucket           training %   eligible %   balanced n   unbalanced n
    #     T2w AXIAL            15.86        16.27          200            ~325
    #     T1w AXIAL            15.87        13.81          200            ~276
    #     FLAIR SAGITTAL       10.81        12.04          200            ~241
    #     SWI AXIAL            11.46        11.63          200            ~233
    #     FLAIR AXIAL           9.97        11.53          200            ~231
    #     T1w SAGITTAL         12.60        10.46          200            ~209
    #     T2w CORONAL           6.87         7.96          200            ~159
    #     T1w CORONAL           7.12         6.07          200            ~121
    #     FLAIR CORONAL         5.13         5.77          200            ~115
    #     T2w SAGITTAL          4.31         4.47          200             ~89
    #
    # Balancing more than doubles the rarest bucket's weight and nearly halves T1w AXIAL's. Training
    # used `series_selection="all"` with every series seen once per epoch, i.e. no balancing at all,
    # so this draw is the closer match to training statistics -- and to the challenge, which scores
    # one distance over its whole hidden set.
    #
    # **Simple random, not proportional-stratified**, by choice: bucket counts land where the draw
    # puts them (multinomial, so the ~89 above is +/-9), which carries the same sampling variance a
    # real submission would and assumes nothing about the allocation.
    #
    # The cost, stated rather than discovered later: unequal bucket counts make `overall_macro`
    # meaningless -- read `overall_pooled` and `overall_weighted` -- and thin the small buckets'
    # per-bucket FID/FVD. Those are *already* rank-deficient at 200/bucket (200 < 512 feature dims,
    # all ten flagged in every completed run), so this worsens a known limitation rather than
    # introducing one. 89 still clears the n<50 `unstable_small_sample` threshold, and a Frechet
    # that cannot be computed now returns null with a reason instead of killing the job.
    #
    # Tag carries `_all<total>_rand<seed>`. Third distinct population, so NO case-level comparison
    # with either the prefix runs or the `representative` ones.
    #
    # **Disabled, not fixed.** Same root cause as `representative`: this stage needs `cli.evaluate`
    # to support drawing a random total over the whole split (`--n-total`/`--case-selection-seed`
    # or equivalent) instead of the per-bucket-balanced `--n-per-bucket`; neither exists today, so
    # `evaluate.sbatch` silently dropped the `R2V_N_TOTAL`/`R2V_CASE_SELECTION_SEED` exports this
    # stage used to set and fell through to `cli.evaluate`'s default of the entire ~29,027-case test
    # split -- far past this stage's own `$WALLTIME`, so the job would run until killed. Re-enable
    # once that CLI support exists; the analysis above is why it's worth adding, not a reason to
    # delete the stage.
    echo "'unbalanced' needs whole-split random sampling (--n-total) in cli.evaluate, which doesn't" >&2
    echo "exist yet. It used to silently fall through to the full ~29,027-case test split instead --" >&2
    echo "far past this stage's own walltime. Disabled until cli.evaluate grows that support." >&2
    exit 1
    ;;
all)
    # **Removed, not fixed.** It submitted all three stages at once and its own header claimed
    # they were chained with --dependency=afterany. They were not, and could not be: stages 2 and
    # 3 read BEST_CFG, which does not exist until a human has read stage 1's metrics_summary.csv.
    # `all` therefore silently pinned 17 of the 42 jobs to the incumbent 4.0 and answered a
    # question nobody asked. A dependency cannot substitute for the judgement in between.
    echo "'all' is gone: stages 2 and 3 need BEST_CFG_* from stage 1, which is a human decision." >&2
    echo "Run:  run_sweep.sh cfg   ->  read metrics_summary.csv  ->  BEST_CFG_A=... run_sweep.sh format" >&2
    exit 1
    ;;
*)
    echo "unknown stage '$STAGE' (cfg | format | headline | representative | unbalanced | all)" >&2; exit 1 ;;
esac
