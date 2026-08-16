#!/bin/bash -l
#SBATCH --job-name=r2v_evaluate
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --account=y100dc
#SBATCH --qos=mq_health
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_evaluate_%j.out
#SBATCH --error=/hnvme/workspace/y100dc19-nvidia-mri-brain/slurm_logs/r2v_evaluate_%j.err
#
# The ONE evaluation job. Builds the dataset, generates, and scores -- no cohort, no prediction set.
#
#   sbatch slurm/05_evaluate.sbatch <task> <run_tag> [adapter.pt]
#
#     report2volume    a trained adapter, conditioned on each case's own report   (needs adapter.pt)
#     reconstruction   the frozen NVIDIA autoencoder, encode then decode
#     generation       the frozen base UNet from a modality label alone, report-blind
#
# Examples:
#   sbatch slurm/05_evaluate.sbatch reconstruction  vae_baseline
#   sbatch slurm/05_evaluate.sbatch generation      nvidia_uncond
#   sbatch slurm/05_evaluate.sbatch report2volume   r2v_A  $WS/runs/r2v_final_A_cxr_bert_cls/adapter_last.pt
#
# **What gets evaluated.** Every case in `--split test`, in a deterministic order with no RNG,
# unless R2V_N_PER_BUCKET caps it -- and the cap takes the FIRST N per bucket in that same order,
# so a cheap run is a prefix of the full one rather than a different sample. Per-case sampler noise
# is `stable_seed(R2V_SEED, case_id)`, a function of the case, so a rerun reproduces every volume.
#
#   R2V_N_PER_BUCKET=8     ~10 min   smoke test; no metric means anything at N=8 and the run says so
#   R2V_N_PER_BUCKET=200   ~4 h      the scale the old cohort pipeline ran at (2,000 cases)
#   unset (full split)     ~60 h     29,027 cases (one per study per bucket) -- CTFlow's whole-split precedent
#
# The walltime above (8 h) fits R2V_N_PER_BUCKET=200 with headroom. **Raise --time for a full-split
# run**; measured 6.7 s/case at 30 inference steps on one H200, and the job is killed, not paused,
# when the clock runs out.
#
# `#SBATCH --export=NONE` means a plain `VAR=x sbatch ...` does NOT reach the job -- pass overrides
# through --export=ALL,... as the final_eval/ scripts do.

unset SLURM_EXPORT_ENV
set -euo pipefail

R2V_REPO="${R2V_REPO:-/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE}"
source "$R2V_REPO/contrastive-pretraining/slurm/_common.sh"
setup

TASK="${1:?usage: sbatch 05_evaluate.sbatch <task> <run_tag> [adapter.pt]}"
TAG="${2:?usage: sbatch 05_evaluate.sbatch <task> <run_tag> [adapter.pt]}"
ADAPTER="${3:-}"

case "$TASK" in
    report2volume|reconstruction|generation) ;;
    *) echo "unknown task '$TASK' (report2volume | reconstruction | generation)" >&2; exit 1 ;;
esac
if [[ "$TASK" == "report2volume" && -z "$ADAPTER" ]]; then
    echo "--task report2volume needs an adapter checkpoint as the third argument" >&2
    exit 1
fi

OUT="$RESULT_ROOT/${TASK}_${TAG}"
preflight "$MANIFEST_CSV" "$REPORT_INDEX_CSV" "$VAE_CHECKPOINT" "$MEDICALNET_CHECKPOINT"
[[ -z "$ADAPTER" ]] || preflight "$ADAPTER"

ARGS=(
    --task "$TASK"
    --manifest "$MANIFEST_CSV"
    --report-index "$REPORT_INDEX_CSV"
    --split "${R2V_SPLIT:-test}"
    --out "$OUT"
    --vae-checkpoint "$VAE_CHECKPOINT"
    # Both are hashed into run_id and both are checked against what the adapter recorded, so a
    # divergence like the 2026-08-10 posterior-shift one stops the job instead of skewing a bucket.
    --posterior-shift-mm "${R2V_POSTERIOR_SHIFT:-15}"
    --normalizer "${R2V_NORMALIZER:-percentile}"
    --geometry-mode per_modality_plane
    --report-sections findings impression
    --medicalnet-checkpoint "$MEDICALNET_CHECKPOINT"
    --seed "${R2V_SEED:-42}"
    --device cuda
    --save-figures "${R2V_SAVE_FIGURES:-3}"
    --overwrite
)

# `--report-format`: NOT a default, and omitting it is silent. The four final adapters were trained
# on the order-agnostic *_meta spec, whose conditioning text carries a [MODALITY]/[PLANE]/[SPACING]
# prefix; composing the text without that prefix is out of distribution for three of the four arms.
# `assert_report_format_matches` refuses the pairing rather than scoring it -- but only because the
# check exists. Configuration D ignores the format and reads the sections instead.
[[ "$TASK" == "report2volume" ]] && ARGS+=(
    --report-format "${R2V_REPORT_FORMAT:-findings_impression_meta}"
    --checkpoint "$ADAPTER"
    --base-checkpoint "$UNET_CHECKPOINT"
    --num-inference-steps "${R2V_STEPS:-30}"
    --report-guidance-scale "${R2V_REPORT_GUIDANCE:-4.0}"
    --modality-guidance-scale "${R2V_MODALITY_GUIDANCE:-10.0}"
)
[[ "$TASK" == "generation" ]] && ARGS+=(--base-checkpoint "$UNET_CHECKPOINT")
[[ -n "${R2V_N_PER_BUCKET:-}" ]] && ARGS+=(--n-per-bucket "$R2V_N_PER_BUCKET")
# R2V_CASE_SELECTION_SEED draws the N per bucket at random instead of taking the first N. The
# prefix is a convenience sample and measurably skewed (see the flag's help); this is how you get a
# representative estimate. It changes run_id, so a seeded run cannot be mistaken for a prefix run.
[[ -n "${R2V_CASE_SELECTION_SEED:-}" ]] && ARGS+=(--case-selection-seed "$R2V_CASE_SELECTION_SEED")
# R2V_N_TOTAL drops per-bucket balancing: draw this many cases from the whole split, so each bucket
# appears at its population frequency (4.5%-16.3%) instead of 1/10th. That is the mixture training
# saw and the one the challenge scores. Mutually exclusive with R2V_N_PER_BUCKET -- the CLI refuses
# both -- so a caller setting this must not also set that.
[[ -n "${R2V_N_TOTAL:-}" ]] && ARGS+=(--n-total "$R2V_N_TOTAL")
[[ -n "${R2V_FRECHET_BATCH:-}" ]] && ARGS+=(--frechet-batch-size "$R2V_FRECHET_BATCH")
# Keep the generated volumes so a later metric does not need a second 4-hour sampling run.
# R2V_SAVE_VOLUMES=1 puts them next to the results; give a path to put them elsewhere.
if [[ "${R2V_SAVE_VOLUMES:-0}" == "1" ]]; then
    ARGS+=(--save-volumes "$OUT/volumes")
elif [[ -n "${R2V_SAVE_VOLUMES:-}" && "${R2V_SAVE_VOLUMES}" != "0" ]]; then
    ARGS+=(--save-volumes "$R2V_SAVE_VOLUMES")
fi
[[ -n "${R2V_SAVE_VOLUMES_DTYPE:-}" ]] && ARGS+=(--save-volumes-dtype "$R2V_SAVE_VOLUMES_DTYPE")
# For the cross-format ablation only: A/B/C trained on the two *_meta formats, so the two
# non-meta ones are out of distribution and `assert_report_format_matches` refuses them by
# default. That refusal is the safety net -- override it only when the OOD degradation IS the
# measurement, which is exactly what run_sweep.sh's format stage is doing.
[[ "${R2V_ALLOW_FORMAT_MISMATCH:-0}" == "1" ]] && ARGS+=(--allow-report-format-mismatch)
# Fail on a missing encoder zoo now, not after the queue wait.
[[ "$TASK" != "report2volume" ]] || preflight "$PRETRAINED_DIR"
[[ -n "${R2V_TRAIN_WORLD_SIZE:-}" ]] && ARGS+=(--train-world-size "$R2V_TRAIN_WORLD_SIZE")

# The blinded classifier is passed only when it exists. Without it the report_consistency group is
# recorded unavailable WITH A REASON -- never faked, never silently dropped.
if [[ -f "$REPORT_CLASSIFIER" ]]; then
    ARGS+=(--report-classifier "$REPORT_CLASSIFIER")
    echo "blinded classifier: $REPORT_CLASSIFIER"
else
    echo "NOTE: $REPORT_CLASSIFIER not found -- report_consistency will be reported unavailable."
    echo "      Fit it with: sbatch slurm/14_train_report_classifier.sbatch"
fi

WANDB_MODE_ARG="${R2V_WANDB:-disabled}"
ARGS+=(--wandb-mode "$WANDB_MODE_ARG")
if [[ "$WANDB_MODE_ARG" != "disabled" ]]; then
    ARGS+=(--wandb-project "${R2V_WANDB_PROJECT:-mr-rate-r2v-eval}"
           --wandb-group "${R2V_WANDB_GROUP:-mr-rate-$TASK}"
           --wandb-name "${R2V_WANDB_NAME:-$TAG}"
           --wandb-panels "${R2V_WANDB_PANELS:-6}")
    [[ -n "${R2V_WANDB_ENTITY:-}" ]] && ARGS+=(--wandb-entity "$R2V_WANDB_ENTITY")
    # Panels embed patient report text, so this must never point at a public project.
    [[ "${R2V_WANDB_REPORTS:-0}" == "1" ]] && ARGS+=(--wandb-log-reports)
    export WANDB_DIR="$OUT"
    mkdir -p "$OUT"
    APPTAINER_ARGS+=(--env "WANDB_DIR=$OUT" --env "WANDB__SERVICE_WAIT=300")
    [[ -n "${WANDB_API_KEY:-}" ]] && APPTAINER_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
fi

if [[ "$WANDB_MODE_ARG" == "online" ]]; then
    # Fail before the GPU work rather than after it. An unreachable W&B degrades to silence (see
    # `setup_proxy`), so the run would burn its whole walltime and log nothing. 401/404 both mean
    # "reached the server". Probed with `requests` -- the library wandb itself uses -- inside the
    # container, so this tests the exact proxy handling the real call will get.
    echo "=== W&B online: probing api.wandb.ai through the proxy ==="
    code="$(apptainer exec "${APPTAINER_ARGS[@]}" "$SIF_IMAGE_TEXT" python3 -c \
        'import requests,sys
try: sys.stdout.write(str(requests.get("https://api.wandb.ai/", timeout=30).status_code))
except Exception as e: sys.stdout.write("000"); sys.stderr.write(f"{type(e).__name__}: {e}\n")' || echo 000)"
    echo "    api.wandb.ai -> HTTP $code"
    if [[ "$code" == "000" ]]; then
        echo "W&B unreachable from inside the container (proxy=$PROXY_URL). Fix the proxy, or use" >&2
        echo "R2V_WANDB=offline and 'wandb sync $OUT/wandb' afterwards." >&2
        exit 1
    fi
    [[ -n "${WANDB_API_KEY:-}" || -f "$HOME/.netrc" ]] || {
        echo "W&B online needs credentials: export WANDB_API_KEY, or 'wandb login' once." >&2; exit 1; }
fi

echo "=== evaluating: task=$TASK tag=$TAG split=${R2V_SPLIT:-test} ==="
echo "    n-per-bucket: ${R2V_N_PER_BUCKET:-<entire split>}"
echo "    n-total     : ${R2V_N_TOTAL:-<per-bucket balanced>}"
echo "    selection   : ${R2V_CASE_SELECTION_SEED:+random, seed $R2V_CASE_SELECTION_SEED}${R2V_CASE_SELECTION_SEED:-first-N prefix}"
echo "    results     : $OUT"

# run_py_text, not run_py: report2volume needs the text-encoder stack, and using one image for all
# three tasks keeps the numbers produced under one environment.
run_py_text mrrate_r2v.cli.evaluate "${ARGS[@]}"

echo
echo "=== acceptance gate (slurm/SUCCESS_CRITERIA.md) ==="
# Never fatal: the metrics are already written and a checker that cannot run must not make a
# completed evaluation look like a failed one. Read its verdict before reading any number.
PYTHONPATH="$CP_ROOT" "$HOST_PYTHON" "$CP_ROOT/slurm/check_run.py" --results "$OUT" || true

echo
echo "=== headline CSVs ==="
head -20 "$OUT/metrics_summary.csv" 2>/dev/null || true
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS 2>&1 || true
