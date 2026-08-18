#!/bin/bash -l
# Two MORE epochs for configuration D, resumed from the completed 2-epoch run.
#
# D won the arm comparison, and its validation curve had not turned over when the run ended --
# N=128 SSIM 0.3394 -> 0.3518 and FVD 22.70 -> 20.53 across the eight passes, monotone in trend,
# no plateau. So the question this job answers is simply "does it keep going", and the four
# checkpoints it writes (one per epoch, plus best_* and last) are what answers it.
#
#   slurm/final/run_D_continue.sh
#
# Everything not set below comes from _final_common.sh, so the scale, batch, validation metrics and
# real-vs-real reference are the same as the run being extended. What differs is stated here.

set -euo pipefail

# **Captured before the source, and this is not decoration.** `_final_common.sh` sets
# `R2V_LR="${R2V_LR:-1e-3}"` itself, so by the time this file runs, `R2V_LR` is always non-empty and
# a `${R2V_LR:-3e-4}` below silently keeps the sweep's 1e-3 -- i.e. submits a peak three times the
# intended one, with nothing in the output saying so. Caught by dry-running the submit.
R2V_LR_OVERRIDE="${R2V_LR:-}"

source "$(dirname "${BASH_SOURCE[0]}")/_final_common.sh"

# ---------------------------------------------------------------- what we resume
#
# `adapter_last.pt`, not one of the `best_*` files: it is the end of epoch 2 with optimizer moments,
# RNG and counters intact, which is the only checkpoint that continues the run rather than restarting
# from a cherry-picked point. (The `best_*` files carry full state too, but "best" was decided partly
# by the single N=512 pass, so resuming one would silently rewind training by an unknown amount.)
SOURCE_RUN="${SOURCE_RUN:-$WORKSPACE/runs/r2v_final_D_report2ct_style}"
RESUME_FROM="${RESUME_FROM:-$SOURCE_RUN/adapter_last.pt}"

# A NEW run directory. Resuming into the source directory would let `--keep-last-n 3` prune the
# original run's step checkpoints and would overwrite its `train_summary.json` -- destroying the
# baseline this job exists to be compared against.
TAG="${TAG:-r2v_final_D_report2ct_style_cont}"

# ---------------------------------------------------------------- learning rate
#
# **`restart`, not `continue`, and this is the whole reason the flag exists.** PolynomialLR reaches
# exactly 0 at its horizon and stays there; the checkpoint stores
# `{total_iters: 4493, last_epoch: 4493, _last_lr: [0.0]}` and an optimizer whose param-group `lr`
# is 0.0. Continuing that schedule would train at LR 0 for 22 hours and write back the weights it
# started with. The trainer now refuses it outright, so this is belt and braces.
#
# **3e-4**, not the swept 1e-3. The sweep's optimum is the right *peak for a run starting from a
# zero-init projection*; this one starts from an adapter that has already annealed to convergence,
# and re-entering at full peak spends its first epoch undoing the last one. 3e-4 with power 2 over
# 4493 optimizer steps has a mean LR of ~1e-4 -- enough to move the adapter, gentle enough not to
# throw away what the first run bought. Set R2V_LR in the environment to override.
R2V_LR="${R2V_LR_OVERRIDE:-3e-4}"
RESUME_LR_SCHEDULE=restart

# ---------------------------------------------------------------- budget
#
# **Two, not three.** Measured cost is 10.85 h of training per epoch, so three epochs is ~32.6 h
# against h200's hard 24 h ceiling (`sinfo`: TIMELIMIT 1-00:00:00). Two fit the same way the
# original 22.17 h run did. A third epoch is a second job resumed from this one's `adapter_epoch004`
# -- worth submitting only if these two still show the curve rising.
EPOCHS=2

# One checkpoint per epoch, which is what makes the third-epoch decision possible and what a
# preemption would fall back to. `adapter_epoch<N>.pt` is numbered absolutely (003, 004) and is
# never touched by `--keep-last-n`, unlike the `adapter_step*.pt` files.
SAVE_EVERY_EPOCHS=1

echo "=== continuing configuration D ==="
echo "    resume from : $RESUME_FROM"
echo "    into        : $WORKSPACE/runs/$TAG"
echo "    lr          : $R2V_LR (PolynomialLR restarted over $EPOCHS epochs)"
echo

submit_final_run D "$TAG"
