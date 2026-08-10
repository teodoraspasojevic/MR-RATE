#!/bin/bash -l
# Final evaluation, configuration C -- RadBERT, unpooled token sequence. (B, L<=512, 768).
#
# The one arm whose training job FAILED (714499, host-RAM OOM in its final N=512 validation), so it
# has no adapter_last.pt and is evaluated from adapter_step0004200.pt -- 4,200 optimizer steps
# against 4,493 for the other three. Quote that caveat wherever this row appears.
source "$(dirname "${BASH_SOURCE[0]}")/_final_eval_common.sh"
submit_final_eval C C_radbert_tokens
