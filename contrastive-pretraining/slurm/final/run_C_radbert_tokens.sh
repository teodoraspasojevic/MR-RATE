#!/bin/bash -l
# Final run, configuration C -- RadBERT, unpooled token sequence. Conditioning tensor (B, L, 768).
#
# The closest domain match (4M radiology reports). RadBERT is where dropping the pooling is least
# arguable: it is a RobertaForMaskedLM with no pooler and no sentence-level objective, so there was
# never a trained summary vector to pool to. It truncates 9.2% of reports at 512 tokens, and with
# the token axis kept, which tokens survive truncation is visible to the attention rather than
# averaged away.
source "$(dirname "${BASH_SOURCE[0]}")/_final_common.sh"
submit_final_run C r2v_final_C_radbert_tokens
