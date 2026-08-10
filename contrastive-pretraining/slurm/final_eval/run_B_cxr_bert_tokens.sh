#!/bin/bash -l
# Final evaluation, configuration B -- CXR-BERT, unpooled token sequence. (B, L<=512, 768).
#
# A with the pooling removed, so the cross-attention has real keys to attend to. Its adapter is
# rebuilt at inference through the kind="tokens" path -- which did not exist before 2026-08-09 and
# silently fell back to RadBERT. Nothing about this arm's numbers is readable from a run made
# before that fix.
source "$(dirname "${BASH_SOURCE[0]}")/_final_eval_common.sh"
submit_final_eval B B_cxr_bert_tokens
