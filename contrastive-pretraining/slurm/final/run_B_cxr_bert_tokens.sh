#!/bin/bash -l
# Final run, configuration B -- CXR-BERT, unpooled token sequence. Conditioning tensor (B, L, 768).
#
# Configuration A with the pooling removed, and a genuine A/B rather than a fix: it trades a
# supervised summary vector for a non-degenerate attention that can localise. Best single-encoder
# scores in docs/TEXT_ENCODERS.md section 6.3, and only 1.56% of MR-RATE reports truncate at 512
# tokens, so the sequence it produces is a near-complete report.
source "$(dirname "${BASH_SOURCE[0]}")/_final_common.sh"
submit_final_run B r2v_final_B_cxr_bert_tokens
