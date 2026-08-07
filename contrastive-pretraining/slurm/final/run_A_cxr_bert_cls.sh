#!/bin/bash -l
# Final run, configuration A -- CXR-BERT, CLS token only. Conditioning tensor (B, 1, 768).
#
# The pooled-conditioning baseline. Softmax over a single key is constant, so to_q/to_k get no
# gradient and the report can only apply a per-channel bias uniformly at every voxel -- 33.8% of the
# trainable parameters are inert. Kept in the lineup because CXR-BERT's CLS is the one staged
# encoder trained by a sentence-level objective, and because CTFlow placed second on the 2026 CT
# leaderboard doing exactly this.
source "$(dirname "${BASH_SOURCE[0]}")/_final_common.sh"
submit_final_run A r2v_final_A_cxr_bert_cls
