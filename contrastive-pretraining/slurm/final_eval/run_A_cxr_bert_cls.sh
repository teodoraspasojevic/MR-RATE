#!/bin/bash -l
# Final evaluation, configuration A -- CXR-BERT, CLS token only. Conditioning tensor (B, 1, 768).
#
# The pooled baseline, and the only arm with a real condition-sensitivity signal in training
# (ssim_advantage +0.0418 at N=512, against ~0 for B and C). Whether that survives on the test
# cohort, and whether the blinded classifier agrees with the report, is what this run measures.
source "$(dirname "${BASH_SOURCE[0]}")/_final_eval_common.sh"
submit_final_eval A A_cxr_bert_cls
