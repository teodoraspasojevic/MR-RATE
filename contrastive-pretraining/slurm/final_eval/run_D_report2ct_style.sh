#!/bin/bash -l
# Final evaluation, configuration D -- Report2CT-style three-encoder fusion. (B, 2, 2560).
#
# The only arm that reads the cohort's report_sections.json: it encodes findings and impression as
# separate cross-attention tokens and cannot recover them from the joined string. A cohort without
# that file makes cli.predict_r2v refuse up front rather than condition D on one token instead of
# two. Also the only arm that is not capacity-matched -- 11.75M trainable against 8.08M.
source "$(dirname "${BASH_SOURCE[0]}")/_final_eval_common.sh"
submit_final_eval D D_report2ct_style
