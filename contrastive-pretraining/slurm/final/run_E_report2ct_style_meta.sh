#!/bin/bash -l
# Final run, configuration E -- Report2CT-style fusion + an acquisition token. (B, 3, 2560).
#
#   sequence axis: [0] findings, [1] impression, [2] "[MODALITY] .. [PLANE] .. [SPACING] .."
#   feature axis:  MedEmbed-large 1024 | Bio_ClinicalBERT 768 | CXR-BERT 768
#
# Like D it takes no --report-format: three tokenizers, no joined string. Unlike D it does not pay
# for that with the metadata -- the prefix A, B and C get at the head of their text is E's third
# conditioning token, composed by the Dataset from the manifest row and the resolved target spacing.
# D and E differ in that one token and nothing else, so the pair is the measurement of whether the
# acquisition metadata is worth a cross-attention key.
source "$(dirname "${BASH_SOURCE[0]}")/_final_common.sh"
submit_final_run E r2v_final_E_report2ct_style_meta
