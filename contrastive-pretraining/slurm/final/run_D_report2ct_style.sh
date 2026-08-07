#!/bin/bash -l
# Final run, configuration D -- Report2CT-style three-encoder fusion. Conditioning tensor (B, 2, 2560).
#
#   sequence axis: [0] findings, [1] impression      (an absent section is masked out)
#   feature axis:  MedEmbed-large 1024 | Bio_ClinicalBERT 768 | CXR-BERT 768
#
# The one configuration that does not take --report-format, and cannot: three tokenizers with no
# token-level correspondence, so each section is pooled separately and never joined into a string.
# The consequence to remember when reading its numbers is that modality and plane reach the UNet
# only as class_labels/spacing_tensor, never as text -- there is nowhere to put the metadata prefix
# the other three runs get.
source "$(dirname "${BASH_SOURCE[0]}")/_final_common.sh"
submit_final_run D r2v_final_D_report2ct_style
