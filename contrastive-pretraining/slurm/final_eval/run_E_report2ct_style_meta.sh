#!/bin/bash -l
# Final evaluation, configuration E -- Report2CT-style fusion + an acquisition token. (B, 3, 2560).
#
# Reads the same per-section text D does, plus the `acquisition` section the Dataset composes from
# each case's own modality, plane and resolved spacing -- so the evaluation conditions E on the
# metadata of the volume it is being scored against, exactly as training did. Nothing extra has to
# be passed for that: `cli.evaluate` hands the sampler the sample's `report_sections_text`, and
# `R2VDatasetConfig.conditioning_sections` carries `acquisition` by default.
#
# Like D it ignores --report-format, and like D it is not capacity-matched to A/B/C (11.75M
# trainable against 8.08M) -- E and D have the same trainable count as each other.
source "$(dirname "${BASH_SOURCE[0]}")/_final_eval_common.sh"
submit_final_eval E E_report2ct_style_meta
