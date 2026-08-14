# Configuration E -- Report2CT-style three-encoder fusion, plus an acquisition token.
#
# Conditioning tensor: (B, 3, 2560).
#   sequence axis: [0] = findings, [1] = impression, [2] = acquisition metadata as text
#                  ("[MODALITY] T1w [PLANE] AXIAL [SPACING] 0.94 0.94 1.09")
#   feature axis:  [0:1024] MedEmbed-large
#                  [1024:1792] Bio_ClinicalBERT
#                  [1792:2560] CXR-BERT                    (1024 + 768 + 768 = 2560)
#
# **D with the one thing D structurally could not have.** A, B and C put
# [MODALITY]/[PLANE]/[SPACING] at the head of their joined string, so the acquisition metadata
# reaches the text encoder as well as reaching the UNet numerically (class_labels, spacing_tensor).
# D encodes each section on its own tokenizer and never joins them, so it has nowhere to put that
# prefix -- see the note at the end of D_report2ct_style.sh, which this configuration answers.
# Here the prefix is its own conditioning token, encoded by the same three encoders through the
# same masked-mean pooling, so nothing about D's two tokens changes: findings stays at sequence
# index 0 and impression at 1, and the third token is appended.
#
# What is genuinely new versus D, and what is not:
#   new    the report branch can attend to modality/plane/spacing, and different voxels can attend
#          to the metadata token and the findings token by different amounts.
#   not    the information itself. Modality already reaches the UNet as a class label and spacing
#          as `spacing_tensor`; plane is implied by the bucket geometry. So the honest expectation
#          is a modest gain from *where* the conditioning enters (a cross-attention key the adapter
#          can learn to weight per voxel), not from new information -- and a null result is a real
#          possible outcome, not a wiring bug. E vs D is the clean measurement of that, because
#          they differ in exactly one token.
#
# The acquisition text is composed by the Dataset from the manifest row and the *resolved* target
# spacing (mrrate_r2v/data/dataset.py), never parsed out of the report, and is the identical string
# `meta_prefix_for` gives A/B/C. It is never empty, so unlike impression (absent for 8.9% of
# studies) its mask entry is always True.
R2V_CONDITIONING=report2ct_style_meta
# No R2V_REPORT_FORMAT, exactly as for D: sections are encoded separately and never joined into one
# string, so there is no section order to be robust to and no joined text to format.
R2V_REPORT_FORMAT=
R2V_MAX_REPORT_TOKENS=512
# Same as D. The third section is one extra forward pass per encoder over a ~12-token string, which
# is noise next to the frozen VAE encode that dominates ~90% of a step -- but it is unmeasured, so
# run `SMOKE=1 slurm/final/run_E_report2ct_style_meta.sh` once before committing a long job, the
# same rule D's batch 8 was set under.
R2V_BATCH_SIZE="${R2V_BATCH_SIZE:-8}"
R2V_GRAD_ACCUM="${R2V_GRAD_ACCUM:-8}"
