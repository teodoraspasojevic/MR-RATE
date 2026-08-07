# Configuration D -- Report2CT-style three-encoder fusion.
#
# Conditioning tensor: (B, 2, 2560).
#   sequence axis: [0] = findings, [1] = impression        (an absent section is masked out)
#   feature axis:  [0:1024] MedEmbed-large
#                  [1024:1792] Bio_ClinicalBERT
#                  [1792:2560] CXR-BERT                    (1024 + 768 + 768 = 2560)
# Each encoder is mask-mean-pooled on its own tokenisation, then the three pooled vectors are
# concatenated. There is no token-level correspondence between three different tokenizers to align,
# which is why pooling comes first.
#
# Verified against Report2CT @ 7b483a856ef159cfd0dada249b110d8f8eebf502 (`vlm3d_inference.ipynb`
# cell 0, `diff_model_train_vlm3D_2560_multi_text.py:275-297`).
#
# **Two documented differences from Report2CT** (see docs/TEXT_ENCODERS.md §9):
#  1. Its third encoder is `medicalai/ClinicalBERT`, a 6-layer DistilBERT. This uses the staged
#     `emilyalsentzer/Bio_ClinicalBERT`, a 12-layer BERT-base. Same width, different checkpoint --
#     hence "style", not "reproduction".
#  2. It fed the raw 2560-vector to a UNet built with cross_attention_dim=2560 and
#     with_conditioning=True, which here would destroy NVIDIA's pretrained MR-Brain weight loading.
#     The existing ContextProjection maps 2560 -> --cross-attention-dim instead.
#
# No R2V_REPORT_FORMAT: sections are encoded separately and never joined into one string.
#
# So the order-agnostic spec A and B use does not apply here -- and does not need to: there is no
# section *order* to be robust to, each section is its own attention token. The trade is the other
# way round, though: this configuration also has nowhere to put the [MODALITY]/[PLANE]/[SPACING]
# prefix, so modality/plane reach the UNet only as `class_labels`/`spacing_tensor` and never as text.
R2V_CONDITIONING=report2ct_style
R2V_REPORT_FORMAT=
R2V_MAX_REPORT_TOKENS=512
# Measured, not guessed: three resident encoders (~555M frozen parameters) and a 2560-wide projection
# cost only +1.7 GB reserved and 1.8% of a step versus A/B, because ~90% of a step is the frozen VAE
# encode. So the batch size is the same 8, at 87.03 GB reserved. Throughput falls at 12 and 16; 24
# OOMs. See docs/TEXT_ENCODERS.md section 9.7.
R2V_BATCH_SIZE="${R2V_BATCH_SIZE:-8}"
R2V_GRAD_ACCUM="${R2V_GRAD_ACCUM:-8}"
