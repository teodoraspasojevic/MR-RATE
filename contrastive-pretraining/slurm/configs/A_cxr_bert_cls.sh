# Configuration A -- CXR-BERT, CLS token only.
#
# Conditioning tensor: (B, 1, 768). `last_hidden_state[:, 0, :]`, the raw CLS state -- not a pooler
# output (the checkpoint has none) and not the CLIP-projected CLS (the `bert_shim` loader drops
# `cls_projection_head`, so this is the pre-projection state).
#
# Why CLS here and mean for RadBERT: `EncoderSpec.pooling` records the choice the text-encoder study
# was run under, and it is "cls" for this checkpoint alone -- CXR-BERT is the one staged encoder
# whose CLS was trained by a sentence-level (CLIP) objective rather than only by MLM.
#
# Why this encoder: best single-encoder scores in docs/TEXT_ENCODERS.md §6.3 at this format --
# pathology probe 0.9796, negation_delta 0.327 (2x RadBERT's 0.152), and 1.56% truncation at 512
# tokens versus RadBERT's 9.20%. Its weakness is domain: chest X-ray anatomy vocabulary is wrong
# for brain.
R2V_CONDITIONING=cxr_bert_cls
# Two formats = sample one per training sample (see mrrate_r2v/textenc/formats.py). The challenge's
# report layout is unknown, so training on one fixed section order teaches the model that order and
# nothing at submission time can detect that it flipped. Both carry the same
# [MODALITY]/[PLANE]/[SPACING] prefix; validation is pinned to the first name.
# Set R2V_REPORT_FORMAT=impression_findings for the single-format run this replaced.
R2V_REPORT_FORMAT="${R2V_REPORT_FORMAT:-findings_impression_meta,impression_findings_meta}"
R2V_MAX_REPORT_TOKENS=512
# Measured on one H200 (140 GB) at the 256^3 fallback bucket, bf16: batch 8 is peak throughput
# (2.573 vol/s, 85-87 GB reserved = 61% of the card). Throughput *falls* at 12 and 16, and 24 OOMs.
# See docs/TEXT_ENCODERS.md section 9.7. Drop to 4/16 if a validation pass ever OOMs (~1% slower).
R2V_BATCH_SIZE="${R2V_BATCH_SIZE:-8}"
R2V_GRAD_ACCUM="${R2V_GRAD_ACCUM:-8}"
