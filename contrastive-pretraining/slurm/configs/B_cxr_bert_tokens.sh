# Configuration B -- CXR-BERT, unpooled token sequence.
#
# Conditioning tensor: (B, L, 768), L = the batch's longest report after tokenisation (<= 512).
#
# **Configuration A with the pooling removed, and unlike C this is a genuine A/B, not a fix.**
# CXR-BERT's CLS *was* trained to summarise: the checkpoint's CLIP objective supervised it at the
# sentence level, which is why the text-encoder study scored it best of the single encoders and why
# A uses `cls` where every other encoder uses `mean`. So B trades a supervised summary vector for a
# non-degenerate attention that can localise, and which of those wins is an empirical question.
#
# What is *not* in question is that A's cross-attention does nothing with its single token: softmax
# over one key is constant, so `to_q`/`to_k` get no gradient and the report can only apply a
# per-channel bias, uniformly at every voxel. A is therefore best read as a pooled-conditioning
# baseline. Notably CTFlow (arXiv 2508.12900, `WongJiayi/CTFlow`) does exactly the same thing --
# `caption_channels: 768`, `model_max_length: 1`, a single L2-normalised CT-CLIP vector into
# cross-attention -- and placed second on the 2026 CT leaderboard on CLIP score, so the degenerate
# form is evidently not fatal. It is still strictly less expressive than what the same parameters
# could do.
R2V_CONDITIONING=cxr_bert_tokens
R2V_REPORT_FORMAT="${R2V_REPORT_FORMAT:-findings_impression_meta,impression_findings_meta}"
# CXR-BERT truncates only 1.56% of MR-RATE reports at 512 tokens (RadBERT: 9.20%), so the token
# sequence it produces is a near-complete report rather than a truncated one.
R2V_MAX_REPORT_TOKENS=512
R2V_BATCH_SIZE="${R2V_BATCH_SIZE:-4}"
R2V_GRAD_ACCUM="${R2V_GRAD_ACCUM:-16}"
