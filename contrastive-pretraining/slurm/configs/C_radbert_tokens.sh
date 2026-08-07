# Configuration C -- RadBERT, unpooled token sequence.
#
# Conditioning tensor: (B, L, 768), L = the batch's longest report after tokenisation (<= 512).
# Padding is masked, not attended: `MaskedCrossAttention` honours `context_mask`, which is the whole
# reason the adapter does not use monai's `CrossAttentionBlock`.
#
# **This is the old pooled-RadBERT arm with the pooling removed, and it exists because that
# pooling made the cross-attention a no-op.** Softmax over one key is identically 1 for every
# query, so with L=1 the query and key projections cannot influence the output and get zero
# gradient, and the adapter collapses into a per-channel bias applied uniformly at every voxel.
# Measured on the real model: L=1 gives `to_q` gradients of 1.2e-12 (roundoff) against 1.2e-05
# for `to_v`, and an injected residual that is spatially constant to 1.8e-05.
# 2,729,344 of the 8,080,000 trainable parameters (33.8%) are inert.
#
# RadBERT is the encoder where dropping the pooling is least arguable: `zzxslp/RadBERT-RoBERTa-4m`
# is a `RobertaForMaskedLM` with no pooler and no sentence-level objective, so there is no trained
# summary vector to pool *to* -- the masked mean was an unweighted average chosen for lack of an
# alternative. Configuration A's CXR-BERT CLS at least was trained by a CLIP objective, which is why
# its unpooled variant (B) is an A/B rather than a straight replacement.
R2V_CONDITIONING=radbert_tokens
# Same order-agnostic spec as A and B. It matters more here, not less: RadBERT truncates 9.2% of reports
# at 512 tokens, and with the token axis kept, *which* tokens survive truncation is now visible to
# the attention rather than averaged away.
R2V_REPORT_FORMAT="${R2V_REPORT_FORMAT:-findings_impression_meta,impression_findings_meta}"
R2V_MAX_REPORT_TOKENS=512
# Halved from B's 8. The cross-attention now has up to 512 keys instead of 1, so both activation
# memory and the attention cost per adapter grow with report length. Raise it back if the measured
# peak leaves room -- there is nothing special about 4.
R2V_BATCH_SIZE="${R2V_BATCH_SIZE:-4}"
R2V_GRAD_ACCUM="${R2V_GRAD_ACCUM:-16}"
