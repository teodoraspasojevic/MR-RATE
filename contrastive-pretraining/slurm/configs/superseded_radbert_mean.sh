# SUPERSEDED -- RadBERT, mask-aware mean pooling. Out of the lettered lineup; use C instead.
#
# Its single mean-pooled token makes the cross-attention degenerate (softmax over one key is
# constant, so to_q/to_k get no gradient and the report can only add a per-channel bias). Unlike
# configuration A, there is not even a trained summary vector to justify the pooling: RadBERT is a
# RobertaForMaskedLM with no sentence-level objective. Kept runnable via
# `R2V_CONFIG=superseded_radbert_mean` so adapters already trained under it still load.
#
# Conditioning tensor: (B, 1, 768). The mean is over real tokens only; padding is divided out, not
# averaged in.
#
# **Mean, not CLS, and this is deliberate.** `zzxslp/RadBERT-RoBERTa-4m` is a `RobertaForMaskedLM`:
# it has no pooler and no sentence-level pretraining objective, so its `<s>` state was never trained
# to summarise a sequence. The text-encoder study scored it on pooled means. Set
# R2V_TEXT_POOLING=cls for an explicit ablation and expect it to underperform for that reason rather
# than for a domain reason.
#
# This is the baseline: the closest domain match (4M radiology reports) and the encoder every earlier
# run in this repository used, so it is what a new configuration has to beat.
R2V_CONDITIONING=radbert_mean
# Order-agnostic + metadata prefix, same reasoning as configuration A. RadBERT truncates 9.2% of
# reports at 512 tokens, so which order survives truncation matters more here than for CXR-BERT --
# which is exactly why the model must not be allowed to depend on one.
R2V_REPORT_FORMAT="${R2V_REPORT_FORMAT:-findings_impression_meta,impression_findings_meta}"
R2V_MAX_REPORT_TOKENS=512
# Measured on one H200 (140 GB) at the 256^3 fallback bucket, bf16: batch 8 is peak throughput
# (2.573 vol/s, 85-87 GB reserved = 61% of the card). Throughput *falls* at 12 and 16, and 24 OOMs.
# See docs/TEXT_ENCODERS.md section 9.7. Drop to 4/16 if a validation pass ever OOMs (~1% slower).
R2V_BATCH_SIZE="${R2V_BATCH_SIZE:-8}"
R2V_GRAD_ACCUM="${R2V_GRAD_ACCUM:-8}"
