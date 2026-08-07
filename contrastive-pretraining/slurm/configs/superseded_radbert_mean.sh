# Configuration B -- RadBERT, mask-aware mean pooling.
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
R2V_REPORT_FORMAT=impression_findings
R2V_MAX_REPORT_TOKENS=512
# Measured on one H200 (140 GB) at the 256^3 fallback bucket, bf16: batch 8 is peak throughput
# (2.573 vol/s, 85-87 GB reserved = 61% of the card). Throughput *falls* at 12 and 16, and 24 OOMs.
# See docs/TEXT_ENCODERS.md section 9.7. Drop to 4/16 if a validation pass ever OOMs (~1% slower).
R2V_BATCH_SIZE="${R2V_BATCH_SIZE:-8}"
R2V_GRAD_ACCUM="${R2V_GRAD_ACCUM:-8}"
