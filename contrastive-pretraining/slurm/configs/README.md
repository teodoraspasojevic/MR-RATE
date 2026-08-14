# The supported conditioning configurations

One file per configuration. Each is a shell fragment sourced by
[`11_train_conditioning.sbatch`](../11_train_conditioning.sbatch), so the flags live in exactly one
place and a run cannot silently disagree with what is documented.

| file | `--conditioning` | encoder(s) | conditioning tensor | report format |
|---|---|---|---|---|
| [`A_cxr_bert_cls.sh`](A_cxr_bert_cls.sh) | `cxr_bert_cls` | `microsoft/BiomedVLP-CXR-BERT-specialized` | `(B, 1, 768)` | order-agnostic + meta |
| [`B_cxr_bert_tokens.sh`](B_cxr_bert_tokens.sh) | `cxr_bert_tokens` | `microsoft/BiomedVLP-CXR-BERT-specialized` | `(B, n, 768)` + `(B, n)` mask | order-agnostic + meta |
| [`C_radbert_tokens.sh`](C_radbert_tokens.sh) | `radbert_tokens` | `zzxslp/RadBERT-RoBERTa-4m` | `(B, n, 768)` + `(B, n)` mask | order-agnostic + meta |
| [`D_report2ct_style.sh`](D_report2ct_style.sh) | `report2ct_style` | MedEmbed-large + Bio_ClinicalBERT + CXR-BERT | `(B, 2, 2560)` | none — sections encoded separately |
| [`E_report2ct_style_meta.sh`](E_report2ct_style_meta.sh) | `report2ct_style_meta` | MedEmbed-large + Bio_ClinicalBERT + CXR-BERT | `(B, 3, 2560)` | none — sections encoded separately |

**`n` is variable, not fixed.** The tokenizer pads each batch to *its own* longest report, capped at
`--max-report-tokens` (512), so `n` differs from batch to batch — measured 133–512 for CXR-BERT and
178–512 for RadBERT on real MR-RATE reports. The padding is carried in the `(B, n)` mask and
excluded from the attention softmax by `MaskedCrossAttention`, so a sample's conditioning never
depends on which reports it was batched with. Nothing resamples `n` to a fixed length; the
`ContextProjection` maps `(B, n, 768) -> (B, n, 512)` and preserves the sequence axis.

**Why A is a baseline rather than the recommendation.** A single conditioning token makes the
cross-attention degenerate: softmax over one key is identically 1 for every query, so `to_q` and
`to_k` cannot influence the output and receive no gradient, and the report can only add a
per-channel bias, uniformly at every voxel. Measured after 40 real training steps, gradient norm of
`to_q`: **1.2e-12 at A (roundoff) versus 4.4e-07 at B and 8.0e-07 at C**. B and C cost nothing
extra for it — 29 s per 40 steps and 13.8 GiB peak, identical to A. A is kept because CXR-BERT's
CLS *was* trained to summarise (CLIP objective) and because it is the exact form CTFlow uses, which
makes it the honest control for B.

`superseded_radbert_mean.sh` is the old pooled-RadBERT arm. It is out of the lettered lineup (its
single mean-pooled token has A's collapse without A's excuse — RadBERT has no sentence-level
objective, so there was no trained summary vector to pool to) but it is kept runnable via
`R2V_CONFIG=superseded_radbert_mean` so existing checkpoints stay loadable.

```bash
sbatch --export=ALL,R2V_CONFIG=A slurm/11_train_conditioning.sbatch      # 4-step smoke run
sbatch --export=ALL,R2V_CONFIG=D,R2V_MAX_STEPS=0 --time=24:00:00 \
       --gres=gpu:h200:4 slurm/11_train_conditioning.sbatch              # real 4-GPU run
```

`R2V_MAX_STEPS=0` means "no step cap" (run `--epochs` to completion). `#SBATCH --export=NONE` in
the job script means a plain `VAR=x sbatch ...` does **not** reach the job — always pass overrides
through `--export=ALL,...` as above.

**E is D plus one token, and that is the whole difference.** D encodes each section with its own
tokenizer and never joins them into a string, so it is the one arm with nowhere to put the
`[MODALITY]/[PLANE]/[SPACING]` prefix A, B and C carry — modality and plane reach it only as
`class_labels`/`spacing_tensor`. E appends that prefix as a third conditioning token, encoded by
the same three encoders through the same masked-mean pooling, leaving findings at sequence index 0
and impression at 1. So D vs E measures exactly one thing: whether the acquisition metadata is
worth a cross-attention key of its own. The information is not new to the model (modality is
already a class label, spacing already a tensor), only its entry point is, so a null result is a
possible honest outcome. The text itself is composed by the Dataset from the manifest row and the
resolved target spacing, never parsed from the report, and is byte-identical to what
`meta_prefix_for` gives A/B/C.

**Configuration D is a Report2CT-*style* fusion, not a reproduction.** Report2CT's third encoder is
`medicalai/ClinicalBERT` (a 6-layer DistilBERT); this substitutes the staged
`emilyalsentzer/Bio_ClinicalBERT` (12-layer BERT-base). Both are 768-wide so the fused width is
2560 either way, but they are different checkpoints. See `docs/TEXT_ENCODERS.md` §9.
