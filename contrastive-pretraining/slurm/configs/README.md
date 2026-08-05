# The three supported conditioning configurations

One file per configuration. Each is a shell fragment sourced by
[`11_train_conditioning.sbatch`](../11_train_conditioning.sbatch), so the flags live in exactly one
place and a run cannot silently disagree with what is documented.

| file | `--conditioning` | encoder(s) | conditioning tensor | report format |
|---|---|---|---|---|
| [`A_cxr_bert_cls.sh`](A_cxr_bert_cls.sh) | `cxr_bert_cls` | `microsoft/BiomedVLP-CXR-BERT-specialized` | `(B, 1, 768)` | `impression_findings` |
| [`B_radbert_mean.sh`](B_radbert_mean.sh) | `radbert_mean` | `zzxslp/RadBERT-RoBERTa-4m` | `(B, 1, 768)` | `impression_findings` |
| [`C_report2ct_style.sh`](C_report2ct_style.sh) | `report2ct_style` | MedEmbed-large + Bio_ClinicalBERT + CXR-BERT | `(B, 2, 2560)` | none — sections encoded separately |

```bash
sbatch --export=ALL,R2V_CONFIG=A slurm/11_train_conditioning.sbatch      # 4-step smoke run
sbatch --export=ALL,R2V_CONFIG=C,R2V_MAX_STEPS=0 --time=24:00:00 \
       --gres=gpu:h200:4 slurm/11_train_conditioning.sbatch              # real 4-GPU run
```

`R2V_MAX_STEPS=0` means "no step cap" (run `--epochs` to completion). `#SBATCH --export=NONE` in
the job script means a plain `VAR=x sbatch ...` does **not** reach the job — always pass overrides
through `--export=ALL,...` as above.

**Configuration C is a Report2CT-*style* fusion, not a reproduction.** Report2CT's third encoder is
`medicalai/ClinicalBERT` (a 6-layer DistilBERT); this substitutes the staged
`emilyalsentzer/Bio_ClinicalBERT` (12-layer BERT-base). Both are 768-wide so the fused width is
2560 either way, but they are different checkpoints. See `docs/TEXT_ENCODERS.md` §9.
