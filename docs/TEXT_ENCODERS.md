# Text encoders and report formats for report-to-volume conditioning

Everything about turning an MR-RATE radiology report into the tensor the conditioned denoiser
attends over: what the reports actually look like, which encoders are staged, which report format
to use, where acquisition metadata should come from, and how the choice was measured.

Companion document: [`R2V.md`](R2V.md) for the generation/evaluation pipeline this feeds, and
[`mrrate_r2v/textenc/README.md`](../contrastive-pretraining/mrrate_r2v/textenc/README.md) for the
production API this document's findings are baked into.

**The selection benchmark (`textbench/` and its three driving CLIs) has been removed** -- the
study below is a completed, one-off report, not a runnable tool anymore. What's kept is the
methodology and the measured conclusions that the current encoder/format choice rests on.

Confidence key: **VERIFIED** (measured here, or quoted from a primary source that was fetched) /
**INFERRED** (well supported but not stated outright) / **ASSUMED** (a provisional choice).

---

## 1. What was added

```
contrastive-pretraining/mrrate_r2v/
├── textenc/                    PRODUCTION: what the trainer and sampler use
│   ├── formats.py              named report formats            (no torch)
│   ├── encoders.py             HFTextEncoder + ENCODER_SPECS
│   ├── fusion.py               ProjectedConcatFusion (used by SectionedFusionEmbedder)
│   ├── conditioning.py         the three supported configurations         (§9)
│   └── README.md
└── cli/
    └── download_text_encoders.py   stage checkpoints (idempotent, pinned revisions)

mrrate_r2v/validation.py            step-based validation: FID + alignment proxy      (§9.6)
slurm/configs/{A,B,C}_*.sh          one file per configuration                        (§9.5)
slurm/train_conditioning.sbatch  trains any of the three
```

The selection study itself ran through `textbench/` (`corpus.py`, `analysis.py`, `negation.py`,
`embed.py`, `tasks.py`, `runner.py`) and three CLIs (`analyze_reports.py`, `embed_reports.py`,
`eval_text_encoders.py`) -- all now removed. The unrelated GPU timing tool, `benchmark_h200.py`
(§9.7), has also been removed since its batch-size recommendations are already baked into
`slurm/configs/*.sh`. §2-§4 and §9.7 below are the two studies' findings; neither can be re-run.

Changed, minimally: `mrrate_r2v/text.py` (the factory resolves zoo names; `encode_reports` is the
one dispatch seam; `rebuild_embedder` is the one inference-side rebuild), `mrrate_r2v/data/dataset.py`
(an optional `report_format`, plus `report_sections_text` alongside the existing `report_text`),
`mrrate_r2v/training.py` (optimizer-step intervals, a real validation hook, W&B, best/retention
checkpoints), `mrrate_r2v/sampling.py` (per-section text reaches the sampler),
`mrrate_r2v/models/adapter.py` (conditioning-compatibility check, `optimizer_step`, `best_metrics`),
`mrrate_r2v/eval/figures.py` + `eval/wandb_logging.py` (the interactive panel),
`mrrate_r2v/cli/train_r2v.py` + `cli/generate_r2v.py`, `slurm/_common.sh`.

**Every default is unchanged**: leave `--conditioning` and `report_format` unset with
`--text-encoder radbert` and the pipeline behaves exactly as before, down to the `cohort_id`. Two
behaviours *did* change, both because they were broken: `--validate-every-steps` and
`--save-every-steps` now count **optimizer** steps rather than micro-steps (§9.6), and `--num-gpus >
1` without `torchrun` now fails instead of silently training N independent single-GPU models (§9.5).

---

## 2. The reports (measured on all 98,334 studies)

Produced with (`cli.analyze_reports`, since removed):

```bash
cd contrastive-pretraining
python -m mrrate_r2v.cli.analyze_reports build-corpus \
    --shards-root /hnvme/workspace/y100dc19-MR-Rate-raw \
    --out /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/report_analysis/reports_all.jsonl
python -m mrrate_r2v.cli.analyze_reports analyze \
    --corpus       /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/report_analysis/reports_all.jsonl \
    --manifest-csv /hnvme/workspace/y100dc19-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv \
    --out          /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/report_analysis/analysis.json
```

### 2.1 Scale and pairing

| split | studies | series | non-empty reports | label sets | clinical | technique | findings | impression |
|---|---|---|---|---|---|---|---|---|
| train | 88,985 | 638,345 | 88,698 | 88,985 | 42,888 | 86,818 | 88,582 | 81,095 |
| val | 3,781 | 27,003 | 3,767 | 3,781 | 1,714 | 3,666 | 3,764 | 3,442 |
| test | 5,568 | 39,906 | 5,554 | 5,568 | 2,582 | 5,388 | 5,550 | 5,050 |

**One report is shared by a median of 6–7 series** (mean 7.17, p95 12, max 83). This is the single
most consequential structural fact: reports are **study-level**, volumes are **series-level**. A
report describes a whole exam, not the T1w axial in front of you. The pipeline already handles it
by conditioning on `(report, modality class, spacing)` jointly and training with
`series_selection="all"` so one report is paired with each of its series — that contrast is what
stops the report adapter from absorbing modality.

### 2.2 Health

| check | count | share |
|---|---|---|
| `report.json` missing | 0 | 0% |
| present but empty | 315 | 0.32% |
| under 10 words | 51 | 0.05% |
| exact duplicates (whitespace-normalised) | 127 reports in 47 groups | 0.13% |
| identical text appearing in more than one split | 12 distinct texts | — |
| `clinical_information` absent | 51,150 | 52.0% |
| `technique` absent | 2,462 | 2.5% |
| `findings` absent | 438 | 0.45% |
| `impression` absent | 8,747 | 8.9% |

The 12 cross-split texts are duplicated *text*, not duplicated patients — the release's own
`patient_split_isolation` check passes with 0 violations, and the repeated strings are short
degenerate ones (the most repeated is a bare anonymisation token appearing 16×). No action needed
beyond knowing they exist.

### 2.3 Lengths

Words, non-empty only:

| field | median | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| raw | 191 | 255 | 327 | 378 | 505 | 14,287 |
| clinical_information | 2 | 5 | 7 | 9 | 14 | 134 |
| technique | 21 | 31 | 36 | 44 | 60 | 261 |
| findings | 143 | 190 | 241 | 269 | 344 | 14,262 |
| impression | 19 | 36 | 60 | 80 | 135 | 623 |
| findings + impression | 161 | 221 | 288 | 335 | 453 | 14,262 |

**The longest reports, inspected rather than just measured.** The 124,685-character outlier
(14,287 words, 842 non-empty lines) has only **three** distinct section heads — it is one exam's
`Findings` section repeated at length, not a multi-exam document. The next four (7.4k–8.2k
characters) are genuine **multi-region** reports: one carries the heads `Clinical information`,
`Technique`, `Findings`, `Impression`, *plus* `THORACIC VERTEBRA MRI EXAMINATION`, `HIP MRI` and a
second `IMPRESSION`. So the long tail is dominated by studies where several body regions were
reported together — and only the cranial part of such a report is relevant to a brain volume.
That is an argument for `findings_impression` over `raw` beyond token cost: `raw` carries other
regions' findings that no brain volume can express.

### 2.4 Structure

`findings` and `impression` are **extractive, not rewritten** (VERIFIED, whitespace-normalised
substring match against the raw report):

| section | verbatim in raw | else >90% token overlap |
|---|---|---|
| clinical_information | 99.1% | 0.5% |
| technique | 96.0% | 3.5% |
| findings | 89.0% | 11.0% |
| impression | 59.2% | 39.0% |

The residual is re-organisation, not new content: the structuring step inserts region sub-headings
(`Cranial:`, `Orbit:`) and normalises the `—` bullets the radiologist used, which is why
`impression` matches verbatim least often while still overlapping >90% of tokens in 39% more
cases. **No section contains content absent from the raw report**, so no format below risks
leaking LLM-invented text into conditioning.

Headings in the raw text are **mostly but not fully standardised**: 845 distinct line-initial
`Word:` heads exist, but four account for almost all of them — `impression` 89.8%, `technique`
88.8%, `findings` 83.4%, `clinical information` 46.9%. The tail is real (`comparison` 6.2%,
`sequences used in the examination` 3.7%, and 800+ single-digit-percent variants including
`patient name`, `protocol no`, `mri device`). 2,793 distinct first lines exist; the top one
(`BRAIN MRI`) covers 29.6%. **Conclusion: parse nothing from the raw text — the released
`findings`/`impression` fields have already done that job, correctly, 89–99% of the time.**

**Where the diagnostic information lives: both, and neither alone is sufficient.** `findings`
carries the detail (median 143 words) and `impression` the conclusion (median 19 words), and
impression is *absent* for 8.9% of studies. Dropping findings loses localisation and incidental
pathology; dropping impression loses the radiologist's synthesis for the 91% that have one.

### 2.5 Acquisition information in the text vs. in the metadata

Share of reports whose text matches each probe (regex, generous by design — a generous pattern
that still misses is strong evidence of real absence):

| property | in raw | in technique | available as structured metadata? |
|---|---|---|---|
| plane word (axial/sagittal/coronal) | 92.1% | 91.7% | **yes** — `manifest.plane` |
| FLAIR mentioned | 90.7% | 89.1% | **yes** — `manifest.modality` |
| DWI/ADC mentioned | 90.2% | 87.2% | yes (not a target bucket) |
| T2 mentioned | 85.9% | 82.6% | **yes** |
| T1 mentioned | 84.6% | 83.7% | **yes** |
| SWI/GRE mentioned | 63.2% | 58.4% | **yes** |
| contrast agent | 62.6% | 57.9% | no |
| "3-plane"/multiplanar | 33.8% | 33.8% | — |
| slice thickness in mm | **3.1%** | 2.5% | **yes** — NIfTI header |
| field strength | **1.9%** | 1.9% | no |
| matrix / dimensions | **0.01%** | 0.00% | **yes** — NIfTI header |
| voxel or pixel spacing | **0.00%** | 0.00% | **yes** — NIfTI header |

This is the decisive table for §5. Reports name **which sequences and planes were acquired** but
they never state **which one this volume is**, and they essentially never state geometry: voxel
spacing appears in **0 of 98,019** reports and a matrix size in 11.

### 2.6 Positive vs. negated content

1,534,925 sentences from findings+impression, classified by explicit lexical rules:

| category | sentences | share |
|---|---|---|
| normal statement (`normal`, `unremarkable`, `within normal limits`) | 701,802 | **45.7%** |
| positive assertion | 383,019 | 25.0% |
| explicit negation (`no`, `not`, `without`, `absent`, `negative for`) | 264,069 | **17.2%** |
| other/descriptive | 186,035 | 12.1% |
| hedged (overlaps the above) | 25,966 | 1.7% |

Per report, the negative-or-normal fraction is **median 0.667, mean 0.651** (p10 0.375, p90 0.889).
3.7% of reports are entirely negative/normal; only 0.24% contain no negative or normal sentence.

**Method and limits, stated plainly.** The classifier is lexical and scope-blind: a sentence like
"no acute infarct but chronic gliosis is present" is counted once, as negated, though it asserts a
finding. It cannot resolve double negation and it does not attach negations to entities. It is
reported because it is fully explainable and identical for every encoder — not as a gold standard.
Read it as an order-of-magnitude statement: **roughly two thirds of MR-RATE's report content is
the absence of disease**, which is exactly the content an encoder that ignores negation destroys.

The label side agrees: 44.3% of studies have **no** positive pathology label, and the mean is 1.12
positives per study over 37 categories.

### 2.7 Token lengths per encoder tokenizer

`python -m mrrate_r2v.cli.analyze_reports tokens --corpus … --sample 20000 --out token_lengths.json`
(seeded sample of 20,000 studies; percentiles are within ±0.3% of the full corpus).

Share of reports truncated at a **512-token** budget:

| encoder | raw | findings | impression | findings_impression | full_structured |
|---|---|---|---|---|---|
| `cxr_bert` | 3.05% | 0.18% | 0.00% | **1.56%** | 2.89% |
| `bioclinical_mbert` / `modernbert` | 13.99% | 1.23% | 0.00% | **6.36%** | 11.99% |
| `medembed_large` / `medembed_small` / `bge_base` | 14.55% | 2.00% | 0.01% | **8.20%** | 13.94% |
| `radbert` | 19.83% | 2.24% | 0.01% | **9.20%** | 16.52% |
| `bio_clinicalbert` | 18.30% | 2.98% | 0.01% | **10.50%** | 17.54% |

At their **native** context, `bioclinical_mbert` and `modernbert` (8192) truncate **0.00%** of
reports in every format. Every other encoder is capped at 512 and pays the column above.

Two findings worth calling out:

- **`raw` costs 1.5–2× the tokens of `findings_impression` and truncates 14–20% of reports.**
  Combined with §2.3's multi-region finding, `raw` is the worst of both worlds.
- **CXR-BERT's radiology-specific vocabulary is ~30% more token-efficient on this text** than
  general BERT wordpiece (mean 225 vs 311 tokens for `findings_impression`). A real advantage that
  its chest-X-ray domain then partly gives back.

---

## 3. Report formats

Implemented in `textenc/formats.py`; select with `R2VDatasetConfig.report_format` or
`--report-format`.

| format | keeps | drops | truncation @512 (radbert) | risk |
|---|---|---|---|---|
| `raw` | everything, headings and all | nothing | 19.8% | other body regions' findings; 800+ heading variants; anonymisation tokens |
| `findings` | the detail | the synthesis | 2.2% | loses the impression for the 91% that have one |
| `impression` | the synthesis | the detail | 0.01% | **empty for 8.9% of studies**; loses localisation and incidental findings |
| `findings_impression` **(default)** | both, marked | indication, protocol | 9.2% | ~9% truncation at 512 |
| `impression_findings` | both, marked, conclusion first | same | 9.2% (same tokens) | none beyond the above — but what is *lost* to truncation is now detail, not conclusion |
| `clinical_findings_impression` | + the indication | protocol | 9.6% | indication is present for only 48% of studies, so the model sees an inconsistent prefix |
| `full_structured` | all four sections | nothing released | 16.5% | protocol text is **train/test-inconsistent** — see below |
| `findings_impression_meta` | + `[MODALITY] … [PLANE] …` | as `findings_impression` | 9.2% | **commits you to having that metadata at challenge inference** |

**Negation survives every format that keeps findings** — asserted by
`tests/test_textenc_formats.py::test_negation_survives_every_format_that_keeps_findings`. No
format performs cleaning, sentence sampling or summarisation, so none can drop a negation cue.
(Contrast the contrastive pipeline's `MRReportDataset`, which randomly subsamples sentences per
`__getitem__` — a contrastive-training trick that would be actively harmful here.)

**Why `full_structured` is a trap.** The `technique` section describes the *protocol of the whole
exam* ("3-plane T1-weighted, FLAIR; axial T2-weighted, B-FFE, SWI and DWI"). Conditioning on it
teaches the model to read acquisition parameters out of free text — and at challenge inference the
report may be reworded, the protocol may differ, or the section may be absent (2.5% already are).
It also inflates tokens by 60% for content that structured conditioning already supplies exactly.
Use `--report-format full_structured` only for a deliberate ablation.

**Why `impression_findings` is worth taking seriously.** It contains *exactly* the same content as
`findings_impression` and costs exactly the same tokens; the two differ only in what survives a
512-token cut. At 9.2% truncation, one report in eleven currently loses its tail — and under
`findings_impression` the tail is the impression, i.e. the radiologist's conclusion. The reorder
is free insurance for any 512-context encoder. For an 8192-context encoder it is a no-op.

### Prior art

The 2025 VLM3D **CT-track winner, Report2CT** (rank 1, Task 4 "Text-Conditional CT Generation";
[arXiv:2509.14780](https://arxiv.org/abs/2509.14780)) used **findings + impression, encoded
separately** and concatenated — no raw report, no clinical/technique sections. Direct quote from
the paper: *"we propose to process the findings and impression sections of each radiology report
separately using three distinct pretrained medical text encoders."* That is independent support
for `findings_impression` being the right content envelope, on the sibling task of this exact
challenge. (VERIFIED — the PDF was fetched and read.)

---

## 4. The encoders

Staged with pinned revisions in `/hnvme/workspace/y100dc19-nvidia-mri-brain/pretrained`:

```bash
python -m mrrate_r2v.cli.download_text_encoders --list       # what exists and what is staged
python -m mrrate_r2v.cli.download_text_encoders --all        # idempotent; skips what is present
```

### 4.1 Comparison table

Architecture/dimension/context/licence/parameters are **VERIFIED** from each repo's
`config.json` + HF API. Training-data claims are quoted from the official model card; anything
not stated on it is marked so.

| | `radbert` | `bioclinical_mbert` | `medembed_large` | `medembed_small` | `bio_clinicalbert` | `cxr_bert` | `modernbert` | `bge_base` |
|---|---|---|---|---|---|---|---|---|
| checkpoint | `zzxslp/RadBERT-RoBERTa-4m` | `thomas-sounack/BioClinical-ModernBERT-base` | `abhinand/MedEmbed-large-v0.1` | `abhinand/MedEmbed-small-v0.1` | `emilyalsentzer/Bio_ClinicalBERT` | `microsoft/BiomedVLP-CXR-BERT-specialized` | `answerdotai/ModernBERT-base` | `BAAI/bge-base-en-v1.5` |
| revision (pinned) | `b8b7433023c4` | `c3648aa87af9` | `963121bfb9c6` | `40a5850d046c` | `d5892b39a4ad` | `5157bdba1437` | `8949b909ec90` | `a5beb1e3e68b` |
| architecture | RoBERTa-base | ModernBERT-base | BERT-large | BERT-small | BERT-base | BERT-base (custom `cxr-bert` type) | ModernBERT-base | BERT-base |
| parameters | ~125M | 149.7M | 335.1M | **33.4M** | ~110M | 109.6M | 149.7M | 109.5M |
| hidden dim | 768 | 768 | **1024** | **384** | 768 | 768 | 768 | 768 |
| max context | 512 | **8192** | 512 | 512 | 512 | 512 | **8192** | 512 |
| tokenizer | RoBERTa BPE (50,265) | ModernBERT BPE (50,368) | BERT wordpiece (30,522) | BERT wordpiece | BERT wordpiece (28,996) | BERT wordpiece, **CXR-specific** (30,522) | ModernBERT BPE | BERT wordpiece |
| token output | (B, L, 768) | (B, L, 768) | (B, L, 1024) | (B, L, 384) | (B, L, 768) | (B, L, 768) | (B, L, 768) | (B, L, 768) |
| pooled available | mask-aware mean | mask-aware mean | mask-aware mean | mask-aware mean | mask-aware mean | **`[CLS]`, CLIP-trained but pre-projection** ¹ | mask-aware mean | mask-aware mean |
| language | English | English | English | English | English | English | English | English |
| biomedical/clinical data | **radiology reports** | PubMed/PMC 50.7B tok + 20 clinical corpora 2.8B tok | medical retrieval fine-tune of BGE-large | as large | MIMIC-III notes (~880M words) | PubMed + MIMIC-III + MIMIC-CXR | none | none |
| **MRI reports included?** | INFERRED yes — 4M all-modality VA radiology reports; the card does not enumerate modalities | **VERIFIED yes** | not stated | not stated | INFERRED — MIMIC-III `NOTEEVENTS` includes radiology reports; modality not enumerated | no (chest X-ray) | no | no |
| **Brain MRI reports included?** | not stated | **VERIFIED yes** — the card's corpus table lists "Brain MRI Stroke, Korea, Radiology Reports, Neurology, 2,603 samples, 0.2M tokens" | not stated | not stated | not stated | no | no | no |
| licence | Apache-2.0 (per `UCSD-VA-health` mirror) | MIT | Apache-2.0 | Apache-2.0 | MIT | MIT (research-use intent stated on card) | Apache-2.0 | MIT |
| access | public, ungated | public, ungated | public, ungated | public, ungated | public, ungated; **no safetensors on the hub** | public, ungated; **needs `trust_remote_code` unless shimmed** | public, ungated | public, ungated |
| truncation @512, `findings_impression` | 9.20% | **6.36% (0.00% at native 8192)** | 8.20% | 8.20% | 10.50% | **1.56%** | 6.36% (0.00% native) | 8.20% |
| GPU memory, bf16 inference, batch 64 @512 | ~1.5 GB | ~1.6 GB | ~3.5 GB | **~0.6 GB** | ~1.4 GB | ~1.4 GB | ~1.6 GB | ~1.4 GB |
| strength for R2V conditioning | closest domain match: radiology report register, negation phrasing, abbreviations | only encoder with documented brain-MRI reports; no truncation at all; modern, fast, MIT | strongest semantic-similarity geometry; winner-precedented | 3× cheaper than anything else, 384-wide context tensor | winner-precedented; broad clinical vocabulary | most token-efficient on radiology text; the only trained (not derived) pooled vector | isolates domain adaptation from architecture | isolates the medical fine-tune from the base retriever |
| weakness | 512 context; anisotropic embedding space; chest/CT-heavy US corpus, no brain-MRI evidence | brain-MRI data is 0.2M of 53.5B tokens (0.0004%) — presence ≠ specialisation | 3× the compute and 2× the width of the alternatives for one extra bit of geometry | smallest capacity; 384 dims may under-serve cross-attention | **pretrained at sequence length 128** — 300–500-token reports are outside its regime; ICU notes ≠ radiology reports | chest X-ray domain; anatomy vocabulary is wrong for brain | no medical exposure | no medical exposure |

¹ **Precisely what `cxr_bert`'s pooled vector is** (VERIFIED at runtime): `last_hidden_state[:, 0, :]`,
the raw 768-wide CLS state. CXR-BERT's CLIP objective acted on that CLS *through* a
`cls_projection_head`, and the `bert_shim` loader drops that head — the load report lists
`cls_projection_head.{dense_to_hidden,dense_to_output,LayerNorm}.{weight,bias}` as UNEXPECTED. So the
CLS was shaped by a sentence-level objective (unlike every other staged checkpoint, whose CLS saw
only MLM), but it is **not** the 128-d CLIP embedding. An earlier version of this row said "trained
CLIP `[CLS]`", which overstated it. Report2CT's mean-pooled path is unaffected by the dropped head,
so the fusion in §9 is faithful there.

Also considered and **not** staged, with reasons: `microsoft/BiomedVLP-BioViL-T` (CXR, same family
as `cxr_bert`, adds nothing); `UFNLP/gatortron-base` (345M, clinical, but 512 context and no
documented radiology-report focus beyond MIMIC); `yikuan8/Clinical-Longformer` (4096 context, but
MR-RATE's p99 is 765 tokens — a Longformer solves a problem this corpus does not have, at
Longformer cost); `ncbi/MedCPT-Query-Encoder` (query-side retrieval encoder, mismatched objective);
`google/flan-t5-base` encoder (the text-to-image convention, but no medical exposure and a third
tokenizer family — `modernbert`/`bge_base` already serve as general-domain controls);
`Simonlee711/Clinical_ModernBERT` (overlaps `bioclinical_mbert`, weaker on the published
benchmarks in the latter's own comparison table).

### 4.2 Selection criteria

Eight staged, chosen to answer specific questions rather than to accumulate models:

1. **Strong single-encoder candidates** — `radbert` (best domain register), `bioclinical_mbert`
   (only documented brain-MRI exposure + no truncation), `medembed_large` (best retrieval
   geometry, winner-precedented).
2. **A genuinely lightweight one** — `medembed_small`, 33M parameters and 384 dims.
3. **The 2025 CT-winner's trio, near-reproducible** — `cxr_bert` + `bio_clinicalbert` +
   `medembed_large` is Report2CT's set *up to one substituted checkpoint*, so its multi-encoder
   claim can be tested on MR rather than assumed to transfer. **Correction (2026-08-05):** an
   earlier version of this line claimed the set was "exactly" Report2CT's. It is not. Report2CT's
   own source names `medicalai/ClinicalBERT` — a 6-layer **DistilBERT**, ~66M parameters — not
   `emilyalsentzer/Bio_ClinicalBERT` (12-layer BERT-base, ~110M). Both are 768-wide, so the fused
   width is 2560 either way, but they are different checkpoints trained on different corpora. See
   §9.
4. **Two controls that make the results interpretable** — `modernbert` is architecture-matched to
   `bioclinical_mbert` (their difference *is* the clinical adaptation) and `bge_base` is
   `medembed_large`'s own base family (their difference *is* the medical fine-tune). Without
   these, "medical encoder wins" is unfalsifiable.

No model above 400M parameters is staged. Total staged size ≈ 4.0 GB.

---

## 5. Where modality, plane, spacing and shape should come from

**Recommendation, per field:**

| value | best source | how to condition | why |
|---|---|---|---|
| **modality** (T1w/T2w/FLAIR/SWI) | manifest (`classified_modality`, DICOM-derived) | **categorical embedding, already implemented** — `conditioning.ModalityEncoder` → NVIDIA's `class_labels` | The frozen NV-Generate-MR-Brain UNet takes an integer modality class natively. The report names *which sequences the exam contained* (T1 84.6%, FLAIR 90.7%, SWI 63.2%) but never which one a given volume is — and one report covers a median of 6–7 series across several modalities. Text cannot disambiguate this even in principle. |
| **acquisition plane** | manifest (`acquisition_plane`) | **implicitly, via the per-bucket geometry** — the bucket's shape/spacing already encodes the plane | 92.1% of reports contain a plane word, but again for the exam, not the volume. Making it explicit text is optional (`findings_impression_meta`); making it structured is free, because the target grid already differs per plane. |
| **voxel spacing** | NIfTI header → `manifest.native_spacing_mm` → the bucket's `GeometrySpec` | **numerical conditioning, already implemented** — NVIDIA's `spacing_tensor` | Present in **0.00%** of reports. Report2CT did the same thing: *"the model receives a voxel-spacing embedding, which is concatenated to the text embeddings before projection."* |
| **volume shape** | the cohort's `GeometrySpec` (FOV/32-multiple rule) | **not conditioned — it is the output grid**, chosen by the bucket | Present in 0.01% of reports. Shape is a property of the request, not of the report. |
| **contrast state** | nothing reliable | **omit** | 62.6% of reports mention a contrast agent, but there is no released structured field, and the report's mention is exam-level. `MRReportToVolumeDataset` already returns `contrast_state="unknown"` and `conditioning.py` deliberately never selects NVIDIA's `mri_t1c` class. |
| **field strength, vendor** | nothing | **omit** | 1.9% and 0.2% of reports; no structured field. |

**Never parse geometry out of text.** It is not there (0 of 98,019 reports state voxel spacing),
and reliable structured metadata exists for every case.

**The challenge-inference constraint, and why this recommendation is robust to it.** The VLM3D MR
task's input schema is **still unpublished**: as of 2026-08-04 the platform's own API returns
`"data_schema": {}` and `"submissions_enabled": false` for the `mr-volume-generation` phase, and
the public task page says only *"Input: Free-text radiology report or text prompt"* with *"DICOM
metadata provided where available"* (UNKNOWN, hedged by the organisers themselves). So:

- **Modality and spacing must be *accepted* when supplied and *defaulted* when not.** The pipeline
  already does this: `ModalityEncoder.id_for` falls back to NVIDIA's `unknown` class rather than
  guessing, and every bucket has a published FOV that supplies a spacing.
- **Do not choose a report format that depends on metadata.** `findings_impression_meta` bakes
  `[MODALITY] T1w [PLANE] AXIAL` into the *text*, so a test-time case without that metadata gets a
  systematically different string from anything the model trained on — a train/test inconsistency
  that a separate categorical embedding does not have (a missing category is just the null class).
  This is why `findings_impression_meta` is implemented, measured, and **not recommended as the
  default**. The metadata-dependent formats are explicitly declared in
  `formats.METADATA_DEPENDENT_FORMATS` and asserted by a test, so the commitment is visible.

**Leakage check.** No recommended source is ground truth that would be unavailable at inference:
modality/plane/spacing are properties of the *request* ("generate an axial T1w at 1mm"), not of
the held-out target volume. The one thing that *would* be leakage — deriving the target's shape
from the real series — is already excluded by the cohort contract, which freezes the grid before
any model runs.

---

## 6. Evaluation

### 6.1 Design

Frozen encoders only. Each encoder × each report format is embedded once and cached, then scored.

```
train split (seeded 20,000 of 88,985)  ──► probes fit here
test  split (all 5,554 reports)        ──► every number reported here
```

Splits are MR-RATE's own and are patient-isolated (0 violations in the release's
`patient_split_isolation` check), so no probe can see a patient twice. Nothing is ever fit on
`test`. Every encoder sees the identical studies, identical labels and identical splits.

Probes run on `concat(mean-pooled, max-pooled)` token states. Mean-only would flatter mean-pooling
encoders; the denoiser attends over *tokens*, so a pooled probe is a proxy either way — stated as
a limitation in §8, not hidden.

| metric | question | label source | strength |
|---|---|---|---|
| `pathology_probe_auroc` | is pathology content linearly recoverable? macro AUROC over labels with ≥1% train prevalence | `labels.json` | **WEAK** |
| `bucket_probe_auroc` | is (modality, plane) recoverable? macro AUROC over the 10 buckets | manifest / DICOM | independent of the text |
| `negation_delta` | how far does flipping a finding's polarity move the embedding, relative to a topic change? | rule-constructed minimal pairs | construction |
| `nn_jaccard_delta` | nearest-neighbour label agreement minus the random-pair baseline | `labels.json` | **WEAK** |
| `sim_spearman` | Spearman(cosine, label Jaccard) over 200k random test pairs | `labels.json` | **WEAK** |
| `embed_dim`, `truncated_pct`, `reports_per_second` | cost | measured | — |

**Justification against the challenge's own scoring.** `MRI_Report_to_Volume.md` names two metric
families: a feature-based FID-like distributional metric, and *"Blinded Classifier Consistency —
whether a classifier trained on real data assigns consistent clinical labels to generated volumes
matching the conditioning report."* `pathology_probe_auroc` is the direct frozen-encoder upper
bound on the second: a label the conditioning embedding cannot linearly express is one the
denoiser has no way to render. `negation_delta` is the same question restricted to the failure
mode that dominates this corpus (§2.6: ~2/3 of content is absence of disease).
`bucket_probe_auroc` covers the acquisition axis, which the FID-like metric is sensitive to
because T1w and FLAIR occupy very different feature distributions. Report2CT evaluated with FID +
CLIP-score; the CLIP-score analogue for a *frozen text encoder with no image tower* is exactly the
retrieval/similarity pair (`nn_jaccard_delta`, `sim_spearman`).

**The weak-label caveat, stated once and carried in `summary.json`.** `labels.json` was produced by
an LLM reading the same report the encoder reads. High probe AUROC proves the embedding retained
what the labeller also extracted from that text — **not** that the label is clinically correct.
These numbers rank encoders against each other validly (identical labels for all) and are invalid
as absolute clinical accuracy. `bucket_probe_auroc` (DICOM-derived) and `negation_delta`
(rule-constructed) do not share this caveat.

**Why `negation_delta` replaced a negation AUROC.** A topic-grouped linear probe decodes polarity
at AUROC > 0.999 for *every* encoder — the cue is a literal token, so decodability discriminates
nothing. `negation_delta` measures geometry instead, on **centred** embeddings, because the shared
mean component differs enormously between checkpoints and an uncentred measure would rank
anisotropy rather than negation sensitivity (asserted by
`test_negation_delta_is_scale_free_across_anisotropic_encoders`).

### 6.2 Running it

**No longer runnable** -- `embed_reports.py`, `eval_text_encoders.py` and the `textbench` package
they drove have been removed, along with the `08_embed_reports.sbatch`/`09_eval_text_encoders.sbatch`
scripts. This subsection is kept only as a record of how the §6.3 results below were produced.

### 6.3 Results

*(Filled in from `metrics_matrix.csv` — see §6.4 for the current run's status.)*

---

## 7. Troubleshooting

Applies to anything still live (`textenc`, `download_text_encoders.py`); rows specific to the
removed selection benchmark have been dropped.

| symptom | cause | fix |
|---|---|---|
| `FileNotFoundError: text encoder '<x>' checkpoint directory not found` | not staged | `python -m mrrate_r2v.cli.download_text_encoders --encoders <x>` |
| `ValueError: max_length=N exceeds what '<x>' supports` | asked for more tokens than the checkpoint has positions | lower `--max-report-tokens`, or use `bioclinical_mbert` (8192) |
| `contains custom code which must be executed` | loading `cxr_bert` through plain `AutoModel` | use `build_encoder("cxr_bert")` — the spec's `bert_shim` loader avoids `trust_remote_code` |
| `Cannot use torch.load ... CVE-2025-32434` | a `.bin`-only checkpoint under torch < 2.6 | `download_text_encoders` converts `bio_clinicalbert` on stage; `text.ensure_local_safetensors` handles `radbert` |
| `AssocGrpGRES` pending forever | GPU job submitted without `--qos=mq_health` | the sbatch scripts set it; keep it if you copy them |

---

## 9. The three supported conditioning configurations

Everything below is production: what the trainer, the validator and the sampler actually run.
Implemented in [`textenc/conditioning.py`](../contrastive-pretraining/mrrate_r2v/textenc/conditioning.py);
selected with one flag.

### 9.1 At a glance

| | A | B | C |
|---|---|---|---|
| `--conditioning` | `cxr_bert_cls` | `radbert_mean` | `report2ct_style` |
| encoder(s) | `microsoft/BiomedVLP-CXR-BERT-specialized` | `zzxslp/RadBERT-RoBERTa-4m` | MedEmbed-large + Bio_ClinicalBERT + CXR-BERT |
| pinned revision | `5157bdba1437` | `b8b7433023c4` | `963121bfb9c6` / `d5892b39a4ad` / `5157bdba1437` |
| architecture | `CXRBertModel` → stock BERT, 12L | `RobertaForMaskedLM`, 12L | BERT-large 24L / BERT-base 12L / BERT 12L |
| parameters (frozen) | 109.6M | ~125M | 335.1M + ~110M + 109.6M ≈ 555M |
| hidden size | 768 | 768 | 1024 + 768 + 768 |
| max context | 512 | 512 (514−2) | 512 each |
| tokenizer | CXR-specific WordPiece (30,522) | RoBERTa BPE (50,265) | three independent tokenizers |
| pooling | **CLS** = `last_hidden_state[:,0,:]` | **masked mean** | **masked mean**, per encoder |
| **conditioning tensor** | **`(B, 1, 768)`** | **`(B, 1, 768)`** | **`(B, 2, 2560)`** |
| **attention mask** | **`(B, 1)`**, all True | **`(B, 1)`**, all True | **`(B, 2)`**, False for an absent section |
| report format | `impression_findings` | `impression_findings` | none — sections encoded separately |
| trainable text params | 0 (frozen default) | 0 | 0 |
| trainable adapter params | 8,080,000 | 8,080,000 | 11,753,600 |
| local checkpoint | `$PRETRAINED_DIR/BiomedVLP-CXR-BERT-specialized` | `.../RadBERT-RoBERTa-4m` | `.../MedEmbed-large-v0.1`, `.../Bio_ClinicalBERT`, `.../BiomedVLP-CXR-BERT-specialized` |
| licence / access | MIT, public ungated | Apache-2.0, public ungated | Apache-2.0 / MIT / MIT, all public ungated |

The adapter parameter count differs only because `ContextProjection` is built from
`embedder.output_dim` — 2560 → `cross_attention_dim` is a wider first Linear than 768 →
`cross_attention_dim`. **No text-encoder dimension is hardcoded anywhere in the generative model.**

### 9.2 Why CLS for A and mean for B

Both are the value already recorded in that encoder's `EncoderSpec.pooling`, i.e. the choice §6 was
measured under. They differ for a reason:

- **CXR-BERT's CLS was shaped by a sentence-level objective** (its CLIP text tower), so it is a
  trained summary vector. It is the *pre-projection* CLS — see footnote ¹ in §4.1.
- **RadBERT has no pooler and no sentence-level objective.** It is a `RobertaForMaskedLM`; its `<s>`
  state was never trained to summarise anything. Using it as the sole conditioning vector is a
  worse readout than the mean, and would confound an A-vs-B comparison with an untrained-readout
  artefact rather than a domain difference. `--text-pooling cls` forces it for an explicit ablation.

### 9.3 How Report2CT's fusion was reproduced — and where it differs

Verified from source, not from the paper: `github.com/sinaamirrajab/report2ct` @
`7b483a856ef159cfd0dada249b110d8f8eebf502`, files `vlm3d_inference.ipynb` (cell 0,
`encode_batch_multi`) and `src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py:275-297`.
Report2CT is itself built on MAISI, the same base family as this pipeline.

**Verified shape trace, B = 2** (the batched form from the train script; the notebook's
`squeeze(0).unsqueeze(0).unsqueeze(0)` is correct only for B = 1):

```
findings[2], impression[2]                        two lists of strings

per encoder, max_length=512, padding=True, truncation=True:
  MedEmbed-large-v0.1     (2, L₁, 1024)   L₁ ≤ 512  ─┐ three tokenizers ⇒ L₁ ≠ L₂ ≠ L₃.
  medicalai/ClinicalBERT  (2, L₂,  768)   L₂ ≤ 512   │ Never aligned — pooling collapses
  BiomedVLP-CXR-BERT      (2, L₃,  768)   L₃ ≤ 512  ─┘ the token axis first.

masked mean pool, each encoder independently (padding divided out, not averaged in):
  (2, L₁, 1024) → (2, 1024)
  (2, L₂,  768) → (2,  768)
  (2, L₃,  768) → (2,  768)

cat(dim=-1) in order [MedEmbed, ClinicalBERT, CXR-BERT]:
  → (2, 2560)          slices [0:1024] [1024:1792] [1792:2560]

twice, for findings and impression → each (2, 2560)
unsqueeze(1)                       → each (2, 1, 2560)
cat(dim=1)                         → (2, 2, 2560)      index 0 = findings, 1 = impression
```

Reproduced exactly: the per-encoder independent masked-mean pooling, the feature-axis concatenation
order, the 2560 width, the separate findings/impression encoding, the section order, the frozen
encoders, and the 512-token budget. **No LayerNorm and no feature normalisation** are applied to the
fused vector, matching the original (its `context * 1e2` scaling was tried and left commented out at
`diff_model_train_vlm3D_2560_multi_text.py:91-92`).

Three documented differences, each because the alternative is worse or impossible here:

| # | Report2CT | here | why |
|---|---|---|---|
| 1 | `medicalai/ClinicalBERT` (DistilBERT, 6L, ~66M) | `emilyalsentzer/Bio_ClinicalBERT` (BERT-base, 12L, ~110M) | The original is not staged. Both are 768-wide so the fused width is 2560 either way — but it is a different checkpoint, which is why this is called `report2ct_style` and never "Report2CT". Staging `medicalai/ClinicalBERT` (ungated, ~260 MB, rev `f7c7f65227cb311f`) would make it exact. |
| 2 | no conditioning mask; a missing section is encoded as `""` | absent section masked out via `(B, 2)` | MAISI's `SpatialTransformer` takes no mask, so Report2CT's absent impression contributes a real attention key. Impression is absent for **8.9%** of MR-RATE studies. With both sections empty the row is all-False, which `prepare_context` already maps to the learned null embedding. |
| 3 | raw 2560-vector into a UNet with `cross_attention_dim=2560, with_conditioning=True` | 2560 → `--cross-attention-dim` via the existing `ContextProjection` | `with_conditioning=True` makes MAISI swap its `SpatialAttentionBlock`s for `SpatialTransformer`s, which changes the module tree and **destroys NVIDIA's pretrained MR-Brain weight loading** (`models/report_conditioned_unet.py:263-268`). Report2CT could afford this; a frozen-base adapter cannot. |

Also worth knowing, because it explains a common misreading: `encode_batch_multi` **computes a
token-level path too** — pad every encoder to exactly 512 positions, zero-pad the feature axis up to
`max_dim=1024`, concatenate along tokens to `(B, 3×512, 1024)` — and then **discards it**
(`c_vec_f, _ = encode_batch_multi(finding)`). Report2CT's actual conditioning is the pooled path
only. Report2CT also caches all embeddings to disk (`<file>multi_2560.json`, holding
`findings_embeddings[2560]` and `impression_embeddings[2560]`), so its encoders never run during
training; here they run live, which the H200 measurements in §9.7 show costs 0.4–2.7% of a step.

### 9.4 Corrections to the shapes that were assumed before this was checked

| assumed | actual | note |
|---|---|---|
| `2048 = 1024 + 512 + 512` | **`2560 = 1024 + 768 + 768`** | CXR-BERT and ClinicalBERT are both **768**-wide, not 512. No staged encoder is 512-wide. |
| an intermediate `(B, 512, 2048)` | **no such tensor exists** in the used path | `padding=True` pads to the batch's longest sequence, not to 512. The literal 512-pad is in the discarded token path. |
| concatenate tokens, *then* mean-pool | **pool each encoder first, then concatenate** | Not equivalent: the three tokenizers give different token counts and different masks, so a joint mean over a padded 512 axis is a different number. |
| final `(B, 2, 2048)` | **`(B, 2, 2560)`** | The `(B, 2, …)` structure and the findings-first order were right. |

### 9.5 Running each configuration

Files: [`slurm/configs/`](../contrastive-pretraining/slurm/configs/README.md), one per
configuration, sourced by
[`slurm/train_conditioning.sbatch`](../contrastive-pretraining/slurm/train_conditioning.sbatch).

```bash
cd contrastive-pretraining

# CPU wiring check, no data, no GPU, seconds -- run this first after any change
python -m mrrate_r2v.cli.train_r2v --dry-run --max-steps 2 --device cpu \
    --conditioning cxr_bert_cls --max-report-tokens 64 --out /tmp/dry_A

# 4-step GPU smoke run, per configuration
sbatch --export=ALL,R2V_CONFIG=A slurm/train_conditioning.sbatch
sbatch --export=ALL,R2V_CONFIG=B slurm/train_conditioning.sbatch
sbatch --export=ALL,R2V_CONFIG=C slurm/train_conditioning.sbatch

# Real single-H200 run, validation every 500 optimizer steps, W&B online
sbatch --export=ALL,R2V_CONFIG=A,R2V_MAX_STEPS=0,R2V_VALIDATE_EVERY=500,R2V_WANDB=online \
       --time=24:00:00 slurm/train_conditioning.sbatch

# Real 4-GPU DDP run on one node
sbatch --export=ALL,R2V_CONFIG=C,R2V_MAX_STEPS=0,R2V_NGPU=4,R2V_VALIDATE_EVERY=500 \
       --gres=gpu:h200:4 --cpus-per-task=32 --time=24:00:00 slurm/train_conditioning.sbatch
```

`#SBATCH --export=NONE` means a bare `VAR=x sbatch ...` does **not** reach the job — always use
`--export=ALL,...`. `R2V_MAX_STEPS=0` removes the step cap.

Equivalent direct invocation, if you are not using Slurm:

```bash
python -m mrrate_r2v.cli.train_r2v \
    --manifest   $DATA/r2v_manifest/manifest_shards_native.csv \
    --report-index $DATA/r2v_manifest/report_index_shards_native.csv \
    --base-checkpoint $WS/models/diff_unet_3d_rflow-mr-brain_v0.pt \
    --vae-checkpoint  $WS/models/autoencoder_v1.pt \
    --conditioning report2ct_style --max-report-tokens 512 \
    --split train --val-split val \
    --batch-size 4 --grad-accumulation-steps 2 --epochs 1 --lr 1e-5 \
    --validate-every-steps 500 --val-quick-samples 32 --val-inference-steps 30 \
    --medicalnet-checkpoint $WS/pretrained/medicalnet/resnet_10_23dataset_statedict.pth \
    --wandb-mode online --wandb-project mr-rate-r2v \
    --out $WS/runs/r2v_C --num-workers 8 --device cuda

# multi-GPU: prefix with torchrun and declare the count
torchrun --nproc_per_node=4 --standalone -m mrrate_r2v.cli.train_r2v --num-gpus 4 ...
```

**`--num-gpus > 1` without `torchrun` now fails immediately** with the correct command in the error.
Previously it was accepted and silently trained N independent single-GPU models.

### 9.6 Validation, W&B, and the interactive panel

Validation is **off by default** (each pass runs a full diffusion sampler). Enable with
`--validate-every-steps N`, counted in **optimizer** steps — never micro-steps, so the number means
the same thing at any `--grad-accumulation-steps`.

This is **conditional** generation: for every validation case `generated_i = G(report_i)`, with
that case's own `ground_truth_i` held alongside. Generation is report-conditioned always, at a fixed
per-case seed, and `ValidationRunner.assert_conditioning_active` **refuses to run** if
`report_guidance_scale == 0` or `report_dropout_probability >= 1` — either would make every
generation report-blind, and FVD/2.5D FID are marginal metrics that would not reveal it.

**Primary metrics** — both compare `{ground_truth_i}` against `{G(report_i)}`:

| W&B key | definition | provenance |
|---|---|---|
| `val/fvd` | Frechet distance over r3d_18 (Kinetics-400) sequence features; per plane, then unweighted mean. Lower better | **MRI-volume adaptation of FVD.** Challenge-precedented in *family* — VLM3D 2025 CT Task 4 scored `FVD_I3D` + `FVD_CT-Net`, inherited from GenerateCT — but the extractor is **r3d_18, not I3D**, so it is never called standard FVD. `eval/video_features.py` states the exact differences from the reference implementation. |
| `val/fid_2p5d` | one mean-pooled InceptionV3 vector per volume per plane over that plane's non-empty slices, then per-plane Frechet, then unweighted mean. Lower better | **Project-specific adaptation.** Neither the challenge nor GenerateCT uses the name "2.5D FID"; GenerateCT's FID is plain *slice-level*, whose limitation it states itself. Volume-weighted by construction, so a thick volume cannot outvote a thin one. |

**Diagnostics** — logged, but not headline curves:

| key | what |
|---|---|
| `val/ssim` | paired 3D SSIM(`generated_i`, `ground_truth_i`), Wang settings (11-wide Gaussian window, σ=1.5, population covariance, K1=0.01, K2=0.03, `data_range=1.0`), cropped to the **ground truth's** foreground box. Standard SSIM; used by neither the challenge nor GenerateCT. |
| `val/sensitivity/*` | does the report change the generation? Swap in another **study's** report at a fixed seed and measure `swap_ssim` / `swap_relative_l1`, plus `ssim_correct` vs `ssim_shuffled` against the true target. `swap_ssim ≈ 1.0` raises a loud error: the model is ignoring its text. Runs every `--val-sensitivity-every-steps` on `--val-sensitivity-samples` cases (default 8), not the whole split. |
| `val/*/rank_level` | 0 = rank-deficient, 1 = marginal, 2 = well-conditioned — see the sample-size warning below. |

### ⚠️ Report–volume semantic fidelity is NOT measured

FVD and 2.5D FID are **marginal** metrics: they ask whether the *set* of generations resembles the
*set* of real volumes. A model that produced a perfectly distributed set of volumes paired to the
**wrong** reports would score identically. `val/ssim` is paired but structural. So nothing in this
suite verifies that generation *i* is semantically faithful to report *i*.

Measuring that needs a frozen, independent, validated cross-modal MRI report–volume model. The one
defensible candidate found is **HLIP** — [`zch0414/clip-vit_base-scan_study-dualdinotxt1568`](https://huggingface.co/zch0414/clip-vit_base-scan_study-dualdinotxt1568),
[arXiv:2505.21862](https://arxiv.org/abs/2505.21862), MIT, 768-d joint space, BiomedBERT text tower
at 256 tokens, **trained on MR-RATE's *training* split** so this project's `val` split is unseen. It
is **not adopted**, and no substitute is faked in its place. `validation.AlignmentMetric` is the seam
it would plug into; the deterministic different-study permutation such a metric needs is already
implemented and tested (`shuffled_report_pairing`). Rejected alternatives and why: CT-CLIP (CT
only), BiomedCLIP (2D PMC figure-captions, no radiology-report claim), BrainG3N (a tokenizer, not an
aligner; non-commercial), Brainfound/BMLIP (weights unverifiable), this repo's own contrastive model
(never trained — no checkpoint exists).

### ⚠️ Measured: the Frechet metrics need far more samples than a training loop can afford

On 512-d features with a **known-zero** ground truth (two disjoint halves of one real population,
so the true distance is 0), the observed Frechet distance is:

| N | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| real-vs-real (truth = 0) | 21576 | 14189 | **6123** | 3692 | 1908 | 892 | 432 |

while a genuine, substantial distributional difference (+0.5 on every feature) registers **128 at
every N**. At N=64 the sample-size bias is ~**48×** a real effect, and it still exceeds it at N=1024.
The conventional FID guidance (N ≥ 2048) is not conservatism.

**Consequence:** `val/ssim` and `val/sensitivity/*` are what a *frequent* validation pass can
actually support. FVD and 2.5D FID belong on the occasional `--validate-full-every-steps` pass at a
large `--val-full-samples`, or offline via `cli.evaluate` over a real cohort — which is where this
repository already computes distribution metrics, at ~2000 cases. `ValidationRunner` warns about
this at startup with the configured N, and every Frechet value carries a `rank_level` flag so it can
never be read as more trustworthy than its sample supports.

Cached reference constants come from `cli.validation_reference` (one-off, `--validation-reference
<json>`), are logged every step so W&B draws them as flat lines, and are never recomputed:
`ssim_identity` (sanity check, must be ~1.0), four SSIM perturbation checks, `ssim_autoencoder`
(the frozen VAE's structural ceiling), and `fvd_real_vs_real` / `fid_2p5d_real_vs_real` (the
finite-sample noise floors above).

The validation subset is **fixed, seeded and bucket-stratified**, and the quick set is a *prefix* of
the full set so the two curves measure the same population. Real features are cached after the first
pass. Under DDP, cases are sharded `index % world_size` (no duplication) and gathered with
`all_gather_object`; only one volume is resident at a time.

W&B: `--wandb-mode online|offline|disabled` (default `disabled`), rank 0 only, and a missing `wandb`
package or credentials degrades to a no-op rather than a crash. Main plots, all on the optimizer-step
axis: `train/loss`, `val/fvd`, `val/fid_2p5d`, with `val/ssim` and `val/sensitivity/*` as
diagnostics and `val/reference/*` as flat lines.

The **interactive panel** is a self-contained `wandb.Html` — no external requests, so it renders
offline. Per case: ground truth and generated side by side in all three planes, one **slice slider**
across all six views, plus findings, impression, modality, plane, step, and the anonymised
`case_id`. Slice indices are matched between the two sources, the intensity window is **one window
taken from the ground truth** (a degenerate prediction must look degenerate), and each plane carries
the physical aspect ratio from `spacing_xyz`.

Open it in W&B → the run → Media → `validation/<case_id>`. **Two axes, both native**: W&B's own step
selector chooses the validation step (the panel key is stable across steps for a given case), and
the panel's slider chooses the slice. A single W&B panel cannot itself expose a step slider, so this
uses the run's step control rather than a custom plugin — the documented limitation.

**The panel embeds report text, so it is off unless `--wandb-log-reports` is passed.** Reports are
patient data even after anonymisation; do not point this at a public project.

### 9.7 H200: measured, not assumed

Environment (job 688298, partition `h200`, node h24-24): **NVIDIA H200, 140.1 GB, compute capability
9.0, 132 SMs**; driver 595.71.05, CUDA 13.2; torch 2.13.0+cu130, cuDNN 9.2.0, **NCCL 2.29.7**;
`bf16_supported = True`; 4 GPUs/node, 128 CPU cores.

**One finding that changes what to optimise: `flash_with_mask` is unavailable.**

```
flash_with_mask          unavailable: RuntimeError      ← the shape this code actually uses
flash_no_mask            ok
mem_efficient_with_mask  ok                             ← what MaskedCrossAttention gets
math_with_mask           ok
```

`MaskedCrossAttention` always passes `attn_mask` when a `context_mask` is given, so the adapters'
cross-attention runs on the **mem-efficient** SDPA kernel, never flash. It is exact and it is fast
(the denoiser is 2–9% of a step), so this is not a problem — but "flash attention is enabled" is
true of the base UNet's self-attention and false of the conditioning path, and the difference is
worth knowing before optimising the wrong thing. `flash_attn` the package is not installed; monai's
`use_flash_attention` routes through SDPA.

Measured at the NVIDIA default 256³ bucket, bf16 autocast, 10 steps after 3 warmup (job 688299):

| config | batch | step s | vol/s | peak alloc GB | peak resv GB | text s | VAE s | UNet s | bound by |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cxr_bert_cls` | 1 | 0.481 | 2.08 | 9.53 | 12.32 | 0.005 | 0.393 | 0.084 | VAE encode |
| `cxr_bert_cls` | 2 | 0.851 | 2.35 | 17.73 | 23.25 | 0.005 | 0.752 | 0.094 | VAE encode |
| `cxr_bert_cls` | 4 | 1.569 | 2.55 | 34.14 | 45.21 | 0.006 | 1.428 | 0.136 | VAE encode |
| `radbert_mean` | 1 | 0.508 | 1.97 | 9.58 | 12.40 | 0.005 | 0.387 | 0.116 | VAE encode |
| `radbert_mean` | 2 | 0.854 | 2.34 | 17.79 | 23.25 | 0.005 | 0.748 | 0.101 | VAE encode |
| `radbert_mean` | 4 | 1.567 | 2.55 | 34.19 | 45.26 | 0.006 | 1.421 | 0.141 | VAE encode |
| `report2ct_style` | 1 | 0.511 | 1.96 | 11.22 | 14.03 | 0.030 | 0.455 | 0.026 | VAE encode |
| `report2ct_style` | 2 | 0.876 | 2.28 | 19.42 | 25.02 | 0.035 | 0.738 | 0.103 | VAE encode |
| `report2ct_style` | 4 | 1.617 | 2.47 | 35.83 | 46.69 | 0.044 | 1.436 | 0.138 | VAE encode |

Extended sweep to find the actual optimum (job 688301, 6 steps after 3 warmup):

| config | batch | step s | vol/s | peak alloc GB | peak resv GB | verdict |
|---|---:|---:|---:|---:|---:|---|
| `cxr_bert_cls` | 8 | 3.109 | **2.573** | 66.95 | 85.29 | **peak throughput** |
| `cxr_bert_cls` | 12 | 4.847 | 2.476 | 99.76 | 115.97 | slower, 83% of the card |
| `cxr_bert_cls` | 16 | 6.909 | 2.316 | 132.57 | 133.25 | slower still, 95% of the card |
| `cxr_bert_cls` | 24 | — | — | — | — | **OutOfMemoryError** |
| `report2ct_style` | 8 | 3.109 | **2.573** | 68.64 | 87.03 | **peak throughput** |
| `report2ct_style` | 12 | 4.934 | 2.432 | 101.45 | 117.68 | slower |
| `report2ct_style` | 16 | 7.073 | 2.262 | 134.27 | 134.77 | slower |
| `report2ct_style` | 24 | — | — | — | — | **OutOfMemoryError** |

**Batch 8 is the measured optimum, not merely the largest that fits.** Throughput rises 2.08 → 2.35
→ 2.55 → 2.573 vol/s to batch 8, then *falls* to 2.476 at 12 and 2.316 at 16 — larger batches are
strictly worse on both axes. At batch 8 the card is 61% reserved, which is the headroom validation
generation needs.

**The dominant cost is the on-the-fly VAE encode — ~90% of every step — not the conditioning and not
the denoiser.** Text encoding is 0.3–0.6% of a step for A/B and 1.8–2.7% for C. Consequences:

1. **The three configurations are within ~3% of each other on throughput and ~3% on memory.** The
   choice between them is a quality decision, not a performance one. (Configuration C does carry
   ~445M more frozen encoder parameters and a 2560-wide projection: +1.4 GB reserved at batch 4 and
   +3.67M trainable adapter parameters.)
2. **Do not micro-optimise the text encoders.** Report2CT precomputed and cached all embeddings;
   here that would save ~3% of a step. Caching *latents* instead would save ~90% — that is the real
   optimisation available, and it is a deliberate design choice documented in `training.py`
   (difference 3: latents on the fly, one fewer on-disk stage).

Batch-size recommendations follow the measurement, with headroom: the recommended size is the
largest whose peak **reserved** memory stays under 75% of the card, because validation generation
runs a full sampler plus a MedicalNet forward pass in the same process and its peak is *not* in the
table above.

| GPUs | batch/GPU | grad accum | effective batch | reserved/GPU | notes |
|---:|---:|---:|---:|---:|---|
| 1 | 8 | 8 | 64 | 85–87 GB | measured optimum; 61% of the card |
| 2 | 8 | 4 | 64 | 85–87 GB | |
| 4 | 8 | 2 | 64 | 85–87 GB | **the recommended real-run shape** |
| 8 (2 nodes) | 8 | 1 | 64 | 85–87 GB | multi-node **untested** — see below |

Drop to `--batch-size 4 --grad-accumulation-steps 16` if a validation pass ever OOMs: it costs ~1%
throughput (2.55 vs 2.573 vol/s) and halves the training-step peak to ~46 GB.

**Honest limits on these numbers.**

- **All measurements are single-GPU.** Multi-GPU DDP throughput, scaling efficiency, and multi-node
  were **not measured**. The DDP code path is exercised only by the unit tests
  (`ShardedBatchSampler`, rank-0 gating) and by the launcher's own refusal to run un-launched — no
  4-GPU job was run. Treat the 2/4/8-GPU rows as arithmetic on the per-GPU measurement, not as
  measured scaling.
- **Batch sizes were probed at 1, 2, 4, 8, 12, 16, 24**, so the optimum is bracketed but not
  resolved between 8 and 12.
- **Every number is the 256³ fallback bucket**, the largest grid in §9.8. The real per-bucket shapes
  are smaller (a T2w/CORONAL volume is 192³, ~42% of the voxels), so these are the worst case and
  real training will be faster and lighter.
- `radbert_mean` was measured only to batch 4; it is within 0.2% of `cxr_bert_cls` at every shared
  size and has an identical parameter count, so batch 8 is expected to match — expected, not
  measured.
- **Validation-pass peak memory was not measured**, which is exactly why the recommendation is the
  75%-of-card size rather than the 95%-of-card one.

Optimisations applied, all stable and all evidence-backed: bf16 autocast (`--no-amp` disables), TF32
for matmul and cuDNN, PyTorch SDPA, pinned host memory, persistent dataloader workers,
`prefetch_factor=4`, gradient accumulation, DDP with `find_unused_parameters=True`, and frozen
encoders under `torch.no_grad` so no activation is retained for a backward pass that never reaches
them. **Not** applied: FP8, FSDP, `torch.compile` — none is needed when 90% of the step is one
frozen VAE call, and each would add a failure mode for no measured gain.

### 9.8 Volume geometry, end to end

Shape and spacing are **fixed per (modality, plane) bucket** — not global, not per sample. One
bucket = one geometry = one `.npz` archive = one FID. Each sits on NVIDIA's published FOV for that
pair: shape = nearest multiple of **32** (the diffusion UNet's constraint), spacing = FOV / shape.

Axis order, the single most bug-prone thing here:

```
NIfTI / on-disk / internal geometry   (D, H, W) = (S, R, A)
crossing the package boundary          (X, Y, Z) = (R, A, S)
PyTorch training tensor                (B, C, X, Y, Z)
```

Convert only via `geometry.dhw_to_xyz` / `xyz_to_dhw`. A skipped conversion is **silent** for a cube
at isotropic spacing (256³ @ 1 mm) and scrambles axes otherwise.

| Stage | Shape | Axis order | Spacing | Orientation | Evidence |
|---|---|---|---:|---|---|
| Raw source (NIfTI in shard tar) | variable per series | `(X, Y, Z)` | variable, from header | as acquired | `manifest.native_shape` / `native_spacing_mm` |
| After reorient → resample → crop/pad → normalize | per-bucket, below | `(D, H, W)` | per-bucket | RAS | `_preprocess_ops.preprocess_nii` |
| Dataset output = **model input** | per-bucket, below | **`(X, Y, Z)`** | per-bucket | RAS | `dataset.py` `permute(0,2,3,1)` |
| Latent | model input ÷ **4** | `(X, Y, Z)` | ×4 | RAS | `sampling.official_latent_divisor([64,128,256,512]) = 4` |
| Model output (denoiser) | = latent | `(X, Y, Z)` | ×4 | RAS | `sample_latent` |
| Decoded + postprocessed | = model input | `(X, Y, Z)` | per-bucket | RAS | `postprocess_mr`, int16, range [0, 1000] |
| Saved `.nii.gz` | = model input | `(X, Y, Z)` | per-bucket | axis-aligned affine, diag = spacing | `sampling.save_volume` |
| Evaluator input | = saved | `(X, Y, Z)` | per-bucket | — | `eval/` reads only `.npy`; **never resizes** |

Per bucket (verified at runtime; every `(D,H,W)` divisible by 32, every `(X,Y,Z)` by 4):

| bucket | FOV mm (D,H,W) | shape (D,H,W) | **shape (X,Y,Z)** | **spacing (X,Y,Z) mm** | latent (X,Y,Z) |
|---|---|---|---|---|---|
| T1w/AXIAL | (174, 240, 240) | (160, 256, 256) | (256, 256, 160) | (0.938, 0.938, 1.087) | (64, 64, 40) |
| T1w/SAGITTAL | (250, 176, 250) | (256, 192, 256) | (192, 256, 256) | (0.917, 0.977, 0.977) | (48, 64, 64) |
| T1w/CORONAL | (240, 240, 200) | (256, 256, 192) | (256, 192, 256) | (0.938, 1.042, 0.938) | (64, 48, 64) |
| T2w/AXIAL | (158, 240, 240) | (160, 256, 256) | (256, 256, 160) | (0.938, 0.938, 0.988) | (64, 64, 40) |
| T2w/SAGITTAL | (240, 162, 240) | (256, 160, 256) | (160, 256, 256) | (1.012, 0.938, 0.938) | (40, 64, 64) |
| T2w/CORONAL | (200, 200, 180) | (192, 192, 192) | (192, 192, 192) | (1.042, 0.938, 1.042) | (48, 48, 48) |
| FLAIR/AXIAL | (175, 250, 250) | (160, 256, 256) | (256, 256, 160) | (0.977, 0.977, 1.094) | (64, 64, 40) |
| FLAIR/SAGITTAL | (250, 176, 250) | (256, 192, 256) | (192, 256, 256) | (0.917, 0.977, 0.977) | (48, 64, 64) |
| FLAIR/CORONAL | (250, 250, 200) | (256, 256, 192) | (256, 192, 256) | (0.977, 1.042, 0.977) | (64, 48, 64) |
| SWI/AXIAL | (145, 230, 230) | (160, 224, 224) | (224, 224, 160) | (1.027, 1.027, 0.906) | (56, 56, 40) |
| SWI/SAGITTAL | (230, 140, 230) | (224, 128, 224) | (128, 224, 224) | (1.094, 1.027, 1.027) | (32, 56, 56) |
| SWI/CORONAL | (230, 230, 155) | (224, 224, 160) | (224, 160, 224) | (1.027, 0.969, 1.027) | (56, 40, 56) |
| MRA/AXIAL | (158, 220, 220) | (160, 224, 224) | (224, 224, 160) | (0.982, 0.982, 0.988) | (56, 56, 40) |
| MRA/SAGITTAL | (250, 158, 250) | (256, 160, 256) | (160, 256, 256) | (0.988, 0.977, 0.977) | (40, 64, 64) |
| MRA/CORONAL | (240, 240, 179) | (256, 256, 192) | (256, 192, 256) | (0.938, 0.932, 0.938) | (64, 48, 64) |
| *fallback* (unlisted pair) | (256, 256, 256) | (256, 256, 256) | (256, 256, 256) | (1.0, 1.0, 1.0) | (64, 64, 64) |

**Does inference have enough information to restore the intended geometry? Yes, and it is not
leakage.** Shape and spacing come from the cohort's frozen `GeometrySpec`, which is a property of
the *request* ("generate an axial T1w at this grid"), not of the held-out target — the cohort
contract freezes the grid before any model runs. `save_volume` writes an axis-aligned affine whose
diagonal is the spacing. **There is no inverse transform back to the native grid**, by design: the
evaluator compares on the cohort grid and excludes a shape mismatch with a reason rather than
resizing.

### 9.9 Checkpoints

Both layouts come from one writer, `models/adapter.save_adapter_checkpoint`. Module ownership is by
**module identity** (`adapter_modules`), cross-checked against `CONDITIONING_PREFIXES`, with a raise
if the two disagree — not substring matching over parameter names.

**Files.** `adapter_last.pt`, `adapter_step<N>.pt` (periodic, `--save-every-steps`, retained
`--keep-last-n` deep), `adapter_best_fid.pt` (lowest FID), `adapter_best_alignment.pt` (highest
alignment). Retention never touches `last` or either `best`.

**Contents** — a full resume plus a lightweight conditioning checkpoint in the same file, because
the frozen 180.5M base is deliberately *not* stored (it is unchanged by construction; storing it
would make every checkpoint ~700 MB of a file the workspace already has, and let a stale copy
diverge):

| key | what |
|---|---|
| `adapter_state_dict` | `context_proj`, `null_context`, `{down,mid,up}_cross_attn` — the trainable conditioning path, and nothing else |
| `optimizer_state_dict`, `lr_scheduler_state_dict`, `scaler_state_dict` | exact-resume optimiser state |
| `step`, `optimizer_step`, `epoch` | micro-step and optimizer-step clocks, kept separately |
| `rng_state` | `torch`, the report-dropout generator, and CUDA RNG for all devices |
| `best_metrics` | `{fid, alignment}`, carried across resumes |
| `config` | `context_dim`, `cross_attention_dim`, `conditioning_levels`, `condition_mid`, **`conditioning_name`**, **`report_format`**, dropout/guidance settings, `num_train_timesteps` |
| `text_encoder` | the full conditioning identity: kind, pooling, section order, encoder order, per-encoder dims, checkpoint paths, `hf_repo`, `trainable` |
| `base_checkpoint` | path + sha256 of NVIDIA's frozen denoiser |
| `scale_factor`, `loss`, `validation` | provenance |

```bash
# resume (compatibility-checked automatically)
python -m mrrate_r2v.cli.train_r2v --conditioning A ... --resume $RUN/adapter_last.pt

# load only the conditioning components for inference
python -m mrrate_r2v.cli.generate_r2v --adapter $RUN/adapter_best_fid.pt \
    --base-checkpoint $WS/models/diff_unet_3d_rflow-mr-brain_v0.pt \
    --vae-checkpoint  $WS/models/autoencoder_v1.pt --cohort $COHORT --out-dir $OUT

# inspect what a checkpoint is, without loading a model
python -c "import torch,json; p=torch.load('$RUN/adapter_last.pt',map_location='cpu',weights_only=False); \
print(json.dumps({'config':p['config'],'text_encoder':p['text_encoder'],'best':p['best_metrics'], \
'optimizer_step':p['optimizer_step']},indent=2,default=str))"
```

**Three refusals, all deliberate.** A different base checkpoint sha256 (`allow_base_mismatch`); a
different conditioning configuration (`allow_conditioning_mismatch`) — including the two that share
a shape, `cxr_bert_cls` and `radbert_mean` are both 768×1 and would otherwise load each other's
weights with no shape error at all; and a cohort whose `report_format` differs from the trained one
(`--allow-report-format-mismatch`).

### 9.10 Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `needs per-section text, but this batch has no 'report_sections_text'` | a sectioned configuration fed joined text | leave `R2VDatasetConfig.conditioning_sections` at its default; do not pass `--report-format` with `report2ct_style` |
| `adapter checkpoint was trained under a different conditioning configuration` | wrong `--conditioning` for that checkpoint | read `config['conditioning_name']` from the file; or `allow_conditioning_mismatch=True` for a deliberate transfer |
| `report-format mismatch: the adapter was trained on ... but this cohort's text was composed with ...` | cohort and adapter disagree | rebuild the cohort with the trained format, or `--allow-report-format-mismatch` |
| `--num-gpus 4 but WORLD_SIZE=1` | launched without `torchrun` | use the `torchrun --nproc_per_node=4` line the error prints |
| `conditioning '<x>' built with output_dim=N, but the config table says M` | a staged snapshot is not the checkpoint the configuration was defined against | check which snapshot is in `$MRRATE_PRETRAINED_DIR` |
| validation runs but no FID is logged | fewer usable cases than `--val-quick-samples` 16, or MedicalNet missing | raise `--val-quick-samples`, or pass `--medicalnet-checkpoint` |
| `MedicalNet feature extractor unavailable` | wrong path | `$PRETRAINED_DIR/medicalnet/resnet_10_23dataset_statedict.pth` |
| no validation panel in W&B | `--wandb-log-reports` not passed | pass it — and only at a private project, since the panel embeds report text |
| W&B silent, training fine | intended: `--wandb-mode disabled`, or `wandb` missing/uncredentialed | check `wandb_run.json` in the run directory |
| `text encoder '<x>' checkpoint directory not found` | not staged | `python -m mrrate_r2v.cli.download_text_encoders --encoders <x>` |
| step count looks wrong vs an older run | intervals now count **optimizer** steps, not micro-steps | divide the old number by `--grad-accumulation-steps` |

---

## 10. Limitations and next experiments

**Limitations of what was measured.**

1. **Pooled-embedding proxy.** Every metric scores `mean`/`max` pooled vectors, but the denoiser
   cross-attends over the full token sequence. An encoder that scatters information across tokens
   is under-credited. Mitigated by using `concat(mean, max)` rather than mean alone; not
   eliminated.
2. **Weak labels for three of five metrics** (§6.1). Valid for ranking, invalid as absolutes.
3. **Negation pairs are constructed, not annotated.** Cue deletion can leave a mildly
   ungrammatical affirmed sentence; the effect is identical across encoders but is not zero.
   Double negation and "cannot be excluded" hedges are excluded rather than transformed.
4. **No end-to-end generation result.** Nothing here proves a better `pathology_probe_auroc`
   produces a better FID or a better blinded-classifier consistency. That link is assumed, and it
   is the assumption most worth testing next.
5. **The challenge input schema is still unpublished** (§5), so the format recommendation is made
   robust to the unknown rather than optimised for a known contract.

**Recommended next experiments, in order of value.**

1. **The one that closes the loop:** train the existing report-conditioning adapter
   (`cli.train_r2v`) twice — best encoder vs. `radbert` baseline, same seed, same steps — and
   compare with `cli.evaluate --task report2volume` on the same test split. Everything needed is
   already wired; this is the only experiment that validates limitation 4.
2. **Token-level conditioning ablation:** the `_tokens` configurations (kept token axis) vs. the
   pooled ones (`cxr_bert_cls`/`radbert_mean`), through the actual adapter. Report2CT used
   pooled-concat; whether token-level cross-attention beats it on MR is untested.
3. **`impression_findings` under a real 512 budget**, end to end. The truncation argument is
   sound but its effect size on generated volumes is unknown.
4. **Unfreeze the top N layers** of the chosen encoder (`trainable=True` is already supported) and
   measure whether adapting the encoder to Turkish-translated report register beats freezing it.
