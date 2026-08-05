# Text encoders and report formats for report-to-volume conditioning

Everything about turning an MR-RATE radiology report into the tensor the conditioned denoiser
attends over: what the reports actually look like, which encoders are staged, which report format
to use, where acquisition metadata should come from, and how the choice was measured.

Companion documents: [`R2V.md`](R2V.md) for the generation/evaluation pipeline this feeds,
[`mrrate_r2v/textenc/README.md`](../contrastive-pretraining/mrrate_r2v/textenc/README.md) for the
production API, [`mrrate_r2v/textbench/README.md`](../contrastive-pretraining/mrrate_r2v/textbench/README.md)
for the selection benchmark.

Confidence key: **VERIFIED** (measured here, or quoted from a primary source that was fetched) /
**INFERRED** (well supported but not stated outright) / **ASSUMED** (a provisional choice).

---

## 1. What was added

```
contrastive-pretraining/mrrate_r2v/
├── textenc/                    PRODUCTION: what the trainer and sampler use
│   ├── formats.py              named report formats            (no torch)
│   ├── encoders.py             HFTextEncoder + ENCODER_SPECS
│   ├── fusion.py               MultiEncoderEmbedder, ProjectedConcatFusion
│   └── README.md
├── textbench/                  SELECTION: never imported by the trainer
│   ├── corpus.py               report/label corpus from the shard tars   (no torch)
│   ├── analysis.py             dataset statistics                        (no torch)
│   ├── negation.py             negation minimal pairs                    (no torch)
│   ├── embed.py                embedding cache
│   ├── tasks.py                the five metrics
│   ├── runner.py               the one scoring path
│   └── README.md
└── cli/
    ├── download_text_encoders.py   stage checkpoints (idempotent, pinned revisions)
    ├── analyze_reports.py          build-corpus | analyze | tokens
    ├── embed_reports.py            GPU: cache embeddings
    └── eval_text_encoders.py       CPU: score the matrix
```

Changed, minimally: `mrrate_r2v/text.py` (the factory now also resolves zoo names),
`mrrate_r2v/data/dataset.py` (an optional `report_format`), `mrrate_r2v/cli/train_r2v.py`
(`--report-format`, the zoo in `--text-encoder`), `slurm/_common.sh` (paths + `run_py_host`).
**Every default is unchanged**: leave `report_format` unset and `--text-encoder radbert` and the
pipeline behaves exactly as before, down to the `cohort_id`.

---

## 2. The reports (measured on all 98,334 studies)

Reproduce with:

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
| pooled available | mask-aware mean | mask-aware mean | mask-aware mean | mask-aware mean | mask-aware mean | **trained CLIP `[CLS]`** | mask-aware mean | mask-aware mean |
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
3. **The 2025 CT-winner's trio, reproducible** — `cxr_bert` + `bio_clinicalbert` +
   `medembed_large` is exactly Report2CT's set, so its multi-encoder claim can be tested on MR
   rather than assumed to transfer.
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

```bash
cd contrastive-pretraining

# GPU, ~1 h: 8 encoders x 8 formats x (20,000 train + 5,554 test) at a common 512-token budget
sbatch slurm/08_embed_reports.sbatch 512 20000 budget512

# CPU, ~15 min: score the matrix, plus any fusion pairs
sbatch slurm/09_eval_text_encoders.sbatch budget512 20000 \
    bioclinical_mbert+radbert medembed_large+radbert medembed_small+radbert \
    cxr_bert+bio_clinicalbert+medembed_large

# the 8192-context question, as a separate run so the caches never mix
sbatch slurm/08_embed_reports.sbatch native 20000 nativectx
sbatch slurm/09_eval_text_encoders.sbatch nativectx 20000
```

Results land in
`/hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/textbench/<run>/results/`:
`metrics_matrix.csv`, `per_label_auroc.csv`, `summary.json`.

### 6.3 Results

*(Filled in from `metrics_matrix.csv` — see §6.4 for the current run's status.)*

---

## 7. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `FileNotFoundError: text encoder '<x>' checkpoint directory not found` | not staged | `python -m mrrate_r2v.cli.download_text_encoders --encoders <x>` |
| `ValueError: max_length=N exceeds what '<x>' supports` | asked for more tokens than the checkpoint has positions | lower `--max-report-tokens`, or use `bioclinical_mbert` (8192) |
| `contains custom code which must be executed` | loading `cxr_bert` through plain `AutoModel` | use `build_encoder("cxr_bert")` — the spec's `bert_shim` loader avoids `trust_remote_code` |
| `Cannot use torch.load ... CVE-2025-32434` | a `.bin`-only checkpoint under torch < 2.6 | `download_text_encoders` converts `bio_clinicalbert` on stage; `text.ensure_local_safetensors` handles `radbert` |
| `KeyError: N cached study ids are absent from the corpus` | `--train-limit`/`--seed`/`--corpus` differ between embed and eval | pass the same values to both — this check exists so a mismatch is loud |
| probe AUROC ≈ 0.5 everywhere | embeddings collapsed, or the corpus key names changed | check `summary.json`'s `nn_jaccard_raw` vs `nn_jaccard_random`; rebuild the corpus with `build-corpus` |
| `AssocGrpGRES` pending forever | GPU job submitted without `--qos=mq_health` | the sbatch scripts set it; keep it if you copy them |
| out of GPU memory | `medembed_large` at batch 64 × 1024 tokens | `--batch-size 32`, or `--dtype bfloat16` |
| `ModuleNotFoundError: sklearn` in the eval job | ran the scoring stage inside a container | scoring uses `run_py_host`; neither SIF has scikit-learn |

**Memory and time, measured.** The embedding stage is the only expensive one: ~1 GPU-hour for the
full 8 × 8 matrix at 512 tokens. Scoring is CPU-only and takes minutes because no model is loaded.
Caches are ~25 MB per (encoder, format, split) in float16; the full matrix is ~3 GB — workspace
only, never `$HOME`, and note `/hnvme`'s **file-count** quota (61k soft): the full matrix writes
~260 files.

---

## 8. Limitations and next experiments

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
   compare with `cli.evaluate --task report2volume` on a frozen cohort. Everything needed is
   already wired; this is the only experiment that validates limitation 4.
2. **Token-level conditioning ablation:** `MultiEncoderEmbedder(mode="token")` vs `mode="feature"`
   through the actual adapter. Report2CT used pooled-concat; whether token-level cross-attention
   beats it on MR is untested.
3. **`impression_findings` under a real 512 budget**, end to end. The truncation argument is
   sound but its effect size on generated volumes is unknown.
4. **Unfreeze the top N layers** of the chosen encoder (`trainable=True` is already supported) and
   measure whether adapting the encoder to Turkish-translated report register beats freezing it.
