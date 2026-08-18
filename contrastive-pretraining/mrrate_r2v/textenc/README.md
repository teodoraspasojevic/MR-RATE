# `textenc` — turning a radiology report into numbers the generator can use

## What problem this solves

The report-to-volume model generates a brain MRI from a written radiology report. But a neural
network cannot read text. Something has to turn

> "No acute infarct. Chronic gliotic foci in the periventricular white matter."

into a tensor of numbers that the image generator can attend over. That is this package's whole
job. It answers two questions:

1. **Which words do we feed in?** The released report has four separate sections, and you do not
   have to use all of them. → `formats.py`
2. **Which pretrained model turns those words into numbers?** → `encoders.py`

A third file, `fusion.py`, handles the case where you want to use two models at once.

Everything here is *production* code — it runs during training and during generation. The sibling
package [`textbench/`](../textbench/README.md) is the experiment that decides which options here
are the good ones. Nothing in `textenc/` imports `textbench/`, and a test enforces that.

---

## The 30-second version

```python
from mrrate_r2v.textenc import build_encoder, format_report

encoder = build_encoder("bioclinical_mbert")            # loads a frozen pretrained model
text    = format_report(record, "findings_impression")  # record: a data.reports.ReportRecord
cond    = encoder.encode([text], device)                # the actual encoding step
```

`cond` is a `TextConditioning` object with three fields:

| field | shape | meaning |
|---|---|---|
| `cond.token_embeddings` | `(B, L, D)` | one vector per token — **this is what the generator attends over** |
| `cond.attention_mask` | `(B, L)` bool | `True` = a real token, `False` = padding. Never invert this. |
| `cond.pooled_embedding` | `(B, D)` | the whole report squashed to one vector, for logging and probes |

`B` = number of reports in the batch, `L` = number of tokens (the longest report in the batch),
`D` = `encoder.output_dim` (768 for most, 1024 for MedEmbed-large, 384 for MedEmbed-small).

In practice you do not assemble this by hand: `--conditioning <name>` selects one of four named
configurations that fix the encoder, the pooling and the format together — see **Part 4**.

---

## Part 1: report formats — which words go in

### Why there is a choice at all

MR-RATE ships each report already split into four sections. Here is one report (**illustrative
example, not a real patient**) shown as the dataset stores it:

```
clinical_information : "Headache for three weeks. Rule out mass lesion."
technique            : "3-plane T1-weighted, FLAIR; axial T2-weighted, DWI, ADC and SWI.
                        T1-weighted after IV 15 ml gadolinium."
findings             : "Brainstem, cerebellum and cerebral parenchyma are normal. A few
                        non-specific gliotic foci are observed in the bilateral frontal
                        subcortical white matter. No diffusion restriction. No pathological
                        contrast enhancement. Ventricles and sulci are age-appropriate."
impression           : "— A few non-specific chronic ischaemic-gliotic foci.
                        — No acute finding."
```

Plus `raw`, the original unsplit report, which also contains a title line, the section headings,
and sometimes other body regions' findings.

Feeding all of it in is not obviously right:

- `technique` describes the **protocol of the whole exam**, not the one volume being generated.
  Conditioning on it teaches the model to read scanner settings out of prose — and at challenge
  inference the wording may differ or the section may be missing.
- `clinical_information` is the *referring question*, not a finding. It is missing for **52%** of
  studies, so the model would see an inconsistent prefix.
- `raw` is 1.5–2× longer in tokens and, in long reports, often contains findings for *other body
  regions* (spine, orbit, hip) that no brain volume can express.
- `impression` alone is short and clean, but it is **missing for 8.9%** of studies and drops all
  localisation.

So the formats are the small set of sensible answers, each a named, deterministic function.

### What each format produces

Using the example above. `[FINDINGS]`-style markers are used instead of plain `Findings:` because
the raw reports already contain 845 different `Word:` headings, so a plain heading is
indistinguishable from report content.

| format | output |
|---|---|
| **`findings_impression`** ← default | `[FINDINGS] Brainstem, cerebellum … age-appropriate.`<br>`[IMPRESSION] — A few non-specific chronic ischaemic-gliotic foci. — No acute finding.` |
| `impression_findings` | `[IMPRESSION] — A few non-specific … — No acute finding.`<br>`[FINDINGS] Brainstem, cerebellum … age-appropriate.` |
| `findings` | `Brainstem, cerebellum … age-appropriate.` |
| `impression` | `— A few non-specific chronic ischaemic-gliotic foci. — No acute finding.` |
| `clinical_findings_impression` | `[CLINICAL] Headache for three weeks. Rule out mass lesion.`<br>`[FINDINGS] …`<br>`[IMPRESSION] …` |
| `full_structured` | `[CLINICAL] …`<br>`[TECHNIQUE] 3-plane T1-weighted, FLAIR; …`<br>`[FINDINGS] …`<br>`[IMPRESSION] …` |
| `raw` | the whole original report verbatim, title line and headings included |
| `findings_impression_meta` | `[MODALITY] T1w [PLANE] AXIAL`<br>`[FINDINGS] …`<br>`[IMPRESSION] …` |

### Choosing between them

| | keeps the detail | keeps the conclusion | tokens | truncated at 512 (RadBERT) | works when a section is missing |
|---|---|---|---|---|---|
| `findings_impression` | yes | yes | medium | 9.2% | yes — empty sections are dropped |
| `impression_findings` | yes | yes | identical | 9.2% | yes |
| `findings` | yes | **no** | lower | 2.2% | yes |
| `impression` | **no** | yes | lowest | 0.01% | **empty for 8.9% of studies** |
| `clinical_findings_impression` | yes | yes | +2% | 9.6% | indication present for only 48% |
| `full_structured` | yes | yes | +18% | 16.5% | protocol text is train/test-inconsistent |
| `raw` | yes, plus noise | yes | +27% | 19.8% | includes other body regions |

**`findings_impression` is the default** because findings and impression carry complementary
information and neither alone is sufficient — findings has the localisation (median 143 words),
impression has the radiologist's synthesis (median 19 words) and is missing 8.9% of the time.
This also matches what the 2025 VLM3D CT-track winner used.

**`impression_findings` is worth knowing about.** It has *exactly the same content and exactly
the same token count* as the default. The only difference is what survives when a 512-token
encoder cuts the tail off: with `findings_impression` you lose the impression (the conclusion);
with `impression_findings` you lose the end of findings (detail). Since ~9% of reports truncate
at 512, this is free insurance. For an 8192-context encoder it changes nothing.

### Rules every format obeys

1. **Negation is never removed.** No format cleans, rewrites, summarises or samples sentences —
   they only *select and mark* released fields. `"No acute infarct"` reaches the encoder intact in
   every format that keeps findings. This is asserted by
   `tests/test_textenc_formats.py::test_negation_survives_every_format_that_keeps_findings`.
   It matters: about two thirds of MR-RATE's report content is the *absence* of disease.
2. **Empty sections are dropped, not emitted as a bare marker.** A study with no impression gets
   `[FINDINGS] …` and nothing else — never a lonely `[IMPRESSION]` with nothing after it.
3. **No format invents text.** The one apparent exception, `findings_impression_meta`, adds only
   values the *caller* supplies from structured metadata. It never parses anything out of the
   report.

### A warning about `findings_impression_meta`

It prepends `[MODALITY] T1w [PLANE] AXIAL` to the text. That is convenient — but it commits you to
having that metadata at challenge inference time, and the challenge's input schema is still
unpublished. If a test case arrives without it, the model sees a string shaped unlike anything it
trained on. Supplying modality as a *separate categorical embedding* (which the pipeline already
does) has no such problem: a missing category is just the null class.

Formats whose output depends on metadata are listed in `formats.METADATA_DEPENDENT_FORMATS`, and a
test fails if one is added without being declared — so the commitment can never be made by
accident.

### Selecting a format

```python
R2VDatasetConfig(report_format="impression_findings")     # in code
```
```bash
python -m mrrate_r2v.cli.train_r2v --report-format impression_findings ...   # on the CLI
```

Leave it unset and the pipeline behaves exactly as it did before formats existed
(`report_sections` joined by `ReportRecord.compose`), down to the same `cohort_id`.

---

## Part 2: encoders — which model turns words into numbers

### What is staged

```bash
python -m mrrate_r2v.cli.download_text_encoders --list   # prints this table, live, with a staged column
python -m mrrate_r2v.cli.download_text_encoders --all    # fetch everything (idempotent, ~4 GB)
```

| name | size | dim | max tokens | what it is |
|---|---|---|---|---|
| `radbert` | 125M | 768 | 512 | RoBERTa further trained on 4M radiology reports |
| `bioclinical_mbert` | 150M | 768 | **8192** | clinical ModernBERT; the only one whose published training data names brain-MRI reports |
| `medembed_large` | 335M | 1024 | 512 | medical retrieval model; used by the 2025 CT-track winner |
| `medembed_small` | **33M** | 384 | 512 | the lightweight option |
| `bio_clinicalbert` | 110M | 768 | 512 | MIMIC-III hospital notes |
| `cxr_bert` | 110M | 768 | 512 | chest-X-ray reports, CLIP-aligned |
| `modernbert` | 150M | 768 | 8192 | general English — a **control**, same architecture as `bioclinical_mbert` |
| `bge_base` | 110M | 768 | 512 | general English — a **control**, MedEmbed's own base model |

The two controls are not padding. `bioclinical_mbert` vs `modernbert` differ *only* in clinical
adaptation, and `medembed_large` vs `bge_base` differ *only* in medical fine-tuning. Without them,
"the medical encoder wins" is a claim you cannot test.

Measured results and licences: [`docs/TEXT_ENCODERS.md`](../../../docs/TEXT_ENCODERS.md).

### Loading one

```python
encoder = build_encoder("radbert")                          # cluster default checkpoint directory
encoder = build_encoder("radbert", checkpoint="/some/dir")  # explicit path
encoder = build_encoder("medembed_small", max_length=256)   # shorter token budget
encoder = build_encoder("radbert", trainable=True)          # allow fine-tuning (default: frozen)
```

The checkpoint directory defaults to `/hnvme/workspace/y100dc19-nvidia-mri-brain/pretrained` and
can be changed globally with the `MRRATE_PRETRAINED_DIR` environment variable.

### Four behaviours worth understanding before you rely on them

**1. Frozen by default, and it stays frozen.** A frozen encoder overrides `.train()`, so calling
`model.train()` on an enclosing module cannot silently switch dropout back on. If you actually
want gradients, pass `trainable=True` when you build it.

**2. Truncation is counted, never silent.** Reports longer than `max_length` get cut. The encoder
tokenises once without truncation first, purely to know what it is about to drop:

```python
encoder.log_truncation_summary()
# {'n_sequences': 5554, 'n_truncated': 517, 'fraction_truncated': 0.093,
#  'max_length': 512, 'longest_seen': 35196, 'dropped_tokens': 118034}
```

At a 512-token budget about **9% of MR-RATE reports lose their tail**. That is one report in
eleven, not a rounding error. Your options: use `bioclinical_mbert` (8192 tokens → 0.00%
truncated), or use `impression_findings` so the conclusion is at the front where the cut cannot
reach it.

**3. `max_length` is checked against the real checkpoint.** RoBERTa spends 2 of its 514 position
slots on an offset, so `radbert`'s true budget is 512. Asking for more raises an error when you
build the encoder, instead of a confusing CUDA index crash at training step 1.

**4. No `trust_remote_code`, anywhere.** `cxr_bert`'s checkpoint declares a custom model type
whose repository code would otherwise be executed on your machine at load time. Instead its
standard BERT weights are loaded into a stock `BertModel` and the unused heads are dropped.

---

## Part 3: fusion — using several encoders at once

The 2025 VLM3D CT-track winner concatenated three text encoders rather than picking one. This
package's version of that is `conditioning.SectionedFusionEmbedder` (Part 4, configurations D/E):
each encoder pools its own report section independently, and the pooled vectors are concatenated
into one wider token per section. `ProjectedConcatFusion` is its optional learned variant — a
per-encoder Linear to a shared width before concatenation, so a 1024-wide and a 384-wide encoder
contribute comparable magnitudes:

```python
from mrrate_r2v.textenc import ProjectedConcatFusion

fusion = ProjectedConcatFusion(input_dims=[1024, 768, 768], projection_dim=256)
```

Fusion costs what it says: three models means three forward passes, roughly triple the time and
memory, and a wider conditioning tensor. Only worth it if it measurably helps — which is what
`textbench` is for.

---

## Part 4: the named configurations

A *configuration* bundles an encoder set, a pooling decision and a report format under one name.
`--conditioning <name>` selects it; nothing else needs setting.

| | `--conditioning` | encoder | conditioning tensor |
|---|---|---|---|
| **A** | `cxr_bert_cls` | CXR-BERT | `(B, 1, 768)` |
| **B** | `cxr_bert_tokens` | CXR-BERT | `(B, n, 768)` + `(B, n)` mask |
| **C** | `radbert_tokens` | RadBERT | `(B, n, 768)` + `(B, n)` mask |
| **D** | `report2ct_style` | MedEmbed-large + Bio_ClinicalBERT + CXR-BERT | `(B, 2, 2560)` |
| **E** | `report2ct_style_meta` | MedEmbed-large + Bio_ClinicalBERT + CXR-BERT | `(B, 3, 2560)` |

**E is D plus one token.** A, B and C put `[MODALITY]/[PLANE]/[SPACING]` at the head of their
joined string (the `*_meta` formats), so the acquisition metadata reaches their text encoder as
well as reaching the UNet as `class_labels`/`spacing_tensor`. D encodes each section on its own
tokenizer and never joins them, so it has nowhere to put that prefix. E gives it a conditioning
token of its own — appended, so findings and impression keep sequence indices 0 and 1 and a D
result and an E result differ in exactly one token. The section text is composed by the Dataset
(`data/dataset.py`) from the manifest row and the resolved target spacing, never parsed from the
report, and is byte-identical to `meta_prefix_for`'s output; inference paths rebuild it with
`formats.with_acquisition_section`. Unlike impression (absent for 8.9% of studies) it is never
empty, so its mask entry is always True.

`superseded_radbert_mean` (pooled RadBERT, `(B, 1, 768)`) is out of the lineup but still runnable,
so adapters trained under it keep loading.

### The one rule: **`n = 1` makes the cross-attention a no-op**

Softmax over a single key is identically 1 for every query, so `to_q` and `to_k` cannot affect the
output and get no gradient — the report collapses to a per-channel bias applied uniformly at every
voxel. Measured after 40 real training steps, `to_q` gradient norm:

```
A (n=1)   1.2e-12   <- roundoff, i.e. dead      33.8% of adapter params inert
B (n≤512) 4.4e-07
C (n≤512) 8.0e-07
D (n=2)   2.7e-07
```

Keeping the token axis costs nothing measurable: 29 s per 40 steps and 13.8 GiB peak, identical to
`n=1`. **A is kept as a pooled baseline** (CXR-BERT's CLS *was* CLIP-trained to summarise, and it is
the exact form CTFlow uses), not as a recommendation.

### `n` is variable, and that is fine

The tokenizer pads each batch to *its own* longest report, capped at `--max-report-tokens` (512),
so `n` changes from batch to batch — and differs per tokenizer (the same text is 194 CXR-BERT
tokens, 243 RadBERT tokens). Reports are tokenised **as a batch**, never encoded separately and
stacked, so there is no shape to reconcile. Padding is carried in the mask and dropped from the
attention softmax, so a sample's conditioning never depends on its batchmates.

Nothing resamples `n`: `ContextProjection` maps `(B, n, D) → (B, n, cross_attention_dim)`.

### Choosing `--max-report-tokens`

The encoder context is the binding constraint, not the report statistics. Measured over 8,000 train
reports at `findings_impression_meta`:

| | mean | median | p95 | p99 | truncated at 512 |
|---|---|---|---|---|---|
| CXR-BERT | 266 | 242 | 469 | 629 | **3.2%** (1.2% of tokens) |
| RadBERT | 350 | 321 | 607 | 805 | **11.8%** (3.9% of tokens) |

Both encoders hard-cap at 512, so p99 coverage is not purchasable — RadBERT loses part of ~1 report
in 8. `bioclinical_mbert` (768-wide, 8192 context, staged) is the only staged way to remove
truncation entirely.

### Setting it

```bash
python -m mrrate_r2v.cli.train_r2v --conditioning cxr_bert_tokens ...   # or via Slurm:
sbatch --export=ALL,R2V_CONFIG=B slurm/11_train_conditioning.sbatch     # A | B | C | D
```

`--report-format` defaults to the configuration's own recommendation and rarely needs setting.
`--text-pooling` applies only to pooled configurations; passing it to a `_tokens` one is an error
rather than a silent no-op.

---

## Where the numbers go next

`encoder.output_dim` is read by `ContextProjection` inside `ReportConditionedUNetMaisi`, which
learns a projection from that width to the denoiser's `cross_attention_dim`. That projection is
the trainable part; the encoder is not. Swapping a 768-wide encoder for a 1024-wide one therefore
needs **no code change** — just rebuild the adapter.

---

## Common errors

| message | cause | fix |
|---|---|---|
| `checkpoint directory not found` | model not staged | `python -m mrrate_r2v.cli.download_text_encoders --encoders <name>` |
| `max_length=N exceeds what '<x>' supports` | asked for more tokens than the model has | lower it, or switch to `bioclinical_mbert` |
| `contains custom code which must be executed` | loaded `cxr_bert` via plain `AutoModel` | use `build_encoder("cxr_bert")` |
| `unknown report format 'findings_only'` | typo — the error lists the valid names | see the format table above |
| `unknown conditioning configuration '<x>'` | not one of the four names | see Part 4 |
| `--text-pooling ... has nothing to act on` | pooling passed to a `_tokens` configuration | drop the flag, or use `cxr_bert_cls` |
| `adapter checkpoint was trained under a different conditioning configuration` | loading e.g. a `cxr_bert_cls` adapter under `cxr_bert_tokens` (both 768-wide) | load it under the configuration named in the checkpoint's `config['conditioning_name']` |
| out of GPU memory | `medembed_large` is 3× the others | smaller batch, or `dtype=torch.bfloat16` |
