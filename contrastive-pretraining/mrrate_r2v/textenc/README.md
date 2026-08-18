# `textenc` — turning a radiology report into numbers the generator can use

## What problem this solves

A neural network can't read text, so something has to turn a radiology report into a tensor
the image generator can attend over. This package answers two questions:

1. **Which words do we feed in?** The released report has four sections, and you don't have to
   use all of them. → `formats.py`
2. **Which pretrained model turns those words into numbers?** → `encoders.py`

A third file, `fusion.py`, handles using more than one encoder at once.

Everything here is production code — it runs during training and generation. The
encoder/format choice baked in here rests on a selection study; see
[`docs/TEXT_ENCODERS.md`](../../../docs/TEXT_ENCODERS.md) for its methodology and findings, and
[`../DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) for the measured evidence behind the defaults
below.

---

## The 30-second version

```python
from mrrate_r2v.textenc import build_encoder, format_report

encoder = build_encoder("bioclinical_mbert")            # loads a frozen pretrained model
text    = format_report(record, "findings_impression")  # record: a data.reports.ReportRecord
cond    = encoder.encode([text], device)                # the actual encoding step
```

`cond` is a `TextConditioning` object:

| field | shape | meaning |
|---|---|---|
| `cond.token_embeddings` | `(B, L, D)` | one vector per token — what the generator attends over |
| `cond.attention_mask` | `(B, L)` bool | `True` = a real token, `False` = padding. Never invert this. |
| `cond.pooled_embedding` | `(B, D)` | the whole report squashed to one vector, for logging/probes |

In practice you don't assemble this by hand: `--conditioning <name>` selects one of five named
configurations that fix the encoder, pooling, and format together — see **Part 4**.

---

## Part 1: report formats — which words go in

MR-RATE ships each report split into sections (`clinical_information`, `technique`, `findings`,
`impression`, plus `raw`, the unsplit original). Feeding all of it in isn't obviously right —
`technique` describes the whole exam's protocol, not the volume being generated;
`clinical_information` is missing for 52% of studies; `raw` often contains other body regions'
findings. So the formats below are the small set of sensible answers, each a named,
deterministic function.

| format | what it keeps |
|---|---|
| **`findings_impression`** ← default | findings, then impression |
| `impression_findings` | impression, then findings (same content/length, different truncation behavior) |
| `findings` | findings only |
| `impression` | impression only (missing for 8.9% of studies) |
| `clinical_findings_impression` | + the referring question |
| `full_structured` | + technique too |
| `raw` | the whole original report, verbatim |
| `findings_impression_meta` | + a `[MODALITY]/[PLANE]/[SPACING]` prefix |

**`findings_impression` is the default** because findings and impression carry complementary
information (localisation vs. the radiologist's synthesis) and neither alone is sufficient; this
also matches what the 2025 VLM3D CT-track winner used. Every format obeys three rules: negation
is never removed (no format cleans, rewrites, or summarises — only selects and marks released
fields); empty sections are dropped rather than emitted as a bare heading; and no format invents
text (the one exception, `*_meta`, only adds values the *caller* supplies from structured
metadata, never parsed from the report).

`findings_impression_meta`'s metadata prefix is convenient, but commits you to having that
metadata at inference time — see [`DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) for the caveat.

### Selecting a format

```python
R2VDatasetConfig(report_format="impression_findings")     # in code
```
```bash
python -m mrrate_r2v.cli.train_r2v --report-format impression_findings ...   # on the CLI
```

Leave it unset and the pipeline behaves exactly as it did before formats existed
(`report_sections` joined together).

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
| `bioclinical_mbert` | 150M | 768 | **8192** | clinical ModernBERT — the only one whose published training data names brain-MRI reports |
| `medembed_large` | 335M | 1024 | 512 | medical retrieval model; used by the 2025 CT-track winner |
| `medembed_small` | **33M** | 384 | 512 | the lightweight option |
| `bio_clinicalbert` | 110M | 768 | 512 | MIMIC-III hospital notes |
| `cxr_bert` | 110M | 768 | 512 | chest-X-ray reports, CLIP-aligned |
| `modernbert` | 150M | 768 | 8192 | general English — a control for `bioclinical_mbert` |
| `bge_base` | 110M | 768 | 512 | general English — a control for `medembed_large` |

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

### Worth knowing before you rely on it

- **Frozen by default, and it stays frozen** — pass `trainable=True` to get gradients.
- **Truncation is counted, never silent**: `encoder.log_truncation_summary()` reports how many
  reports got cut. At a 512-token budget, ~9% of MR-RATE reports lose their tail — use
  `bioclinical_mbert` (8192 tokens) if that matters, or `impression_findings` so the conclusion
  sits where the cut can't reach it.
- **No `trust_remote_code`, anywhere** — even for encoders whose checkpoint declares custom
  model code.

---

## Part 3: fusion — using several encoders at once

`conditioning.SectionedFusionEmbedder` (configurations D/E below) pools each encoder's own
report section independently and concatenates the pooled vectors into one wider token per
section. `ProjectedConcatFusion` is its optional learned variant — a per-encoder `Linear` to a
shared width before concatenation:

```python
from mrrate_r2v.textenc import ProjectedConcatFusion
fusion = ProjectedConcatFusion(input_dims=[1024, 768, 768], projection_dim=256)
```

Fusion costs roughly triple the time and memory of a single encoder — only worth it if it
measurably helps (see `docs/TEXT_ENCODERS.md`).

---

## Part 4: the named configurations

A *configuration* bundles an encoder set, a pooling decision, and a report format under one
name. `--conditioning <name>` selects it; nothing else needs setting.

| | `--conditioning` | encoder | conditioning tensor |
|---|---|---|---|
| **A** | `cxr_bert_cls` | CXR-BERT | `(B, 1, 768)` — pooled |
| **B** | `cxr_bert_tokens` | CXR-BERT | `(B, n, 768)` + mask |
| **C** | `radbert_tokens` | RadBERT | `(B, n, 768)` + mask |
| **D** | `report2ct_style` | MedEmbed-large + Bio_ClinicalBERT + CXR-BERT | `(B, 2, 2560)` |
| **E** | `report2ct_style_meta` | same as D | `(B, 3, 2560)` |

**Prefer a token-sequence configuration (B, C, D, or E) over a pooled one (A).** A single
conditioning token makes the adapter's cross-attention a no-op — see
[`DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) for the measured gradient evidence. A is kept as a
pooled baseline, not a recommendation.

### Setting it

```bash
python -m mrrate_r2v.cli.train_r2v --conditioning cxr_bert_tokens ...   # or via Slurm:
sbatch --export=ALL,R2V_CONFIG=B slurm/train_conditioning.sbatch     # A | B | C | D | E
```

`--report-format` defaults to the configuration's own recommendation and rarely needs setting.
`--text-pooling` applies only to pooled configurations.

---

## Common errors

| message | cause | fix |
|---|---|---|
| `checkpoint directory not found` | model not staged | `python -m mrrate_r2v.cli.download_text_encoders --encoders <name>` |
| `max_length=N exceeds what '<x>' supports` | asked for more tokens than the model has | lower it, or switch to `bioclinical_mbert` |
| `contains custom code which must be executed` | loaded `cxr_bert` via plain `AutoModel` | use `build_encoder("cxr_bert")` |
| `unknown report format 'findings_only'` | typo — the error lists the valid names | see the format table above |
| `unknown conditioning configuration '<x>'` | not one of the five names | see Part 4 |
| `--text-pooling ... has nothing to act on` | pooling passed to a `_tokens` configuration | drop the flag, or use `cxr_bert_cls` |
| `adapter checkpoint was trained under a different conditioning configuration` | loading a mismatched adapter | load it under the configuration named in the checkpoint's `config['conditioning_name']` |
| out of GPU memory | `medembed_large` is 3× the others | smaller batch, or `dtype=torch.bfloat16` |
