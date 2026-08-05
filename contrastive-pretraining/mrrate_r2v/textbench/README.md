# `textbench` — the experiment that picks the encoder and the report format

## What this package is for

[`textenc/`](../textenc/README.md) offers **8 pretrained text encoders** and **8 report formats**.
That is 64 combinations. Somebody has to decide which one to actually train the generator with.

This package is that decision, made with measurements instead of intuition. It produces one table:

```
encoder              report_format         pathology  bucket  negΔ    nn Δ   trunc%   rep/s
bioclinical_mbert    findings_impression      0.87     0.94   0.31   +0.19    0.00      112
radbert              findings_impression      0.85     0.93   0.22   +0.16    9.20       61
medembed_small       impression               0.81     0.88   0.29   +0.14    0.01      210
...
```

…and you pick a row. (Those numbers are illustrative — the real ones are in
[`docs/TEXT_ENCODERS.md`](../../../docs/TEXT_ENCODERS.md).)

**Nothing here runs during training or generation.** It is a one-off study. No module in the
production path can even import it — `tests/test_textbench.py::test_production_modules_never_import_the_benchmark`
fails if one does.

---

## The obvious problem, and how it is solved

We want to know *"which encoder produces the best generated brain MRI?"* But answering that
directly means training the whole diffusion model 64 times. That is impossible.

So we ask a cheaper question that stands in for it:

> **Does the encoder's output still contain the information the generator would need?**

If an encoder's 768 numbers do not distinguish "infarct present" from "infarct absent", then no
amount of downstream training can make the generator render that difference. The information is
already gone. Measuring what *survives the encoder* is a ceiling on what the generator can do,
and it costs CPU-minutes instead of GPU-weeks.

That is the whole idea. Everything below is machinery for asking it carefully.

---

## How it works, end to end

```
   ┌─ STEP 1 ─ collect the text ─────────────────────────────────────────────┐
   │  cli.analyze_reports build-corpus                                        │
   │     shard tars ──► reports_all.jsonl                                     │
   │     one line per study: the 4 report sections + its 37 pathology labels   │
   │     CPU, ~3 minutes, run once ever                                       │
   └──────────────────────────────────────────────────────────────────────────┘
                                    │
   ┌─ STEP 2 ─ encode it ────────────────────────────────────────────────────┐
   │  cli.embed_reports                                                       │
   │     for each encoder, for each format, for train and test:               │
   │        format the report ──► run the frozen encoder ──► save the vectors │
   │     ──► embeddings/<encoder>__<format>__<split>.npz                      │
   │     THE ONLY EXPENSIVE STEP. ~1 GPU-hour, or a few CPU-hours.            │
   └──────────────────────────────────────────────────────────────────────────┘
                                    │
   ┌─ STEP 3 ─ score it ─────────────────────────────────────────────────────┐
   │  cli.eval_text_encoders                                                  │
   │     load the saved vectors, fit small linear probes, compute 5 metrics   │
   │     ──► metrics_matrix.csv                                               │
   │     CPU, ~15 minutes. No model is loaded.                                │
   └──────────────────────────────────────────────────────────────────────────┘
```

**Why steps 2 and 3 are separate.** The encoding is slow; the scoring is fast. By saving the
vectors and baking *no labels* into them, adding a new metric or swapping the label set is a
15-minute CPU job instead of another day of encoding. This is the same trick the contrastive
pipeline uses for `extract_features.py` + `linear_probe.py`.

---

## The five metrics, in plain terms

Each row of the output table gets five quality numbers plus three cost numbers.

### 1. `pathology_probe_auroc` — "can you still tell what disease was described?"

Take the encoder's output for a report. Fit a **linear probe** — the simplest possible
classifier, a single weighted sum — to predict "does this report mention a cerebral infarct?"
Train the probe on the training split, score it on the test split, repeat for every pathology
that occurs in at least 1% of reports, and average.

- **0.5** = the encoder's output tells you nothing about pathology.
- **1.0** = pathology is perfectly and linearly readable.

Why linear and not something stronger: we want to know whether the information is *easily
accessible*, not whether it is buried somewhere recoverable. Cross-attention in a diffusion model
is closer to a linear read than to a deep classifier.

**This is the metric closest to how the challenge scores you.** The challenge uses a "blinded
classifier consistency" check — does a classifier looking at your generated volume assign the
labels the report described? If the conditioning embedding cannot express a label, the generator
cannot render it.

### 2. `bucket_probe_auroc` — "can you still tell which scan it was?"

Same idea, but predicting the 10 `(modality, plane)` combinations — `T1w__AXIAL`,
`FLAIR__SAGITTAL`, and so on.

This one is special because the labels come from **DICOM metadata**, not from the text. It is the
one quality metric with a label source completely independent of the reports.

### 3. `negation_delta` — "does 'no infarct' land somewhere different from 'infarct'?"

The one that matters most clinically, and the one encoders most often fail.

About two thirds of MR-RATE's report content is the *absence* of disease. "No acute infarct" and
"acute infarct" share almost every word. An encoder that maps them to nearly the same place will
make the generator draw a stroke into a scan that explicitly ruled one out.

To measure it, we take real negated sentences from the reports and build a **controlled
counterfactual** by deleting only the negation cue:

```
real from the report : "No acute infarct was detected on diffusion ADC sections."
counterfactual       : "Acute infarct was detected on diffusion ADC sections."
```

Then: how far apart does the encoder put those two, compared to how far it puts the original from
a sentence about a *completely different* finding?

```
negation_delta = (distance moved by flipping polarity) / (distance moved by changing the topic)
```

- **0.0** = flipping "no infarct" to "infarct" moves the embedding not at all. Polarity ignored.
- **1.0** = flipping polarity moves it as far as changing the finding entirely.

Higher is better. It is a ratio of two distances measured from the same starting point, so it is
comparable across encoders whose absolute distance scales differ wildly (they do: some checkpoints
put every report at cosine ~0.95 of every other, some at ~0.71).

> **Those counterfactual sentences are a ruler, not data.** They are clinically false by
> construction. They exist in memory for one `encode()` call and are thrown away — never saved as
> text, never attached to a study, never given a label, never used as conditioning text or
> training data. What reaches disk is a float16 array of vectors and a one-word topic tag. Three
> tests enforce this, and the production packages cannot import this module. See the top of
> `negation.py`.

We *also* report `negation_auroc` (can a probe classify polarity at all), but it comes out above
0.999 for every single encoder — the negation cue is a literal word, so it is trivially
detectable. It is kept as a floor check, not as a way to tell encoders apart. **Rank on
`negation_delta`.**

### 4. `nn_jaccard_delta` — "are clinically similar reports close together?"

For each test report, find its nearest neighbour in embedding space, then check how much their
pathology labels overlap (Jaccard). Subtract the same quantity for randomly paired reports.

The subtraction is essential: 44% of MR-RATE studies have no positive pathology at all, so an
encoder that collapsed every report to a single point would score a *high* raw overlap. After
subtracting the random baseline it scores 0, which is the honest answer.

### 5. `sim_spearman` — "does cosine similarity track clinical similarity?"

Over 200,000 random report pairs, correlate embedding cosine similarity against label-set
overlap. Answers the smooth version of metric 4: not just "is the closest one similar" but "does
distance mean anything at all".

### Plus three costs, which are part of the decision

`embed_dim` (how wide the conditioning tensor is), `truncated_pct` (share of reports that lost
their tail), `reports_per_second`. A model that wins by 0.005 AUROC while being 3× slower and
truncating 10% of reports is not the winner.

---

## Two honest caveats you must carry with the numbers

### The weak-label caveat

`pathology_probe_auroc`, `nn_jaccard_delta` and `sim_spearman` use `labels.json`, which was
produced by **an LLM reading the same report the encoder is reading**.

So a high score proves the embedding kept the information the labeller also extracted from that
text. It does **not** prove the label is clinically correct. These three numbers are valid for
**ranking encoders against each other** — every encoder faces identical labels, identical splits,
identical studies — and invalid as absolute measures of clinical accuracy.

`bucket_probe_auroc` (labels from DICOM) and `negation_delta` (labels from rule-based
construction) do not have this problem. This warning is written into `summary.json` so it travels
with the numbers rather than living only here.

### The pooled-embedding caveat

The metrics score a *summary vector* per report (mean and max over tokens), but the generator
cross-attends over the **full token sequence**. An encoder that spreads information thinly across
many tokens is undersold by these metrics. Using `concat(mean, max)` rather than mean alone
reduces the effect; it does not remove it.

---

## Fair comparison: what is held constant

- **Same studies for every encoder.** 20,000 seeded-random training studies, all 5,554 test
  reports.
- **Probes never see the test split.** They fit on `train`, are scored on `test`.
- **No patient appears in both.** MR-RATE's splits are patient-isolated (the release's own check
  reports 0 violations).
- **Same token budget.** `--max-length 512` is applied to every encoder in the main run, so no
  model wins by being allowed to read more. The 8192-context models get a *separate* run to
  measure what long context is worth — never mixed into the same table.
- **A missing result is reported, not averaged over.** If a cache is absent the pair is listed in
  `skipped_missing_cache`, not silently dropped from a mean.
- **The cache stores its own study ids.** If you score against a different corpus or a different
  `--train-limit`, it fails loudly instead of quietly comparing different data.

---

## Running it

```bash
cd contrastive-pretraining

# once ever: pull the reports and labels out of the shard tars (~3 min, CPU)
python -m mrrate_r2v.cli.analyze_reports build-corpus \
    --shards-root /hnvme/workspace/y100dc19-MR-Rate-raw \
    --out /hnvme/workspace/y100dc19-nvidia-mri-brain/cache/r2v/report_analysis/reports_all.jsonl

# dataset statistics: lengths, sections, headings, negation, what the text says about acquisition
python -m mrrate_r2v.cli.analyze_reports analyze \
    --corpus       .../reports_all.jsonl \
    --manifest-csv /hnvme/workspace/y100dc19-MR-Rate-raw/r2v_manifest/manifest_shards_native.csv \
    --out          .../analysis.json

# token lengths and truncation rate, per encoder tokenizer x report format
python -m mrrate_r2v.cli.analyze_reports tokens --corpus .../reports_all.jsonl --out .../token_lengths.json
```

On the cluster, the two expensive stages:

```bash
sbatch slurm/08_embed_reports.sbatch 512 20000 budget512          # step 2 -- GPU
sbatch slurm/09_eval_text_encoders.sbatch budget512 20000 \       # step 3 -- CPU
       bioclinical_mbert+radbert medembed_large+radbert           #   (optional fusion pairs)
```

The scoring job runs on the host python rather than in a container, because it needs
scikit-learn, which neither container image has. It loads no model and touches no GPU.

### Testing a two-encoder combination

`--fusion encA+encB` concatenates two already-cached encoders' vectors and scores the result as if
it were a ninth encoder. Because the vectors are already on disk, this costs **no extra
encoding** — it is the cheap way to find out whether fusion is worth the doubled inference cost
before committing to it in training.

---

## What each file does

| file | needs torch? | job |
|---|---|---|
| `corpus.py` | no | read `report.json` + `labels.json` out of the shard tars; attach `(modality, plane)` from the manifest |
| `analysis.py` | no | dataset statistics — lengths, sections, headings, negation rates, acquisition content |
| `negation.py` | no | build the negation counterfactual pairs (read the containment note at the top) |
| `embed.py` | yes | run the encoder, save `.npz` vectors atomically |
| `tasks.py` | numpy + sklearn | the five metrics |
| `runner.py` | numpy + sklearn | **the single scoring path** — CLI and tests both call `run_benchmark` |

`corpus.py`, `analysis.py` and `negation.py` deliberately avoid torch so the analysis half runs on
any interpreter. `__init__.py` re-exports nothing, so a heavy dependency in one module cannot make
another unimportable.

## Output files

| file | contents |
|---|---|
| `metrics_matrix.csv` | one row per (encoder, format), one column per metric — **the deliverable** |
| `per_label_auroc.csv` | the pathology probe broken out by individual disease |
| `summary.json` | the same numbers plus every run's provenance, truncation stats and the weak-label warning |

## Privacy

`study_uid` — already the released anonymised id — is the only key kept. The analysis emits counts
and statistics, never verbatim clinical text; a test asserts a whole report cannot appear in a
serialised analysis output.
