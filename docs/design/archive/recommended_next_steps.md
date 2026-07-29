# Recommended Next Steps — MR-RATE Report-to-Volume Project

Context: this audit found MR-RATE's native-space release to be anonymized, converted, classified, and defaced, but **not** skull-stripped, reoriented, resampled, or intensity-normalized — see `mr_rate_local_audit.md` for the full evidence base. Terminology note: the target task is **report-to-volume generation / text-conditioned 3D medical image synthesis**, architecturally closest to **3D latent diffusion** (cf. NVIDIA's `NV-Generate-MR-Brain`, built on this same dataset), not a "VLM" in the discriminative-alignment sense already implemented in `contrastive-pretraining/`.

---

## Minimum Viable Pilot

Goal: prove out the training-unit design and geometry strategy on a small, fast-iterating slice before any large-scale engineering investment.

1. **Pick one modality + one plane** (recommend T1w axial — the largest single stratum, ~231,800 series dataset-wide) and restrict to `image_present=True` studies outside the known-gap batches (avoid 04/14/15/16/27 for this first pass).
2. **Build a small (~500-1,000 study) manifest** using the schema in `proposed_model_manifest_schema.json`, joining `series.parquet`, the official `splits.csv`, and the reports CSV — filtered to studies with `has_report=True` and no `source_read_error`.
3. **Implement one geometry strategy** (recommend: resample to a fixed physical spacing, then center-crop/pad to a fixed FOV, explicitly accounting for the defacing-induced anterior asymmetry the way the existing contrastive loader's "posterior shift" does) and validate it visually on a handful of cases before scaling.
4. **Train a small-scale 3D VAE + diffusion pilot** on this slice, conditioned on report text (a frozen text encoder is fine for a pilot — the existing BiomedVLP-CXR-BERT is a reasonable starting point, though a general clinical-text encoder may condition better on full narrative reports than a chest-X-ray-tuned one).
5. **Sanity-check outputs**: does the generator produce plausible brain anatomy at all before investing in report-conditioning fidelity? A pilot that can't produce a passable unconditional brain MRI isn't ready for text-conditioning evaluation.

## Production-Scale Preprocessing

1. **Resolve the SHARDS_PATH question first**: since it's already built to match a named "MR Volume Generation" challenge format, confirm (once network/challenge-doc access is available) whether it's the right foundation to build on before investing in a parallel pipeline from DATA_PATH's raw tars.
2. **Build the QC-filtering pass**: exclude the 2,707 corrupt/missing series and the 43 `ok_zero_series` studies; flag (don't silently drop) the 83-series-in-one-study outlier and similar extremes for manual review.
3. **Decide and document the geometry strategy dataset-wide**, not per-pilot-slice — given the near-total heterogeneity found (36/37 unique shapes in-sample), this needs a principled default (fixed spacing + crop/pad, or a geometry-conditioned generator) rather than per-series ad hoc handling.
4. **Decide and validate the intensity-normalization strategy per dtype** (uint16 vs. float32 showed a >100x difference in observed maxima in this audit's sample) — do not reuse the contrastive loader's normalizer defaults without re-validating them for a generative objective.
5. **If a shared-reference-frame geometry is chosen**, plan the ~17.6TB coreg-derivative download explicitly (currently absent from DATA_PATH) rather than discovering the need mid-project.
6. **Build the report-deduplication/near-duplicate detection pass** that does not exist anywhere in the upstream pipeline (a genuine gap, more consequential for a generative model than the existing discriminative one).

## Training

1. Start from the **report + explicit conditioning (modality/plane/contrast-state/skull-state) → one series** unit-of-training (see `report2volume_gap_analysis.md` for the tradeoff discussion versus study-level or naive per-series-with-shared-report alternatives).
2. Architecturally, prioritize **3D latent diffusion** (VAE + diffusion/flow-matching in latent space, text cross-attention conditioning) given the directly relevant NV-Generate-MR-Brain prior art on this exact dataset; keep 3D GAN and autoregressive-token alternatives as fallback options if diffusion training proves too resource-intensive at your available compute scale.
3. Use the **official patient-level splits.csv** directly (independently verified clean in this audit) rather than re-deriving splits — but add an explicit patient-level isolation check in your own data loader, since no upstream loader code currently enforces this at runtime.
4. Consider **curriculum by stratum**: start with the largest, cleanest strata (T1w/T2w/FLAIR axial, uint16) before extending to the long tail (MRA at 0.02% of series, float32 series, rare plane/spacing combinations).

## Evaluation

Per `mr_rate_local_audit.md` §10, avoid a single unstratified FID. Recommended layered protocol:
1. **VAE reconstruction quality** (if using a latent-diffusion architecture) — per-modality, per-plane reconstruction error, checked before any diffusion training begins.
2. **Generative distribution quality, stratified** by modality and plane at minimum (dtype too, given the intensity-scale finding) — unstratified aggregate metrics will hide minority-stratum failure.
3. **Anatomical validity** — structural plausibility checks (e.g., via the existing brain-mask/segmentation tooling, or a held-out segmentation model) rather than relying on perceptual metrics alone.
4. **Report-image semantic consistency** — e.g., using the existing contrastive MR-RATE model (`contrastive-pretraining/`) as a zero-shot judge of whether generated volumes match their conditioning report, given it was trained on this exact image-report distribution.
5. **Pathology fidelity** — spot-check whether reports describing specific pathologies (from the 37/32/14-category label sets already available) produce visually consistent findings, acknowledging the label sets themselves have documented reliability caveats (5 of 37 categories were dropped from the shipped set for low labeling agreement).
6. **Diversity and memorization/privacy risk** — this is real patient-derived medical data; a generative model that memorizes and reproduces training volumes (or, combined with report text, could support re-identification) is a genuine risk category, not a theoretical one, given this dataset's CC BY-NC-SA license and IRB-approved-but-still-sensitive nature.
7. **Downstream utility** — e.g., does synthetic data improve a held-out pathology classifier's performance when added to real training data.
8. **Radiologist review** — qualitative, ideally stratified by modality/pathology, as the final check before any claim of clinical-plausibility.

## Unresolved Questions Requiring Clinical or Research Decisions

These cannot be resolved by further code/data auditing — they require a decision from you (and possibly clinical or challenge-organizer input):

1. **Does the target "MR Volume Generation" challenge (per SHARDS_PATH's `.forithmus/config.json`) have a specific required data format, geometry, or evaluation protocol?** This audit could not check (no network access) and this should be resolved before further engineering investment in the shard pipeline.
2. **Should the generator target skull-containing (native, as-is) or skull-stripped output?** Native-space MR-RATE is uniformly defaced-but-not-stripped; producing skull-stripped output requires an explicit additional preprocessing decision, not something the released data does for you.
3. **Is study-level report information sufficient for series-level conditioning, or does this project need to invest in extracting series-specific findings from the study-level report text (e.g., via an LLM pass that attributes findings to likely sequences)?** This is a real modeling-quality tradeoff, not just an engineering one.
4. **What is an acceptable privacy/memorization risk threshold for this project**, given the dataset's CC BY-NC-SA license, IRB approval scope (Istanbul Medipol University, per the dataset card), and the unresolved anonymization-method-auditability gap found in this audit?
5. **Should contrast-enhancement state be inferred (via the weak SeriesNumber heuristic, or a purpose-built classifier) before training, or should the project accept "contrast state unknown" as a conditioning gap for v1?**
6. **Is the coreg derivative (currently absent locally, 17.6TB) worth downloading**, i.e., does the chosen geometry strategy actually need a shared per-study reference frame, or is per-series native geometry (with explicit conditioning) sufficient?
7. **How should the 5 known-gap batches (04/14/15/16/27) be handled** — excluded entirely from v1 training, or included for their intact subset with the gap treated as expected missingness?
