# MICCAI 2026 VLM3D MRI Brain Report-to-Volume Challenge — Provisional Contract

**Date:** 2026-07-28 (originally written no-network; re-run same day with full internet access — see §0.1). **Scope:** read-only documentation analysis, no cluster jobs, no dataset/model changes. **Constraint:** no patient/study identifiers or report text are reproduced below — only schema/field names and platform mechanics.

Confidence key used throughout: **VERIFIED** (directly supported by a local document/code, or a directly-fetched web page, cited) / **INFERRED** (strongly suggested by evidence — local or a search-engine-synthesized summary of an unfetchable page — but not an explicit direct quote) / **ASSUMED** (a provisional choice made here because the real spec is silent) / **UNKNOWN** (cannot currently be determined).

---

## 0. Resolved location and documents inspected

Resolved path: `docs/challange_docs/` (relative to `MR-RATE/`, i.e. `/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE/docs/challange_docs/`). All 11 files present were read in full:

| File | Subject |
|---|---|
| `Quick_Start.md` | Platform overview; participant and host onboarding workflow; challenge/task type catalog |
| `Data_Schemas_Mock_Data.md` | How input/output schema is auto-detected; mock-data generation; output validation |
| `Evaluation_Pipeline.md` | Full submission pipeline (upload → scan → validate → registry → execute → checkpoint → evaluate → score → settle); evaluation-container contract |
| `File_Submissions.md` | Non-Docker prediction-file submission (JSON/CSV/ZIP) |
| `Docker_Submission.md` | Docker submission contract, directory layout, compute tiers, size/time limits |
| `Leaderboard_Ranking.md` | Ranking methods, hidden metrics, best-per-participant rule |
| `Phases_Editions.md` | Phase/edition structure, activation prerequisites, submission limits |
| `Reproducibility.md` | Artifact retention policy (Docker images, outputs, checkpoints) after archiving |
| `CLI_Tool.md` | `forithmus` CLI command reference |
| `Checkpoints_and_Continuation.md` | Checkpoint/resume mechanics, incl. under preemption |
| `Billing_Credits.md` | Wallet, sponsor pools, pricing, refund policy |
| `Spot_Instances.md` | Preemptible-VM mechanics, retry/fallback behavior |

Also consulted for cross-verification (not part of `docs/challange_docs/`, but locally available and directly relevant):
- `CLAUDE.md` (repo root) — architecture of both submodules, MR-RATE report/label schema.
- `README.md` (repo root), `data-preprocessing/docs/dataset_guide.md` — grepped for `VLM3D`/`Report-to-Volume`/`R2V`/`challenge`; **zero matches**.
- `docs/design/mr_rate_local_audit.md`, `logs/mr_rate_audit_metrics.json`, `docs/design/report2volume_gap_analysis.md`, `docs/design/recommended_next_steps.md`, `docs/design/audit_progress.md`, `logs/mr_rate_dataset_contract.json` — the pre-existing MR-RATE local audit, read per task instructions and independently re-verified where reused below.
- `/hnvme/workspace/y100dc19-MR-Rate-raw/validation/.forithmus/config.json` and `/hnvme/workspace/y100dc19-MR-Rate-raw/dataset.json` — outside the repo (a separate cluster workspace, "SHARDS_PATH" in the prior audit), re-read directly in this session (not just trusted from the prior audit's summary) because it is the only locally-reachable artifact naming the actual target challenge.

**Web sources consulted 2026-07-28 (§0.1), by access outcome:**
- **Directly fetched (higher confidence):** `huggingface.co/datasets/Forithmus/MR-RATE` (dataset card), `github.com/forithmus/MR-RATE` (README), `huggingface.co/nvidia/NV-Generate-MR-Brain` (model card), `developer.nvidia.com` NV-Generate-MR-Brain blog post, `zenodo.org/records/15052708` (2025/CT-edition challenge deposit), `voxel51.com/blog/mr-rate-brain-mri-dataset-fiftyone`, `conferences.miccai.org/2026/en/challenges.asp` (official MICCAI 2026 challenge listing), `conferences.miccai.org/2026/en/CALL-FOR-CHALLENGES.html`.
- **Blocked (HTTP 403 on every attempt) — not read, findings below are second-hand only:** `research.forithmus.com/collections/vlm3d-challenge`, `research.forithmus.com/getting-started`, `forithmus.com/`, `forithmus.com/leaderboard`, `ucd.ie/medicine/miua2026/...` (program page and program PDF).
- **DNS no longer resolves (decommissioned):** `reportgen.vlm3dchallenge.com`, `ctgen.vlm3dchallenge.com`, `abnclass.vlm3dchallenge.com` (2025-edition per-task pages, still indexed by search engines as cached snippets, no longer live). `vlm3dchallenge.com` root now 301-redirects permanently to `research.forithmus.com/collections/vlm3d-challenge`.
- **Search-engine query synthesis only (weakest tier, used where nothing else was available):** claims about the MIUA 2026 preliminary round's exact scope, and about the 2025 edition's exact task-page wording, are search-summary reconstructions of pages this session could not fetch directly — graded INFERRED throughout, never VERIFIED.

### Central finding (VERIFIED)

**None of the 11 files in `docs/challange_docs/` mention MRI, brain, reports, volumes, VLM3D, "Report-to-Volume," or any MICCAI-specific task detail.** They document **Forithmus Research Hub, a generic multi-task challenge-hosting platform** — the same mechanics apply whether a challenge on it is 3D segmentation, classification, detection, regression, report generation, image generation, reconstruction, or "custom" (`Quick_Start.md:151-192`, especially the "Common task types" list at `Quick_Start.md:172-191`). Every requirement extracted from these 11 files is therefore a statement about **how the hosting platform works**, not about **what this specific challenge requires**.

The only locally-available artifact that names the actual target challenge sits outside this repository and outside `docs/challange_docs/`: `/hnvme/workspace/y100dc19-MR-Rate-raw/validation/.forithmus/config.json`, independently re-read in this session:

```json
{
  "challenge": "mr-volume-generation",
  "challenge_title": "MR Volume Generation",
  "phase": "main",
  "phase_id": "0edb5ef1-2030-4556-ab59-b26f94d3646a",
  "phase_name": "Main",
  "submission_type": "docker",
  "data_schema": {},
  "ranking_config": {}
}
```

This confirms (VERIFIED, direct read): the challenge is named `mr-volume-generation` / "MR Volume Generation", is configured as a single phase (`"phase": "main"`), and is configured for **Docker** submission. It also confirms, importantly, that **`data_schema` and `ranking_config` are both empty objects** — i.e., even this artifact (built by the user's own pipeline against the real platform, per the prior audit) has no populated input/output schema or metric/ranking configuration. This directly corroborates the task premise that the schema and submission format are not yet specified anywhere locally. No rules document, task-description PDF, CFP, or dataset-specific challenge README exists anywhere in the searched **local** locations.

**Update (2026-07-28, re-run with internet access — see §0.1 for full detail):** a public task page for this exact challenge *does* exist on the open internet (`research.forithmus.com/collections/vlm3d-challenge`, and the challenge is officially listed on MICCAI's own 2026 site), so "no spec exists" was only ever true of this repository's local filesystem, not of the world. This session's tools could not fetch that page directly (HTTP 403 on every attempt), so the task-specific schema/metrics remain unread rather than confirmed nonexistent. §0.1 documents everything recoverable via search-engine snippets and third-party corroboration instead.

Everything in §1-16 below was originally built from: (a) the generic platform mechanics in `docs/challange_docs/` (VERIFIED for *any* challenge on this platform), (b) the one small, schema-less config artifact above (VERIFIED, but minimal), and (c) inference from this repo's own MR-RATE dataset/pipeline (already audited) plus the parent directory name `VLM3D-MRI-R2V-MICCAI-26` (INFERRED/ASSUMED, never itself a source of authority). **This was a no-network analysis; §0.1 below re-runs the same question with full internet access and documents what changed.**

---

## 0.1 Addendum — internet cross-check (2026-07-28, same day, re-run with network access)

**Access limitations, stated up front:** direct fetches of every page on the actual hosting platform failed — `research.forithmus.com/*` (both `/collections/vlm3d-challenge` and `/getting-started`), `forithmus.com/*` (root and `/leaderboard`), and the original `vlm3dchallenge.com` per-task subdomains (`reportgen.`, `ctgen.`, `abnclass.` — all `getaddrinfo ENOTFOUND`, i.e. DNS no longer resolves for them) all returned **HTTP 403** or a DNS failure to this session's fetch tool. `vlm3dchallenge.com` itself now 301-redirects permanently to `research.forithmus.com/collections/vlm3d-challenge`. The University College Dublin MIUA-2026 program pages also 403'd, and Wayback Machine has no snapshot of the Forithmus URL. **No claim below is a direct primary-source read of the actual task specification** — everything is either (a) a successful fetch of a *third-party* page (HuggingFace, GitHub, NVIDIA, Zenodo, MICCAI's own site) or (b) a search-engine-synthesized summary of snippets from pages this session could not fetch directly, which is a weaker evidence tier and is labeled as such below.

### What is now VERIFIED that wasn't before

- **The challenge is real, named, and officially part of MICCAI 2026.** MICCAI's own official 2026 challenges listing (`conferences.miccai.org/2026/en/challenges.asp`, fetched directly) lists: *"VLM3D — Vision-Language Modeling in 3D Medical Imaging | Oct. 1 | Contact: Ibrahim Ethem Hamamci (ibrahim.hamamci@uzh.ch)"*. This resolves Q1's biggest open question: the "MR Volume Generation" phase named in `.forithmus/config.json` is one track of this officially-registered challenge, not a rehearsal/internal name only.
- **VLM3D is a second edition**, not a new event: the first edition ran as an officially registered MICCAI 2025 challenge in Daejeon, South Korea (challenge day reported as **Sept 24, 2025**), built entirely on the chest-CT **CT-RATE** dataset (same organizing lineage as MR-RATE, same lead contact). Real results exist from that edition (a "Deepnoid"/M4CT team win in two of the four CT tasks, per a Korean industry-press writeup) — this is corroborating, not challenge-defining, evidence for the *2026 MR-RATE edition's* format, but it establishes the challenge has run once before with a genuine four-task structure and a real leaderboard/results cycle.
- **MICCAI 2026 dates and venue:** Sept 27–Oct 1, 2026, Strasbourg, France (relocated from Abu Dhabi). VLM3D's own slot within it is listed as **Oct 1, 2026** — i.e. **still about two months after today (2026-07-28)**.
- **A second, earlier public-facing event already occurred:** MIUA 2026 (Medical Image Understanding and Analysis), Dublin, **20–22 July 2026** — i.e. **six days before today**. Search-engine summaries (not a direct page read — UCD's own pages 403'd) describe this as an explicit **"preliminary round"** of VLM3D, with "early winners of the challenge leaderboards... invited to present" at MIUA, and describe the brain-MRI/MR-RATE track as included. **This is graded INFERRED, not VERIFIED** — it is a search-synthesized claim from a page this session could not fetch directly, and no exact schema/task list for the MIUA round could be independently confirmed.
- **The CT-track sibling of "MR Volume Generation" is named "Text-Conditional CT Generation"** (Task 4 of the 2025 CT edition, cached page content, described as "synthesize realistic 3D chest CT volumes from free-form radiology text prompts... matching anatomical context and reflecting all described pathologies with realistic Hounsfield distributions"). By direct structural analogy (the platform mirrors CT tasks 1:1 into MR tasks — "MR Abnormality Classification," "MR Report Generation" are both attested in search summaries alongside "MR Volume Generation"), this is the strongest evidence yet that **Q1's task is indeed report/text-conditioned generation**, not unconditional or label-only generation — upgrading A1 from a repo-directory-name inference to a cross-validated structural inference. It is still not a verbatim quote of the actual MR-track task page, so it stays INFERRED rather than VERIFIED.
- **MR-RATE's own public dataset card (HuggingFace, fetched directly) externally corroborates every split count already found locally** (train 75,000 patients/88,985 studies/638,345 series; val 3,425/3,781/27,003; test 5,000/5,568/39,906) and confirms license as **CC BY-NC-SA 4.0** specifically (version number not previously pinned locally), academic/research-only use, GDPR/HIPAA compliance obligations, and required attribution. The HF card itself explicitly states no challenge/competition or submission-format details are present on the card — consistent with the local finding that the dataset artifact and the challenge artifact are two separate things.
- **Code vs. data licensing is now split precisely:** the GitHub `forithmus/MR-RATE` README (fetched directly) confirms code is Apache 2.0, while data/model weights remain CC BY-NC-SA 4.0 — refines the previously-cited "CC BY-NC-SA" into two distinct license scopes.
- **A concrete, closely-related generative model's actual I/O contract is now known** — `nvidia/NV-Generate-MR-Brain` (HF model card, fetched directly): 3D latent diffusion, 240M params, MONAI Core 1.5/PyTorch, four modalities (**T1, T2, FLAIR, SWI — not MRA**, consistent with MRA's 0.02% rarity in MR-RATE itself), whole-brain or skull-stripped output selectable, max resolution **512×512×256 at 0.45×0.45×0.7mm**, and — importantly — its documented inputs are `num_output_samples`, `modality` (an **integer label**, not free text), optional `output_size`, optional `spacing`. **The publicly documented model card does not mention a free-text report input at all.** This is a meaningful new tension, flagged in §"Revised assumptions" below: if the actual MR Volume Generation baseline is architecturally closer to this model than to a text-conditional one, "report-to-volume" may be this project's own framing/ambition rather than the literal, currently-implemented challenge baseline — though the CT-track naming precedent ("Text-Conditional CT Generation") argues the *challenge task itself* is still likely to require text-conditioning even if today's public reference model doesn't yet expose it that way.
- **MR-RATE was first made public 2026-03-18** (dataset + code), per search summaries of secondary sources — gives a concrete anchor for "how long this has been available" that wasn't pinned down before.
- **The CT edition used a genuinely external hidden test set**, not the public CT-RATE test split: "Internal test set: 2,000 cases; External test set (Boston University Hospital): 1,024 cases" (cached page content, 2025 edition). This is the most decision-relevant update to the assumption ledger — see A7 revision below.

### What remains a live, unresolved tension (report honestly, don't resolve by assertion)

**Is the challenge currently accepting submissions right now, contradicting the task's original premise?** The evidence points both ways and neither side is a direct primary-source confirmation:
- *For "yes, something is already live":* the MIUA 2026 preliminary round (Jul 20-22) reportedly already produced an early leaderboard with invited presenters, six days before today.
- *For "no, still not fully configured":* the freshest available local platform artifact, `/hnvme/workspace/y100dc19-MR-Rate-raw/validation/.forithmus/config.json`, is dated **2026-07-27T14:30 — one day before today** — and even then shows `data_schema: {}` and `ranking_config: {}` still empty for the `"main"` phase of `mr-volume-generation`.

These are not necessarily contradictory (the MIUA round could have used a different, earlier phase/schema than the one snapshotted; or MIUA's "preliminary round" may have been a promotional/leaderboard-only checkpoint inside one continuously-open main phase whose schema was still being finalized as of yesterday; or the search-engine summary of the MIUA page may be imprecise, since that page could not be fetched directly to check). **Recommendation: do not assume either that submissions are closed or that they are fully open — verify directly against the platform (which this session's tools cannot access) before making time-sensitive plans.**

---

## 1–16. Answers

### 1. What task is the challenge describing?
- **VERIFIED (platform-generic):** the platform's task catalog includes "Image Generation — Text prompts in, generated images out" and "Report Generation — Medical images in, text reports out" as two of several supported *categories* (`Quick_Start.md:186-189`); it does not describe this challenge specifically.
- **INFERRED:** the configured challenge is titled "MR Volume Generation" (`.forithmus/config.json`, re-verified above), and the repository's own parent directory is named `VLM3D-MRI-R2V-MICCAI-26` — jointly suggesting a **text/report-conditioned 3D brain MRI volume generation** task ("R2V" = report-to-volume), i.e. the *inverse* direction of the platform's generic "Report Generation" category and closer to its generic "Image Generation" category, but for 3D medical volumes rather than 2D images.
- **UNKNOWN (locally):** the authoritative MICCAI VLM3D task definition (unconditional vs. conditional generation, single-volume vs. multi-sequence target, exact input contract) is not present in any local document.
- **Web update (2026-07-28, §0.1):** VLM3D is officially listed on MICCAI 2026's own challenges page (finals Oct 1, 2026, contact Ibrahim Ethem Hamamci, University of Zurich) — the task is real and MICCAI-affiliated, not just a name in a local config file. The CT-track sibling task is explicitly named "Text-Conditional CT Generation," which by structural analogy (the MR track mirrors CT tasks 1:1) is the strongest evidence yet that this is genuinely report/text-conditioned generation, not unconditional generation — still INFERRED-by-analogy, not a direct quote of the MR-track page itself, which this session could not fetch (403).

### 2. What is provided to participants?
- **VERIFIED (platform-generic):** test data mounted read-only at `/input/` inside the Docker container, optionally accompanied by a host-supplied `/input/metadata.json` (`Docker_Submission.md:16-19`); **ground truth is never given to participants** (`Quick_Start.md:12`); participants develop against locally-generated, schema-matching mock data (`forithmus generate` → `mock_input/`, `Quick_Start.md:42-46`, `Data_Schemas_Mock_Data.md:38-47`).
- **UNKNOWN:** the concrete contents of `/input/` for this challenge (is a report supplied per case as text/JSON? is any partial/reference volume given? is conditioning metadata such as modality/plane included?) — unresolvable while `data_schema: {}`.

### 3. What must participants generate?
- **VERIFIED (platform-generic):** predictions must be written to `/output/`, one file (or file set) per case, non-root container, network disabled during execution (`Docker_Submission.md:6-27`, `Evaluation_Pipeline.md:31-32`).
- **INFERRED:** given the "MR Volume Generation" title and the platform's first-class NIfTI support (`Data_Schemas_Mock_Data.md:26-27`), the output is most likely one NIfTI volume per case (paralleling the segmentation example `case_001.nii.gz` used throughout `Docker_Submission.md`/`File_Submissions.md`).
- **UNKNOWN:** exact file naming convention, required header conventions (affine/orientation), and whether one file constitutes a complete answer or several files (e.g. per-modality) are required per case.

### 4. Is the target one volume, multiple modalities, or a study?
- **UNKNOWN.** No document states case granularity. The platform generically supports 1:1 (segmentation-style) or many:1 (classification/regression-style) input→output mapping, auto-detected from the host's baseline run (`Data_Schemas_Mock_Data.md:18-19`) — meaningless here since no baseline has been run.
- **ASSUMED (provisional):** one target series/volume per case. Rationale: MR-RATE's own reports are **study-level** while volumes are **series-level** (already documented in `docs/design/mr_rate_local_audit.md §6`, `docs/design/report2volume_gap_analysis.md` row 1) — a known, unresolved granularity mismatch for *any* report-conditioned generation task on this dataset, independent of the challenge. Treating the challenge's "case" as one target series (with study-level report reused as its condition, or a study-level target of several co-registered series) are both plausible; this document assumes the former as the simpler default and flags the latter as a live alternative.

### 5. Is the modality given, inferred from the report, or chosen by the participant?
- **UNKNOWN.** Not addressed anywhere locally.
- **ASSUMED (provisional):** modality is *given* as a conditioning field (most consistent with the platform's per-case `metadata.json` mechanism, `Docker_Submission.md:19`, and with MR-RATE's own metadata already carrying a `classified_modality` field per series, per `CLAUDE.md`'s data-preprocessing section and the prior audit). Report-inferred or participant-chosen modality are live alternatives that would change model design materially (see High-Risk Unknowns, §12).
- **Web update (2026-07-28, §0.1):** `nvidia/NV-Generate-MR-Brain`'s public model card documents `modality` as an **integer label input**, not inferred from text — the closest real, documented generative model on this exact dataset supports this assumption directly, strengthening it from pure inference to model-precedented inference. Note, however, that this same model's card documents no free-text report input at all, which is a live tension with A1's text-conditioning assumption (see §0.1, question 12).

### 6. Is acquisition plane specified?
- **UNKNOWN**, same status and reasoning as Q5. MR-RATE already carries an `acquisition_plane` field per series (VERIFIED in the prior audit, computed in `modality_filtering.py`), so it is *available* if the challenge organizers choose to expose it — but nothing local confirms they will.

### 7. Are spacing, shape, orientation, or FOV specified?
- **UNKNOWN.** `data_schema: {}` in the one available config artifact means none of these are configured yet. The platform's schema-detection mechanism explicitly derives "shapes, dimensions, and data types" from the host's real upload (`Data_Schemas_Mock_Data.md:9`), i.e. this is populated only once the organizers upload real test data — which per the task premise has not happened.
- **Relevant background (VERIFIED, from the prior audit, not from challenge docs):** MR-RATE's native-space release has near-total geometric heterogeneity — 36/37 unique shapes and 34/37 unique spacings in a stratified 37-file byte-level sample (`docs/design/mr_rate_local_audit.md §5.2`). If the challenge draws on native-space MR-RATE data as-is, a fixed shape/spacing contract is unlikely without an explicit resampling step somewhere in the pipeline (organizer-side or participant-side, currently unspecified either way).

### 8. Are reports raw, anonymized, translated, or structured?
- **UNKNOWN for the challenge specifically** — no challenge document mentions reports at all.
- **INFERRED (from this repo's own MR-RATE pipeline, cross-repo, not challenge-specific):** if the challenge's report input is drawn from the released MR-RATE reports artifact, it would be anonymized (LLM token-replacement, `CLAUDE.md` reports-preprocessing section, step 01), translated Turkish→English (step 02–03), and structured into four named sections (step 04) — this is a property of the *dataset*, not a confirmed property of the *challenge's* input contract.

### 9. Which report sections appear intended as input?
- **UNKNOWN for the challenge.**
- **ASSUMED (provisional):** MR-RATE's reports CSV schema is `study_uid, report, clinical_information, technique, findings, impression` (VERIFIED header, `docs/design/mr_rate_local_audit.md §6`, `logs/mr_rate_audit_metrics.json.reports_csv_schema_check`). If reused, `findings`/`impression` are the most plausible primary conditioning signal (diagnostic content); `clinical_information`/`technique` are more likely auxiliary/contextual fields (indication, scanner protocol) that a generator may or may not condition on. This is a design assumption, not a documented requirement.

### 10. Are pathology labels, masks, or other metadata available?
- **UNKNOWN whether exposed to challenge participants** — not addressed in any challenge document.
- **VERIFIED (dataset-level, not challenge-level, from the prior audit):** MR-RATE separately ships a 37-column (32-shipped, 14-merged-clinical-group) pathology-label CSV, per-series brain masks (99.99% coverage) and defacing masks (99.99% coverage) — see `logs/mr_rate_audit_metrics.json.series_parquet_aggregates`. Whether the challenge exposes any of these as input, holds them out for evaluation only, or excludes them is not determinable locally.

### 11. What training, validation, and hidden-test data are described?
- **VERIFIED (platform-generic):** hosts upload hidden test data + ground truth kept "strictly separate," accessible only to the evaluation container (`Quick_Start.md:112-114`); most challenges use a single phase (VERIFIED via `.forithmus/config.json`: `"phase": "main"`); multi-phase challenges can pair a public-leaderboard preliminary phase with a private-test final phase (`Phases_Editions.md:11-13`) — not confirmed to apply here (only one phase is currently configured).
- **INFERRED, with an important caveat:** the SHARDS_PATH repackaging already partitions data into train/val/test shard sets matching MR-RATE's own official `splits.csv` exactly (train 88,985 studies / val 3,781 / test 5,568 — `logs/mr_rate_audit_metrics.json.shards_path_inventory.splits`, independently re-confirmed via `dataset.json` above), and its own config names the "mr-volume-generation" challenge. It is tempting to assume the challenge's hidden test set *is* MR-RATE's published test split.
- **This assumption is flagged as high-risk (see §12):** MR-RATE's test split (5,568 studies) was **already released publicly** as part of the HF dataset (`docs/design/mr_rate_local_audit.md §6`), whereas the platform's own documentation describes "hidden test data" as something hosts upload and that "never leaves the platform" (`Quick_Start.md:3-14`). A genuinely hidden evaluation set is normally *not* identical to a previously-published test split, precisely to prevent participants from having already seen it. Whether the organizers reuse the public MR-RATE test split as-is, hold out a disjoint unpublished set, or do both is **UNKNOWN**.
- **Web update (2026-07-28, §0.1) — this is no longer just a theoretical risk:** the 2025 CT edition of this same challenge (same organizing team) confirmed used a genuinely external hidden test set — "internal test set: 2,000 cases; external test set (Boston University Hospital): 1,024 cases" — *not* CT-RATE's own published test split. By direct precedent, assume the MR-RATE hidden test set is similarly external/non-public until told otherwise (see revised A7).

### 12. What evaluation metrics are mentioned?
- **VERIFIED (platform-generic only):** the evaluation container must write `/output/metrics.json` as a flat key→numeric-value map (`Evaluation_Pipeline.md:59-79`); the example shown (`dice`, `hausdorff_95`, `sensitivity`, `specificity`, `precision`) is a **segmentation example**, not this challenge's metric set; ranking can use single-metric, custom-formula, mean-score, mean-rank, or weighted-rank aggregation, with metric matching case-insensitive (`Leaderboard_Ranking.md:1-22`); metrics can be marked "hidden" (computed but not shown publicly) (`Leaderboard_Ranking.md:15-17`).
- **UNKNOWN:** the actual metric(s) for MR volume generation. `ranking_config: {}` in the one available config artifact confirms none are configured yet. Candidates a generative-imaging challenge would plausibly use (FID/KID-style distributional metrics, reconstruction similarity such as SSIM/PSNR/LPIPS, downstream-task metrics like segmentation Dice on generated volumes, or report-image semantic consistency) are **not documented anywhere locally** — any specific metric named in prior project discussion should be treated as ASSUMED, not VERIFIED, until an organizer-provided evaluation container or metric list appears.

### 13. Is clinical/radiologist evaluation mentioned?
- **UNKNOWN / not mentioned.** No document in `docs/challange_docs/` references clinical or radiologist review as part of the platform's or this challenge's evaluation. (Note: `docs/design/recommended_next_steps.md` recommends radiologist review as part of *this project's own* internal evaluation protocol — that is a project recommendation, not an organizer requirement, and should not be conflated with the latter.)

### 14. What submission format is currently described?
- **VERIFIED:** `submission_type: "docker"` is explicitly set for this challenge in the one available config artifact (re-verified above) — so the generic file-submission alternative (`File_Submissions.md`) is presumptively **not** the intended path here, though this could change before the challenge opens.
- **VERIFIED (generic Docker contract, applies once schema is populated):** `/input/` read-only (images + optional `metadata.json`), `/output/` for predictions, optional `/checkpoint/` for resumability, `USER` (non-root) instruction required, `amd64` architecture, image ≤15GB via web UI / ≤50GB via CLI registry limit, SIGTERM handled within 30 seconds, network disabled during execution (`Docker_Submission.md` throughout; `Evaluation_Pipeline.md:20-58`).
- **UNKNOWN:** case-ID pattern, exact output filenames/format, and any per-case auxiliary output (e.g. a manifest/JSON alongside each volume) — all schema-dependent and not yet configured.

### 15. What licenses, eligibility, or compute restrictions exist?
- **VERIFIED (platform-generic):** public challenges are open to all registered users with free compute up to per-submission limits and free storage up to 1TB data / 1TB registry (`Quick_Start.md:156-163`); private challenges are invite-only, fully sponsor-funded, with no participant-visible data (`Quick_Start.md:164-170`); compute tiers available depend on what the host enables (`cpu-4`, `cpu-16`, `gpu-t4`, `gpu-a100-40`, `gpu-a100-80`, `Docker_Submission.md:156-174`); minimum time budget 5 minutes, pre-charged and refunded for unused time (`Docker_Submission.md:176-178`, `Billing_Credits.md:19-21`); spot instances are opt-in per phase, up to 3 auto-retries before falling back to on-demand (`Spot_Instances.md:14-19`); after archiving, Docker images/outputs/test data/checkpoints are deleted after 30 days unless exported (`Reproducibility.md:17-32`).
- **UNKNOWN:** whether this specific challenge is public or private, which compute tiers/spot support are enabled for it, its submission-limit configuration (total/daily/weekly/monthly, `Phases_Editions.md:41-49`), any opens_at/closes_at window, eligibility rules (institutional/geographic), and — separately — the **data usage license/eligibility terms set by the MICCAI VLM3D organizers themselves** (distinct from the underlying MR-RATE dataset's own CC BY-NC-SA license noted in the prior audit, `docs/design/recommended_next_steps.md` item 4, which governs the *dataset*, not necessarily the *challenge submission rules*).
- **Web update (2026-07-28, §0.1):** the dataset license is now pinned precisely — CC BY-NC-SA **4.0** for data/weights, Apache 2.0 for code (previously only "CC BY-NC-SA" without version, from the local audit). MICCAI 2026 itself runs Sept 27–Oct 1, 2026 in Strasbourg (VLM3D slotted Oct 1); a MIUA 2026 "preliminary round" (Dublin, Jul 20-22) reportedly already ran six days before this update. Challenge-specific eligibility/participation terms (as opposed to dataset license) remain unfound.

### 16. What remains unknown?
Consolidated in §11 (High-Risk Unknowns) and §13 (Questions to Organizers) below; in short: the exact input/output schema (`data_schema: {}`), the metric/ranking configuration (`ranking_config: {}`), whether public MR-RATE test data equals the real hidden evaluation set, conditioning-field availability and provenance (modality/plane/contrast/geometry: given vs. inferred vs. participant-chosen), report-section scope, case granularity (series vs. study), phase/timeline/eligibility details, and the challenge-specific license/terms.

---

## Verified challenge requirements

These hold regardless of what the final task-specific schema turns out to be, because they are properties of the hosting platform itself (VERIFIED, generic):

1. Docker is the confirmed submission type for this challenge (`submission_type: "docker"`, independently re-read from `.forithmus/config.json`).
2. Container contract: `/input/` read-only test data (+ optional `metadata.json`), `/output/` predictions, optional `/checkpoint/` for resumability; non-root `USER`; `amd64`; network disabled at execution time (`Docker_Submission.md`, `Evaluation_Pipeline.md:31-32`).
3. Ground truth is never exposed to participants, at any point (`Quick_Start.md:12`).
4. Evaluation is via a separate host-supplied evaluation container reading `/input/predictions/` and `/input/ground_truth/`, writing a flat `/output/metrics.json` (`Evaluation_Pipeline.md:59-79`).
5. SIGTERM must be handled within 30 seconds (timeout or spot preemption) with state saved to `/checkpoint/`; `/output/` does **not** persist across runs/continuations (`Checkpoints_and_Continuation.md:1-20`, `Docker_Submission.md:9`).
6. Leaderboard shows only the best-scoring submission per participant/team per phase; ranking method is one of single-metric / custom-formula / mean-score / mean-rank / weighted-rank, host-configured (`Leaderboard_Ranking.md`).
7. This challenge is currently configured as a single phase named "Main" (`.forithmus/config.json`).
8. Compute tiers, time budgets, and (if enabled) spot instances are billed via pre-charge/refund from a wallet or sponsor pool (`Billing_Credits.md`).
9. Post-archival retention: images/outputs/test data/checkpoints deleted after 30 days unless exported; scores/leaderboard/submission metadata kept indefinitely (`Reproducibility.md`).
10. Recommended workflow: `forithmus init` → `forithmus generate` (mock data) → build/test locally → `forithmus test` (schema validation) → `forithmus submit` (`Quick_Start.md`, `CLI_Tool.md`).

## Incomplete or ambiguous requirements

| Item | What's known | What's missing |
|---|---|---|
| Input schema | Generic `/input/` contract; `metadata.json` mechanism exists | `data_schema: {}` — no fields, formats, shapes, or case-ID pattern defined |
| Output schema | Generic `/output/` contract; NIfTI is a supported format | No required filename pattern, per-case file count, or header convention |
| Metrics/ranking | Generic metric-key + ranking-method mechanism | `ranking_config: {}` — no metric names, directions, or weights defined |
| Conditioning fields | MR-RATE dataset *has* modality/plane fields available | Not confirmed whether/how they reach participants for this challenge |
| Test-set identity | Public MR-RATE test split exists and matches SHARDS_PATH's test partition | Not confirmed this *is* the actual hidden evaluation set (see §11 above) |
| Phase/timeline | Single phase "Main" configured | No `opens_at`/`closes_at`, submission-limit, or public/private status found |
| License/eligibility for the challenge itself | MR-RATE dataset license (CC BY-NC-SA) documented elsewhere | Challenge-specific participation/eligibility terms not found locally |

## Assumption ledger

| # | Assumption | Rationale | Risk if wrong | Revisit trigger |
|---|---|---|---|---|
| A1 | Task = report/text-conditioned single-series 3D brain MRI volume generation | Challenge title "MR Volume Generation" + repo directory name `VLM3D-MRI-R2V-MICCAI-26` | Wrong architecture family chosen (e.g. if actually multi-series/study-level target) | Official task description or `data_schema` populated |
| A2 | One target volume (series) per case, not a whole study | Simpler default; matches platform's per-case file convention (`case_001.nii.gz`) | Under-builds if organizers require multi-series output per case | `data_schema` shows `many:1` or nested per-study structure |
| A3 | Modality is given as an input conditioning field, not inferred or chosen | Matches MR-RATE's existing `classified_modality` field and the platform's `metadata.json` mechanism | Model design assumes a field that may not be provided, or is provided differently (e.g. embedded in report text only) | Real `/input/metadata.json` example or schema doc appears |
| A4 | Acquisition plane is likewise given, not inferred | Same reasoning as A3; MR-RATE already computes `acquisition_plane` per series | Same as A3 | Same as A3 |
| A5 | `findings`/`impression` are the primary report-conditioning text; `clinical_information`/`technique` are auxiliary | These are MR-RATE's diagnostic-content sections vs. contextual sections | Under- or over-weighting report content the challenge actually intends as primary | Any organizer-provided example case with report text |
| A6 | Output format is NIfTI (`.nii.gz`), one file per case | Platform's first-class NIfTI support; consistent with every Docker example shown | Wasted format-conversion work if organizers require NumPy or another container format | Real `/input/`/`/output/` example from the platform |
| A7 | Hidden test set is drawn from (or equals) MR-RATE's published test split | SHARDS_PATH's shard counts exactly match `splits.csv`'s official test partition | If wrong, local test-set performance estimates are meaningless for the real leaderboard; also a genuinely "hidden" set should arguably not be identical to already-published data | Organizer statement on test-set provenance, or an official small-N public "validation" case release differing from the known split |
| A8 | Submission is Docker-only for this challenge (file-submission alternative not applicable) | `submission_type: "docker"` explicitly set in the one available config | Wasted effort if a file-submission path is later enabled for a simpler intermediate phase | Config change or organizer announcement |
| A9 | Geometry (spacing/shape/orientation) is not fixed by the challenge and must be handled by the participant's own pipeline | `data_schema: {}`; MR-RATE native data is itself highly heterogeneous | Building for a fixed canonical grid the organizers don't require (or vice versa) | Populated `data_schema` with explicit shape/spacing fields |

**A7 revised downward in confidence (2026-07-28, web cross-check, §0.1):** the 2025 CT edition of this same challenge used a genuinely external, non-public hidden test set ("internal test set: 2,000 cases; external test set (Boston University Hospital): 1,024 cases" — cached page content, not primary-source-confirmed), *not* CT-RATE's own published test split. By direct precedent from the same organizing team, **A7 is now the single most likely-to-be-wrong assumption in this ledger** — treat any locally-computed metric against MR-RATE's public test split as a development/sanity check only, never as a stand-in for the real leaderboard score.

## Provisional training-input contract

Not an official spec — a working assumption set for internal pipeline design, subject to full revision once `data_schema` is populated.

```
Per training/validation case (ASSUMED shape, pending real schema):
  study_uid / case_id           : str            # join key; internal only, never logged/exposed downstream
  report                        : structured text # raw + {clinical_information, technique, findings, impression}
                                                    # per MR-RATE's existing schema (VERIFIED schema, ASSUMED reuse)
  modality                      : str | None       # ASSUMED given (A3)
  acquisition_plane             : str | None        # ASSUMED given (A4)
  contrast_state                : str | None        # UNKNOWN — not available as a real MR-RATE label at all
                                                     # (prior audit: is_contrast_enhanced dropped from release)
  target_volume                 : NIfTI path        # ASSUMED one series per case (A2)
  native_spacing/shape/orientation: as-recorded      # heterogeneous; NOT assumed fixed (A9)
  split                         : {train,val,test}   # from official splits.csv, patient-level (VERIFIED clean)
```

## Provisional generated-output contract

```
Per case (ASSUMED, pending real schema):
  <case_id>.nii.gz               # ASSUMED single NIfTI volume per case (A6)
  # affine/header convention UNKNOWN — safest default is to match the input case's own
  # affine if one is supplied at inference time, otherwise a documented canonical grid
  # (choice itself still open, see High-Risk Unknowns)
```

Per the generic evaluation contract, whatever is written under `/output/` is what an organizer-provided evaluation container will read from `/input/predictions/` (`Evaluation_Pipeline.md:36-38`) — the exact required filename/format is presently unconstrained by anything locally available beyond "match the schema once it exists."

## Provisional submission adapter interface

Documented as an interface sketch only — **not implemented**, per task instructions. Its purpose: keep the challenge-independent internal representation (below) decoupled from whatever the real Forithmus `/input//output/` schema turns out to be, so that only this adapter needs to change when the schema is published.

```python
# Provisional design sketch — NOT implemented.

class SubmissionAdapter(Protocol):
    """Translates between the challenge-independent internal representation
    and whatever the real, currently-unpublished Forithmus /input//output/
    contract for `mr-volume-generation` turns out to require."""

    def read_case(self, input_dir: Path) -> "ChallengeExample":
        """Parse one case from /input/ (images/, optional metadata.json)
        into the internal representation. Must not assume any specific
        conditioning field is present -- treat schema-dependent fields as
        optional until the real schema is confirmed."""
        ...

    def write_prediction(self, example: "ChallengeExample",
                          generated: "GeneratedVolume",
                          output_dir: Path) -> None:
        """Serialize a generated volume to /output/ using whatever
        filename/format convention the real schema specifies. Placeholder
        default: <case_id>.nii.gz, single volume, matching A6/A2."""
        ...
```

## Internal representation (challenge-independent)

Extends the skeleton given in the task, with fields marked by provenance/confidence so a future implementer knows which are load-bearing today vs. speculative. **Design only — not implemented.**

```python
# Provisional design sketch — NOT implemented.

@dataclass
class ReportInput:
    raw: str | None                      # ASSUMED available (reused MR-RATE `report` column)
    clinical_information: str | None     # ASSUMED available (MR-RATE schema)
    technique: str | None                # ASSUMED available (MR-RATE schema)
    findings: str | None                 # ASSUMED available (MR-RATE schema); likely primary condition (A5)
    impression: str | None               # ASSUMED available (MR-RATE schema); likely primary condition (A5)
    language: str = "en"                 # ASSUMED (post-translation, per MR-RATE pipeline)

@dataclass
class ChallengeExample:
    case_id: str                          # VERIFIED concept (platform case-ID convention); internal join key only
    report: ReportInput
    requested_modality: str | None        # ASSUMED given, not inferred/chosen (A3) -- UNKNOWN for real schema
    requested_plane: str | None           # ASSUMED given (A4) -- UNKNOWN for real schema
    requested_contrast_state: str | None  # UNKNOWN -- no such label exists in released MR-RATE metadata at all
    requested_spacing: tuple[float, float, float] | None   # UNKNOWN -- data_schema not yet populated
    requested_shape: tuple[int, int, int] | None            # UNKNOWN -- data_schema not yet populated
    target_volume: Path | None            # present for train/val; ASSUMED absent at real inference time
    split: str | None                     # train / val / test, ASSUMED reused from official splits.csv
    metadata: dict                        # escape hatch for any organizer field not yet anticipated here

@dataclass
class GeneratedVolume:
    case_id: str
    volume: Path                          # ASSUMED single NIfTI per case (A6)
    affine: "np.ndarray | None"           # convention UNKNOWN, see generated-output contract above
```

## Evaluation expectations

- **VERIFIED mechanism, UNKNOWN content:** metrics arrive as a flat `metrics.json` map (`Evaluation_Pipeline.md:69-79`); ranking is one of the five generic methods (`Leaderboard_Ranking.md`); some metrics may be marked "hidden" and invisible to participants (`Leaderboard_Ranking.md:15-17`).
- **UNKNOWN:** the actual metric(s) for a volume-generation task. No distributional-similarity, reconstruction-fidelity, downstream-task, or semantic-consistency metric is named anywhere locally. Any specific metric assumed for internal pilot work (e.g. from `docs/design/recommended_next_steps.md`'s evaluation-protocol recommendation) is this project's own proposal, not a confirmed organizer requirement, and should be kept swappable.
- **Not mentioned anywhere locally:** clinical/radiologist review as an organizer-run evaluation step (Q13). Treat any radiologist-review step as internal-only quality assurance, not a documented challenge requirement.

## High-risk unknowns

0. **Submission-window status may already have changed (added 2026-07-28, §0.1).** Search-derived evidence suggests a "preliminary round" of this exact challenge, potentially including the brain-MRI/MR-RATE track, already ran at MIUA 2026 (Dublin, Jul 20-22 — six days before this update), with an early leaderboard and invited presenters. The freshest local platform artifact (dated Jul 27) still shows an empty schema, so this is a genuine, unresolved tension rather than a confirmed status change — but it means "not currently accepting submissions" should be re-verified directly rather than treated as still true by default. The official MICCAI 2026 finals slot for VLM3D is confirmed for **Oct 1, 2026** (about two months out), so at minimum the *final* deadline is not imminent even if an interim round already occurred.
1. **Test-set identity (§11/A7).** If the challenge's hidden evaluation set is *not* MR-RATE's published test split (the more standard practice for "hidden" data per `Quick_Start.md:3-14`'s own framing), any locally-computed leaderboard-style number is not predictive of the real score, and there is a live possibility of undisclosed additional hold-out data. **This risk is now upgraded from "plausible" to "likely," per the 2025 CT edition's confirmed use of a genuinely external (Boston University Hospital) hidden test set rather than CT-RATE's own public split** (§0.1) — the same organizing team is highly likely to follow the same pattern for MR-RATE.
2. **Conditioning-field provenance (Q5/Q6/A3/A4).** Whether modality/plane/contrast are given, inferred from report text, or left to the participant fundamentally changes model architecture (conditional generation vs. joint report-and-metadata-to-image vs. text-only-to-image). This is currently a pure assumption.
3. **Case granularity (Q4/A2).** Series-level vs. study-level targets changes the entire output contract and how a study-level report is used (whole-report → one series is known to be a label-noise risk already flagged in `docs/design/report2volume_gap_analysis.md` row 1, independent of the challenge).
4. **Geometry contract (Q7/A9).** Given MR-RATE's near-total native-space heterogeneity (36/37 unique shapes in-sample, `docs/design/mr_rate_local_audit.md §5.2`), whether the challenge fixes a canonical grid or leaves this to participants materially changes preprocessing design and evaluation comparability.
5. **Output serialization convention.** Affine/orientation convention for generated volumes is unconstrained locally; a wrong default could fail schema validation outright once real validation exists.
6. **License/eligibility terms for challenge participation** specifically (as opposed to the underlying dataset's CC BY-NC-SA license) are not present locally and could restrict use (e.g. commercial-use, external-data, or pretrained-encoder restrictions common in MICCAI challenges).

## Questions to ask the organizers

0. **Is the "MR Volume Generation" track currently open for submissions, and did the MIUA 2026 (Jul 20-22) preliminary round already include it?** (Added 2026-07-28 — see §0.1's unresolved timeline tension; this is now the most time-sensitive open question.)
1. What exactly is provided under `/input/` per case — report text/JSON, any reference imaging, and any conditioning metadata (modality, plane, contrast, target geometry)?
2. Is the target one series/volume per case, or a full multi-sequence study?
3. Is the hidden test set identical to, a superset of, or disjoint from MR-RATE's publicly released test split (5,568 studies)?
4. What are the official evaluation metric(s) and ranking configuration for this challenge?
5. Which report sections (raw vs. `clinical_information`/`technique`/`findings`/`impression`) constitute the intended conditioning input?
6. Is contrast-enhancement state ever supplied, given that it is not present in MR-RATE's released metadata?
7. Is a canonical target geometry (spacing/shape/orientation) specified, or is participant-chosen geometry acceptable as long as outputs match a required shape at write-time?
8. Is this challenge public or private, and what compute tiers, submission limits, and phase timeline (opens_at/closes_at) apply?
9. Are external pretrained models/encoders and additional (non-MR-RATE) training data permitted?
10. What license/usage terms govern participation and any generated-model outputs, separate from the MR-RATE dataset's own CC BY-NC-SA terms?
11. Is the hidden test set an external, institution-specific hold-out (as the 2025 CT edition used a Boston University Hospital external set) rather than MR-RATE's own published test split? (Added 2026-07-28, §0.1.)
12. Is `nvidia/NV-Generate-MR-Brain` (modality-label + optional-geometry conditioned, no documented free-text input) the intended reference/baseline architecture for this task, or is the real task strictly report-text-conditioned as the CT-track's "Text-Conditional CT Generation" naming would suggest? (Added 2026-07-28, §0.1.)

## Requirements likely to change when submissions open

- **This section's premise ("when submissions open," future tense) may already be partly overtaken by events** — see §0.1: a preliminary round six days before this update may have already exercised part of this challenge on the real platform. Treat everything below as "may already be happening or may still be pending," not as a clean future/present split.
- `data_schema` and `ranking_config` will almost certainly be populated with concrete field names, shapes, dtypes, and metric keys (`Data_Schemas_Mock_Data.md` describes exactly this auto-detection process running against a real host upload and baseline submission) — every "UNKNOWN" schema item in this document is expected to resolve at that point, not be a permanent gap.
- Compute-tier availability, spot-instance support, and submission limits are phase-level configuration the host can change any time before "Enable submissions" is flipped (`Phases_Editions.md:15-40`).
- Whether Docker remains the sole submission type, or a file-submission path is added for a simpler intermediate phase, is host-configurable and not fixed by anything seen so far.
- Public/private visibility and any sponsor-pool-funded free compute could be announced or changed up to submission opening.

---

## Closing statement — most logical working assumptions until the official spec is released

Until the organizers publish a populated schema, proceed under the following minimal, most-defensible assumption set: **one MR-RATE-style report (with `findings`/`impression` as primary conditioning text) paired with a single target 3D brain MRI series, generated as one NIfTI volume per case, with modality and acquisition plane supplied as given conditioning fields rather than inferred or participant-chosen, evaluated on a genuinely external hidden test set that should not be assumed identical to MR-RATE's already-published test split, submitted as a Docker container following the generic `/input/`-read-only → `/output/`-write contract already fully specified by the Forithmus platform documentation.** Every element of this sentence beyond the Docker contract itself is an assumption (A1–A9 above), not a verified requirement, and the internal representation above is deliberately built so that only the adapter layer — not model or training code — needs to change when the real schema, metric set, and test-set provenance are eventually published.

**2026-07-28 update:** the challenge's identity, task framing (text-conditional generation, by structural analogy to the CT track), and MICCAI 2026 timing (finals Oct 1, 2026) are now cross-validated externally and materially more trustworthy than on first pass — but the single most important open question changed from "what will the spec eventually say" to **"has part of this already gone live without our noticing"** (§0.1, high-risk unknown #0). Verifying current submission status directly against the platform — which this session's tools could not reach — should now be the immediate next step, ahead of any further design work.
