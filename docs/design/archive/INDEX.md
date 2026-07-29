# Archived design records

**You do not need these to use the pipeline** — [`docs/R2V.md`](../../R2V.md) is the guide. These are
the evidence and reasoning behind the defaults, kept because the analysis was expensive and the
citations are real. Read one when you want to know *why* something is the way it is, or before
changing a default that looks arbitrary.

They describe an earlier code layout. Where they name `data_r2v.py`, `r2v_storage.py`, or
`evaluation/evaluate_*.py`, the current equivalents are:

| Then | Now |
|---|---|
| `scripts/data_r2v.py` | `mrrate_r2v/data/{dataset,geometry,manifest,reports}.py` |
| `scripts/r2v_storage.py` | `mrrate_r2v/data/storage.py` |
| `scripts/build_r2v_manifest.py` + the standalone pyarrow copy | `mrrate_r2v/cli/build_manifest.py` (one script; the duplicate is gone) |
| `evaluation/evaluate_vae.py` | `mrrate_r2v/cli/predict_vae.py` + `mrrate_r2v/cli/evaluate.py` |
| `evaluation/evaluate_generation.py` | `mrrate_r2v/cli/predict_generation.py` + `mrrate_r2v/cli/evaluate.py` |
| `evaluation/evaluate_r2v.py` | `mrrate_r2v/cli/predict_r2v.py` + `mrrate_r2v/cli/evaluate.py` |
| `evaluation/cohort.py` | `mrrate_r2v/cohort.py` (now a persisted artifact, not just a function) |
| `evaluation/geometry.py` | `mrrate_r2v/eval/geometry_contract.py` |
| `evaluation/metrics.py` | `mrrate_r2v/eval/paired.py` |
| `evaluation/distribution_metrics.py` | `mrrate_r2v/eval/distribution.py` |

---

## Still-current reference material (one level up, in `docs/design/`)

These are **not** archived — they document external systems, not this code's structure:

| Document | What it is |
|---|---|
| `challenge_contract.md` | what the VLM3D R2V challenge requires |
| `nv_generate_mr_brain_audit.md` | audit of NVIDIA's MAISI-v2 / rflow-mr-brain model and its conditioning |
| `mr_rate_local_audit.md` | audit of the local MR-RATE copy: schemas, counts, data quality |
| `fau_hpc_execution_profile.md` | partitions, QoS, walltimes, storage, and job templates for this cluster |

---

## The archive

| Document | What it establishes | Read it before |
|---|---|---|
| `06_report_to_volume_dataset_implementation.md` | every Dataset default: report representation, series-selection policy, why the contrastive loader's collate/sampling was replaced | changing `series_selection`, `report_sections`, or the sample schema |
| `07_archive_backed_mrrate_storage.md` | that random access into un-extracted tars is fast and safe, with measured timings; the node-local cache design and its budget | touching `storage.py` or considering extracting the dataset |
| `08_dataset_recommendation_manifest_and_axis_order.md` | the axis-order trace end-to-end: why R2V returns (X,Y,Z) and the contrastive loader (D,H,W), with file:line citations into both models | changing any axis convention |
| `09_older_evaluation_implementation_audit.md` | component-by-component audit of a previous evaluation implementation, including the blind-resize bug this one replaces | reintroducing anything from that implementation |
| `10_evaluation_geometry_contract_and_shape_mismatch_policy.md` | the full four-verdict geometry policy with citations into the implementation and tests | loosening `compare_geometry` or adding a "just resize it" path |
| `mr_rate_dataset_and_dataloader_implementation.md` | the original dataset/dataloader analysis that preceded doc 06 | historical context only |
| `report2volume_gap_analysis.md` | what was missing for R2V at the outset | historical context only |
| `recommended_next_steps.md` | a planning snapshot | historical context only |
| `audit_progress.md` | progress log of the local dataset audit, with measured parse rates | sizing a preprocessing job |

## Superseded and deleted

Three documents were removed rather than archived, because their content is now in the code and the
current guides and keeping them would mean two sources of truth:

- `EVALUATION_STATUS.md` — a session-resume file mixing real findings with stale Slurm job IDs. Its
  durable content is in `docs/R2V.md` and the module READMEs; its job table described jobs that were
  cancelled and never ran.
- `REPORT_TO_VOLUME.md` — a 690-line beginner guide, superseded by `docs/R2V.md` plus
  `mrrate_r2v/data/README.md`.
- `evaluation/README.md` — superseded by `mrrate_r2v/eval/README.md`.
