# MR-RATE Dataset, Preprocessing, and DataLoader Implementation Audit

Code-level audit of `contrastive-pretraining/` as checked out in this fork (branch `main`, HEAD `d72f47c`, working tree clean except untracked `CLAUDE.md`). Every claim below is either cited to `relative/path/file.py:Lx-Ly` (read directly in this session) or explicitly marked as inferred/unknown. Where feasible, claims were **verified by executing the actual code** (against real local filesystem roots, and against synthetic data) rather than by reading alone — those are marked "confirmed by execution."

---

## 1. Executive Answer

The MR-RATE contrastive-pretraining stack has exactly **one live image Dataset implementation**, `MRReportDataset` (`contrastive-pretraining/scripts/data.py:314-664`), used for training, and one inference twin, `MRReportDatasetInfer` (`contrastive-pretraining/scripts/data_inference.py:37-239`), used by every inference/feature-extraction/probing entrypoint. Both:

- discover subjects via a shared function, `discover_subjects()` (`data.py:123-163`), that auto-detects one of **two** supported on-disk layouts by inspecting only the *first alphabetically-sorted* top-level directory;
- return **one study** per item — never one series — as a variable-length stack of *all* of that study's volumes;
- apply an identical, hardcoded-by-default preprocessing chain: NIfTI load (nibabel) → RAS canonical reorientation → physical-spacing resample (trilinear) → intensity normalization → center-crop/pad with a posterior shift that compensates for defacing → cast to `bfloat16`;
- are consumed by a `collate_fn` (`data.py:667-671`) / `collate_fn_infer` (`data_inference.py:242-251`) that **only ever look at `batch[0]`** — i.e. they hard-assume `batch_size=1` and will *silently discard* every other item in a larger batch with no error, warning, or count mismatch. This was reproduced live (Section 11).

**None of your local dataset roots are directly usable by this code today** — confirmed by literally running the repository's real `discover_subjects()` against `DATA_PATH`, `DATA_PATH/MR-RATE-atlas`, and `SHARDS_PATH`: all return **0 subjects**, because both are still packaged as un-extracted `.tar`/`.zip`/WebDataset-`.tar` archives, and `discover_subjects()` only walks real on-disk directories (`os.listdir`/`os.path.isdir`, `data.py:136-163`). Getting either root working requires **extraction** (turning archives into a directory tree), not "resharding" — the existing `batch00`…`batch27` partitioning is already exactly what layout 2 expects and needs no reorganization once extracted. See Sections 4–6 for the precise, tested distinction.

---

## 2. Active Dataset Classes and Entrypoints

### 2.1 `MRReportDataset` — `contrastive-pretraining/scripts/data.py:314`

| Field | Value |
|---|---|
| Purpose | Live training dataset: variable-N-volume subject → (image stack, sampled sentences, mask) |
| Entrypoints | `mr_rate_trainer.py:224` (the **only** production consumer) — used by `run_train.py` (the sole training CLI) |
| Training/inference/preprocessing | **Training** |
| Required constructor args | `data_folder` (unless `use_preprocessed=True`, then `preprocessed_dir` required instead — `data.py:383-389`), `jsonl_file` (always required) |
| Optional args + defaults | `max_sentences_per_image=34`, `target_spacing=(1.0,0.5,0.5)`, `target_shape=(256,384,384)`, `posterior_shift_mm=15.0`, `space="native_space"`, `normalizer="zscore"`, `normalizer_kwargs=None`, `splits_csv=None`, `split="train"`, `pathology_labels_csv=None`, `rebalance_strategy=None`, `rebalance_base_weight=1.0`, `rebalance_eps=1e-6`, `preprocessed_dir=None`, `use_preprocessed=False`, `cache_allow_mismatch=False` (`data.py:330-350`) |
| Required external files | A JSONL reports file (always); optionally a `splits_csv` and/or `pathology_labels_csv`; either a live NIfTI tree or a `.npz` cache tree with `_manifest.json` |
| `__len__` meaning | Number of *subjects* (studies) that have both ≥1 discovered volume **and** a matching JSONL report entry (`data.py:606-607`, built by `_prepare_samples`/`_prepare_samples_from_cache`) |
| `__getitem__` behavior | Loads (live or cached) every volume for the subject, normalizes+crops each, stacks them; randomly subsamples or zero-pads the subject's sentences to exactly `max_sentences_per_image` (`data.py:645-664`) |
| Output structure | 3-tuple `(volume_stack, selected_sentences, mask)` — **not a dict** |
| Tensor shapes | `volume_stack`: `[N, 1, D, H, W]`; `mask`: `[max_sentences_per_image]` |
| Dtypes | `volume_stack`: `torch.bfloat16`; `mask`: `torch.bool`; `selected_sentences`: `list[str]` (Python, not a tensor) |
| Variable dimension | `N` (number of volumes for that subject; official docs say 2–12+, code enforces no minimum or maximum) |
| Failure/exclusion behavior | A subject with 0 discovered volumes is never added to `self.samples` (silent, no error); a subject in the JSONL with no matching volumes is likewise silently absent; a NIfTI read failure inside `__getitem__` raises an uncaught exception (no try/except anywhere in `_load_volume_stack`, `data.py:622-643`) that will crash the whole training step |

### 2.2 `MRReportDatasetInfer` — `contrastive-pretraining/scripts/data_inference.py:37`

| Field | Value |
|---|---|
| Purpose | Deterministic inference/feature-extraction twin of `MRReportDataset` |
| Entrypoints | `extract_features.py:368`, `inference.py` (via `self.ds`, constructed before line 272), `mil_probe_online.py:231-243` (`build_dataset`) |
| Training/inference/preprocessing | **Inference / feature extraction / MIL probing** |
| Required constructor args | Same pattern as `MRReportDataset`: `data_folder` or `preprocessed_dir`+`use_preprocessed=True`, `jsonl_file` |
| Optional args + defaults | `target_spacing=(1.0,0.5,0.5)`, `target_shape=(256,384,384)`, `posterior_shift_mm=15.0`, `space="native_space"`, `normalizer="zscore"`, `normalizer_kwargs=None`, `labels_file=None`, `splits_csv=None`, `split="test"`, `preprocessed_dir=None`, `use_preprocessed=False`, `cache_allow_mismatch=False` (`data_inference.py:45-61`) |
| Required external files | Same as `MRReportDataset`, plus optionally a `labels_file` (binary pathology CSV) |
| `__len__` meaning | Identical semantics to `MRReportDataset` |
| `__getitem__` behavior | Loads **all** sentences for the subject (no random sampling/truncation — confirmed by execution, Section 11), returns a `real_volume_mask` of all-`True` (`data_inference.py:229-239`) |
| Output structure | 5-tuple `(volume_stack, sentences, subject_id, real_volume_mask, labels)` |
| Tensor shapes | `volume_stack`: `[N,1,D,H,W]`; `real_volume_mask`: `[N]` |
| Dtypes | `volume_stack`: bf16; `real_volume_mask`: bool; `sentences`: `list[str]` (length = however many the study actually has, **not** padded to a fixed count); `subject_id`: `str`; `labels`: `np.ndarray` (`float32`, shape `(0,)` if no `labels_file`) |
| Variable dimension | `N` (volumes), and (unlike training) sentence-list length is also variable per subject |
| Failure/exclusion behavior | Same silent-exclusion pattern as `MRReportDataset` |

### 2.3 `RaggedTokenDataset` — `contrastive-pretraining/scripts/mil_probe.py:161`

Not part of the image-loading path. Operates on **already-extracted, memory-mapped per-series token features** produced by `extract_features.py --feature_level tokens` (never reads NIfTI itself). `__getitem__` (`mil_probe.py:238-243`) returns `(tokens: Tensor[variable,dim], labels: Tensor[n_classes], index: int)`. Its collate function, `collate_ragged` (`mil_probe.py:259-268`), **correctly unpacks the whole batch** via `zip(*batch)` and builds a `cu_seqlens` cumulative-length tensor — this is the one collate function in the repo that is *not* hardcoded to batch_size=1, provided as a useful contrast to Section 11's finding. Used only by `mil_probe.py:485-531` (offline MIL probing on cached features); irrelevant to a new generative-training design except as a template for a "correct" ragged collate.

### 2.4 Non-Dataset loaders worth noting
- `TensorDataset` (stdlib) is used in `linear_probe.py:143` over pre-extracted feature matrices (`np.ndarray` → `torch.from_numpy`) — not image loading, no NIfTI/report involvement at all.
- `mil_probe_online.py:42` defines `DatasetMetadata`, a plain dataclass-like container for label bookkeeping, **not** a `torch.utils.data.Dataset` — it wraps `MRReportDatasetInfer` (`build_dataset`, `mil_probe_online.py:231-243`) rather than replacing it.

### 2.5 Primary path actually used by the documented training command

`python scripts/run_train.py ...` → `MrRateTrainer.__init__` (`mr_rate_trainer.py:224-239`) → `MRReportDataset`. This is the only class exercised by the repository's own training CLI; nothing else is "active" in the training sense. All obsolete-looking or parallel code (`RaggedTokenDataset`, `TensorDataset` usage) belongs to separate, later-stage probing pipelines that consume *frozen features*, not raw volumes, and are not alternate paths to the same job.

---

## 3. End-to-End Data Call Graph

### 3.1 Training: `run_train.py` → batch at the model

```
run_train.py:225-250              MrRateTrainer(clip, data_folder=..., jsonl_file=..., batch_size=1, ...)
  mr_rate_trainer.py:224-239        self.ds = MRReportDataset(data_folder, jsonl_file, space, normalizer, splits_csv, split,
                                                                pathology_labels_csv, rebalance_strategy, preprocessed_dir, use_preprocessed, ...)
    data.py:376                       self.split_uids = self._load_splits(splits_csv, split)     # study_uid allow-list, or None
    data.py:379                       self.subject_to_sentences = self._load_jsonl(jsonl_file)    # report/split filtering happens HERE
    data.py:382-389                   self.samples = self._prepare_samples(data_folder)           # or _prepare_samples_from_cache()
      data.py:445                       for sub in discover_subjects(data_folder, space): ...      # filesystem discovery, report/split AGNOSTIC
        data.py:136-163                   os.listdir/os.path.isdir walk -> layout auto-detect -> list of {subject_id, image_paths}
      data.py:447-453                   keep sub only if sub['subject_id'] in subject_to_sentences  # <-- split/report filter applied AFTER discovery
    data.py:395-400                   self.sample_weights = self._compute_sample_weights(pathology_labels_csv, rebalance_strategy, ...)
  mr_rate_trainer.py:252-273        sampler = self.ds.get_weighted_sampler() if rebalancing else None; shuffle = (sampler is None)
                                     self.dl = DataLoader(self.ds, num_workers=0, batch_size=1, sampler=sampler, shuffle=shuffle,
                                                           drop_last=True, collate_fn=collate_fn, persistent_workers=(num_workers>0))
  mr_rate_trainer.py:283-292        self.model, self.dl, self.optim, self.scheduler = accelerator.prepare(...)   # DDP wrap + (Accelerate-internal) sampler rewrite
  mr_rate_trainer.py:295            self.dl_iter = cycle(self.dl)                                   # infinite iterator (data.py:16-20)
  mr_rate_trainer.py:406            images, sentences, masks = next(self.dl_iter)                   # -> DataLoader worker calls __getitem__ then collate_fn
    data.py:645-664                   MRReportDataset.__getitem__(index)
      data.py:622-643                   _load_volume_stack(sample)  -> live: load_and_resample_nii + normalize_volume + crop_or_pad per volume
                                                                        cached: np.load(.npz)['volumes'] -> bf16, unsqueeze(1)
      data.py:654-664                   sentence sampling: random.sample(...) if n>=max_sentences else pad with ""
    data.py:667-671                   collate_fn(batch): images, sentences, masks = batch[0]; return images.unsqueeze(0), sentences, masks.unsqueeze(0)
  mr_rate_trainer.py:411-427         images/masks -> device; pad volume dim to max_vols ACROSS DDP RANKS (all_reduce MAX) for SyncBatchNorm/DDP consistency
  mr_rate_trainer.py:430-435         tok = tokenizer(sentences, padding=True, truncation=True)       # BertTokenizer, BiomedVLP-CXR-BERT vocab
  mr_rate_trainer.py:442-451         loss = self.model(text_input=tok, image=images, sentence_mask=masks, real_volume_mask=real_vols, return_loss=True, ...)
    mr_rate.py:418                     b, r, c, d, h, w = image.shape          # b=studies(=1), r=volumes/series, c=channel(=1)
    mr_rate.py:460-464                 visual_tokens, token_mask = self._encode_visual_tokens(image, real_volume_mask, vis_proj_layer, ...)
    mr_rate.py:483-499                 all_gather_batch(text_latents), all_gather_batch(visual_tokens)   # DDP all-gather BEFORE the contrastive loss
```

### 3.2 Inference: `inference.py` → batch at the model

```
inference.py (parse_args)         --batch_size (default 1, CLI-EXPOSED, see Section 11) --data_folder/--preprocessed_dir --jsonl_file --labels_file --pathologies_file
inference.py (engine ctor)          self.ds = MRReportDatasetInfer(data_folder, jsonl_file, space, normalizer, labels_file, splits_csv, split, preprocessed_dir, use_preprocessed, ...)
inference.py:272-280                eval_loader = DataLoader(self.ds, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False, collate_fn=collate_fn_infer, pin_memory=True)
inference.py:288-289                for batch in eval_loader: imgs, sentences, subject_id, real_volume_mask, labels = batch
  data_inference.py:242-251           collate_fn_infer(batch): images, sentences, subject_id, mask, labels = batch[0]   # <-- SAME batch[0]-only bug, but batch_size IS user-settable here
```

### 3.3 Feature extraction: `extract_features.py`

```
extract_features.py:368-379       ds = MRReportDatasetInfer(...)              # same construction pattern
extract_features.py:403-406       loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False, drop_last=False, collate_fn=collate_fn_infer, pin_memory=True)
```
`batch_size=1` is a **literal constant** here, not a CLI variable — this entrypoint cannot trigger the batch-size bug.

### 3.4 Offline cache build: `preprocess_volumes.py`

```
preprocess_volumes.py:185           subjects = discover_subjects(args.data_folder, args.space)     # same shared function, report/split AGNOSTIC (no jsonl/splits args exist here)
preprocess_volumes.py:191-198       subjects = subjects[shard_index::num_shards]  (optional)  ->  subjects[:limit]  (optional)
preprocess_volumes.py:224-232       tasks built per subject: (subject_id, image_paths, out_path, target_spacing, target_shape, posterior_shift_voxels, normalizer, dtype, compress, overwrite)
preprocess_volumes.py:250-267       ProcessPoolExecutor(num_workers) or serial -> _process_subject(task) per subject
preprocess_volumes.py:119-126         arr = preprocess_nii(path, target_spacing, target_shape, posterior_shift_voxels, normalizer_obj)   # SAME function data.py uses live
preprocess_volumes.py:130-138         stacked = np.stack(vols); atomic tmp-then-rename np.savez(_compressed)(tmp, volumes=stacked)
```

### 3.5 CLI-default propagation gaps (explicitly requested to be flagged)

| Parameter | Exists on `MRReportDataset`? | Exposed by `run_train.py` CLI? | Consequence |
|---|---|---|---|
| `batch_size` | Yes (`MrRateTrainer`, default 1) | **No** — hardcoded `batch_size=1` at `run_train.py:235` | The `collate_fn` batch-size bug (Section 11) is **unreachable from this entrypoint**, by accident of this hardcoding, not by any guard in the Dataset/collate code itself |
| `target_spacing` | Yes, default `(1.0,0.5,0.5)` | **No** — never appears in `run_train.py`'s argparse or in the `MrRateTrainer(...)` call (`run_train.py:225-250`) | Cannot be changed via the documented training CLI at all; only reachable by editing source or calling `MrRateTrainer`/`MRReportDataset` programmatically |
| `target_shape` | Yes, default `(256,384,384)` | **No** — same as above | Same |
| `posterior_shift_mm` | Yes, default `15.0` | **No** — same as above | Same |
| `max_sentences_per_image` | Yes, default `34` | **No** — same as above | Same |
| `gradient_accumulation_steps` | Yes, default `1` | **No** — not in argparse, not passed | Always 1 through this entrypoint |
| `num_workers` | Yes, default `0` | **No** — not in argparse, not passed | Always 0 through this entrypoint (this happens to already match the smoke-test constraint used in this audit) |
| `normalizer_kwargs` | Yes | **No** — `run_train.py` never constructs or passes it | Non-default `PercentileNormalizer`/custom kwargs are unreachable via CLI; only the class default `{}` is ever used through `run_train.py` |
| `--batch_size` | — | **Yes, in `inference.py:401`** (default 1) | This is the one entrypoint where a user *can* trigger the batch-size bug, by passing `--batch_size 2` or higher |

---

## 4. Supported Filesystem Layouts (from `discover_subjects()`, `data.py:123-163` — confirmed by execution)

`discover_subjects(data_folder, space)`:
1. Lists top-level entries of `data_folder`, keeps only directories, sorts them (`data.py:136-139`). If none exist, returns `[]` immediately (`data.py:140-141`).
2. Takes the **first** sorted directory only and checks `os.path.isdir(<that dir>/<space>)` (`data.py:144-145`). This single check decides the layout for the **entire** call — every other directory is then processed under whichever branch that check selected.

### Layout 1 — "space-based" (selected when the first top-level dir has a `<space>` subfolder)
```
<data_folder>/
└── <study_uid>/
    └── <space>/              # e.g. native_space, coreg_space, atlas_space (literal string match)
        └── img/               # ALWAYS literally "img", regardless of which space — data.py:150
            └── <series>.nii.gz
```
- `data_folder` must point at the level whose **immediate children are study directories**.
- Study directories = every top-level directory under `data_folder` (no naming convention required — `data.py:149`).
- Image subdirectory is **always** `img`, never `coreg_img`/`atlas_img`, in this layout (`data.py:150`) — the `SPACE_TO_IMG_SUBDIR` mapping is **not consulted** in this branch.
- No batch-name convention, no nesting beyond one nested `<space>/img/` level is supported.

### Layout 2 — "HF batch-based" (selected when the first top-level dir does NOT have a `<space>` subfolder)
```
<data_folder>/
└── <batch_name>/                       # any name; no "batchNN" convention enforced in code
    └── <study_uid>/
        └── <img_subdir>/                # native_space->img, coreg_space->coreg_img, atlas_space->atlas_img (data.py:100-104, 155)
            └── <series>.nii.gz
```
- `data_folder` must point at the level whose immediate children are **batch** directories.
- `img_subdir` for an unrecognized `space` string silently falls back to `'img'` (`SPACE_TO_IMG_SUBDIR.get(space, 'img')`, `data.py:155`) rather than raising — a typo'd `--space` value degrades to native-style discovery with no warning.
- Study directories are read via a second, un-sorted-at-top-level-only pass: `for study_uid in sorted(os.listdir(batch_path))` (`data.py:158`) — sorted per batch, correctly.
- Zip files, tar files, or Hugging Face cache directories (e.g. `~/.cache/huggingface`) are **not** supported directly in either layout — only real, already-extracted subdirectories containing `*.nii.gz` files are recognized; `list_nii_files` (`data.py:112-120`) does a plain `os.listdir` + suffix filter with no archive awareness.
- Native, coreg, and atlas files **may coexist** side-by-side under one study directory (this is exactly the `merge_downloaded_repos.py` output layout from the official repo, per the prior audit task) — `discover_subjects` simply picks whichever `img_subdir` matches the requested `--space` for that one call; it does not read multiple spaces at once.
- **Duplicate studies across batches**: not deduplicated — if the same `study_uid` string appeared under two different batch directories, both would be appended to the returned list as **separate entries with the same `subject_id`**; downstream, `_prepare_samples` (`data.py:437-454`) builds a plain list (not a dict keyed by ID), so both would survive into `self.samples`, and the JSONL lookup `subject_to_sentences[sid]` (`data.py:452`) would silently attach the *same* report to both — a real duplication risk if a merge or partial re-run ever produces overlapping batch contents.

### Cache layout (from `preprocess_volumes.py:15-23`, `data.py:456-491`)
```
<preprocessed_dir>/
└── <space>/
    ├── _manifest.json          # CACHE_MANIFEST_NAME, data.py:109
    └── <study_uid>.npz         # key 'volumes': float16 [N, D, H, W]
```
This layout is **not** discovered via `discover_subjects()` at all — it has its own, separate listing function (`_prepare_samples_from_cache`, `data.py:468-491`) that just globs `*.npz` under `<preprocessed_dir>/<space>/`.

### Brittle assumptions and ambiguous cases (confirmed empirically with synthetic directory trees — Section 11.0)

| Scenario | Tested? | Result |
|---|---|---|
| Layout detection depends on inspecting only the first directory | Yes | Confirmed by code (`data.py:144-145`) and by design; see the two cases below for real consequences |
| An unrelated/interloper directory sorts alphabetically before real batch directories (e.g. a hypothetical fully-extracted `MR-RATE-atlas/` sitting next to `batch00`..`batch27` at the same level) | Yes | **Less dangerous than initially assumed.** In the tested case, the interloper is (correctly) still processed under the layout-2 branch and simply contributes 0 subjects (its internal structure doesn't happen to match `<img_subdir>` at the expected depth) — it does **not** prevent the real batch directories from being discovered normally, because layout-2's per-batch loop (`data.py:156-162`) runs independently over every top-level directory, not just the first |
| A per-study space inconsistency in layout 1 (e.g. one study has `native_space/` but not `atlas_space/`, another has `atlas_space/` but not `native_space/`) | Yes | **Confirmed misclassification.** If the alphabetically-first study lacks the requested `<space>` subfolder, `discover_subjects` wrongly falls back to layout-2 semantics for every study — the second study's real, correctly-present data is **not found** (0 results), even though it exists on disk in a valid layout-1 shape. This is a genuine, reproducible bug class, not just a theoretical risk |
| Empty study directory (dir exists, `img/` missing or has no `.nii.gz`) | Yes | Silently skipped, no error, no log line — confirmed |
| Symbolic links to study directories | Yes | Work transparently (`os.path.isdir` follows symlinks); a symlink and its target are **counted as two separate subjects** if they have different directory names, which is a latent duplication risk if a future merge step ever uses symlinks (the current `merge_downloaded_repos.py`, per the prior audit, uses `move`, not symlinks, so this risk is not currently realized in the shipped pipeline) |
| Partially merged layouts (mixed native + one derivative present, other derivative absent, at the same directory level) | Inferred from code + the per-study-inconsistency test above | Same misclassification risk applies whenever the *alphabetically-first* top-level entry is not representative of the whole tree |
| Zip files or HF cache dirs passed as `data_folder` | Confirmed by execution against real DATA_PATH (un-extracted tars) | Returns 0 subjects, no error — `os.path.isdir()` on a `.tar`/`.zip` file is `False`, so it's silently excluded from `first_level_dirs`, and if literally nothing is a directory, `discover_subjects` returns `[]` at `data.py:140-141` |

---

## 5. Compatibility With Your Local Datasets

### Compatibility matrix

| Local root / layout | Intended space | Compatible directly? | Required CLI args | Required action | Reshard needed? | Evidence |
|---|---|---|---|---|---|---|
| `DATA_PATH` (`/hnvme/workspace/b180dc29-MR-RATE`), currently 28 un-extracted `batchNN.tar` (each a tar-of-per-study-`.zip`) + `MR-RATE-atlas/` (same tar-of-zips pattern) | native_space (root); atlas_space (`MR-RATE-atlas/`) | **Incompatible** (0 subjects, confirmed by execution) | n/a until extracted | Extract every `batchNN.tar`, then extract every per-study `.zip` inside it, into a real directory tree | **No** — post-extraction the natural result is exactly `<extract_root>/batchNN/<study_uid>/img/*.nii.gz`, i.e. layout 2 as-is; the batch partitioning does not need to change | `discover_subjects(DATA_PATH, "native_space")` → 0; `discover_subjects(DATA_PATH, "coreg_space")` → 0; `discover_subjects(DATA_PATH, "atlas_space")` → 0 (all confirmed live) |
| `DATA_PATH/MR-RATE-atlas` alone | atlas_space | **Incompatible** (0 subjects, confirmed) | n/a until extracted | Same two-level extraction | No | Same live test, all three spaces → 0 |
| `SHARDS_PATH` (`/hnvme/workspace/y100dc19-MR-Rate-raw`), WebDataset-style `train/validation/test/shard-*.tar` + `series.parquet`/`studies.parquet` | Built for a named "MR Volume Generation" challenge (per prior audit's `.forithmus/config.json` finding), not one of `native_space`/`coreg_space`/`atlas_space` in this code's sense | **Incompatible** (0 subjects, confirmed) — and structurally cannot become compatible via extraction alone | n/a | Either (a) fully extract+reorganize into a `discover_subjects`-shaped tree (loses the existing shard/parquet indexing this data already has), or (b) write a **new** Dataset/IterableDataset that reads directly from the shard `.tar` files and `series.parquet`/`studies.parquet` manifests — no such reader exists anywhere in this repo today (confirmed by grep in the prior audit task) | **No**, in the sense that the WebDataset shard count/boundaries wouldn't need to change even under option (a) — but building a *new* loader (option b) is a small-to-medium engineering task, not a "reshard" | Live `discover_subjects` test on `SHARDS_PATH` and `SHARDS_PATH/train` → 0 in all cases |

Compatibility classification used above, per the categories requested:
- **directly compatible**: none of the three roots qualify.
- **compatible after pointing data_folder at a different level**: also does not apply here — even pointing `data_folder` at, say, `DATA_PATH/MR-RATE-atlas` changes nothing, because the blocking issue is that nothing is extracted, not that the wrong directory level was chosen.
- **compatible only after archive extraction**: **DATA_PATH and DATA_PATH/MR-RATE-atlas** — this is the correct classification for both.
- **compatible only after using the repository merge script**: not applicable to either local root as currently structured (`merge_downloaded_repos.py` merges *already-extracted* derivative trees into a base tree; DATA_PATH's derivatives aren't extracted yet, so there is nothing to merge yet — extraction must happen first regardless).
- **compatible only after a small directory-layout adapter**: this describes what this audit's own smoke test built (Section 11) — a thin, synthetic re-housing of already-available files into the exact shape `discover_subjects` expects. For DATA_PATH, once extracted, no adapter is even needed (it already lands in layout 2). For SHARDS_PATH, an adapter alone is not enough because the underlying container format (WebDataset tar members) is fundamentally different from "files sitting in real directories" — this is the one case that plausibly needs a genuinely new reader, not just a directory shim.
- **incompatible**: none are permanently incompatible; both are fixable, by different amounts of work (see below).

### Definitions, as requested

- **Extraction**: decompressing an archive (`.tar`, `.zip`, `.tar.gz`) into real files on disk, with **no change to which studies belong to which batch or shard**. This is what DATA_PATH needs (twice — once for the outer `batchNN.tar`, once per inner per-study `.zip`).
- **Merging**: `merge_downloaded_repos.py`'s specific operation of moving already-extracted derivative-repo study subdirectories (`coreg_img/`, `atlas_img/`, `transform/`, etc.) into the corresponding study directory of the base native tree, so one study folder holds all spaces. Only applicable after extraction; not currently blocking either local root today since DATA_PATH's only present derivative (atlas) isn't extracted either.
- **Reorganization**: any change to directory *shape* beyond plain extraction — e.g., renaming batch folders, moving studies between batches, flattening nested structure. **Not needed** for DATA_PATH; would be needed for SHARDS_PATH if you choose the extraction route rather than a new reader.
- **Preprocessing cache**: the `.npz`-per-subject output of `preprocess_volumes.py` — a *performance* optimization (skips repeated NIfTI decode/resample at train time) that requires the exact same source discovery to already work; it does not substitute for extraction/reorganization, it happens strictly after.
- **Sharding** (as the term is used for the *release*): the existing `batch00`…`batch27` division of the official dataset, chosen so each folder complies with a Hugging Face per-folder file-count limit (per the prior audit's dataset-guide reading) — this is a fixed, already-correct partitioning that the current code's layout 2 consumes as-is.
- **Resharding**: changing that partitioning — e.g., merging/splitting batches, or (for SHARDS_PATH) changing which studies land in which `shard-NNNNNN.tar`. **This audit found no reason to do this.** The code iterates whatever batch directories exist without caring about their size or count (`data.py:156-162`); nothing about `discover_subjects` requires a particular number of batches or a particular studies-per-batch count.

### Do I need to reshard my local MR-RATE data?

**No.** The blocking problem for DATA_PATH is that its official `batch00`…`batch27` release sharding is still compressed (tar-of-zips), not that the sharding itself is wrong — once extracted, `data_folder=<extract_root>` with the existing batch boundaries is layout 2 exactly as the code expects, with zero reorganization. For SHARDS_PATH, the blocking problem is a **container-format mismatch** (WebDataset tar members vs. real files-on-disk), which is a different kind of problem than resharding — the number and boundaries of its 3,762 shards are irrelevant to whether the current code can read them; a new reader (or full extraction, discarding the shard structure's efficiency) is what's needed, not different shard boundaries.

### Proposed target tree (extraction route for DATA_PATH), migration plan — **not performed in this audit**

```
<extract_root>/
├── batch00/
│   └── <study_uid>/
│       └── img/
│           └── <study_uid>_<series_id>.nii.gz
├── batch01/
│   └── ...
└── batch27/
    └── ...
```
Non-destructive migration outline (for future execution, not run here): (1) copy or stream-extract `batchNN.tar` to a **new** output location (never in place inside `DATA_PATH`, which must stay read-only); (2) for each per-study `.zip` inside, extract to `<extract_root>/batchNN/<study_uid>/`; (3) verify per-batch study counts against `series.parquet`/`studies.parquet` (already available from the prior audit) before treating a batch as complete; (4) run `discover_subjects(extract_root, "native_space")` as a cheap sanity check (as done in Section 11) before pointing real training at it.

---

## 6. Image / Report / Split Schemas

### 6.1 Report JSONL (`_load_jsonl`, `data.py:421-435` and `data_inference.py:117-131` — identical logic in both files)

- **Required keys**: `valid_json` (bool, must be truthy), `extracted_sentences` (list, must be non-empty), `volume_name` (str — used as the join key against discovered `subject_id`s).
- **Optional keys**: any others are read but ignored (the repo's own test fixture, `tests/test_data.py:158-173`, includes `original_findings` and `raw_output` alongside the three required keys, confirming extra keys are harmless).
- **Validity filter**: a line is skipped (via a bare `except Exception: continue`, `data.py:433-434`) if it isn't valid JSON, if `valid_json` is falsy/missing, or if `extracted_sentences` is empty/missing — **silently**, with no count of how many lines were dropped.
- **Matching identifier**: `volume_name` is matched **exactly** against the `subject_id` returned by `discover_subjects` (which is the study directory's name) — this is a **study-level**, not series-level, identifier; the same sentence list is used for every volume in that study.
- **Sentence representation**: pre-extracted **sentences**, not raw report text and not report *sections* — the file is expected to already be a per-sentence breakdown (consistent with the parent repo's use of this format as an LLM-extracted-sentence cache, though the extraction step itself lives outside `contrastive-pretraining/`).
- **Malformed lines**: dropped silently (see above); a single malformed line does not stop the whole file from loading.
- **Empty reports**: a JSONL entry with `extracted_sentences: []` fails the `len(...) > 0` check (`data.py:428`) and is dropped — such a subject simply never appears in `self.samples`, with no explicit "empty report" flag anywhere.
- **Sentence sampling and padding** (training path, `data.py:654-664`): if a subject has ≥`max_sentences_per_image` (default 34) sentences, exactly 34 are drawn via `random.sample` (uniform, **without replacement, re-sampled independently on every `__getitem__` call** — confirmed by execution: a 40-sentence synthetic subject returned a *different-composition* 34-sentence list depending on when it's called, since the seed is not fixed per-index); if fewer, the list is padded to 34 with empty strings `""`, and `mask` marks the real vs. padded positions.
- **Maximum sentence count**: fixed at `max_sentences_per_image` (default 34) for the **training** dataset only.
- **Raw reports vs. extracted sentences**: the inference dataset (`data_inference.py`) returns **all** extracted sentences, unpadded, uncapped — confirmed by execution (a 40-sentence synthetic subject returned a length-40 list from `MRReportDatasetInfer`, vs. length-34 from `MRReportDataset`).

Synthetic example (matches the schema exactly; no real data):
```json
{"volume_name": "<STUDY_UID>", "valid_json": true, "extracted_sentences": ["Synthetic example sentence."]}
```

### 6.2 Split CSV (`_load_splits`, `data.py:410-419` and `data_inference.py:106-115` — identical in both)

- **Required columns**: `split`, `study_uid` (only these two are read; `csv.DictReader` tolerates and ignores any other columns, e.g. the `batch_id`/`patient_uid` columns the real released `splits.csv` also has, per the prior audit).
- **Allowed split values**: whatever string is passed as the `split=` constructor argument (default `"train"` for `MRReportDataset`, `"test"` for `MRReportDatasetInfer`) is matched by exact string equality against the CSV's `split` column (`data.py:417`) — there is no enum/validation; a typo'd `--split` value silently yields an empty allow-list, and thus zero training subjects, with no error.
- **Matching identifier**: `study_uid`, matched against the same `subject_id` used for image discovery and report matching.
- **Missing study behavior**: a `study_uid` absent from the (optional) splits CSV is neither included nor excluded by the splits filter *itself* — but if a `splits_csv` was passed, `_load_splits` builds an explicit allow-set (`data.py:413-418`) and `_load_jsonl` only keeps `uid`s inside that set (`data.py:430-431`), so in practice a study not present in the splits CSV is **excluded** whenever a splits CSV is provided at all.
- **Patient-level separation**: **assumed, not verified by any code in this repo.** Neither `_load_splits` implementation ever reads or checks a `patient_uid` column — confirmed by the prior audit's independent read of the actual released `splits.csv` (patient-level integrity holds in the *data*, 0 leakage across 83,425 patients) and confirmed here by re-reading the loader code: it is architecturally incapable of checking this, since it never looks at `patient_uid` at all.

### 6.3 Pathology-label CSV (`_load_pathology_labels`, `data.py:560-586`; `_load_labels`, `data_inference.py:133-143`)

- **Required ID column**: `study_uid` if present, else `subject_id` (`data.py:572-576`); a CSV with neither raises `ValueError`.
- **Label columns**: every remaining column, coerced with `float(row[c])` (`data.py:583-584`) — a non-numeric value in any label column raises an uncaught `ValueError` at load time (not a per-row skip).
- **Missing-value behavior**: no explicit NaN/empty-string handling — an empty CSV cell would fail `float("")` and crash the whole dataset construction; this is a real, untested brittleness (no local example CSV was checked for empty cells since the audit avoided reading real label/report content).
- **Prevalence calculation**: `label_matrix.mean(axis=0)` (`data.py:531`) over only the subset of `self.samples` that also appear in the labels CSV (`data.py:522-523`) — i.e. prevalence is computed on the **dataset-filtered** subject set, not the raw CSV's full population.
- **Sampler construction**: `WeightedRandomSampler(weights=self.sample_weights, num_samples=len(self.samples), replacement=True)` (`data.py:599-604`), built from per-subject weights computed via one of three strategies (`inverse_freq`/`sqrt_inverse_freq`/`max_inverse_freq`, `data.py:501-503, 533-549`).
- **Does this affect the tensor returned by `__getitem__`?** **No.** `pathology_labels_csv` affects only *which subjects get drawn more often* (via the sampler) — it is never joined into, or returned as part of, `__getitem__`'s output tuple in `MRReportDataset`. (`MRReportDatasetInfer` is the exception: if a separate `labels_file` is passed to *it*, the label vector *is* returned as the 5th tuple element — a different code path for a different purpose.)

---

## 7. Exact Live Preprocessing (execution order, confirmed by reading + live execution)

Entry point: `preprocess_nii(path, target_spacing, target_shape, posterior_shift_voxels, normalizer_obj)` (`data.py:232-241`), used identically by the live dataset (`data.py:637-639`, via the class-bound wrapper methods) and by `preprocess_volumes.py:122-125` for the offline cache — **this is a single shared function**, so live and cached preprocessing cannot drift by construction (not just "tested for equality" — they call the exact same code).

1. **Load**: `nib.load(str(path))` (`data.py:171`) — library used: **nibabel**.
2. **Canonical reorientation**: `nib.as_closest_canonical(nii_img)` (`data.py:173`) — reorders/flips array axes to the closest RAS-aligned orientation without resampling (a pure axis permutation/flip, not interpolation).
3. **Read array + NaN/Inf handling**: `img_data = nii_img.get_fdata().astype(np.float32)` then `np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)` (`data.py:175-176`) — **this happens before resampling**, so any NaN/Inf in the source data is zeroed before it can propagate through interpolation.
4. **Spacing source and reorder**: `voxel_sizes = nii_img.header.get_zooms()` on the *already-canonicalized* image (`data.py:178`) — so zooms are already in RAS axis order (dim0=R, dim1=A, dim2=S). Reordered to `current_spacing = (zooms[2], zooms[0], zooms[1])` = `(S-spacing, R-spacing, A-spacing)` (`data.py:181`).
5. **Axis transpose**: `img_data.transpose(2, 0, 1)` turns `(X,Y,Z)=(R,A,S)` into `(Z,X,Y)=(S,R,A)` (`data.py:185`) — matching the spacing reorder in step 4.
6. **Resample**: `resize_array(tensor, current_spacing, target_spacing)` (`data.py:23-29, 187`) — computes `scaling_factor[i] = current_spacing[i] / target_spacing[i]`, `new_shape[i] = round(original_shape[i] * scaling_factor[i])`, then `F.interpolate(tensor, size=new_shape, mode='trilinear', align_corners=False)`. **Interpolation mode: trilinear. `align_corners=False`.** This resamples to the *target physical spacing*, not to a fixed voxel count — voxel count after this step still varies per volume, because each volume's original physical size differs.
7. **Normalize**: one of `ZScoreNormalizer`/`PercentileNormalizer`/`MinMaxNormalizer` (`data.py:32-94`), applied to the resampled float32 array — still variable shape at this point.
8. **Crop/pad**: `crop_or_pad(data, target_shape, posterior_shift_voxels)` (`data.py:191-229`) — plain center crop on D and H; on W (the AP axis — see the anatomy discussion below), the crop center is shifted by `posterior_shift_voxels = round(posterior_shift_mm / target_spacing[2])` (`data.py:358`, `preprocess_volumes.py:161`) **toward lower index (posterior direction in RAS)** to compensate for the released images already having their anterior/face region zeroed by defacing. Whatever remains short of `target_shape` after cropping is then zero-padded, split symmetrically before/after each axis (`data.py:217-228`), via `F.pad(..., value=0)`.
9. **Cast**: only at the very end, in the class method wrapper (`data.py:617-620`, `data_inference.py:208-211`) — `torch.from_numpy(arr).unsqueeze(0).to(torch.bfloat16)` — adds the channel dim and casts float32→**bfloat16**. Normalization and cropping both still happen in float32; **bf16 is the very last step**, not applied during resampling or normalization.
10. **Stack**: per-subject volumes are loaded **in sorted-filename order** (`list_nii_files`, `data.py:112-120`, alphabetical `os.listdir` + sort) — there is **no modality-aware ordering or selection**; `torch.stack(volume_tensors, dim=0)` (`data.py:643`) produces `[N,1,D,H,W]`.
11. **4D/multichannel NIfTI**: **not handled** — `get_fdata()` would return a 4D array if the source file had a 4th dimension, and the subsequent `.transpose(2,0,1)` (a 3-argument permutation) would raise a `ValueError` on a 4D array. There is no guard or branch for this anywhere in `data.py`; a genuinely 4D or multi-echo NIfTI in the source tree would crash the dataset at load time, not be silently handled.
12. **Masks**: the live/cached image pipeline **never loads brain-mask or defacing-mask files at all** — `discover_subjects` only looks in the `img`/`<img_subdir>` folder (`data.py:150, 159`), never `seg/`. Skull-state, brain-mask presence, etc. are entirely invisible to this loader; it operates purely on the released, already-defaced-but-not-skull-stripped image array as-is (consistent with the prior audit's Phase-4 finding that native-space release images are defaced, not skull-stripped).

### 7.1 Anatomical axis meaning — verified from code, not assumed from variable names

For the tensor `[N, 1, D, H, W]` returned by `__getitem__`:
- `N` = number of MRI series (volumes) for this **study** (variable, sorted by filename — not necessarily a meaningful clinical order).
- `1` = image channel (always 1 for this loader; a separate "3-channel replication for `fusion_mode=early`" happens **downstream in the model**, not in the Dataset).
- `D` = the axis that was `S` (Superior–Inferior) in the RAS-canonical image, per the `transpose(2,0,1)` at `data.py:185` and the `current_spacing`/comment at `data.py:180-181` — this one **does** mean what its name suggests: depth = superior–inferior extent.
- `H` = the axis that was `R` (Right–Left / Left–Right) in RAS — **this is not vertical "height" in any viewing sense; it is the left–right extent.** The variable name `H` is a generic array-axis label, not an anatomical claim, and should not be read as "superior-inferior" or "coronal height."
- `W` = the axis that was `A` (Anterior–Posterior) in RAS — confirmed both by the transpose/spacing-reorder code and by the fact that `posterior_shift_voxels` is applied specifically to this axis (`data.py:209-213`, comment "W axis (Y/AP): shift center posteriorly").

**So `H` and `W` are, respectively, the left–right and anterior–posterior axes — not "height" and "width" in any 2D-image sense.** This matters because a new report-to-volume model's data documentation should not silently inherit a `[N,C,D,H,W]` docstring that implies a video-like (time, channel, height, width) convention; the actual physical axes are (series, channel, S, R, A).

### 7.2 Does `target_spacing=(1.0,0.5,0.5)`, `target_shape=(256,384,384)` mean a 256×192×192mm physical FOV?

**No — and this is an important correction to a plausible-looking assumption.** `target_spacing` and `target_shape` are both given in the same `(D,H,W)` = `(S,R,A)` axis order (confirmed: `target_shape=(256,384,384)` is documented in `data.py:78-80`/`preprocess_volumes.py` argparse `metavar=("D","H","W")`, and `target_spacing`'s components are used index-for-index against that same axis order in `resize_array`/`crop_or_pad`). So the physical FOV is `shape[i] × spacing[i]` **per axis, not shape[0]×spacing[0] applied to every axis**:
- D (Superior–Inferior): `256 × 1.0mm = 256mm`
- H (Right–Left): `384 × 0.5mm = 192mm`
- W (Anterior–Posterior): `384 × 0.5mm = 192mm`

So the physical FOV is **256mm (S–I) × 192mm (R–L) × 192mm (A–P)** — the "256×192×192mm" figure in the prompt is numerically the right set of three numbers, but only correct once you know it's **(D,H,W)-axis-paired**, i.e. 256mm along Superior–Inferior and 192mm along **both** Right–Left and Anterior–Posterior (not, e.g., 256mm along Right–Left as a naive `[N,C,H,W]` video/image convention might suggest). This was verified directly against the code's own axis bookkeeping (`data.py:180-181, 185, 209-213`), not assumed from variable names.

### 7.3 Reorientation vs. resampling vs. resizing vs. crop/pad vs. co-registration vs. atlas registration

| Operation | What it changes | Who performs it here |
|---|---|---|
| **Reorientation** | Axis order/flips only (which array index corresponds to which anatomical direction); no interpolation, no change to voxel values or physical spacing | `nib.as_closest_canonical` — **the Dataset does this**, live, every load (`data.py:173`) |
| **Resampling** | Physical voxel spacing (mm/voxel), via interpolation; changes voxel *count* as a side effect of holding physical size roughly fixed | Trilinear `F.interpolate` — **the Dataset does this**, live (`data.py:28, 187`) |
| **Resizing** (as commonly meant: force a fixed voxel count regardless of physical size) | Voxel count directly, ignoring physical spacing | **The Dataset does NOT do a naive resize** — its "resample" step is spacing-driven, not shape-driven; shape is only forced afterward, via crop/pad |
| **Crop/pad** | Trims or zero-fills the spatial extent to an exact target voxel count, after resampling | **The Dataset does this**, live (`data.py:191-229`) |
| **Co-registration** | Aligns one study's *other* series to its own T1w center modality (per-study, native-resolution alignment) | **Not performed by this Dataset at all** — this is exclusively the separately-run `registration.py` pipeline (per the prior audit), producing the `coreg_space` derivative that this loader can *read* (via `--space coreg_space`) but never *computes* |
| **Atlas registration** | Aligns the center modality (and, transitively, the whole study) to the MNI152 template — a population-level, not per-study, alignment | Same as co-registration: **not performed here**, only optionally *read* via `--space atlas_space` |

**What must already be true of the source data before this Dataset can be used**: DICOM→NIfTI conversion, de-identification/defacing, and (if `--space coreg_space`/`atlas_space` is requested) the corresponding registration must already be done upstream — none of that is in scope for `data.py`. What the Dataset *does* do, every time, regardless of `--space`: canonical reorientation, physical-spacing resampling, intensity normalization, and shape-fixing crop/pad.

---

## 8. Exact Cached Preprocessing (`preprocess_volumes.py`)

- **Cache creation command** (from the script's own docstring, `preprocess_volumes.py:28-34`):
  ```
  python scripts/preprocess_volumes.py --data_folder <dir> --out_dir <dir> --space coreg_space --normalizer zscore --num_workers 8
  ```
- **Output structure**: `<out_dir>/<space>/_manifest.json` + `<out_dir>/<space>/<study_uid>.npz` (`preprocess_volumes.py:15-23`).
- **Manifest fields** (`build_cache_manifest`, `data.py:244-257`): `version` (=1), `layout` (="per_subject_stack"), `space`, `target_spacing`, `target_shape`, `posterior_shift_mm`, `normalizer`, `normalizer_kwargs`, `dtype`.
- **`.npz` keys**: exactly one, `volumes`, shape `[N, D, H, W]` (confirmed by execution: a 2-volume synthetic subject wrote a `(2, 256, 384, 384)` array).
- **Array dtype**: whatever `--dtype` requested (`float16` default, `float32` alternative — CLI only allows these two, `preprocess_volumes.py:83-84`); confirmed by execution (`float16` observed on disk).
- **Compression**: off by default (`np.savez`, not `np.savez_compressed`) — "optimized for fast training reads" per the script's own `--compress` help text (`preprocess_volumes.py:85-87`); `--compress` opts into `savez_compressed`.
- **Overwrite/resume behavior**: existing `.npz` files are skipped unless `--overwrite` is passed (`preprocess_volumes.py:113-114`); writes are atomic (`tmp.npz` then `os.replace`, `preprocess_volumes.py:134-138`), so an interrupted job never leaves a half-written file that a subsequent run would mistake for complete.
- **Multiprocessing / manual sharding**: `--num_workers` controls a `ProcessPoolExecutor` (or serial loop if ≤1) that parallelizes **subjects within one job** (`preprocess_volumes.py:250-267`). `--num_shards`/`--shard_index` instead partition the **already-sorted** subject list via a strided slice, `subjects[shard_index::num_shards]` (`preprocess_volumes.py:192-193`) — **confirmed: this is purely a work-distribution mechanism across separate job invocations writing into the same `<out_dir>/<space>/` directory; it does not change the on-disk format, file count semantics, or naming in any way.** This directly answers the prompt's question: `--num_shards` assigns subsets of studies to separate preprocessing jobs; it is not a permanent storage-format change.
- **Cache validation** (`validate_cache_manifest`, `data.py:268-311`, used identically by both `MRReportDataset` and `MRReportDatasetInfer`): compares `CACHE_CONFIG_KEYS = (space, target_spacing, target_shape, posterior_shift_mm, normalizer, normalizer_kwargs)` (`data.py:262-265`) between the on-disk `_manifest.json` and the currently-requested config.
- **Which differences cause rejection**: any mismatch in the six keys above raises `ValueError` by default (`data.py:300-311`).
- **Which differences may be bypassed**: (a) any of the six keys above, if `cache_allow_mismatch=True`/`--cache_allow_mismatch` is passed — downgraded to a printed warning instead of an exception (`data.py:305-306`); (b) `dtype` is **never checked at all** — it is not a member of `CACHE_CONFIG_KEYS`, so a cache written as `float16` and one nominally expected to be `float32` (or vice versa) produce no warning even without `cache_allow_mismatch`, because the loader always upcasts whatever it reads to `bfloat16` regardless (`data.py:630-633`) — this is a **safe**, intentional omission, not an oversight, since the dtype difference is fully absorbed downstream.
- **Risk of bypassing validation**: silently training on volumes preprocessed with a different spacing/shape/posterior-shift/normalizer than the one the run believes it's using — e.g. a stale cache from an earlier experiment would load without error and produce numerically wrong (differently normalized or geometrically shifted) inputs with no crash to reveal it.
- **Live vs. cached equality**: not merely "tested" in the abstract — they call the **identical `preprocess_nii` function** (`data.py:232-241`, imported directly by `preprocess_volumes.py:50`), so they cannot drift by construction, only by *configuration* mismatch (which `validate_cache_manifest` is specifically designed to catch). This audit additionally **executed** both paths on the same synthetic subject and confirmed `torch.allclose` agreement within `atol=1e-2` (the residual difference is explained by the float16-on-disk → bfloat16-at-load rounding, not a logic difference) — see Section 11, Part F.

### Is `--num_shards` a permanent storage-format change or a job-partitioning convenience?

**Job-partitioning convenience only**, confirmed by code (`preprocess_volumes.py:192-193`) — it slices the subject list, nothing about the output directory structure, filenames, or manifest changes based on `--num_shards`/`--shard_index`.

### Does creating the `.npz` cache count as "resharding"?

**No.** It is a **performance optimization** (skip repeated NIfTI decode+resample+normalize+crop at every training epoch) that preserves the exact same subject set and study/batch semantics as the source tree it was built from — it is not necessary for the dataset to function (the live path works without it), only for training throughput on large/slow-to-read derivatives (the script's own stated motivation, `preprocess_volumes.py:1-7`, is specifically the ~2GB coregistered/atlas files starving GPUs on live reads).

### Cache storage estimate (aggregate, using the actual defaults and a confirmed real measurement)

Using the formula `bytes_per_volume = D × H × W × bytes_per_element` with the **default** `target_shape=(256,384,384)` and `float16` (2 bytes/element):

```
bytes_per_volume = 256 × 384 × 384 × 2 = 75,497,472 bytes ≈ 72.0 MiB per volume
```

This was **confirmed empirically** in Section 11 (Part F): a synthetic 2-volume subject produced a `.npz` of 150,995,212 bytes ≈ `2 × 75,497,472` bytes plus ~268 bytes of `.npz`/zip-container overhead — matching the formula almost exactly.

Applying this to the **prior audit's aggregate counts** (label only, not a re-measurement in this task): the released dataset has 705,254 series total (documented) / 636,218 series observed locally in `series.parquet` (prior audit). At ~72.0 MiB/volume and `float16`:
- Full documented dataset (705,254 series): **≈ 48.4 TiB** of `.npz` cache, uncompressed, at the current defaults.
- Locally-present series only (636,218): **≈ 43.7 TiB**.

**These are aggregate, order-of-magnitude estimates only** (labeled explicitly as such per the task's instructions) — actual per-volume size varies slightly with `.npz` container overhead, and `--compress` would reduce it at the cost of slower reads; `--dtype float32` would **double** both figures.

---

## 9. Anatomy of One Dataset Sample and One DataLoader Batch (confirmed by live execution — Section 11)

```
dataset[0]                                          # MRReportDataset, synthetic 2-volume subject
└── tuple (len=3)
    ├── volume_stack: torch.Tensor
    │   ├── shape: [2, 1, 256, 384, 384]             # N=2 (this subject's volume count), C=1, D=256, H=384, W=384
    │   ├── dtype: torch.bfloat16
    │   └── value range: [-1.0, 1.0]                 # zscore normalizer clip+rescale
    ├── selected_sentences: list[str], len=34         # ALWAYS 34 (max_sentences_per_image default) -- 2 real + 32 "" padding, this subject
    └── mask: torch.Tensor
        ├── shape: [34]
        ├── dtype: torch.bool
        └── sum: 2                                    # number of real (non-padded) sentence slots
```

```
next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_fn)))
└── tuple (len=3)
    ├── images: torch.Tensor
    │   ├── shape: [1, 2, 1, 256, 384, 384]           # leading 1 = batch dim, added by collate_fn.unsqueeze(0); rest UNCHANGED from the sample
    │   └── dtype: torch.bfloat16
    ├── sentences: list[str], len=34                   # IDENTICAL object to the single sample's list -- not re-batched, no list-of-lists
    └── masks: torch.Tensor, shape=[1, 34], dtype=torch.bool
```

**The only difference between one sample and one batch, at `batch_size=1`, is a single `unsqueeze(0)` on the two tensors — `sentences` is untouched.** There is no meaningful "collation" happening; `collate_fn` is an unwrap-and-reshape operation for exactly one item, not a general batching function.

### Batch size behavior — confirmed by live execution (Section 11, Part D)

- **Configured batch size**: `1`, everywhere it matters for training (`run_train.py:235`, hardcoded, not CLI-exposed) and in most inference/feature paths (`extract_features.py:404`, `mil_probe_online.py:309`, both hardcoded `1`).
- **Does batch_size > 1 "work"?** It **runs without raising an exception**, but it is **not correct**: `collate_fn`/`collate_fn_infer` only ever read `batch[0]` (`data.py:669`, `data_inference.py:244`). Live execution with `batch_size=2` on a 2-subject synthetic dataset: the `DataLoader` internally fetched **both** subjects (both were fully loaded/preprocessed — confirmed by the printed `[Dataset] Loading subject ...` lines appearing for both), grouped them into one internal batch of 2, called `collate_fn` once, and received back a tensor representing **only 1 study** — the second subject's fully-computed data was silently discarded with **no error, no warning, no count mismatch visible to the caller** (the returned `images.shape[0]` is still 1, same as at `batch_size=1`).
- **Variable numbers of series**: padded or masked, but **only across DDP ranks**, not across studies in one batch — `mr_rate_trainer.py:414-427` pads the volume-count (`N`) dimension up to the maximum seen across all processes in that step (via `dist.all_reduce(..., op=MAX)`), building a `real_vols` boolean mask the model uses to ignore the padding. This exists because different ranks may pull subjects with different `N` in the same step; it is **not** a batch-of-studies padding mechanism (since the study-batch dimension is always exactly 1 per rank in every configured entrypoint).
- **`inference.py` is the one entrypoint that is actually vulnerable in practice**: `--batch_size` is a real, CLI-exposed integer (`inference.py:401`, default `1`) that flows straight into a `DataLoader(..., collate_fn=collate_fn_infer)` (`inference.py:272-280`) with the identical `batch[0]`-only bug. Running `inference.py --batch_size 4` would silently process only `ceil(len(dataset)/4)` studies instead of `len(dataset)`, discarding roughly 3 of every 4 fetched (and fully I/O- and compute-loaded) studies with no error and no partial-completion message.
- **Distributed training's effect on effective batch size**: per-rank batch is always 1 study; `Accelerator.prepare()` (`mr_rate_trainer.py:283-292`) wraps the model in DDP and (per Accelerate's own, not this repo's, documented behavior) will additionally shard/distribute the `DataLoader`'s sampling across ranks so different ranks see different subjects each step — this last point is **Accelerate library behavior, not verified from this repo's own source**, and is flagged here as inferred rather than confirmed. What **is** confirmed from this repo's code: the contrastive loss explicitly `all_gather`s `text_latents` and `visual_tokens` across all DDP ranks *before* computing the loss (`mr_rate.py:483-499`), so more GPUs directly increases the number of distinct studies contrasted against each other in the loss, even though each individual rank only ever processes one study per step.
- **Gradient accumulation**: implemented (`mr_rate_trainer.py:203, 452, 477`), defaulting to `1` and never overridden by `run_train.py`'s CLI (not in its argparse list) — so it is always 1 through the documented training command.
- **Batch dimension inside the model**: `b, r, c, d, h, w = image.shape` (`mr_rate.py:418`) — `b` (batch = studies) is always 1 in every configured entrypoint; `r` (series/volumes per study) is a **secondary, pooled** axis consumed via `fusion_mode`-specific logic (`early` uses only `r=0`; `mid_cnn`/`late`/`late_attn` pool across all `r` using `real_volume_mask`) — series is never treated as the SGD batch dimension anywhere in this code.

---

## 10. Distributed-Training Behavior (summary; see Sections 8–9 above for full citations)

- `Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True), InitProcessGroupKwargs(timeout=10h)])` (`mr_rate_trainer.py:182-187`).
- `SyncBatchNorm` conversion happens twice, redundantly: once unconditionally in `run_train.py:203` (`convert_bn_to_syncbn(clip)`, applied to every run regardless of process count) and again conditionally inside the trainer only if `num_processes > 1` (`mr_rate_trainer.py:277-279`) — the `run_train.py`-level conversion runs even for single-process training, which is harmless (SyncBatchNorm behaves like plain BatchNorm with one process) but is a minor redundancy worth noting.
- The model is moved to device and DDP-wrapped (`accelerator.prepare_model(..., device_placement=False)`, `mr_rate_trainer.py:283`) **before** the dataloader/optimizer/scheduler are `prepare()`'d (`mr_rate_trainer.py:284-292`).
- Volume-count (`N`) padding across ranks (`mr_rate_trainer.py:414-427`) exists specifically so that `SyncBatchNorm`'s cross-rank statistics synchronization and DDP's gradient all-reduce see tensors of a consistent shape across ranks in a given step, since different ranks' subjects can have different `N`.
- Checkpointing: model-only (`MrRate.<step>.pt`) and full-resume (`MrRate.full.<step>.pt`) checkpoints are both written every `save_model_every` steps by the main process only (`mr_rate_trainer.py:500-510`); the full-checkpoint file is **overwritten** each time ("to save disk space" per the comment) rather than versioned.

---

## 11. Smoke Test Results (executed this session — synthetic data only)

**Method**: none of the local real dataset layouts are directly readable by `discover_subjects()` (confirmed, Section 5), and this audit does not extract real archives. To still exercise the *actual, unmodified* dataset/collate/cache code end-to-end, a small synthetic layout-2 tree was built in the session scratchpad (never in `DATA_PATH`, `SHARDS_PATH`, or `OUTPUT_PATH`) with 2 synthetic subjects (`SYN_STUDY_0001`: 2 volumes; `SYN_STUDY_0002`: 4 volumes, both randomly-generated 24×28×16 arrays with an anisotropic synthetic affine — no real patient data was read, copied, or referenced anywhere in this test), plus a synthetic JSONL (placeholder sentences only) and, for the cache test, a synthetic `.npz` cache built via the real `preprocess_nii`/`build_cache_manifest` functions. Run with `/apps/python/3.12-conda/envs/pytorch2.5.1/bin/python3` (has torch 2.5.1 + nibabel 5.4.2 + numpy), `num_workers=0` throughout, wrapped in `timeout 180`. Total wall time: a few seconds.

| Test | Dataset class | Space | Source mode | N series | Result |
|---|---|---|---|---|---|
| A | — (`discover_subjects` directly) | native_space | live tree | — | 2 subjects found |
| B | `MRReportDataset` | native_space | live NIfTI | 2 and 4 | `ds[0]`: `[2,1,256,384,384]` bf16, range `[-1,1]`, 0 NaN/Inf, 34 sentences (2 real + 32 padded), mask sum=2. `ds[1]`: `[4,1,256,384,384]` bf16, 34 sentences (randomly subsampled from 40), mask sum=34 |
| C | `MRReportDataset` + `collate_fn` | native_space | live NIfTI | 2 | batch_size=1 → `images:[1,2,1,256,384,384]`, `masks:[1,34]`, `sentences` unchanged len-34 list |
| D | `MRReportDataset` + `collate_fn` | native_space | live NIfTI | 2 (subj 1), 4 (subj 2) | batch_size=2 → **both subjects fully loaded/preprocessed, but the returned batch represents only 1 study** (`images.shape=[1,2,1,256,384,384]`) — the second subject's fully-computed volume stack was silently discarded, confirmed live |
| E | `MRReportDatasetInfer` + `collate_fn_infer` | native_space | live NIfTI | 4 | all 40 sentences returned (no padding/truncation), `real_volume_mask` all-True shape `[4]`, `labels` empty `ndarray` shape `(0,)`; after collate: `images:[1,2,1,256,384,384]`, `mask:[1,2]` |
| F | `MRReportDataset` (`use_preprocessed=True`) vs. live | native_space | `.npz` cache built via real `preprocess_nii` | 2 | Cache write: `float16` `[2,256,384,384]`, file size 150,995,212 bytes (matches the `D×H×W×2` formula almost exactly). Cached-load vs. live-load `torch.allclose(atol=1e-2)` → **True** |

No exceptions or warnings were raised in any of the above except the intentional/expected ones already documented (the batch-size-2 silent drop is not an exception — it is silent by design, which is itself the finding).

---

## 12. Edge Cases and Brittle Assumptions (consolidated)

1. **`collate_fn`/`collate_fn_infer` silently drop everything past `batch[0]`** — the single most consequential finding in this audit for anyone reusing this loader machinery. Confirmed live (Section 11, Part D). Reachable today only via `inference.py --batch_size N>1`.
2. **Layout auto-detection inspects only the first alphabetically-sorted top-level directory** (`data.py:144-145`) — safe when the tree is uniform, but produces a **confirmed, reproducible misclassification** (0 results despite valid data existing) when per-study space availability is inconsistent in a layout-1 tree (Section 4, tested live with synthetic data).
3. **No 4D/multi-echo NIfTI support** — `.transpose(2,0,1)` on a 4D array raises; there is no shape check before this call anywhere in `data.py`.
4. **No archive-format support** — zip/tar/HF-cache directories are invisible to `discover_subjects`; confirmed live against the real, currently-un-extracted `DATA_PATH` and `SHARDS_PATH` (0 subjects each).
5. **Duplicate `study_uid` across batches is not deduplicated** — both would survive into `self.samples` as separate list entries sharing one JSONL report.
6. **Uncaught exceptions during `__getitem__`** — a single unreadable/corrupt NIfTI (per the prior audit, ~0.4% of the released dataset's series are known-corrupt) will raise inside `_load_volume_stack` with no try/except anywhere in the call chain, crashing the whole training step rather than skipping that one subject.
7. **`--space` typos silently degrade rather than error**: an unrecognized `space` string falls back to `img_subdir='img'` in layout 2 (`data.py:155`) with no validation or warning.
8. **Stale `.npz` caches are only caught if `dtype` isn't the sole difference** — six of seven cache-manifest fields are strictly checked; `dtype` is deliberately never checked (safe, since the loader always upcasts to bf16 regardless).
9. **Geometry/training hyperparameters are not exposed by the documented training CLI** (`target_spacing`, `target_shape`, `posterior_shift_mm`, `max_sentences_per_image`, `gradient_accumulation_steps`, `num_workers`, `normalizer_kwargs`) — only reachable by editing `run_train.py` or calling `MrRateTrainer`/`MRReportDataset` programmatically.
10. **Patient-level split isolation is architecturally unverifiable by this loader** — `_load_splits` never reads `patient_uid`, so even if a future `splits.csv` regression introduced leakage, nothing in this code path would catch it.

---

## 13. What Can Be Reused for Report-to-Volume Generation

**Directly reusable, largely as-is:**
- `discover_subjects()` (`data.py:123-163`) — layout detection and NIfTI discovery are report/task-agnostic already (the docstring itself says "Report/split agnostic," `data.py:126`); works unchanged for a generative dataset once a real directory tree exists.
- The per-volume preprocessing primitives — `load_and_resample_nii`, the three `Normalizer` classes, `crop_or_pad`, `preprocess_nii` (`data.py:32-94, 166-241`) — are pure functions independent of the contrastive objective; directly reusable, though the *default parameter values* (spacing/shape/posterior-shift) were tuned for the discriminative VJEPA2/contrastive setup and should be re-validated for a generative target resolution/FOV.
- The `.npz` cache mechanism and its manifest-validation contract (`build_cache_manifest`, `validate_cache_manifest`, and `preprocess_volumes.py` wholesale) — reusable unchanged as a performance layer for any new Dataset, since it's decoupled from the contrastive-specific parts.
- The split-CSV and pathology-label-CSV loaders — reusable for split filtering and (if desired) rare-pathology-aware sampling in a generative context too.

**Tied specifically to contrastive vision-language training, not reusable as-is:**
- The random per-`__getitem__` sentence *subsampling* to a fixed count (`data.py:654-664`) — appropriate for a contrastive loss that only needs *some* sentence-image pairs per step, but actively harmful for generation, where the *whole* report (or a deliberately chosen subset like `findings`+`impression`) should condition the image deterministically, not be randomly re-sampled every epoch.
- `collate_fn`'s `batch[0]`-only behavior and the whole batch_size=1-assuming design (Sections 9, 12) — a generative training loop will very likely want a real batch of independent (report, single-volume) pairs; this collate function must not be reused unmodified.
- The **study-level** granularity of `__getitem__` (returns *all* of a study's volumes stacked, one report shared across all of them) — directly conflicts with the report-to-**volume** framing in the desired schema, which wants **one series** per training example (see below).
- `MRRATE`'s `real_volume_mask`/`fusion_mode` machinery (`mr_rate.py`) — entirely specific to pooling multiple series into one contrastive embedding; not applicable to a per-series generative target.

### Is the current return value suitable for `report → one MRI volume`?

**No, not without a new Dataset/adapter.** Concretely, checked against the code:
- **One study or one series?** `MRReportDataset.__getitem__` returns **one study**, as a stack of *all* its volumes (`data.py:645-664`) — never a single series in isolation.
- **Does pairing one report with all series cause ambiguity?** Yes, and the code does exactly this: the same `sample['sentences']` list is attached to every volume in the stack (`data.py:449-452`) with no per-series attribution — a report describing one sequence's findings is presented as if it equally describes every other volume in the study, exactly the label-noise risk already flagged in the prior audit's dataset-level report.
- **Are modality and plane available to the loader?** **No.** `discover_subjects`/`_prepare_samples` carry only `subject_id` and a list of `image_paths` (`data.py:147-163, 444-454`) — the `series_id`/`classified_modality`/`acquisition_plane` fields that exist in the *official metadata CSV* (per the prior audit) are never read, joined, or exposed anywhere in this loading path. A generative loader needs to join this in itself.
- **Is original spacing/FOV discarded after fixed-grid preprocessing?** **Yes, completely.** `crop_or_pad`'s output is a bare array of `target_shape` with no accompanying record of the original shape, spacing, or FOV (`data.py:191-229`, `crop_or_pad` class method `data.py:617-620`) — nothing in `__getitem__`'s return tuple carries this information forward.
- **Are series filenames retained after loading?** **No.** `image_paths` exists only inside the internal `self.samples` list used to build the stack; `__getitem__` never returns the path, filename, or `series_id` for any volume — the returned tensor's `N` axis has no accompanying label of *which* volume is at which index (only their sorted-filename order, which itself is not returned as a label).
- **Should report `technique` be a condition?** Given `technique` (per the prior audit's confirmed reports-CSV schema) explicitly names which sequences were acquired, it is a strong candidate condition/leakage risk either way — worth an explicit decision, but this codebase currently only sees a flat, pre-extracted sentence list with no section labels (`clinical_information`/`technique`/`findings`/`impression` distinctions are not preserved in the JSONL schema this loader consumes at all).
- **Is the current random sentence selection appropriate for generation?** **No** (see above) — determinism and (likely) full-report or full-section conditioning are preferable for a generative target; the inference dataset's "return everything" behavior is closer to what generation needs than the training dataset's random-34-subsample.
- **Does a generative model need full findings/impression rather than sampled sentences?** Very likely yes, though this is a **design recommendation**, not something this audit can settle definitively from code alone — flagged as a decision for you, consistent with the "unresolved questions" framing in the prior audit's deliverables.
- **Is the fixed-grid preprocessing appropriate for a first generative baseline?** Plausible as a *starting point* (it's exactly what the existing, working pipeline already does, and is directly reusable), but the *specific* defaults (1.0×0.5×0.5mm spacing, 256×384×384 shape) were chosen for a discriminative contrastive encoder, not validated for a generative target — re-evaluate before treating them as final, per the prior audit's Section 10 recommendation.

---

## 14. Required Dataset Changes for Generation — Proposed Interface (pseudocode only, not implemented)

```python
class MRReportToVolumeDataset(Dataset):
    """
    One item = one (report-conditioning, single series) pair.
    Reuses: discover_subjects() for filesystem walking, preprocess_nii()'s
    building blocks (load_and_resample_nii / normalize / crop_or_pad) for the
    per-volume transform, and the splits-CSV loader for train/val/test filtering.
    Does NOT reuse: MRReportDataset's per-study sentence stack, its random
    sentence subsampling, or its N-volume stacking (this class flattens the
    study-level subject list into a series-level example list instead).
    """
    def __init__(self, data_folder, metadata_csv, reports_csv, splits_csv, split,
                 space="native_space", target_spacing=..., target_shape=..., ...):
        # 1. subjects = discover_subjects(data_folder, space)            # REUSE, unchanged
        # 2. metadata = read metadata_csv -> per-series modality/plane/is_center_modality
        #    (NEW: this loader must join series_id -> metadata row; MRReportDataset never does this)
        # 3. reports = read reports_csv -> per-study report/technique/findings/impression
        #    (NEW: keep sections separate, do not flatten to a single sentence list)
        # 4. split_uids = _load_splits(splits_csv, split)                 # REUSE, unchanged
        # 5. self.examples = flatten: one row per (series_path, metadata row, study's report)
        #    filtered to split_uids and to whatever modality/plane subset is in scope for v1
        ...

    def __len__(self):
        return len(self.examples)  # NOTE: this is now a SERIES count, not a STUDY count

    def __getitem__(self, index):
        ex = self.examples[index]
        resampled = load_and_resample_nii(ex.path, self.target_spacing)      # REUSE
        original_shape, original_spacing = ... # capture BEFORE crop_or_pad, which MRReportDataset discards
        normalized = normalizer_obj.normalize(resampled)                     # REUSE
        cropped = crop_or_pad(normalized, self.target_shape, self.posterior_shift_voxels)  # REUSE
        image = torch.from_numpy(cropped).unsqueeze(0).to(dtype)             # [C, D, H, W]  -- NOTE: no N axis
        return {
            "image": image,                                  # Tensor[1, D, H, W]
            "input_text": ex.report_text_or_section,          # NEW: deterministic choice of section(s), not random sentences
            "modality": ex.classified_modality,                # NEW: joined from metadata_csv, never available in MRReportDataset
            "plane": ex.acquisition_plane,                      # NEW
            "spacing": torch.tensor(original_spacing),          # NEW: currently discarded by crop_or_pad
            "shape": torch.tensor(original_shape),               # NEW: currently discarded
            "fov": torch.tensor([s*p for s, p in zip(original_shape, original_spacing)]),  # NEW
            "study_key": ex.study_uid,                           # NEW: never returned by MRReportDataset
            "series_key": ex.series_id,                           # NEW: never returned by MRReportDataset
        }

def collate_fn_generation(batch):
    """A REAL collate function -- must not repeat data.py's batch[0]-only bug."""
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),      # [B, 1, D, H, W]
        "input_text": [b["input_text"] for b in batch],                  # list[str], len=B
        "modality": [b["modality"] for b in batch],
        "plane": [b["plane"] for b in batch],
        "spacing": torch.stack([b["spacing"] for b in batch], dim=0),    # [B, 3]
        "shape": torch.stack([b["shape"] for b in batch], dim=0),         # [B, 3]
        "fov": torch.stack([b["fov"] for b in batch], dim=0),              # [B, 3]
        "study_key": [b["study_key"] for b in batch],
        "series_key": [b["series_key"] for b in batch],
    }
```

Minimal new pieces needed beyond what exists today: (1) a metadata-CSV join (series_id → modality/plane), entirely absent from the current loader; (2) a report-section-aware text field instead of a flattened, randomly-subsampled sentence list; (3) preservation of original shape/spacing/FOV through to the returned item, which `crop_or_pad` currently discards; (4) a per-series (not per-study) example list; (5) a batch-size-correct collate function.

---

## 15. Verified Facts, Inferred Facts, and Remaining Unknowns

**Verified by direct code reading with line citations (this session):** every claim in Sections 2, 3, 4, 6, 7, 8, 10, 12 not otherwise flagged.

**Verified by live code execution (this session, Section 11 and the discover_subjects tests in Section 4):** the batch_size=2 silent-drop behavior; the layout-1-inconsistency misclassification; the symlink/empty-directory handling; 0-subjects against every real local root; cached-vs-live numerical equivalence; the `.npz` size formula.

**Inferred, not verified from this repo's own code:**
- Accelerate's exact DataLoader-sharding behavior across DDP ranks under `accelerator.prepare()` (Section 9) — this is documented, standard behavior of the `accelerate` library, not something this repo's own source demonstrates directly.
- Whether report `technique` *should* be a generation condition, and whether full findings/impression is strictly necessary vs. sampled sentences — these are design recommendations, explicitly not settleable from code alone.

**Remaining unknowns:**
- Whether any *other*, uncommitted or externally-hosted training entrypoint exists that calls `MRReportDataset`/`MRReportDatasetInfer` with `batch_size` set via a different mechanism than the two CLIs read in this audit (`run_train.py`, `inference.py`) — this audit only traced the entrypoints present in this checked-out repo.
- The real-world frequency of the layout-1 per-study-space-inconsistency misclassification bug on the actual released dataset — this was demonstrated on synthetic data; whether it is ever triggered by real MR-RATE trees (e.g. studies with some but not all derivative spaces present) was not checked against real extracted data in this audit (none was extracted).
- Behavior on truly 4D/multi-echo NIfTI files in the real dataset — inferred to crash from the code path, not observed against a real 4D file (the prior audit's 37-sample byte-level check found only 3D volumes, too small a sample to rule out 4D series existing elsewhere in the 636,218-series dataset).

---

## 16. Source-Code Index (file:line references used in this report)

- `contrastive-pretraining/scripts/data.py`
  - `resize_array`: L23-29 — `discover_subjects`: L123-163 — `list_nii_files`: L112-120 — `SPACE_TO_IMG_SUBDIR`: L100-104 — `load_and_resample_nii`: L166-188 — `crop_or_pad`: L191-229 — `preprocess_nii`: L232-241 — `build_cache_manifest`: L244-257 — `CACHE_CONFIG_KEYS`: L262-265 — `validate_cache_manifest`: L268-311 — `MRReportDataset`: L314-664 (`__init__` L330-408, `_load_splits` L410-419, `_load_jsonl` L421-435, `_prepare_samples` L437-454, `_prepare_samples_from_cache` L468-491, `_compute_sample_weights` L493-558, `_load_pathology_labels` L560-586, `get_weighted_sampler` L588-604, `__len__` L606-607, `_load_volume_stack` L622-643, `__getitem__` L645-664) — `collate_fn`: L667-671
- `contrastive-pretraining/scripts/data_inference.py`
  - `MRReportDatasetInfer`: L37-239 (`__init__` L45-104, `_load_splits` L106-115, `_load_jsonl` L117-131, `_load_labels` L133-143, `_prepare_samples` L145-164, `_prepare_samples_from_cache` L166-196, `__len__` L198-199, `_load_volume_stack` L213-227, `__getitem__` L229-239) — `collate_fn_infer`: L242-251
- `contrastive-pretraining/scripts/preprocess_volumes.py`
  - `parse_args`: L54-100 — `_process_subject`: L103-148 — `main`: L151-276 (discovery L182-198, manifest L200-221, work-list L224-232, run L234-267)
- `contrastive-pretraining/scripts/mr_rate_trainer.py`
  - `MrRateTrainer.__init__`: L125-c.390 (dataset L224-239, sampler/DataLoader L249-273, distributed prep L275-295) — `train_step`: L393-513 (batch fetch L406, DDP volume-padding L414-427, tokenize L430-435, forward L440-451, backward/step L467-484) — `train`: L515-522
- `contrastive-pretraining/scripts/run_train.py`
  - argparse: L16-79 — model/encoder construction: L109-186 — `MrRateTrainer(...)` call: L225-250 (note `batch_size=1` hardcoded at L235)
- `contrastive-pretraining/scripts/inference.py`
  - `infer(batch_size=1)`: L243 — `eval_loader`: L272-280 — CLI `--batch_size`: L401 — `results = engine.infer(batch_size=args.batch_size)`: L514
- `contrastive-pretraining/scripts/extract_features.py`
  - Dataset construction: L368-379 — `loader` (hardcoded `batch_size=1`): L403-406
- `contrastive-pretraining/scripts/mil_probe_online.py`
  - `build_dataset`: L231-243 — `make_loader` (hardcoded `batch_size=1`): L300-318
- `contrastive-pretraining/scripts/mil_probe.py`
  - `RaggedTokenDataset`: L161-260ish (`__init__` L164-233, `__len__` L235-236, `__getitem__` L238-243) — `collate_ragged`: L259-268 — loader construction: L485-531
- `contrastive-pretraining/scripts/linear_probe.py`
  - `TensorDataset`/`DataLoader` construction: L142-144
- `contrastive-pretraining/mr_rate/mr_rate/mr_rate.py`
  - `forward`: L403-c.470 (shape unpack L418, `_encode_visual_tokens`/`_encode_visual_instances` dispatch L454-464) — DDP `all_gather_batch` calls before loss: L483-499
- `contrastive-pretraining/tests/test_data.py`
  - `synthetic_dataset` fixture (corroborates layout-2 + JSONL + splits-CSV schema): L132-189
