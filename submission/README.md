# VLM3D `mr-volume-generation` — Docker submission

The container the challenge platform runs: `/input/prompts.json` in, one synthetic 3D brain MRI
volume per prompt out at `/output/<case_id>.nii.gz`.

```
Dockerfile           thin image (no weights), built from the REPOSITORY ROOT
entrypoint.sh        bridges /weights -> /opt/app/{models,pretrained}, execs predict.py
predict.py           the run loop: prompts -> generate -> /output, with /checkpoint resume
routing.py           picks a (modality, plane) for a prompt that carries none
make_weights_zip.sh  assembles weights_arm<X>.zip for `forithmus submit --weights`
mock_input/          5 synthetic prompts for a local dry run
```

`predict.py` contains **no model code**. It calls `mrrate_r2v.cli.generate_r2v.build_sampler` and
`.conditioning_text_for` unchanged, so the container's loading path is the one `cli.evaluate` was
validated with. Everything genuinely new is the platform I/O contract, the modality/plane routing,
and recovering `findings`/`impression` from one flat prompt string.

---

## Status: what the spec actually says

Checked against the platform's JSON API on **2026-08-13** (`research.forithmus.com/api/...` — the
web UI 403s automated fetches, the API does not).

**VERIFIED — the MR track is still not submittable.** `mr-volume-generation` has one phase,
`main`, with `submissions_enabled: false`, `accepting_submissions: false`, `data_schema: {}`,
`ranking_config: {}`, `score_json_path` still at the platform default `$.metrics.dice`,
`has_eval_image: false`, `baseline_stale: true`. All three **CT** tracks have a working
`main-2026` phase and are accepting submissions. So the container contract below is inferred from
the CT sibling track, not read off the MR phase.

**VERIFIED — it moved today.** The phase's `updated_at` is `2026-08-13T13:12Z` and
`gt_processing_status` flipped `failed` → `complete` (it was `failed` on 2026-08-04). Ground truth
is now uploaded and processed; what is still missing is a scored baseline run, which is what
populates `data_schema` (the platform auto-detects the schema from the host's data plus a baseline
submission's output). 67 participants, 23 submissions, empty leaderboard.

**VERIFIED — the deadline did not move.** Final submission `2026-08-19T22:00Z`, results
`2026-08-21`, MICCAI on-site `2026-09-29`. `registry_purge_after` is set to `2026-08-19T21:08Z`,
consistent with that. **Six days.**

**VERIFIED — nobody has answered the schema questions.** The MR forum has exactly one thread,
"MR-RATE data and Challenge Setup" (2026-07-16), asking which MR-RATE version to use, whether one
model must cover all modalities and how the target modality is specified, how to separate brain
from spine, and what output geometry is expected. **No reply, four weeks later.** Those are
precisely the unknowns this scaffold has to paper over.

### The contract this container implements (INFERRED from `ct-volume-generation` `main-2026`)

The CT track's `data_schema` is fully populated, and the organizers' own
[`VLM3D-Dockers`](https://github.com/forithmus/VLM3D-Dockers) repo documents it:

| | Contract |
|---|---|
| Input | `/input/prompts.json` — JSON array of `{"input_image_name": "<uuid>.mha", "report": "<text>"}`, `entry_count: 100`, case-id field `input_image_name` |
| Output | loose `*.nii.gz` in `/output`, one per prompt, named `<input_image_name-without-extension>.nii.gz`. **Not zipped.** |
| CT dtype/affine | `float32`, `affine = diag(-1, -1, 1, 1)`, `spacing (1,1,1)`, `_shapes_vary: true` |
| Mounts | `/input` ro, `/output` rw, `/weights` ro (from `--weights weights.zip`), `/checkpoint` rw, `/tmp` scratch |
| Runtime | no network egress, non-root `USER` required, amd64 only, image ≤ 15 GB compressed (weights.zip separate, ≤ 100 GB) |
| Shutdown | SIGTERM → 30 s to save `/checkpoint`; `/output` does **not** persist across runs |
| Metrics (CT) | `FVD_CTNet` (primary, asc), `FID_2p5D_{XY,XZ,YZ,Avg}` (asc), `CLIPScore{,_I2I,_mean}` (desc) |

The published MR **task page** says something different from the CT metric set: "feature-based
similarity (FID-like)" plus "blinded classifier consistency", ranked by a "two-sided permutation
test between all team pairs". Neither the CT metric keys nor the MR page's description is confirmed
for the MR phase, whose `ranking_config` is empty. Both point the same way for design purposes:
**distributional, unpaired metrics — no voxelwise fidelity term.**

### Unknowns, and what each one costs us

| Unknown | Our choice | Cost if wrong |
|---|---|---|
| Is modality/plane supplied per case? | Not supplied — `routing.py` assigns them (see below) | Low. `read_prompts` already accepts extra keys; if a `modality` field appears, read it and skip routing. |
| Target geometry: fixed grid or ours? | Per-bucket trained grid, div-32, FOV from NVIDIA's published table (`geometry_for`) | Low-medium. CT's own schema has `_shapes_vary: true` and its FID resamples everything to 512³@1 mm, so variable shapes are tolerated. A hard fixed-grid requirement means one resample step. |
| dtype / affine | `float32`, affine = `diag(spacing)` | Low. One line each in `predict.py` / `mrrate_r2v.sampling.save_volume`. |
| Brain vs spine (the task text says "brain **or** spine") | Brain only | Low. Spine is <1 % of MR-RATE and no label ships; the task page's own title and overview are brain. |
| Is the hidden test set MR-RATE's public test split? | Assumed *not* | Medium. The 2025 CT edition used an external Boston University hold-out. Affects `routing.py`'s marginal (below), not correctness. |
| Prompt count / time budget | 100 prompts assumed, ~7 s/case measured on an H200 | Low. Checkpoint/resume handles a timeout; pick `--time-budget` generously. |

---

## The one real modelling decision: modality and plane

The prompt gives a report and nothing else, but the model needs a modality and a plane — they pick
the geometry bucket, they reach the UNet as `class_labels` and `spacing_tensor`, and for adapters
A/B/C (trained on `findings_impression_meta`) they are literal `[MODALITY]`/`[PLANE]`/`[SPACING]`
tokens in the conditioning text. There is no "unspecified" the model has seen.

An MR-RATE report is **study-level** — it describes every series in the exam — so it is not
evidence for which single series a given prompt stands for. Reading the modality out of the text is
therefore mostly noise. What *is* actionable: FID/FVD compare our **set** against the hidden
**set**, so making the two sets' modality/plane mixtures agree is a real, cheap win that needs no
per-case truth.

`R2V_ROUTING=marginal` (default) does a largest-remainder allocation of the prompt list over the
ten trained buckets, matching MR-RATE's own test-split mixture (counted from
`manifest_shards_native.csv`, 34,453 test series):

```
T1w AXIAL 15.9%   T2w AXIAL 15.9%   T1w SAGITTAL 12.6%   SWI AXIAL 11.5%   FLAIR SAGITTAL 10.8%
FLAIR AXIAL 10.0%  T1w CORONAL 7.1%  T2w CORONAL 6.9%     FLAIR CORONAL 5.1%  T2w SAGITTAL 4.3%
```

The assignment is a function of the prompt set (SHA-256 of the case id decides ordering), not of
iteration order or an RNG, so a rerun or a resume produces the same volumes. `R2V_ROUTING=report`
honours a sequence/plane named unambiguously in the text and fills the rest from the remaining
quota; `R2V_ROUTING=fixed` + `R2V_FIXED_BUCKET=T1w:AXIAL` emits one bucket throughout.

---

## Build, test, submit

**There is no Docker on Helma** — only Apptainer, which cannot produce the `docker save` tarball
the platform ingests. Build on a machine with Docker (workstation or laptop), and on Apple silicon
use `docker buildx build --platform linux/amd64` or the validator rejects the image *after* the
upload.

```bash
# 1. weights.zip, on Helma (arm C shown; A/B/D also valid — see make_weights_zip.sh)
./submission/make_weights_zip.sh C
#    -> /hnvme/workspace/y100dc19-nvidia-mri-brain/submission/weights_armC.zip  (~2.8 GB)

# 2. image, from the REPOSITORY ROOT, on a machine with Docker
docker buildx build --platform linux/amd64 -f submission/Dockerfile -t mrgen-thin:latest .

# 3. local dry run against the mock prompts
docker run --rm --gpus all \
    -v "$PWD/submission/mock_input":/input:ro \
    -v /tmp/mrgen-out:/output \
    -v /path/to/unzipped/weights:/weights:ro \
    -v /tmp/mrgen-ckpt:/checkpoint \
    mrgen-thin:latest

# 4. schema check + submit (the CLI is the authority once the MR phase opens)
pip install --upgrade forithmus && forithmus login
forithmus init mr-volume-generation      # writes .forithmus/ with the real schema
forithmus generate                       # real mock prompts.json for THIS phase
forithmus test mrgen-thin:latest --timeout 1200

docker save mrgen-thin:latest | gzip > submission.tar.gz
forithmus submit submission.tar.gz --tier gpu-a100-80 --time-budget 240 \
    --weights .../weights_armC.zip \
    -d "NV-Generate-MR-Brain + report-conditioned adapter (arm C)"
```

Iterating on code only? `--reuse-weights <previous-submission-id>` instead of `--weights`, and only
from a submission that **scored** — a failed one gets garbage-collected and the pointer breaks at
mount time.

## Knobs

All read from the environment, so a re-run needs no rebuild — set them with `docker run -e` locally
and bake the final values into the `Dockerfile` before submitting (the platform passes no flags).

| Variable | Default | |
|---|---|---|
| `R2V_ADAPTER` | `adapter.pt` | filename in `/weights`; `make_weights_zip.sh` renames the arm's checkpoint to this |
| `R2V_ROUTING` | `marginal` | `marginal` / `report` / `fixed` |
| `R2V_FIXED_BUCKET` | — | e.g. `T1w:AXIAL`, with `R2V_ROUTING=fixed` |
| `R2V_NUM_INFERENCE_STEPS` | `30` | NVIDIA's own default |
| `R2V_REPORT_GUIDANCE_SCALE` | `4.0` | `0` disables the report term |
| `R2V_MODALITY_GUIDANCE_SCALE` | `10.0` | NVIDIA's `cfg_guidance_scale` for mr-brain |
| `R2V_SEED` | `1234` | per-case seed is `sha256(seed:case_id)`, matching `eval.live.stable_seed` |
| `R2V_OUTPUT_DTYPE` | `float32` | values stay NVIDIA's MR `[0, 1000]` range either way |
| `R2V_CHECKPOINT_EVERY` | `5` | cases between `/checkpoint` saves |

`predict.py` writes a per-run manifest (routing mixture, per-case geometry and timing, adapter step,
text-encoder identity, guidance scales) to `/checkpoint/run_manifest.json` — **not** to `/output`,
where a stray file could fail output-schema validation.

## Environment fidelity

Pins in the `Dockerfile` are the exact versions the four adapters were trained and locally evaluated
under: Python 3.12, torch 2.5.1+cu121, monai 1.6.0, transformers 5.14.1, numpy 2.5.1, scipy 1.18.0,
nibabel 5.4.2. Floating any of them turns a leaderboard score into a number we cannot reproduce
locally, which is the point of pinning rather than a style preference.

The in-container tree mirrors the repository — `/opt/app/NV-Generate-CTMR` beside
`/opt/app/contrastive-pretraining/mrrate_r2v` — because `mrrate_r2v/models/nvidia.py` locates the
vendored NVIDIA code as `parents[3]/NV-Generate-CTMR`. Flattening `contrastive-pretraining/` away
makes that resolve to `/NV-Generate-CTMR` and the import fails at container start.

Text-encoder directories are resolved from `MRRATE_PRETRAINED_DIR=/opt/app/pretrained`, never from
the absolute cluster paths recorded in the adapter checkpoint. That is what lets arm D's three
encoders load with no per-arm wiring in `entrypoint.sh`, and it is why `predict.py` passes
`text_checkpoint=None`: a single `--text-checkpoint` is applied to whichever encoder the
configuration names, which is how a CXR-BERT adapter ends up running on same-width RadBERT weights
and generating confident nonsense.

## Before submitting

- [ ] MR phase actually open (`submissions_enabled: true`) — re-check the API; it was false today
- [ ] `forithmus init` + `generate` + `test` against the **real** phase schema, not `mock_input/`
- [ ] arm chosen and its `weights_arm<X>.zip` built and listed (`unzip -l`, no parent prefix)
- [ ] image is amd64 and non-root; `torch.cuda.is_available()` true inside it
- [ ] all N prompts produced a volume (`predict.py` exits non-zero if not)
- [ ] `-d` describes the method — it is the algorithm name on the public leaderboard
