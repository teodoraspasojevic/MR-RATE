# VLM3D `mr-volume-generation` — Docker submission

The container the challenge platform runs: `/input/prompts.json` in, one synthetic 3D brain MRI
volume per prompt out at `/output/<case_id>.nii.gz`.

```
Dockerfile           thin image (no weights), built from the REPOSITORY ROOT
entrypoint.sh        bridges /weights -> /opt/app/{models,pretrained}, execs predict.py
predict.py           the run loop: prompts -> generate -> /output, with /checkpoint resume
make_weights_zip.sh  assembles weights_arm<X>.zip for `forithmus submit --weights`
package_image.sh     docker save -> classic (non-OCI) format via skopeo, verified
mock_input/          8 synthetic prompts for a local dry run
```

`predict.py` contains **no model code**. It calls `mrrate_r2v.cli.generate_r2v.build_sampler` and
`.conditioning_text_for` unchanged, so the container's loading path is the one `cli.evaluate` was
validated with. Everything genuinely new is the platform I/O contract, decoding (modality, plane)
from each case id, and recovering `findings`/`impression` from one flat prompt string.

---

## Status: what the spec actually says

**The MR phase opened and published its contract** — `docs/challange_docs/MRI_Report_to_Volume.md`
is the organizers' own "How to Submit" writeup and is now the authority, superseding the
CT-sibling-track inference this container used to run on. Key confirmed facts:

| | Contract |
|---|---|
| Input | `/input/prompts.json` — JSON array of `{"input_image_name": "<id>", "report": "<text>"}`, one entry per target series |
| Case id | `input_image_name = {study_uid}_{modality}-raw-{plane}[-N]`, e.g. `WNPYIQCPIN_flair-raw-sag`, `WNPYIQCPIN_t1w-raw-sag-2` — modality ∈ `{t1w, t2w, flair, swi}`, plane ∈ `{axi, sag, cor, obl}`, optional `-N` for a repeated acquisition |
| Test set shape | 690 entries across 100 studies (not every study has every modality; study composition varies) |
| Output | one `/output/{input_image_name}.nii.gz` per prompt, filename **must** match exactly (suffix included) — the evaluator pairs by filename and parses modality from it |
| Metrics | per-case MSE, PSNR, SSIM (per modality) plus a 2.5D FID over squeezenet features — a **paired**, per-case comparison, not only a distributional/unpaired one |
| Mounts | `/input` ro, `/output` rw, `/weights` ro (from `--weights weights.zip`), `/checkpoint` rw |
| Runtime | no network egress at runtime; write restartable progress to `/checkpoint` and handle SIGTERM (spot preemption / time-budget timeout resume instead of restarting) |

Because scoring is now confirmed **paired and per-case**, getting one case's (modality, plane)
wrong tanks that case's own MSE/PSNR/SSIM, not just the population-level mixture — which is why
`modality_plane_for` (below) reads the ground truth directly out of the id rather than guessing.

### Remaining unknowns, and what each one costs us

| Unknown | Our choice | Cost if wrong |
|---|---|---|
| Target geometry: fixed grid or ours? | Per-bucket trained grid, div-32, FOV from NVIDIA's published table (`geometry_for`) | Low. The doc doesn't mandate a fixed output grid; `save_volume`'s affine carries the true spacing either way. |
| dtype / affine | `float32`, affine = `diag(spacing)` | Low. One line each in `predict.py` / `mrrate_r2v.sampling.save_volume`. |
| Brain vs spine (the task text says "brain **or** spine") | Brain only | Low. Spine is <1 % of MR-RATE and no label ships; every published `input_image_name` example and the vocabulary breakdown are brain-track modalities. |
| Prompt count / time budget | 690 prompts confirmed, ~7 s/case measured on an H200 | Low. Checkpoint/resume handles a timeout; pick a generous time budget. |

---

## The one real modelling decision: modality and plane

The model needs a modality and a plane for every case — they pick the geometry bucket, they reach
the UNet as `class_labels` and `spacing_tensor`, and for adapters A/B/C (trained on
`findings_impression_meta`) they are literal `[MODALITY]`/`[PLANE]`/`[SPACING]` tokens in the
conditioning text. There is no "unspecified" the model has seen.

`docs/challange_docs/MRI_Report_to_Volume.md` confirms `input_image_name` (our case id, and the
name our output must match exactly) encodes it directly: `{study_uid}_{modality}-raw-{plane}`,
optionally with a `-{n}` duplicate-series suffix. `predict.modality_plane_for` decodes it, so every
case gets the modality/plane its own reference volume actually has — no report-based or
population-level guessing. An MR-RATE report is study-level (it describes every series in the
exam), so it was never good evidence for a single series, and the id makes it unnecessary anyway.
A case id that doesn't carry the pattern raises `ValueError` and stops the run — per the challenge
doc, a missing output already scores as invalid, so failing loudly on a contract violation costs
nothing extra.

`obl` (OBLIQUE) is in the doc's own name vocabulary (7 of 690 entries) and is handled, not rejected
— `PLANE_CODES` maps it, `GeometryPolicy.resolve` already has a documented fallback FOV for it (it
is excluded from NVIDIA's published per-bucket table), and `mrrate_r2v/data/README.md` lists
`OBLIQUE` as one of four real `acquisition_plane` values. A case id can also name a modality/plane
the adapter never trained on (e.g. `SWI` outside `AXIAL`, excluded from the ten trained buckets —
see `mrrate_r2v/README.md`); `modality_plane_for` still returns it rather than coercing it into a
trained bucket, since `ModalityEncoder.id_for` falls back to the unconditional class for an
unmapped modality and `GeometryPolicy.resolve` falls back to a default FOV, so generation degrades
gracefully instead of silently mislabelling the case.

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
docker buildx build --platform linux/amd64 -f submission/Dockerfile -t mr-r2v-vlm3d-challenge:latest .

# 3. local dry run against the mock prompts
#    /output and /checkpoint must be writable by the image's non-root UID (1000): a bind mount
#    replaces whatever the Dockerfile chowned at build time with the HOST directory's own
#    permissions, so a freshly `mkdir -p`'d host dir owned by you (not uid 1000) makes predict.py
#    fail with `PermissionError: [Errno 13] ... '/checkpoint/outputs'` before it reads a prompt.
mkdir -p /tmp/mrgen-out /tmp/mrgen-ckpt && chmod 777 /tmp/mrgen-out /tmp/mrgen-ckpt
docker run --rm --gpus all \
    -v "$PWD/submission/mock_input":/input:ro \
    -v /tmp/mrgen-out:/output \
    -v /path/to/unzipped/weights:/weights:ro \
    -v /tmp/mrgen-ckpt:/checkpoint \
    mr-r2v-vlm3d-challenge:latest

# 4. schema check + submit (the CLI is the authority once the MR phase opens)
pip install --upgrade forithmus && forithmus login
forithmus init mr-volume-generation      # writes .forithmus/ with the real schema
forithmus generate                       # real mock prompts.json for THIS phase
forithmus test mr-r2v-vlm3d-challenge:latest --timeout 1200

# `docker save` on this host (containerd-backed buildkit, Docker 29.x) produces an OCI-format
# tarball (index.json + oci-layout + blobs/sha256/<hash>, no extension) -- confirmed empirically
# 2026-08-14. Forithmus's validator only understands the classic format (manifest.json only,
# Config as <hash>.json, layers as <hash>.tar); an OCI tarball fails with
# "Container validation failed: Image config blobs/sha256/<hash>... not found in tarball" -- see
# Submission_Lessons.md Issue 1. package_image.sh converts via skopeo and verifies the result.
sudo apt-get install -y skopeo   # once per machine
./submission/package_image.sh mr-r2v-vlm3d-challenge:latest submission.tar.gz

forithmus submit submission.tar.gz --tier gpu-a100-80 --time-budget 240 \
    --weights .../weights_armC.zip \
    -d "NV-Generate-MR-Brain + report-conditioned adapter (arm C)"
```

Iterating on code only? `--reuse-weights <previous-submission-id>` instead of `--weights`, and only
from a submission that **scored** — a failed one gets garbage-collected and the pointer breaks at
mount time.

## Knobs

All read from the environment. **As of the image built from this Dockerfile, every one of these is
baked in as an `ENV` default** (see the block right after `PYTHONPATH`/`R2V_MODELS_DIR`), matching
what the real platform will run — it passes no runtime flags or env vars at all, so `docker run` for
the actual submission needs nothing but `-v .../weights.zip-contents:/weights:ro` (plus the
platform's own fixed `/input`, `/output`, `/checkpoint` mounts). `docker run -e VAR=value` still
overrides a baked default for local experiments (Docker's normal `ENV`-vs-`-e` precedence), but
changing what the **submitted** image does means editing the value below and rebuilding — an `-e`
flag typed locally never reaches the platform.

| Variable | Baked-in value | |
|---|---|---|
| `R2V_ADAPTER` | `adapter.pt` | filename in `/weights`; `make_weights_zip.sh` renames the arm's checkpoint to this |
| `R2V_NUM_INFERENCE_STEPS` | `30` | NVIDIA's own default |
| `R2V_REPORT_GUIDANCE_SCALE` | `3` | `0` disables the report term; predict.py's own fallback (unset) is `4.0` |
| `R2V_MODALITY_GUIDANCE_SCALE` | `10.0` | NVIDIA's `cfg_guidance_scale` for mr-brain |
| `R2V_GEOMETRY_MODE` | `policy` | `policy` = per-modality/plane `GeometryPolicy` (as trained); `baseline` = the platform's own reference container's fixed 64^3-latent grid at (1.5, 1.9, 1.9)mm, for A/B testing |
| `R2V_OUTPUT_DTYPE` | `float32` | values stay NVIDIA's MR `[0, 1000]` range either way |

Sampling is intentionally unseeded (no `R2V_SEED` knob): the platform's own reference baseline
(`mrgen_example_docker/inference.py`) draws unseeded noise every run too, and a CT-track sibling
submission's own seed sweep found no seed reliably beats another — between-seed score spread was
indistinguishable from within-seed-repeat spread.

`predict.py` backs up each finished volume to `/checkpoint/outputs` and its filename to
`/checkpoint/done.json` (rank-scoped under multi-GPU) as it goes, so a restart resumes instead of
regenerating everything.

## Environment fidelity

Pins in the `Dockerfile` are the exact versions the four adapters were trained and locally evaluated
under: Python 3.12, torch 2.5.1+cu121, monai 1.5.2, transformers 5.14.1, numpy 2.5.1, scipy 1.18.0,
nibabel 5.4.2. Floating any of them turns a leaderboard score into a number we cannot reproduce
locally, which is the point of pinning rather than a style preference.

Text-encoder directories are resolved from `MRRATE_PRETRAINED_DIR=/opt/app/pretrained`, never from
the absolute cluster paths recorded in the adapter checkpoint. That is what lets arm D's three
encoders load with no per-arm wiring in `entrypoint.sh`, and it is why `predict.py` passes
`text_checkpoint=None`: a single `--text-checkpoint` is applied to whichever encoder the
configuration names, which is how a CXR-BERT adapter ends up running on same-width RadBERT weights
and generating confident nonsense.

## Multi-GPU (DDP)

`entrypoint.sh` detects the visible GPU count and launches `torchrun --standalone --nproc_per_node=N
/opt/app/predict.py` whenever `N > 1`; a single GPU (every compute tier `Docker_Submission.md`
currently lists) runs exactly the old way, plain `python predict.py`, byte-for-byte unchanged.
Under `torchrun`, `predict.py`'s `ddp_setup()` gives each rank its own full model copy on its own
`cuda:{local_rank}` device and a disjoint, striped slice of the prompt list (`rank::world_size`);
the checkpoint index becomes rank-scoped (`done_rank0.json`, ...); only rank 0 does the
final aggregate "did every prompt produce a volume" check, after a `dist.barrier()`. Verified
2026-08-14 on 2 GPUs: correct work split, and outputs **bit-identical** to a single-GPU run of the
same prompts (`sha256sum` match on every case) — the per-case seed derivation doesn't depend on
which rank generated it.

**Known limitation, accepted rather than fixed**: `torchrun`'s own elastic agent treats *receiving*
SIGTERM as immediately fatal at its own top level, independent of whether the workers underneath
already exited cleanly — reproduced twice: once where the run had already fully succeeded moments
before the signal arrived, once mid-generation with 0 cases done. In both cases `predict.py`'s own
SIGTERM handler ran correctly in every worker (checkpoint saved, confirmed on disk) but the
container's **overall exit code was 1**, not 0. Since the platform's timeout/spot-preemption flow
(`Checkpoints_and_Continuation.md`) depends on a clean exit(0) after SIGTERM to mark a run
`timed_out` (resumable) rather than crashed, **checkpoint/resume and spot instances do not work
correctly under the multi-GPU path** — only under the single-GPU path (unaffected, re-verified the
same day).

This is a real gap, not a hypothetical one: the CT track's own submission (`mr_challenge_template/
entrypoint.sh`) explicitly submits against a `gpu-4xa100` tier, which isn't in our current
`Docker_Submission.md` tier snapshot — so a multi-GPU tier evidently can exist. That same reference
code, though, doesn't attempt SIGTERM handling or checkpointing anywhere at all (no `signal.signal`,
no `/checkpoint` read or write) — an interrupted run there just dies with whatever was already
written to `/output`, no resume, full restart on retry. **We're deliberately matching that
trade-off for the multi-GPU path** rather than building a custom launcher to fix it: if a multi-GPU
tier is used, request a generous enough time budget that hitting the wall is rare, and accept that
an interrupted multi-GPU run does not cleanly resume (a retry restarts from prompt 1). The
single-GPU path keeps full, correct checkpoint/resume regardless — use it whenever resume actually
matters.

## Before submitting

- [ ] `forithmus init mr-volume-generation` + `generate` + `test` against the **real** phase schema,
      not `mock_input/` — `mock_input/` is for a quick local smoke test only
- [ ] every `input_image_name` in the real prompts still matches `modality_plane_for`'s pattern
      (`{study_uid}_{modality}-raw-{plane}[-N]`); an id it can't parse raises and stops the run
- [ ] arm chosen and its `weights_arm<X>.zip` built and listed (`unzip -l`, no parent prefix)
- [ ] image is amd64 and non-root; `torch.cuda.is_available()` true inside it (entrypoint.sh now
      checks this itself and fails loud before touching a checkpoint if it's not)
- [ ] `submission.tar.gz` built via `package_image.sh` (classic Docker format, not raw `docker save`
      — this host's `docker save` emits OCI tarballs the validator rejects, see Submission_Lessons.md
      Issue 1)
- [ ] if the selected tier has >1 GPU: SIGTERM/checkpoint does not work under `torchrun` (see
      "Multi-GPU (DDP)" above, accepted trade-off) — pick a generous time budget so a timeout is
      unlikely, since an interrupted run there restarts from scratch rather than resuming
- [ ] all N prompts produced a volume (`predict.py` exits non-zero if not)
- [ ] `-d` describes the method — it is the algorithm name on the public leaderboard
