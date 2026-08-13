# MR challenge reference bundle (pulled from your colleague's CT docker_ft repo)

This is the **code-and-architecture-only** subset of `docker_ft`, the submission repo
your colleague built for the "CT Volume Generation" MICCAI challenge (main-2026 phase,
`report -> CT volume`, ranked on FVD/FID-2.5D/CLIPScore — see
`.forithmus/challenge_config_reference.json` for the exact schema, which is likely a
close preview of how your MR challenge's evaluation will be specified once it opens).

**Nothing in here is a trained weight.** Every `.pt` / `.safetensors` / `.bin` file,
`weights.zip`, the `sample_outputs_*` volumes, and the nine `submission_v10_*.tar.gz`
bundles (~95GB total) were left out on purpose — those are CT-trained and won't help
an MR model. What's included is the scaffolding and model code you can use as a
reference/template:

- `Dockerfile`, `entrypoint.sh`, `.dockerignore`, `make_weights_zip.sh` — the
  Docker submission harness (base image, conda/pip pinned deps, non-root user,
  offline HF env vars, weight-symlinking pattern). Note `make_weights_zip.sh`
  still has her hardcoded HPC paths (`/vol/idea_ramses/ba49mefe/...`) — treat it
  as a pattern, not a script to run as-is.
- `CTFlow/common/*.py` — the core model/dataset code (the spatio-temporal
  flow-matching DiT backbone, LoRA, schedulers, dataset loaders).
- `ct_clip_src/` — the CT-CLIP implementation (CTViT + text-image contrastive
  model), used in her earlier v2/v3 pipeline as a candidate-scoring filter.
  Not wired into the current entrypoint (`_v4_d3_long.py`), kept for reference.
- `inference_ft_vae11k*.py` (10 variants) — her experiment history for the
  inference pipeline: candidate sampling, selection rules, resampling, denoise
  iterations. Heavily commented with what was tried and the observed effect on
  leaderboard metrics — worth reading before you design your own inference
  loop, even though the specifics (frame-count gating, CT-specific denoise
  filter) won't transfer directly.
- `models/*/config.json` (and the two BiomedVLP-CXR-BERT tokenizer/vocab
  files) — architecture configs only, no weight tensors. Useful to see the
  hyperparameters (e.g. `lvfm_STDiT-L2_16f8_2_2_2_bsz16_v2_black_rate_0.3_v2`)
  without pulling the actual checkpoints.
- `.forithmus/` — the challenge platform config and a 5-case mock
  input/output schema example, useful as a template for smoke-testing your
  own container locally before submitting.

## Getting this onto Helma

I can't reach Helma directly from this session (no network path from either
the cloud sandbox or your Mac's local bridge), so move it yourself once you
have this folder. Standard HPC convention — and probably the right split
here since this bundle is all small text/code files — is:

- **home directory**: this whole bundle (it's code, git-trackable, and
  under 1MB total).
- **workspace**: reserved for your own model's weights later, not needed for
  this bundle.

From a terminal that can reach Helma (e.g. on the FAU network/VPN):

```bash
# from wherever you unzip this bundle
rsync -avz ./mr_challenge_template/ <your-user>@<helma-login-host>:~/mr_challenge_template/
```

Replace `<your-user>@<helma-login-host>` with your actual Helma login, and
`~/mr_challenge_template/` with wherever in your home directory you want it.
If your HPC setup uses separate env vars for home vs. large-file storage
(common pattern: `$HOME` vs. something like `$WORK`/`$HPCVAULT`/`$SCRATCH`),
run `echo $HOME` on Helma once to confirm the exact path before syncing.
