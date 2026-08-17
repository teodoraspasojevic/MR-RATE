# Forithmus Docker Submission: Lessons Learned

This document summarizes the issues encountered and solutions found while submitting the Phase 7 TTA + Specialists model to the Forithmus CT Abnormality Classification challenge.

## Overview

- **Model**: Phase 7 TTA + Specialists ensemble (CRG 0.5533, AUC 0.8706)
- **Platform**: Forithmus Research Hub (https://research.forithmus.com)
- **Challenge**: CT Abnormality Classification
- **Submission format**: Docker tarball + weights.zip

---

## Issue 1: Docker Image Format (OCI vs Classic)

### Symptom
```
Container validation failed: Image config blobs/sha256/71b3fe92... not found in tarball
```

### Cause
Modern Docker (with containerd backend, used by Docker Desktop) produces **OCI format** tarballs by default. Forithmus's validator expects **classic Docker format**.

**OCI format indicators:**
- Contains `index.json`
- Contains `oci-layout`
- Config path: `blobs/sha256/<hash>` (no extension)
- Layers in `blobs/sha256/`

**Classic Docker format indicators:**
- Only `manifest.json` (no `index.json`)
- No `oci-layout`
- Config path: `<hash>.json` (with `.json` extension)
- Layers as `<hash>.tar` files

### Why local testing passed
Docker itself handles both formats seamlessly. The issue only appears on Forithmus because they use a custom validator that only understands classic format.

### Solution
Use `skopeo` to convert OCI to classic Docker format:

```bash
# Install skopeo
sudo apt-get install -y skopeo

# Convert from Docker daemon to classic Docker archive
skopeo copy docker-daemon:phase7-tta:latest docker-archive:submission_docker.tar

# Compress
gzip submission_docker.tar

# Verify format (should NOT show index.json or oci-layout)
tar -tf submission_docker.tar.gz | grep -E "(index.json|oci-layout)"
```

### Verification
Check manifest.json format:
```bash
tar -xzf submission_docker.tar.gz manifest.json -O
```
- **OCI**: `"Config":"blobs/sha256/71b3fe92..."`
- **Classic**: `"Config":"71b3fe92....json"`

---

## Issue 2: Incomplete HuggingFace Cache

### Symptom
```
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files 
in the disk cache and outgoing traffic has been disabled.
```

### Cause
The HuggingFace cache for COLIPRI only contained `model.safetensors` but was missing:
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- Other metadata files

This happened because when COLIPRI was originally used on the cluster, the `colipri` package loaded config from its bundled defaults instead of caching from HuggingFace.

### Why it matters
Forithmus runs containers **offline** (no internet access). All model files must be pre-cached in `weights.zip`.

### Solution
Force complete download by running the model loading code:

```bash
export HF_HOME=/path/to/checkpoints/colipri_cache
python3 -c "
from colipri import get_model, get_processor
model = get_model()
processor = get_processor()
print('Complete cache downloaded')
"

# Verify all files exist
find /path/to/checkpoints/colipri_cache -name "*.json"
```

### Required cache structure
```
colipri_cache/hub/models--microsoft--colipri/
├── blobs/
│   └── <hash>           # model.safetensors content
├── refs/
│   └── main             # branch reference
└── snapshots/<commit>/
    ├── config.json      # ← REQUIRED
    ├── model.safetensors
    ├── tokenizer.json   # ← REQUIRED
    └── ...
```

---

## Issue 3: GitHub Actions Workflow Permissions

### Symptom
```
[remote rejected] master -> master (refusing to allow a Personal Access Token 
to create or update workflow without `workflow` scope)
```

### Cause
GitHub PAT needs `workflow` scope to push `.github/workflows/*.yml` files.

### Solution
Update token at https://github.com/settings/tokens and check the `workflow` scope.

---

## Issue 4: Disk Space on GitHub Actions

### Symptom
```
gzip: stdout: No space left on device
```

### Cause
Docker image (~8GB) + gzipped export (~5GB) exceeds runner's free disk space.

### Solution
Free disk space before building:

```yaml
- name: Free disk space
  run: |
    sudo rm -rf /usr/share/dotnet
    sudo rm -rf /opt/ghc
    sudo rm -rf /usr/local/share/boost
    sudo rm -rf /usr/share/swift
    sudo rm -rf /usr/local/lib/android
    sudo rm -rf /opt/hostedtoolcache/CodeQL
    sudo rm -rf /usr/local/share/powershell
    sudo rm -rf /usr/share/az_*
    sudo docker system prune -af
```

---

## Issue 5: WSL/Windows Path Issues with forithmus test

### Symptom
```
".forithmus/test_data/input" includes invalid characters for a local volume name
```

### Cause
`forithmus test` doesn't handle paths correctly on Windows or WSL with Windows mounts.

### Solution
1. Use native Linux paths (not `/mnt/c/...`)
2. Or test manually:

```bash
docker run --rm --gpus all \
  -v /absolute/path/input:/input:ro \
  -v /absolute/path/output:/output \
  -v /absolute/path/weights:/weights:ro \
  image-name:latest
```

---

## Submission Checklist

### Before Building
- [ ] Serialize all model checkpoints (heads, calibrators, specialists)
- [ ] Download complete HuggingFace caches (verify `.json` files exist)
- [ ] Copy FORA checkpoint and VJEPA2 cache
- [ ] Package `weights.zip` with all required files

### Building Docker Image
- [ ] Use GitHub Actions or machine with Docker
- [ ] Free disk space if using GitHub Actions
- [ ] Convert to classic Docker format using `skopeo`

### Before Submitting
- [ ] Verify tarball loads: `docker load -i submission.tar.gz`
- [ ] Check manifest format: `tar -xzf submission.tar.gz manifest.json -O`
- [ ] Test container locally with mounted weights
- [ ] Verify no OCI markers: `tar -tf ... | grep -E "(index.json|oci-layout)"`

### Submission Command
```bash
forithmus submit submission_docker.tar.gz \
    --weights weights.zip \
    --tier gpu-l4 \
    --time-budget 240 \
    -d "Phase7 TTA+specialists ensemble"
```

---

## GPU Tier Selection

| Tier | VRAM | Notes |
|------|------|-------|
| `cpu-4` | — | CPU only |
| `gpu-t4` | 16GB | Budget option |
| `gpu-l4` | 24GB | Sufficient for this model (~12GB VRAM) |
| `gpu-v100` | 16/32GB | Fallback if L4 capacity unavailable |
| `gpu-a100` | 40/80GB | Overkill for this model |

Our model needs ~12GB VRAM peak, so `gpu-l4` is sufficient. If you get "GPU capacity error in us-central1", try `gpu-v100` instead.

---

## Key Files

| File | Purpose |
|------|---------|
| `submission/Dockerfile` | Container definition |
| `submission/inference/predict.py` | Main entry point |
| `submission/inference/models.py` | COLIPRI/FORA loading |
| `submission/inference/pipeline.py` | Phase 7 ensemble logic |
| `submission/scripts/serialize_pipeline.py` | Train and save checkpoints |
| `submission/scripts/package_weights.sh` | Package weights.zip |
| `.github/workflows/build-docker.yml` | GitHub Actions build |

---

## Issue 6: COLIPRI Meta Tensor Initialization

### Symptom
```
NotImplementedError: Cannot copy out of meta tensor; no data! 
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
```

With warnings during loading:
```
UserWarning: copying from a non-meta parameter in the checkpoint to a meta 
parameter in the current model, which is a no-op.
```

### Cause
The `colipri` package uses HuggingFace's lazy/meta tensor initialization for memory efficiency. In offline mode, the model is created with meta tensors (placeholders with no data), and `load_state_dict()` fails to properly materialize them because it needs `assign=True`.

When you call `.to(device)` on a model with meta tensors, it fails because meta tensors have no data to copy.

### Solution
In `inference/models.py`, detect meta tensors after loading and manually materialize:

```python
# Check if model has meta tensors
has_meta = any(p.device.type == "meta" for p in model.parameters())
if has_meta:
    model = model.to_empty(device="cpu")
    
    # Reload from safetensors with assign=True
    from safetensors.torch import load_file
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=False, assign=True)
```

### Note
The microsoft/colipri HuggingFace repo only contains `model.safetensors` — no `config.json`. This is normal; the `colipri` package bundles its own configuration.

---

## Issue 7: bfloat16 → numpy Causes 100% NaN (v6 failure)

### Symptom
Every single volume fails with:
```
ERROR: Input contains NaN
```
The error comes from the isotonic calibrator's `.predict()` method, which raises when inputs contain NaN.

### Cause
`torch.cuda.amp.autocast()` produces **bfloat16** tensors for efficiency. numpy doesn't natively support bfloat16, so calling `.cpu().numpy()` directly on a bfloat16 tensor produces NaN or garbage values.

This affected FORA embedding extraction:
```python
# Bad - produces NaN
with torch.cuda.amp.autocast():
    emb = model(x).cpu().numpy()
```

### Solution
Always convert to float32 before numpy conversion:
```python
# Good - explicit float32 conversion
with torch.cuda.amp.autocast():
    emb = model(x).float().cpu().numpy()
```

### Files affected
- `inference/predict.py` — `extract_colipri_embeddings()` and `extract_fora_embeddings()`
- `inference/pipeline.py` — `_run_heads()` and `_run_specialist()`

### Why local testing might pass
If testing on CPU or without autocast, tensors are already float32.

---

## Issue 8: V100 lacks native bfloat16 support (v7 failure)

### Symptom
Same as Issue 7 — 100% NaN errors — but with the `.float()` fix applied. The job completes but all predictions are fallback 0.5, giving AUC ~0.5.

### Cause
V100 GPUs don't have native bfloat16 support. The VJEPA2 model was loaded with `torch_dtype=torch.bfloat16`:
```python
# vjepa_encoder.py
self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
```

When V100 tries to do matrix operations on bfloat16 tensors, it produces NaN values. The `.float()` fix at the end doesn't help because NaN is already in the tensor.

### Solution
Change from bfloat16 to float16 (which V100 supports natively via tensor cores):

```python
# vjepa_encoder.py - use float16 for V100 compatibility
self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
```

Also update preprocessing to match:
```python
# fora_preprocessing.py
tensor = tensor.to(torch.float16)  # was torch.bfloat16
```

### Why it worked on cluster A100s
A100 GPUs have native bfloat16 support, so the same code works fine on FAU's A100 nodes.

### GPU dtype support matrix
| GPU | float16 | bfloat16 |
|-----|---------|----------|
| V100 | ✅ Native | ❌ Emulated/broken |
| A100 | ✅ Native | ✅ Native |
| L4 | ✅ Native | ✅ Native |
| T4 | ✅ Native | ❌ Emulated/broken |

---

## Issue 9: FORA Checkpoint Saved in bfloat16 (v9 failure)

### Symptom
Same 100% NaN errors as v7/v8, even after applying the float16 fix to `vjepa_encoder.py`.

### Cause
The FORA checkpoint (`RadRate.24500.pt`) was trained on A100 GPUs and saved with **843 bfloat16 tensors** for the VJEPA2 backbone weights. When the checkpoint is loaded, these bfloat16 weights overwrite the freshly-loaded float16 model weights.

The float16 fix in `vjepa_encoder.py` only affects the initial model loading from HuggingFace — the subsequent `model.load(checkpoint_path)` call replaces those weights with bfloat16 from the checkpoint.

### Diagnosis
```python
import torch
ckpt = torch.load('RadRate.24500.pt', map_location='cpu')
bf16_count = sum(1 for v in ckpt.values() if hasattr(v, 'dtype') and v.dtype == torch.bfloat16)
print(f'bfloat16 tensors: {bf16_count}')  # 843
```

### Solution
Convert checkpoint tensors from bfloat16 to float16 before packaging:

```python
import torch

ckpt = torch.load('RadRate.24500.pt', map_location='cpu')
for k, v in ckpt.items():
    if hasattr(v, 'dtype') and v.dtype == torch.bfloat16:
        ckpt[k] = v.to(torch.float16)
torch.save(ckpt, 'RadRate.24500.pt')
```

Then repackage `weights.zip`:
```bash
cd submission/checkpoints
zip -r0 ../weights.zip heads/ calibrators/ specialists/ fora/ colipri_cache/ metadata.json
```

### Important
The Docker image itself doesn't need to be rebuilt — only `weights.zip` needs to be updated with the converted checkpoint.

---

## Issue 10: Error Handling Must "Fail Loud"

### Symptom
All volumes return 0.5 probability (fallback), resulting in AUC ~0.5.

### Cause
Our inference code wrapped the main loop in `try/except` and output 0.5 fallback predictions when errors occurred. When every volume hit a NaN error, every volume got 0.5.

### Forithmus Requirement
From the submission guidelines:
> "Fail loud: do not wrap the inference loop in try/except — an OOM should crash the run, not silently skip volumes."

### Solution
Remove the try/except wrapper around the inference loop. Let errors crash the run so you immediately see what went wrong instead of getting silent 0.5 predictions.

```python
# BAD - silently continues with fallback
try:
    probs = pipeline.predict(...)
    predictions.append({"probabilities": probs})
except Exception as e:
    predictions.append({"probabilities": {label: 0.5 for label in LABELS}})

# GOOD - crashes on error so you see the problem
probs = pipeline.predict(...)
predictions.append({"probabilities": probs})
```

---

## Checkpointing Behavior

Forithmus supports checkpointing via the `/checkpoint/` mount:
- Write progress to `/checkpoint/progress.json` periodically
- On job restart, read checkpoint and resume from last saved position
- The platform persists `/checkpoint/` between job restarts within the same submission

Example checkpoint format:
```json
{"processed": 150, "predictions": [...]}
```

If the job times out or crashes, you can continue from where it left off by submitting again (the platform handles this automatically for the same submission).

---

## GPU Tier Pricing (as of 2026)

| Tier | VRAM | Host RAM | Price | Notes |
|------|------|----------|-------|-------|
| `gpu-t4` | 16 GB | 16 GB | $0.65/hr | Budget, float16 only |
| `gpu-l4` | 24 GB | 16 GB | $0.85/hr | Good balance |
| `gpu-l4-xl` | 24 GB | 32 GB | $1.30/hr | Extra host RAM |
| `gpu-v100` | 32 GB | varies | $3.43/hr | float16 only (no bfloat16!) |
| `gpu-a100-40` | 40 GB | varies | $4.41/hr | Full dtype support |
| `gpu-a100-80` | 80 GB | varies | $6.03/hr | Full dtype support |

**Cost planning**: You're pre-charged for the time budget, then refunded unused time. With limited budget, use shorter time budgets and rely on checkpointing.

---

## weights.zip Structure

Files must be at the **root** of the zip (no parent folder):

```
# CORRECT - files at root
weights.zip
├── heads/
├── calibrators/
├── specialists/
├── fora/
└── colipri_cache/

# WRONG - nested in parent folder
weights.zip
└── checkpoints/
    ├── heads/
    └── ...
```

Create correctly:
```bash
cd submission/checkpoints
zip -r0 ../weights.zip heads/ calibrators/ specialists/ fora/ colipri_cache/ metadata.json
```

At runtime, contents are mounted at `/weights/`.

---

## Issue 11: COLIPRI Meta Tensor Loading in Docker (v10-v11 failure)

### Symptom
NaN errors persist even after fixing bfloat16 issues. The error occurs at the first volume:
```
ValueError: Input contains NaN.
```
The NaN comes from COLIPRI embeddings, not FORA.

### Cause
The `colipri` package uses `accelerate.init_empty_weights()` internally:

```python
# Inside colipri.get_model():
with init_empty_weights():
    model = instantiate(config)
model = load_checkpoint_and_dispatch(model, checkpoint_path)
```

This creates **meta tensors** (placeholder tensors with no data) that should be filled by `load_checkpoint_and_dispatch()`. However, in the offline Docker environment, this dispatch fails silently — the model appears to load successfully but parameters contain uninitialized garbage or NaN values.

### Why it worked on the cluster
On the cluster, the `colipri` package loads without meta tensors (possibly due to different accelerate version or environment settings). The model loads normally with real tensors.

### Diagnosis
In Docker logs, you see:
```
Warning: Model has meta tensors, materializing...
Loaded vision weights from model.safetensors
Loaded text weights from model.safetensors
All weights materialized successfully
```

But despite "successfully" materializing, the model still produces NaN embeddings.

### Solution
Bypass `colipri.get_model()` entirely. Instead, instantiate the model **without** `init_empty_weights()` and load weights directly with `safetensors.load_model()`:

```python
from colipri import get_processor
from colipri.checkpoint import download_weights, load_model_config
from hydra.utils import instantiate
from safetensors.torch import load_model as sf_load_model

# Get checkpoint path
checkpoint_path = download_weights()

# Load config
config = load_model_config()

# Create model NORMALLY (not with init_empty_weights)
model = instantiate(config)

# Load weights directly with safetensors
sf_load_model(model, str(checkpoint_path))

processor = get_processor()
```

This creates real tensors on CPU, then loads weights directly — no meta tensors involved.

### Key Insight
When debugging model loading issues in Docker:
1. "Model loaded successfully" doesn't mean weights are correct
2. Meta tensors can appear to load but contain garbage
3. Always verify with `torch.isnan(param).any()` checks
4. Consider bypassing high-level loading APIs that use accelerate

---

## Issue 12: Raw Probabilities + Fixed-0.5 Scoring = F1 Collapse (v-phase7 test failure)

### Symptom
Validation looked great but the leaderboard test scores were far worse, with F1 collapsing while AUC barely moved:

| Metric | Val (ours) | Test (leaderboard) |
|--------|-----------|--------------------|
| CRG    | 0.5533    | 0.4291             |
| AUC    | 0.8706    | 0.8322             |
| **F1** | **0.6136**| **0.2969**         |
| Acc    | ~0.90     | 0.8303             |

**Signature: F1 collapses, AUC stays ~stable, accuracy stays high.** That specific combination is the fingerprint of an operating-point / threshold problem — NOT a model-quality problem. AUC is threshold-independent (rank-only), so it barely changes; F1 depends entirely on where you threshold.

### Cause
Two things compounded:
1. **We submitted raw calibrated probabilities and let the challenge threshold them at a fixed 0.5.** Our F1-optimal per-class thresholds are 0.21–0.52 (17 of 18 classes below 0.5). The challenge silently raised every operating point to 0.5, killing recall on rare classes.
2. **Domain shift** pushed test scores lower (probabilities shift toward 0), which under a fixed 0.5 cutoff makes the recall collapse even worse.

Proven on the val scores (`analysis/phase7/scores_tta_plus_specialists.csv`):
- tuned per-class thresholds → F1 **0.6136** (what we reported)
- raw probs @ fixed 0.5 (what the challenge actually did) → F1 **0.4929**
- simulate a monotone downward shift `p**gamma`, gamma=2 → raw@0.5 F1 **0.306** ≈ observed test **0.2969** (mechanism confirmed)

### Solution: bake the operating point into the probabilities you submit
Never rely on the challenge's 0.5 cutoff. Remap probabilities before writing them so the intended decision boundary lands exactly at 0.5. Two AUC-preserving (monotone) options, in `inference/threshold_remap.py`:

- **`bake_to_half(p, t)`** — piecewise-linear remap that sends threshold `t → 0.5`, `0 → 0`, `1 → 1`. Recovers full tuned-threshold F1 on val (0.6136), but the fixed thresholds are still fragile under domain shift (drops to ~0.46 at gamma=2).
- **`prevalence_matched_thresholds` / mode `'prevalence'`** — rank-based: set each class threshold to the `(1 - prior)` quantile of the *current batch's* scores, so the predicted positive rate matches the training prevalence. Because it's rank-based, it is **invariant to any monotone shift** — F1 stays flat at **0.5969** across all simulated shifts. Slightly below f1-optimal on clean val, but immune to the exact failure that killed the test score.

We ship `OPERATING_POINT_MODE=prevalence` by default (`predict.py`), with priors in `weights/operating_point/priors.json` and F1 thresholds in `thresholds_f1.json`. The remap is applied once over the full accumulated batch right before writing `results.json`. Fail-safe: if the artifact is missing it warns and submits raw rather than crashing.

### Key Insight
- If F1 tanks on test but AUC holds and accuracy stays high, suspect the **operating point**, not the model.
- When the scorer thresholds at a fixed 0.5 on imbalanced classes, **submitting raw probabilities is a bug** — bake your thresholds in.
- Prefer **rank/prevalence-based** thresholds over fixed value thresholds: they survive domain shift because they only depend on score ordering, not absolute values.

---

## Useful Commands

```bash
# Check tarball format
tar -tzf submission.tar.gz | head -20
tar -xzf submission.tar.gz manifest.json -O

# Convert OCI to Docker format
skopeo copy docker-daemon:image:tag docker-archive:output.tar

# Test container locally
docker run --rm --gpus all \
  -v ~/test/input:/input:ro \
  -v ~/test/output:/output \
  -v ~/test/weights:/weights:ro \
  image:tag

# Monitor Forithmus submission
forithmus status
```

---

## Issue 13: transformers / torch version incompatibility (DTensor import crash)

### Symptom
Container crashes immediately on COLIPRI load with:
```
ImportError: cannot import name 'DTensor' from 'torch.distributed.tensor'
```
Full traceback from `transformers/distributed/sharding_utils.py`.

### Cause
`transformers>=4.51.0` imports `torch.distributed.tensor.DTensor`, which only exists in PyTorch ≥ 2.5.
Our Docker image pins **torch==2.4.1** (cu121), so the import fails.

Because `requirements.txt` specified `transformers>=4.40.0` without an upper bound, a newer transformers was installed at Docker build time.

### Solution
Pin transformers below 4.51 in `requirements.txt`:
```
transformers>=4.40.0,<4.51.0
accelerate>=0.28.0,<1.4.0
```

Rebuild the Docker image — no other changes needed.

### Note
This is **our bug**, not the platform's. Always pin a compatible upper bound for packages when using a fixed torch version.

### GPU dtype support matrix
| torch version | min transformers that imports DTensor |
|---------------|---------------------------------------|
| 2.4.x         | ≥ 4.51 breaks                        |
| 2.5.x         | ≥ 4.51 works                         |

---

## Issue 14: ModuleNotFoundError: No module named 'numpy'

### Symptom
Container fails immediately on startup or first inference import with:
```
ModuleNotFoundError: No module named 'numpy'
```

### Cause
Python dependencies were installed into a user-local site-packages path for one
user, but the container runtime executed as a different user. As a result,
runtime Python could not see `numpy` even though build logs showed it was
installed.

Typical trigger pattern:
- build-time installs use `pip --user` (or equivalent user-local target)
- runtime switches to a non-root user
- import path no longer includes the build user's local package directory

### Why this is easy to miss
- Build logs show `Successfully installed numpy ...`
- Local tests can pass if run as the same user that installed packages
- Submission runtime may execute under a different user context

### Solution
Install inference dependencies system-wide in the Docker image, and avoid
`pip --user` for runtime-critical packages.

In our submission Dockerfile, this is why we explicitly keep:
- system-wide `pip install -r requirements.txt`
- system-wide editable installs for FORA packages
- runtime user switch only after dependencies are installed

### Quick diagnostics inside container
```bash
whoami
python -c "import sys, numpy; print(sys.executable); print(numpy.__version__); print(numpy.__file__)"
python -m pip show numpy
```

If `pip show numpy` succeeds for one user but import fails for another, it is an
install-scope mismatch.

### Prevention checklist
- Do not use `pip --user` in the submission image build for required runtime deps.
- Keep dependency installs before final `USER` switch.
- Add a startup import sanity check (`numpy`, `torch`, `sklearn`) in entrypoint.
- Rebuild and re-export the Docker tarball after dependency-install changes.
