Docker submissions are the primary submission format. You package your algorithm as a container image. The platform runs it with test data mounted read-only, collects your predictions, and evaluates them automatically.

## Container requirements

Required
Non-root user: include a USER instruction in your Dockerfile
Read input from /input/ (mounted read-only by the platform)
Write predictions to /output/
Handle SIGTERM for graceful shutdown (save state within 30 seconds)

Optional
Save progress to /checkpoint/ for continuation across runs, required for spot instances
Support for GPU via CUDA (if a GPU tier is selected)

### Directory structure inside the container

/input/              # Test data (read-only, mounted by the platform)
  images/            #   e.g., case_001.nii.gz, case_002.nii.gz, ...
  metadata.json      #   Optional metadata provided by the host

/output/             # Write your predictions here
                     #   e.g., case_001.nii.gz (segmentation mask)
                     #   or predictions.json (classification results)

/checkpoint/         # Optional: save progress for continuation
                     #   Persists across runs; /output/ does NOT persist

### Complete Dockerfile example

```
FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir numpy nibabel torch

# Create a non-root user (required by the platform)
RUN useradd -m runner

# Switch to non-root user
USER runner
WORKDIR /home/runner

# Copy your algorithm code
COPY predict.py .

# Set the entrypoint
CMD ["python", "predict.py"]
```

### Complete predict.py example

```
import json
import os
import shutil
import signal
import sys

import nibabel as nib
import numpy as np

# ── Configuration ──
INPUT_DIR = "/input/images"
OUTPUT_DIR = "/output"
CHECKPOINT_DIR = "/checkpoint"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "progress.json")
OUTPUT_BACKUP = os.path.join(CHECKPOINT_DIR, "outputs")

# ── Ensure directories exist ──
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_BACKUP, exist_ok=True)

# ── Track progress for checkpointing ──
processed = []
shutting_down = False

def save_checkpoint():
    """Save current progress to checkpoint directory."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed": processed}, f)
    # Back up outputs (they are cleared between runs)
    for fname in processed:
        src = os.path.join(OUTPUT_DIR, fname)
        dst = os.path.join(OUTPUT_BACKUP, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

def handle_sigterm(sig, frame):
    """Handle SIGTERM: save checkpoint and exit cleanly."""
    global shutting_down
    shutting_down = True
    print("SIGTERM received, saving checkpoint...")
    save_checkpoint()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

# ── Restore from checkpoint if continuing ──
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        processed = json.load(f)["processed"]
    print(f"Resuming from checkpoint: {len(processed)} cases already done")
    # Restore backed-up outputs
    for fname in processed:
        bak = os.path.join(OUTPUT_BACKUP, fname)
        out = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(bak):
            shutil.copy2(bak, out)

# ── Process each case ──
all_cases = sorted(os.listdir(INPUT_DIR))
for fname in all_cases:
    if fname in processed or shutting_down:
        continue

    # Load input
    img = nib.load(os.path.join(INPUT_DIR, fname))
    data = img.get_fdata()

    # Run your algorithm (replace with your model)
    prediction = (data > 0.5).astype(np.float32)

    # Save prediction
    out_img = nib.Nifti1Image(prediction, img.affine, img.header)
    nib.save(out_img, os.path.join(OUTPUT_DIR, fname))

    # Update progress
    processed.append(fname)
    print(f"Processed {fname} ({len(processed)}/{len(all_cases)})")

    # Save checkpoint periodically (every 10 cases)
    if len(processed) % 10 == 0:
        save_checkpoint()

# Final checkpoint
save_checkpoint()
print(f"Done. Processed {len(processed)} cases.")
```

### Build and export

```
# Build the image
docker build -t my-algorithm .

# Test locally (optional but recommended)
docker run --rm -v ./test_input:/input:ro -v ./test_output:/output my-algorithm

# Export as tar.gz for upload
docker save my-algorithm | gzip > my-algorithm.tar.gz

# Upload via web (up to 15GB) or CLI (no size limit)
forithmus submit my-algorithm.tar.gz
```

### Compute tiers

Select a compute tier when submitting. Available tiers depend on what the challenge host has enabled.

cpu-4
4 vCPU, 16 GB RAM
Lightweight algorithms, classical ML
cpu-16
16 vCPU, 64 GB RAM
CPU-intensive processing, large volumes
gpu-t4
NVIDIA T4, 16 GB VRAM, 8 vCPU, 32 GB RAM
Deep learning inference, medium models
gpu-a100-40
NVIDIA A100 40 GB VRAM, 12 vCPU, 85 GB RAM
Large models, multi-step inference
gpu-a100-80
NVIDIA A100 80 GB VRAM, 12 vCPU, 170 GB RAM
Very large models, foundation models

### Time budget

You also select a time budget (minimum 5 minutes). The full cost is pre-charged at submission time based on the selected tier and duration. When your container finishes, the unused time is automatically refunded to your wallet. You only pay for the time your container actually used. See the platform for current pricing per tier.

### Container size limits

The web upload interface accepts Docker images up to 15 GB. For larger images, use the CLI (forithmus submit) which supports chunked, resumable uploads with no practical size limit. The maximum container size in the registry is 50 GB.