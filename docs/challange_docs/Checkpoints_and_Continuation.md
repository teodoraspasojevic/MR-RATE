For algorithms that need more time than a single run allows, the platform supports checkpointing. Save your progress, resume later with a new time budget, and continue from where you left off.

How it works

1
Your container runs and processes cases
Periodically save your progress to /checkpoint/. This includes your progress tracker (which cases are done) and copies of your output files.
2
Time limit reached: SIGTERM sent
When the time budget expires, your container receives SIGTERM. You have 30 seconds to save a final checkpoint. After 30 seconds, the container is forcefully terminated.
3
Checkpoint stored to GCS
The contents of /checkpoint/ are uploaded to Google Cloud Storage. The submission status changes to "timed_out" with a "Continue" button.
4
User clicks "Continue" with new time budget
You select a new time budget and are pre-charged for it. The checkpoint is restored to /checkpoint/ and your container starts again.
5
Container resumes from checkpoint
Your code reads the checkpoint, restores backed-up outputs to/output/, skips completed cases, and processes the rest.
Important: the /output/ directory is cleared between runs. Only /checkpoint/ persists. Always back up your output files to the checkpoint directory, and restore them to /output/ when resuming.
Complete code example

```
import json
import os
import shutil
import signal
import sys

CHECKPOINT_DIR = "/checkpoint"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "progress.json")
OUTPUT_BACKUP = os.path.join(CHECKPOINT_DIR, "outputs")
OUTPUT_DIR = "/output"
INPUT_DIR = "/input/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_BACKUP, exist_ok=True)

processed = []
shutting_down = False

def save_checkpoint():
    """Persist progress and back up outputs."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed": processed}, f)
    for fname in processed:
        src = os.path.join(OUTPUT_DIR, fname)
        dst = os.path.join(OUTPUT_BACKUP, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

def handle_sigterm(sig, frame):
    """SIGTERM handler: save checkpoint and exit."""
    global shutting_down
    shutting_down = True
    print("SIGTERM received, saving checkpoint...")
    save_checkpoint()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

# Restore from checkpoint if resuming
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        processed = json.load(f)["processed"]
    print(f"Resuming: {len(processed)} cases already done")
    for fname in processed:
        bak = os.path.join(OUTPUT_BACKUP, fname)
        out = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(bak):
            shutil.copy2(bak, out)

# Process remaining cases
all_cases = sorted(os.listdir(INPUT_DIR))
for fname in all_cases:
    if fname in processed or shutting_down:
        continue
    # ... your algorithm here ...
    # Save result to OUTPUT_DIR
    processed.append(fname)
    if len(processed) % 10 == 0:
        save_checkpoint()

save_checkpoint()
print(f"Done: {len(processed)}/{len(all_cases)} cases")
```

Continuation billing

Each continuation is billed as a separate run. Costs accumulate across continuations based on the selected tier and total time used. Unused time in each continuation is refunded individually. See the platform for current pricing per tier.

Spot instances and checkpoints

Spot instances use the same checkpoint mechanism. If a spot VM is preempted, the platform sends SIGTERM, saves your checkpoint, and automatically retries on a new VM. Your checkpoint code works identically for both timeouts and preemptions. See the Spot Instances section for details.

←
