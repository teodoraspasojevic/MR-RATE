Every submission goes through a multi-step pipeline from upload to leaderboard. Each step includes validation and error handling. Here is the complete sequence.

1
Upload & Scan
2
Validate
3
Registry Push
4
Execute
5
Checkpoint
6
Evaluate
7
Score
8
Settle
Step by step

1
Upload and virus scan
Your Docker image (.tar.gz) is uploaded and scanned by ClamAV antivirus. All layers are inspected. If malware is detected, the submission is rejected immediately.
2
Container validation
The platform checks: non-root user configured (USER instruction), supported architecture (amd64), image size within limits, and no setuid binaries.
3
Registry push
The validated image is pushed to the internal container registry. This image is used for execution and stored for reproducibility (if enabled by the host).
4
Execution
Your container runs on the selected compute tier. Test data is mounted read-only at /input/. Network access is completely disabled. The time budget is enforced. If a previous checkpoint exists, it is restored to /checkpoint/.
5
Checkpoint save
If the container times out or is preempted (spot instances), the platform sends SIGTERM and waits 30 seconds for your container to save state to /checkpoint/. The checkpoint is stored to GCS for later continuation.
6
Evaluation
The host's evaluation container runs. It receives your predictions at /input/predictions/ and ground truth at /input/ground_truth/. It must write a metrics.json to /output/ containing the computed scores.
7
Score extraction and leaderboard update
Metrics are extracted from metrics.json according to the ranking configuration. Your best score per phase is shown on the public leaderboard. You receive a notification when scoring completes.
8
Billing settlement
The actual runtime is calculated and the unused portion of the pre-charged time budget is refunded to your wallet (or the challenge sponsor pool).
Error handling

Virus detected
Submission rejected. Upload a clean image. No charges applied.
Validation failure
Missing USER instruction, unsupported architecture, or oversized image. Fix and resubmit. No charges applied.
Container crash
Status becomes "failed". Check logs for the error. Unused time is refunded.
Timeout
Status becomes "timed_out". If you saved checkpoints, click "Continue" to resume with a new time budget.
Evaluation error
The eval container failed (wrong output format, missing files). Error details shown in the submission status.
Score extraction error
Metric keys from ranking config not found in metrics.json. Host should verify eval container output.
For hosts: evaluation container

Your evaluation container receives:

/input/
  predictions/    # Participant's output files
  ground_truth/   # Your ground truth files

/output/
  metrics.json    # Write evaluation results here
Example metrics.json:

{
  "metrics": {
    "dice": 0.847,
    "hausdorff_95": 12.3,
    "sensitivity": 0.912,
    "specificity": 0.965,
    "precision": 0.891
  }
}

