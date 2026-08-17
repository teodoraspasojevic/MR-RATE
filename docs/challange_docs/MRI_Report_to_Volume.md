## Overview

Generating realistic 3D brain MRI volumes from text descriptions enables data augmentation, simulation, and privacy-preserving data sharing for neuroimaging research. This task pushes the frontier of conditional medical image synthesis.

In this task, participants build models that generate clinically plausible 3D brain MRI volumes conditioned on a radiology report or text description. The task leverages MR-RATE, providing paired MRI volumes and reports for training.

Task at a Glance
Input: Free-text radiology report or text prompt
Output: Synthetic 3D brain MRI volume
Evaluation: Feature-based similarity (FID-like) + consistency checks via blinded classifier
Ranking: Point-based permutation test across all metrics

## Data Description

This task uses MR-RATE, a large-scale dataset pairing 3D brain and spine MRI volumes with expert radiology reports.

Split
# Patients
# Studies
# Series
Source
Train (public)
75,000
88,985
638,345
Istanbul Medipol University Hospital
Validation (public)
8,425
9,349
66,909
Istanbul Medipol University Hospital
Internal Test (private)
2,000
2,568
19,395
Istanbul Medipol University Hospital
Total
85,425
100,902
724,649
Istanbul Medipol University Hospital
Data Format
Input: Free-text radiology report describing expected brain and spine MRI findings
Output: Synthetic 3D brain or spine MRI volume matching the report description
Multi-sequence exams (T1-weighted, T2-weighted, FLAIR, SWI, MRA) from multi-vendor 1.5T/3T clinical scanners
Non-contrast and contrast-enhanced acquisitions
DICOM metadata provided where available
Data Sources
Primary source: Istanbul Medipol University Hospital
Scans acquired by trained MRI technologists
Reports written by board-certified radiologists
Cases sampled from routine clinical practice to preserve realistic pathology prevalence

## Evaluation Metrics

Submissions are evaluated using feature-based similarity and clinical consistency metrics on the test set.

Image Quality Metrics
Metric	Purpose
Feature-based Similarity (FID-like)	Distributional similarity between generated and real MRI volumes in feature space
Clinical Consistency
Metric	Purpose
Blinded Classifier Consistency	Whether a classifier trained on real data assigns consistent clinical labels to generated volumes matching the conditioning report
Ranking Method
A robust point-based scheme:

For each metric, compute case-level scores and perform a two-sided permutation test between all team pairs.
Award points for statistically significant wins.
Final rank is determined by total points aggregated across primary metrics with predefined weights.
Missing outputs are treated as invalid and receive the lowest possible score.

## Rules

Eligibility
Open to academic and industry participants worldwide.
Challenge organizers and their direct collaborators may not submit competitive entries.
Each team may submit under one account only.
Submission Format
Submissions are Docker containers that accept a text report as input and produce a 3D brain MRI volume as output.
Containers are evaluated server-side on the hidden test sets.
Participants may submit multiple runs. Only the final valid submission will be used for ranking.
Maximum container size: 50 GB.
Data Usage
Participants may use additional publicly available data for training.
Use of private or proprietary datasets must be disclosed.
The test set ground truth must not be used in any form during training.
Code and Methods
Participants are encouraged (but not required) to release their code.
A brief method description is required for inclusion in the joint challenge paper.

How to Submit

You submit a self-contained Docker container that generates 3D brain MR volumes from report-text prompts and writes them as .nii.gz files to /output. The container runs offline on the platform's GPUs (no internet at runtime).

A runnable reference implementation — a MAISI-style latent-diffusion generator with CXR-BERT text conditioning — and this track's production evaluation container live in the VLM3D-Dockers repo under mr_challenges/mrgen_example_docker/ and mr_challenges/mrgen_evaluation/. Start from those.

1. Install the CLI

pip install --upgrade forithmus
forithmus login
2. Initialize and test locally

forithmus init mr-volume-generation
forithmus generate     # synthetic local test set + expected output shape
docker build -t my-generator:latest .
forithmus test my-generator:latest --timeout 1200
Input format — read this carefully

Mounted at /input (read-only): a single prompts.json — a list with one entry per target series:

[
  {"input_image_name": "WNPYIQCPIN_flair-raw-sag",
   "report": "BRAIN MRI (CONTRAST-ENHANCED)\n\nClinical information: ..."},
  ...
]
report is the conditioning text: the study's full free-text radiology report (repeated for every series of that study).
input_image_name = {study_uid}_{modality}-raw-{plane} — it tells you exactly which volume to generate and is the required output filename. The evaluator pairs your output to its ground-truth target by this name only, so cross-modality comparisons cannot happen — but a misnamed or missing volume scores worst-case.
Entries map 1:1 to the ground-truth series that actually exist. Not every study has every modality (e.g. some lack SWI or FLAIR) — generate exactly the entries in prompts.json, nothing more, nothing less.
Only T1w / T2w / FLAIR / SWI series are scored (and only these appear in prompts.json).
Complete name vocabulary
Every input_image_name is {study_uid}_{modality}-raw-{plane}[-N], where:

modality ∈ t1w (214 entries), t2w (216), flair (168), swi (92);
plane ∈ axi (380), sag (178), cor (125), obl (7 — oblique acquisitions exist, handle them);
-N suffix (-2, -3): repeated acquisitions of the same sequence in one study — 108 entries carry one. Keep the suffix verbatim in your output filename; stripping it breaks the pairing.
690 entries total across 100 studies. Study composition varies (68 studies have all four modalities; some lack SWI or FLAIR) — always iterate prompts.json, never assume a fixed per-study set.

Output format

Write one /output/{input_image_name}.nii.gz per prompt.
The filename must exactly match its target ({study_uid}_{modality}-raw-{plane}.nii.gz) — the evaluator pairs generated and reference volumes by name and parses the modality from it. Missing or misnamed volumes receive worst-case scores.
Model weights

Ship code in the image and weights as a separate zip: forithmus submit image.tar.gz --weights weights.zip. The zip is extracted before start and its path exported as FORITHMUS_WEIGHTS_DIR. Re-use previously uploaded weights with --reuse-weights <submission-id>.

Compute and limits

Pick a tier (forithmus tiers) and time budget at submission; unused time is refunded. Jobs exceeding their budget are stopped.
Generation is a long batch job — write restartable progress to /checkpoint and handle SIGTERM so spot preemptions and timeouts resume instead of restarting (the example container shows the pattern).
Scoring

Each generated volume is compared against its reference: MSE, PSNR, SSIM per modality plus a 2.5D FID over squeezenet features. Evaluation streams one volume pair at a time and runs after your container finishes.
Submit your model

Upload a Docker container (.tar.gz). You're pre-charged for the time budget, then refunded for unused time after completion.

How submissions work

CPU 4 vCPU
CPU
4 vCPU, 16 GB RAM
$0.23/hr

CPU 8 vCPU
CPU
8 vCPU, 32 GB RAM
$0.46/hr

CPU 16 vCPU
CPU
16 vCPU, 64 GB RAM
$0.91/hr

CPU 32 vCPU
CPU
32 vCPU, 128 GB RAM
$1.82/hr

T4
GPU
NVIDIA T4, 16 GB VRAM
$0.65/hr

L4
GPU
NVIDIA L4, 24 GB VRAM
$0.85/hr

V100
GPU
NVIDIA V100, 32 GB VRAM
$3.43/hr

L4 (XL host)
GPU
NVIDIA L4, 24 GB VRAM, 32 GB host RAM
$1.30/hr

A100 40GB
GPU
NVIDIA A100, 40 GB VRAM
$4.41/hr

A100 80GB
GPU
NVIDIA A100, 80 GB VRAM
$6.03/hr

H100
GPU
NVIDIA H100, 80 GB VRAM
$13.00/hr

2x A100
2X GPU
2x NVIDIA A100 40GB, 80 GB VRAM
$8.82/hr

4x A100
4X GPU
4x NVIDIA A100 40GB, 160 GB VRAM
$17.63/hr

8x A100
8X GPU
8x NVIDIA A100 40GB, 320 GB VRAM
$35.26/hr

8x H100
8X GPU
8x NVIDIA H100 80GB, 640 GB VRAM
$105.40/hr
TIME BUDGET

minutes
(1 hr)
Container stops after this time. If your script writes checkpoints to /checkpoint/, you can continue from where it left off.
CPU 4 vCPU
On-demand · 1 hr
$0.23
Evaluation
Scored by platform
Free
W
Wallet
$99.77 remaining after
-$0.23
You pay
$0.23
Save $0.14 with spot pricing
Pre-charged. Unused time refunded after completion.
Model weights (optional)
Upload new
Reuse previous
Large model weights? Upload them separately to keep your Docker image small. The weights folder will be mounted at /weights inside your container. Leave empty if your image already contains the weights.

