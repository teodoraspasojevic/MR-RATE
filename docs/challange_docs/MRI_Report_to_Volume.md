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
