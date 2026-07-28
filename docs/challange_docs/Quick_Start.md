# Overview

Forithmus Research Hub is a platform for medical imaging AI challenges. It brings together researchers, clinicians, and industry teams to develop, evaluate, and benchmark algorithms on hidden data. Whether you want to compete on a public leaderboard or validate your algorithm against proprietary data that never leaves the platform, Forithmus handles compute, security, scoring, and leaderboard management automatically so you can focus on the science.

How it works

1
Host creates a challenge
Upload hidden test data, ground truth, and an evaluation container. Configure which metrics matter and how submissions are ranked. A guided stepper ensures nothing is missed.
2
Participant submits an algorithm
Upload a Docker container or prediction file. For Docker submissions, the container receives test data at /input/ and writes predictions to /output/. Participants never access the ground truth.
3
Platform evaluates and scores
The evaluation container scores predictions against ground truth. Results appear on the leaderboard automatically.

# Quick Start: Participants
This guide walks you through the complete workflow from finding a challenge to seeing your score on the leaderboard. The entire process is driven by the forithmus client tool, which handles authentication, schema generation, local testing, and submission. Submissions can also be done through the weh UI.

Step by step

1
Browse challenges
Visit the Challenges page to find an open challenge. Read the description, and check the evaluation criteria.
2
Join the challenge
Click "Join" on the challenge page. For public challenges, unless the host specifies otherwise, you are accepted immediately. For private challenges, you need to be invited.
3
Install the CLI
Install the Forithmus CLI tool via pip.
pip install forithmus
4
Log in
Authenticate via your browser. The CLI opens a login page and waits for confirmation.
forithmus login
5
Initialize your workspace
Download the challenge schema and scaffold a project directory. Replace challenge-slug with the slug shown on the challenge page.
forithmus init challenge-slug
This creates a folder with the challenge schema, a sample Dockerfile, and a starter predict.py.
6
Generate mock data
Create structurally correct mock input data that matches the challenge schema. Use this to develop and test your algorithm locally.
forithmus generate
Mock data appears in the mock_input/ directory with matching case IDs and correct file formats, shapes, and data types.
7
Build your Docker container
Develop your algorithm, then build and export your Docker image.
docker build -t my-image .
docker save my-image | gzip > my-image.tar.gz
8
Test locally
Run your container against the mock data. The CLI validates that your output matches the expected schema (file count, naming convention, data types, shapes).
forithmus test my-image
If there are validation errors, the CLI provides clear, actionable error messages explaining what went wrong and how to fix it.
9
Fix any issues
Review the validation output. Common issues include missing output files, incorrect file names, wrong data types, or running as root. Fix, rebuild, and test again until all checks pass.
10
Submit
Upload your container to the platform. The CLI handles chunked uploads with resume support for large images.
forithmus submit my-image.tar.gz
You will be prompted to select a compute tier and time budget. The cost is pre-charged from your wallet (or the challenge sponsor pool).
11
View results
Once evaluation completes, your score appears on the challenge leaderboard. You can also check submission status anytime with forithmus status or on the web UI.
Complete CLI workflow example

# Install the CLI
pip install forithmus

# Authenticate
forithmus login

# Initialize workspace for a liver segmentation challenge
forithmus init liver-segmentation-2026

# Generate mock input data matching the challenge schema
forithmus generate

# Build your Docker image
docker build -t liver-seg .

# Test locally against mock data (validates output format)
forithmus test liver-seg

# Export and submit
docker save liver-seg | gzip > liver-seg.tar.gz
forithmus submit liver-seg.tar.gz

# Check submission status
forithmus status
Alternative: file submissions

Some challenges accept prediction files directly (JSON, CSV, or ZIP) instead of Docker containers. This is useful for simpler tasks like classification or regression where you process the data externally. Upload your prediction file through the web UI or with forithmus submit predictions.json. See the File Submissions section for details.

Tip: always run forithmus test before submitting. It catches most issues locally, saving you time and compute credits. The validator explains exactly what is wrong and how to fix it.

# Quick Start: Hosts
Creating a challenge is a guided process. A stepper UI walks you through every requirement, and many fields are auto-detected from your uploads.

Guided setup steps

1
Create the challenge
Click "Create Challenge" from your dashboard. Choose public or private. Set a title, description, category, and optional deadline. Upload a banner image. For private challenges, you will also set up billing.
2
Upload test data
Upload a ZIP file containing the test cases participants will process. The platform scans your data and auto-detects the input schema: file formats, folder structure, shapes, data types, JSON keys, and CSV columns. For large datasets, use the CLI:
forithmus upload-data test-data.zip
3
Upload ground truth
Upload a ZIP with the reference labels. This is kept strictly separate and is only accessible to the evaluation container. The platform scans it the same way.
forithmus upload-gt ground-truth.zip
4
Upload evaluation container
Upload the Docker image that scores predictions against ground truth. It receives predictions at /input/predictions/ and ground truth at /input/ground_truth/, and must write a metrics.json to /output/.
forithmus upload-eval eval-container.tar.gz
5
Run a baseline submission
Submit a baseline Docker container (even a simple one) to verify the full pipeline works end-to-end.
6
Review data schema
The platform auto-configures the input and output schema based on your uploads and the baseline run. Review the detected formats, field names, file structures, and mapping (1:1 for segmentation, many:1 for classification). Edit if needed.
7
Set ranking
The metrics discovered from your baseline are auto-populated in the ranking configuration. Choose a ranking method (single metric, custom formula, mean score, mean rank, or weighted rank). Set the direction for each metric (higher or lower is better).
8
Configure evaluation compute
Choose which compute tiers participants can use (CPU, GPU T4, GPU A100). Set maximum time budgets and optional spot instance support.
9
Write submission instructions
Write clear instructions for participants: expected input/output format, any constraints, tips for getting started. This appears on the challenge page under the Submit tab.
10
Enable submissions
The stepper shows all prerequisites as a checklist. Once every item is green, flip the switch to start accepting submissions. Participants can now join and submit.
CLI for large uploads: the web interface supports uploads up to 15GB. For larger datasets or Docker images, use the CLI commands forithmus upload-data, forithmus upload-gt, and forithmus upload-eval. They support chunked, resumable uploads with no size limit.
What gets auto-detected

From your data uploads
File formats (NIfTI, PNG, DICOM, JSON, CSV, NumPy)
Folder structure and naming conventions
Shapes, data types, value ranges, JSON keys and CSV column names
Number of cases and case ID pattern
From the baseline run
Output format and structure
Input-to-output mapping (1:1 or many:1)
Metric keys from metrics.json
Metric value ranges and types

# Challenge Types
Forithmus supports two visibility modes (public and private) and any task type. The data schema is auto-detected from your uploads.

Public vs Private

Public challenges

Open to all registered users
Free platform compute up to per-submission limits
Leaderboard visible to everyone
Free storage up to 1TB data and 1TB registry
Great for conference challenges, community benchmark, and reproducible research

Private challenges

Invite-only: host adds members manually
Host pays via subscription, all costs from challenge pool
Data never visible to participants (mounted read-only inside containers)
Leaderboard visible only to members
Ideal for external validation
Common task types

The platform supports any input/output format, the data schema is auto-detected from your uploads and can be customized. Here are common tasks hosted on the platform:

3D Segmentation
NIfTI volumes in, segmentation masks out.
2D Segmentation
PNG or TIFF images in, mask images out.
Classification
Images or volumes in, class labels out (JSON/CSV).
Detection
Images in, bounding boxes with confidence scores out.
Regression
Images or tabular data in, numeric predictions out.
Report Generation
Medical images in, text reports out.
Image Generation
Text prompts in, generated images out.
Reconstruction
Degraded data in, reconstructed volumes out.
Custom
Any input/output format. Define your own schema.