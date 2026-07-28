The forithmus is a command-line tool for interacting with the platform. It handles authentication, workspace setup, mock data generation, local testing, and submissions. It is the recommended way to work with challenges, especially for large file uploads.

### Installation

```
pip install forithmus
```

Requires Python 3.9 or later. Works on Linux, macOS, and Windows (WSL recommended for Docker integration). Docker must be installed for the test command.

All commands

forithmus login
Authenticate via your browser. Opens a login page, waits for confirmation, and stores your token locally.
forithmus logout
Remove your stored authentication token.
forithmus whoami
Display your username and account email.
forithmus challenges
List all challenges you have access to (joined and public).
forithmus init <slug>
Download the challenge schema and scaffold a project directory with a Dockerfile and starter code.
forithmus generate
Create mock input data matching the challenge schema. Produces structurally correct files with matching case IDs, shapes, and data types.
forithmus test <image>
Run your Docker container locally against mock data. Validates output format, file count, naming, shapes, and data types against the schema.
forithmus validate <dir>
Check an output directory against the schema without running Docker. Useful for quick iteration.
forithmus submit <file>
Upload and submit a Docker image (.tar.gz) or prediction file (JSON, CSV, ZIP). Supports chunked, resumable uploads.
forithmus status
Check the processing status of your most recent submission.
forithmus upload-data <file>
Host command: upload test data (ZIP). Chunked upload with resume support for large datasets.
forithmus upload-gt <file>
Host command: upload ground truth (ZIP).
forithmus upload-eval <file>
Host command: upload evaluation container (Docker .tar.gz).
Validation error messages

When you run forithmus test and validation fails, the tool provides clear, actionable error messages. Each error explains what went wrong, what was expected, and how to fix it. For example:

Validation found 2 issues:

  1. Missing output file: case_003.nii.gz
     Expected 3 output files to match 3 input cases.
     Check that your algorithm processes all files in /input/images/.

  2. Wrong data type in case_001.nii.gz
     Expected: float32, Got: int16
     Cast your predictions to float32 before saving.
Large uploads: the CLI is the recommended way to upload Docker images over 15 GB and datasets over 5 GB. Uploads are chunked and resumable, so interruptions do not require starting over.