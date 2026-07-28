The platform auto-detects input and output schemas from host uploads and the baseline submission. This schema powers mock data generation, output validation, and clear error messages for participants.

How input schema is detected

When the host uploads test data, the platform scans the contents and detects:

File formats (NIfTI, NumPy, PNG, DICOM, JSON, CSV)
Folder structure and naming conventions
Image shapes, dimensions, and data types
JSON keys and nested structures
CSV column names and data types
Number of cases and case ID pattern
How output schema is detected

After the baseline submission scores successfully, the platform performs the same analysis on the prediction outputs:

Output file formats, shapes, and data types
Input-to-output mapping: 1:1 (one output per input, typical for segmentation) or many:1 (all inputs produce one output, typical for classification or regression)
File naming conventions and required fields
Host review

The auto-detected schema is presented to the host for review. Hosts can edit field descriptions, add constraints, mark fields as optional, and adjust the detected mapping. The schema is locked once submissions are enabled (disable submissions to make changes).

Supported formats

NIfTI
.nii, .nii.gz
NumPy
.npy, .npz
PNG
.png
DICOM
.dcm, folder series
JSON
.json
CSV
.csv
Mock data generation

Running forithmus generate creates structurally correct mock input data that matches the challenge schema:

Files have the correct format, shape, data type, and naming convention
Case IDs match the pattern from the real test data
JSON structures have all required keys with plausible placeholder values
CSV files have all required columns with correct types
Image data contains random noise with correct dimensions and ranges
Mock data appears in the mock_input/ directory within your project workspace. Use it for development and with forithmus test for local validation.

Output validation

Running forithmus test my-image runs your container against mock data and validates the output. It checks:

Correct number of output files
File names match expected pattern
File formats match schema
Shapes and data types match schema
JSON has all required keys
CSV has all required columns
Container ran as non-root user
Validation errors include clear messages explaining what was expected and what was found, with suggestions for how to fix the issue.