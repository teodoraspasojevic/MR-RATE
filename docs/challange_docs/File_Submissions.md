For simpler challenges, you can upload prediction files directly instead of building a Docker container. Process the data yourself (externally or locally) and upload just the results. No Docker knowledge needed.

### Supported formats

JSON
Array of prediction objects with case_id and prediction fields. Good for classification and structured outputs.
CSV
Tabular predictions with required columns. Good for regression, survival analysis, and tabular tasks.
ZIP
Archive of prediction files (NIfTI masks, PNG images, NumPy arrays). Good for segmentation and image outputs.

### Example: classification (JSON)

```
[
  {"case_id": "case_001", "label": 1, "confidence": 0.95},
  {"case_id": "case_002", "label": 0, "confidence": 0.12},
  {"case_id": "case_003", "label": 2, "confidence": 0.87}
]
```

### Example: regression (CSV)

```
case_id,prediction,confidence
case_001,3.42,0.95
case_002,1.17,0.78
case_003,5.89,0.91
```

### Example: segmentation (ZIP)

```
predictions.zip
  case_001.nii.gz    # Binary mask matching input dimensions
  case_002.nii.gz
  case_003.nii.gz
```

### Pricing

File submissions have a small evaluation fee per submission. No compute tier selection needed. The evaluation container runs on platform-managed infrastructure. See the platform for current pricing.

### File size limits

JSON: up to 100 MB
CSV: up to 100 MB
ZIP: up to 5 GB (web upload) or unlimited via CLI
Check the challenge's Submission Format section for exactly which fields, columns, file naming conventions, and formats are required.