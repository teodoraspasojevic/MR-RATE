The platform provides tools for hosts to preserve Docker images and prediction outputs for reproducibility. Storage settings control what is kept and for how long.

Keep Docker images

When enabled, the platform retains Docker images in the internal registry for potential re-evaluation or verification.

Only the best submission per participant is kept (not every submission)
When this setting is disabled, images are removed after a 7-day grace period
Free up to 1 TB for public challenges; see platform for private storage pricing
Keep prediction outputs

When enabled, the platform retains prediction output files from the participant's container.

Public challenges: outputs under 10 GB per submission are kept
Private challenges: no size limit (storage billed from pool)
Free within data limits for public challenges; see platform for private storage pricing
Retention after archiving

When a challenge is archived (closed permanently), the following retention policy applies:

Kept forever

Scores and metric values
Leaderboard positions
Submission metadata (timestamps, compute tiers, runtimes)
Challenge description and settings
Deleted after 30 days

Docker images
Prediction output files
Test data and ground truth
Checkpoint files
Export before archiving: if you need to preserve Docker images, prediction outputs, or test data, export them within 30 days of archiving. After 30 days, these files are permanently deleted to free storage.