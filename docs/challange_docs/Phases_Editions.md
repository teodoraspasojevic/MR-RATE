Phases and Editions
Phases let you organize a challenge into submission stages, each with its own data, evaluation, limits, and leaderboard. Editions support yearly recurring challenges with separate leaderboards but shared infrastructure.

What is a phase?

A phase is a submission stage within a challenge. Each phase has independent settings:

Its own test data and ground truth
Its own evaluation container
Its own submission limits (total, daily, weekly, monthly)
Its own leaderboard and ranking configuration
Optional time window (opens_at and closes_at)
Most challenges need only one phase. Multi-phase challenges are useful for staged competitions (for example, a Preliminary phase with public leaderboard and a Final phase with private test data).

Phase activation prerequisites

✓
Upload test data
ZIP file with test cases uploaded and processed
✓
Upload ground truth
ZIP file with reference labels uploaded and processed
✓
Upload evaluation container
Docker image that scores predictions
✓
Run baseline submission
At least one submission scored successfully (verifies full pipeline)
✓
Review data schema
Input and output schema reviewed and confirmed
✓
Set metrics and ranking
At least one metric configured with a ranking method selected
✓
Write submission instructions
Instructions for participants on the Submit tab
✓
Enable submissions
Flip the switch once all prerequisites are green
Submission limits

Hosts can set submission limits per phase to control how many attempts each participant (or team) can make:

Limit types
Total: maximum submissions over the entire phase
Daily: maximum per calendar day
Weekly: maximum per calendar week
Monthly: maximum per calendar month
Time windows
opens_at: date and time when submissions become accepted
closes_at: date and time when submissions stop being accepted
Both are optional (leave blank for always-open)
Editions

Editions support yearly recurring challenges. Each edition has its own phases, leaderboards, and submission history, but shares the same challenge page and infrastructure.

Create a new edition each year (e.g., "2025", "2026")
Previous editions are locked for new submissions
Previous edition leaderboards remain visible
Participants can compare their performance across years
Each edition can have different data, evaluation, and ranking