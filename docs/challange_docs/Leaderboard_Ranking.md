The leaderboard shows the best score per participant per phase. Hosts configure ranking using one of five methods. Metrics are auto-detected from the baseline submission and can be fine-tuned.

Ranking methods

1. Single metric
Rank by one primary metric. The host sets the direction (higher is better or lower is better). Example: rank by Dice coefficient, higher is better. Simple and transparent.
2. Custom formula
A math expression combining multiple metrics with weights. Example: 0.5 * dice + 0.3 * precision + 0.2 * recall. Supports standard operators (+, -, *, /) and parentheses.
3. Mean score
Average of all metric values after normalization. Each metric is scaled to [0, 1] based on the observed range across all submissions, then averaged. Good when all metrics are equally important.
4. Mean rank
Average position across all metrics. For each metric, participants are ranked independently. The final ranking is the average of these per-metric positions. Robust to outliers.
5. Weighted rank
Weighted average of per-metric positions. Same as mean rank, but with host-assigned weights to reflect relative importance. Example: dice weight 3, hausdorff weight 1.
Hidden metrics

Hosts can mark certain metrics as "hidden". Hidden metrics are computed and stored, but are only visible to challenge admins. They do not appear on the public leaderboard. This is useful for internal tracking or metrics that might bias participant behavior.

Best score per participant

The leaderboard shows only the best submission per participant (or team) per phase. "Best" is determined by the ranking method. If a participant submits multiple times, only the highest-ranking submission is displayed.

Metric matching is case-insensitive. If your evaluation container outputs dice and you configure DICE in the ranking settings, they will match automatically.