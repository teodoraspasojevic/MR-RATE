Billing and Credits
The platform uses a simple wallet system. Top up your wallet, and costs are deducted automatically. Challenge hosts can also fund sponsor pools so participants submit for free.

Wallet

Every user has a wallet with a dollar balance
Top up via Stripe (credit card or bank transfer)
Choose from preset amounts or enter a custom amount
Credits never expire
Manage your wallet from Dashboard > Billing
Sponsor pools

Challenge hosts (or sponsors) can fund a pool for their challenge. When a participant submits, the cost is deducted from the pool first. If the pool is empty, the cost falls back to the participant's wallet.

Hosts fund pools from the challenge Billing settings
Participants see whether a pool is available before submitting
Deduction order: pool first, then participant wallet
Pool balance visible to challenge admins
Pre-charge and refund

When you submit a Docker container, the full cost (tier price per hour times time budget) is pre-charged from your wallet or the sponsor pool. After execution completes, the unused portion of the time budget is automatically refunded. You only pay for the time your container actually used.

Compute pricing

Pricing varies by compute tier and whether you use on-demand or spot instances. Spot instances offer significant discounts. See the platform for current pricing across all tiers.

File submissions

File submissions (JSON, CSV, ZIP) have a small evaluation fee per submission. No compute tier selection needed. See the platform for current pricing.

Evaluation costs

Public challenges

The platform covers evaluation compute up to a per-submission limit. If the evaluation exceeds this limit, the excess is charged to the challenge pool.
Private challenges

All evaluation costs are charged to the challenge sponsor pool.
Storage billing

Public challenges (free tiers)
Test data and ground truth: free up to 1 TB
Docker registry: free up to 1 TB
Contact the platform for extended storage needs
Private challenges
GCS data storage and Docker registry billed per GB/month
Billed weekly from the challenge pool
See the platform for current storage pricing
Refunds

Unused compute time: refunded automatically after scoring completes
Failed submissions (crash before any processing): full refund
Cancelled submissions: refund for unused time at point of cancellation
Validation failures (virus, non-root, architecture): no charge, no refund needed