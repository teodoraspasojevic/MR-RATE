Spot instances are cheaper preemptible VMs that can be interrupted by the cloud provider at any time. They offer significant cost savings for algorithms that handle checkpointing well.

How spot instances work

1
Host enables spot in challenge settings
Spot instances are an opt-in feature. The challenge host enables them in the compute settings for each phase.
2
Participant chooses spot at submission time
When submitting, you can choose between on-demand (guaranteed) and spot (cheaper but preemptible) for your compute tier.
3
Container runs on a spot VM
Your container starts as usual. From your code's perspective, there is no difference between spot and on-demand.
4
If preempted: SIGTERM, checkpoint, auto-retry
If the cloud provider reclaims the VM, the platform sends SIGTERM to your container (same as a timeout). You have 30 seconds to save to /checkpoint/. The checkpoint is stored, and the platform automatically retries on a new spot VM (up to 3 times).
5
Fallback to on-demand
If all 3 spot retries are preempted, the platform automatically falls back to an on-demand VM for the remaining run. Your checkpoint is restored and execution continues.
Cost savings

Spot instances offer significant discounts compared to on-demand pricing across all compute tiers. See the platform for current spot and on-demand pricing per tier.

When to use spot instances

Good fit

Training-heavy algorithms that checkpoint naturally
Long inference runs over many cases
Budget-constrained submissions
Algorithms with robust SIGTERM handling
Not ideal

Very short runs (under 10 minutes) where preemption overhead matters
Algorithms that cannot checkpoint (all progress lost on preemption)
Time-critical submissions near a deadline
Same code for both: your container code does not need to distinguish between spot and on-demand. The SIGTERM and checkpoint mechanism is identical. If your code handles SIGTERM gracefully, it works on spot instances automatically.