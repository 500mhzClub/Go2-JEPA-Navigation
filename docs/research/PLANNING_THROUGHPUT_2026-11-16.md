# Object-planning throughput diagnosis — 16 November 2026

The first O0 planning operation was intentionally retired after its observed
throughput projected beyond the existing allocation. The first29saved tasks
averaged6.528seconds each, suggesting about5.44hours for one3,000-task arm.
B001's existing total allowance is2GPU-hours. This prefix estimate is uncertain,
but spending the entire allowance at that rate would not complete even O0.

The exact owned child received SIGTERM and exited. Its original C009v2 supervisor
confirmed returncode-15, unchanged inputs and both closed GPU intervals.55complete
saved case rows remain at their original path. The operation lasted435.236seconds;
its review is `technical_blocked`, while the separate retirement audit passes.
No complete scientific result was produced, so no planning success conclusion
is drawn from this prefix. The stage budget and full task scope are unchanged.

[PyTorch asynchronous execution](https://docs.pytorch.org/docs/2.12/notes/cuda.html)
explains that CPU/GPU copies synchronize. Source inspection finds100per-matrix
host copies per latent-cost evaluation in the original slot matcher. Their share
of wall time is not yet measured; transformer rollout may dominate instead.
[SciPy assignment documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.linear_sum_assignment.html)
defines the matching objective to retain. [Numerical accuracy notes](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
warn that batching reductions can change rounding, so a faster implementation
cannot be assumed equivalent.

[C015](records/C015.json) preregisters two transfer layouts: one full cost-matrix
batch copy with original individual GPU sums, and the same copy with batched
index transfers/sums. Full64 model inference, CEM settings, sampled tasks,
matching objective and endpoint/collision definitions stay fixed. Implementation
is confined to new external helpers in an isolated performance checkout; all
original scientific and reviewed runtime sources remain unchanged.

Eight CPU software groups pass:3/5/8slots,1/8/100candidate batches, random/zero/
permuted/tied fixtures, both models' saved train/validation numerical arrays,
inherited process/accounting identity and temporary-ledger refusal checks.
Actual GPU cost/rank/action equivalence and runtime remain unmeasured.

The next bounded diagnostic uses both selected n3seed0 predictors, train and
validation episode0, and horizons4/6/8. It saves every proposal, cost, ranking,
assignment matrix and final action across20CEM iterations. Synchronized timings
use separate uninstrumented passes. Qualification requires cost agreement at
1e-12relative/absolute tolerance plus exact rankings, commands and physical
scores. Independent trace inspection and substantive review precede any use.
The diagnostic has900seconds including45cleanup within the existing reserve;
no GPU allowance is added. A speedup alone does not qualify scientific planning.

Evidence: `outputs/gated/C015`; preserved original partial science under
`outputs/gated/planning/o0_n3_s0` and operation under
`outputs/gated/C009/v2/runs/planning/o0_n3_s0`. Paired probes remain reviewed
negatives with a descriptive O1 error improvement. Complete paired planning,
offline P26, replications and final report/archive are still required.
