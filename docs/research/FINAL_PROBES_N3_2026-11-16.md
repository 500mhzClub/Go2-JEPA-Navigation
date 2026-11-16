# Final n3 baseline probes — 16 November 2026

The first matched final-checkpoint physical probes show a clear tradeoff. L0
represents and predicts the controlled red ball precisely, while its fixed linear
readout barely recovers the autonomous white balls. L1's physical supervision
substantially improves white-ball readout, with worse red-ball precision. Neither
result establishes planning success or a benefit to Go2.

Both models finished 303,000 updates and passed the final artifact and inference
checks. These C006 CPU evaluations use the unchanged C001 probe functions:
4,096 training windows for a frozen ridge readout, 5,000 test windows from 1,000
episodes, three-frame history and four future steps. Train/test sample indices
and targets match exactly across models. The checkpoint seed is zero.

| Position RMSE, pixels | L0 | L1 |
|---|---:|---:|
| Current controlled red ball | 0.704 | 2.682 |
| Four-step controlled red ball | 0.873 | 3.019 |
| Current autonomous white balls | 20.782 | 12.718 |
| Four-step autonomous white balls | 20.762 | 13.638 |
| Future red, holding current latent | 5.168 | 5.710 |
| Future white, holding current latent | 20.746 | 14.059 |

RMSE is the square root of mean squared Euclidean XY error; white errors average
over balls before taking the square root. White coordinates are sorted
lexicographically at each time and describe positional ranks, not persistent
object identities.

L0's current red mean coordinate Pearson correlation is 0.9996, versus 0.0016
for white. All six white current-coordinate bootstrap intervals span zero, as
do its six future-coordinate intervals. L1 reaches 0.7943 mean white correlation
currently and 0.7533 after rollout; the respective per-coordinate ranges are
0.628–0.886 and 0.597–0.844. Thus L1's improvement does not establish uniformly
high-fidelity physical state. Its four-step white error is only modestly better
than holding the current latent, leaving substantial dynamics error unresolved.

The reference paper's [section 3, section 4.4 and Appendix A.10](https://arxiv.org/html/2609.04264v3)
separate latent geometry, recoverable information and planning. Revisited after
each case, these methods support a narrow interpretation: low latent prediction
loss can coexist with poor access to important physical variables. A failed
linear readout does not prove that all information is absent. The paper's n3
baseline emphasizes autonomous objects instead, so our red/white selectivity is
not an exact replication of its direction. Our probe uses 4,096 samples and
ridge 0.001 rather than 50,000 and 1e-6, and the original local L1 training has
the already-disclosed predictor-width and auxiliary-head gradient-routing
differences. The literature does not justify treating L1 as a universal winner.

Both models' red rollout errors beat holding the current latent. Reversing
future actions produces errors close to factual actions and slightly improves
L1 here; reversal can preserve net displacement and is a weak order control.
This does not establish correct action sequencing. No new control, fitting,
planner adjustment or test-driven selection was introduced.

These are descriptive results conditional on one trained seed. Reported
per-coordinate intervals use 2,000 whole-episode bootstrap samples and omit
training-seed uncertainty. No paired-difference interval or significance claim
is made. Numerical grouping differences are below 7.8e-14; ridge residuals are
below 6.5e-16. Independent recomputation from raw cases reproduces all point
correlations and coordinate RMSEs. Checkpoint, input and saved artifact hashes
match; learned weights remain unchanged. CPU execution is an explicit C006
adaptation, without claiming universal CPU/GPU equality, consistent with
[PyTorch numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html).

Each CPU case exited normally in about 22.6 seconds, below 1.67 GiB sampled RSS,
without initializing CUDA, sending signals or adding GPU time. Scientific and
operational results have separate exact-digest literature reviews under
`outputs/gated/probes/{l0_n3_s0,l1_n3_s0}` and
`outputs/gated/C006/supervision/{l0_n3_s0,l1_n3_s0}`. The C006 implementation and
software checks are captured by hashes in its reviewed software result.

The next CPU cases are L0/L1 n1, then L0/L1 n6. Meanwhile the n3 seed0 RGB slot
encoder has completed its first epoch and reached 3,169 verified updates out of
23,752; it is still training. Selected-checkpoint numerical validation, object
binding, O0/O1 dynamics and fixed paired planning remain necessary. The object
experiment asks whether image-based grouping can preserve useful state for all
objects without privileged position targets. The later frozen-s0 Genesis
comparison tests relevance to the Go2 pipeline; no robot transfer is yet shown.
