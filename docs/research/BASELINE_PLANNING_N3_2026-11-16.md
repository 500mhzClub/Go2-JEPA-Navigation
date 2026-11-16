# Completed n3 baseline planning pair — 16 November 2026

Both final seed0 models have completed the same 1,000 test episodes at each
horizon. L1, the model with physical auxiliary supervision, solves about 94–95%
of endpoint tasks. The ordinary L0 baseline solves about 98%. Auxiliary
supervision therefore has no observed endpoint-planning advantage in this pair.
All 6,000 saved cases have passed independent scoring and substantive review.

| Horizon | L0 endpoint success | L1 endpoint success | L1 95% interval | L0 collision-free success | L1 collision-free success |
|---|---:|---:|---:|---:|---:|
| 4 | 97.8% | 94.7% | 93.3–96.0025% | 85.0% | 81.9% |
| 6 | 98.2% | 94.2% | 92.7–95.6% | 81.7% | 77.9% |
| 8 | 97.7% | 94.7% | 93.3–96.1% | 77.7% | 74.8% |

L1 trails by 3.1, 4.0 and 3.0 percentage points in endpoint success. These are
observed differences on matching tasks and CEM seeds. The per-model intervals
use the fixed 2,000 episode bootstrap; they omit training-seed uncertainty.
This report does not add a paired-difference significance claim. Five groups of
200 episodes are evaluation groups, not five trained seeds.

The [final physical probes](FINAL_BASELINE_PROBES_2026-11-16.md) showed that L1
improves autonomous white-ball readout at n3 while reducing controlled red-ball
precision. The planning ranking is consistent with that tradeoff, but does not
isolate its cause. White balls pass through the red ball without changing its
dynamics. Accurate red endpoint planning consequently need not require accurate
white prediction. Collision-free success is lower for both models and also
lower for L1; it is a discrete-frame diagnostic without a collision planning cost.
Neither endpoint success nor this diagnostic establishes robot avoidance.

The [reference paper, sections 4.3–5.2 and A.10](https://arxiv.org/html/2609.04264v3),
was revisited after completion and independent inspection. Its reported weak
baseline and auxiliary benefit have not reproduced locally. The original local
auxiliary-head update includes predicted-branch gradients, unlike section 4.3's
stability precaution; predictor width, training and inference/evaluation also
differ. These are candidate explanations, not tested causes. Preserve the
unexpected ranking and fixed benchmark; no test-driven correction was made.

The declared L1 H4 reference of at least 50% is met. L0's below-10% reference is
not met. As documented in the [first L0 report](FIRST_BASELINE_PLANNING_2026-11-16.md),
the original O1 advantage target of 30 percentage points is unreachable on these
n3 seed0 cases: even a perfect O1 can improve only 2.2 points. O1 itself remains
unmeasured; this bound does not populate that missing comparison.

The unchanged protocol uses final 303,000-update checkpoints, train-only
calibrated float64/math inference, three history frames, independently encoded
goals and the original 100-candidate/10-elite/20-iteration CEM. No training,
calibration or planner tuning occurred during evaluation. Matching recorded,
direct and true-model controls reach 100%. The learned-only summary's
`controls_passed=false` concerns absent control arms in that file; reviewed
controls are separate inputs.

The independent inspector reconstructs every action trajectory and endpoint and
collision flag, checks immutable labels and matching control identities, and
reproduces every group and bootstrap summary. L1's maximum reconstruction
discrepancy is 8.733e-6 pixels, below the frozen tolerance, with identical
classifications. Saved singleton numerical arrays pass; model weights are
unchanged and CUDA was not initialized. These checks follow
[PyTorch's limits on numerical equivalence](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html).
They do not demonstrate cross-device bitwise equality.

L1 exited normally after 1,584.14 seconds, with sampled peak RSS 1.5675 GiB,
no signals or errors and no GPU interval. Scientific evidence and review are
under `outputs/gated/planning/l1_n3_s0`, operational evidence under
`outputs/gated/B003/operations/l1_n3_s0`, and independent inspection under
`outputs/gated/B003/inspection/v2/l1_n3_s0`.

Next is the fixed L0n1 evaluation, followed by L1n1 and the n6 pair. The RGB object
encoder continues on the GPU; selected-checkpoint checks, O0/O1 dynamics,
binding/probes/planning, P26, additional seeds/counts and the final evidence
bundle remain. No object-model benefit or Go2 transfer has yet been demonstrated.
