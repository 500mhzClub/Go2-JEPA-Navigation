# Final baseline inference — 16 November 2026

All six original L0/L1 seed0 experiments completed 303,000 steps after sequential
recovery. Their actual A015 checkpoint audits and A014 completion reviews pass.
All six final A003 numerical evaluations and A016 operational reviews now also
pass. No original trainer or final evaluator is active at this report's capture.
Physical prediction, learned planning, the object-based comparison and P26 remain
unfinished; this is completion of the baseline training and inference gates.

The final L1 n6 continuation took 57.84 minutes and exited 0. Its audit verifies
148,000 resumed updates, exact CPU restoration, matching full/final learned state
and complete finite logs. L1 n1/n3 resumed 198,000 updates each; the other four
resumed 148,000 each. The original recipe and pinned trainer remain unchanged.
Six successful serial continuations support operational recovery after reboot,
without identifying the original hang's component or guaranteeing future stability.

## Numerical evidence

Each final checkpoint passed 135 comparisons before and 135 after calibration:
six grouping comparisons, two CPU/GPU forward comparisons and 127 base-model
gradient comparisons per phase. All 1,620 comparisons pass the unchanged
rtol 1e-4 / atol 1e-5; the largest absolute discrepancy is 2.083e-13. All inspected
state, outputs and gradients are finite. L1's auxiliary head is finite and exact,
but is not part of the latent forward/gradient parity check.

Inference uses the common float64/math policy selected earlier by A002.
Calibration pools moments from 4,096 training windows, encoder before predictor.
Independent CPU inspections confirm every learned tensor is preserved after
permitted dtype conversion: only the six declared BatchNorm buffers change.
Exports are marked inference-only and bind to the audited final source digests.

| Final checkpoint | Saved-statistic validation latent MSE | Calibrated MSE | Reduction |
|---|---:|---:|---:|
| L0 n1 | 0.000687687 | 0.000548770 | 20.20% |
| L0 n3 | 0.000639934 | 0.000602617 | 5.83% |
| L0 n6 | 0.007301325 | 0.007198348 | 1.41% |
| L1 n1 | 0.004086544 | 0.003811207 | 6.74% |
| L1 n3 | 0.008608173 | 0.008484887 | 1.43% |
| L1 n6 | 0.033426268 | 0.033033438 | 1.18% |

These are the fixed 64-window diagnostics, not full-test physical errors. Here
`n` counts autonomous white balls, in addition to one action-controlled red ball.
The final L1 n1 saved/calibrated gap is much smaller than at 105,000 or 220,000 in
[A017](CHECKPOINT_PROGRESS_2026-11-15.md). Do not extrapolate the earlier large
normalization mismatch unchanged to final weights. Both-batch diagnostics have
higher errors for five final models but slightly lower error for L1 n6; batch
statistics are not a universal remedy. Learned representations and normalization
targets differ across arms, so the table does not rank physical quality.

[Batch Normalization section 3.1 / Algorithm 2](https://arxiv.org/pdf/1502.03167v3)
was revisited after each result: fixed training-derived population statistics
support deterministic inference. Exact pooled moments and dependency-ordered
calibration are declared local adaptations. [PyTorch numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
motivates explicit grouping and CPU/GPU checks. Coverage is 64 train/validation
windows, grouping on 8 validation windows and parity/gradients on 4; this is not
an all-input correctness guarantee or evidence for physical/planning generalization.

## Scientific interpretation and remaining tests

The strongest physical evidence remains A017's within-L1 n1 comparison from
105,000 to 220,000: four-step position RMSE improved about 20% for the controlled
ball and 24% for the autonomous ball. Action ablation worsened controlled-ball
prediction substantially. Those results are conditional on one seed and the
reused validation corpus; final 303,000-step physical probes have not yet run.

Meanwhile, latent-distance correlations with physical distances declined in
that comparison. [The reference paper's section 3, section 4.4 and Appendix A.10](https://arxiv.org/html/2609.04264v3)
were revisited: latent-distance geometry, information recoverable by probes and
planning success are separate measurements. Our interpretation is that improved
physical readout leaves the planner's distance objective unresolved. It is not
evidence that planning must fail. L0's lower latent error establishes neither
superiority nor omission of an important object.

Next are the approved n3 seed0 RGB slots, selected-checkpoint validation,
cache and O0/O1 stages, binding/probes and fixed paired planning. The scientific
question is whether an object-based representation preserves useful object
state and action-conditioned dynamics without privileged position supervision.
The frozen-s0 offline Genesis comparison then checks relevance to the Go2
pipeline. Extra seeds/counts and the final evidence bundle follow within budget.
No Go2 improvement, object-model advantage or robot transfer has yet been shown.

## Provenance and resources

Actual reviews are under `outputs/gated/A015/finals/RUN`,
`outputs/gated/A014/runs/RUN`, `outputs/gated/A003_recovered/RUN` and
`outputs/gated/A016/runs/RUN`. Each review binds the exact result hash; the A003
results also bind source and calibrated checkpoints. The evaluator remained
pinned to 161fe8b for all six cases. Primary-source consultation and individual
measurement inspection preceded every passing review and successor.

All six evaluators exited 0 without timeout or signals. Their outer intervals
charge 34.340678 seconds total; inner intervals are contained and overlap counts
once. All ledger intervals are closed. Total additional GPU use is 0.301059 hours,
leaving 47.698941 of 48 hours; original training is excluded as authorized.

A first L0 n1 review-assembly attempt encountered a symlink path comparison
error. It is preserved and reviewed under
`outputs/gated/A016/runs/l0_n1_s0/review_preparation_attempt1`. Canonical-path
lookup resolved it without changing or rerunning the experiment or its ledger.
The older interim archive predates all final audits and these evaluations;
the final reproducibility bundle must include them and the remaining science.
