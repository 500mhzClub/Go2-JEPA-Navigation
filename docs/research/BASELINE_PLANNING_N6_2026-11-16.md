# Completed six-white-ball planning pair — 16 November 2026

Both seed0 models completed all 1,000 matching episodes at horizons 4/6/8.
All 6,000 saved cases passed independent scoring and separate literature reviews.
L1, which adds physical auxiliary supervision, performs much worse than L0.
This is a measured negative, with normally completed execution.

| Horizon | L0 endpoint success | L1 endpoint success | L1 95% episode interval | L0 collision-free success | L1 collision-free success |
|---|---:|---:|---:|---:|---:|
| 4 | 97.1% | 2.9% | 1.9–4.0% | 71.2% | 2.2% |
| 6 | 93.5% | 1.8% | 1.1–2.7% | 62.6% | 1.3% |
| 8 | 82.6% | 1.0% | 0.4–1.7% | 52.1% | 0.7% |

Observed endpoint deficits are 94.2, 91.7 and 81.6 percentage points. L1 median
endpoint errors are 5.658/7.872/10.048 pixels, versus L0 0.263/0.388/0.503.
This is substantial error beyond the fixed one-pixel success threshold.
The L0 endpoint intervals remain 96.0–98.1%, 91.8975–95.0025%, 80.3–84.9%.
These are per-model episode intervals and omit training-seed uncertainty;
no paired-difference significance claim is added. Different horizons have their
own fixed goal tasks, so the horizon trend is descriptive.

The [final probes](FINAL_BASELINE_PROBES_2026-11-16.md) already show weak L1n6
controlled-ball readout: current RMSE 20.071 pixels versus L0 5.924, and future
21.099 versus its own held-current 20.326. The planning failure is consistent
with that weakness, but neither measurement identifies the mechanism.
Weak linear readout alone does not prove information is absent.

[Sections 4.3–5.2 and appendix A.10 of the reference paper](https://arxiv.org/html/2609.04264v3)
were revisited after actual completion and independent inspection. Its n6
auxiliary advantage has not reproduced. The original local head receives
predictor-branch gradients; section 4.3 excludes those updates for stability.
This difference, architecture, optimization, calibration/evaluation and seed
variation remain candidate explanations. No intervention isolates them and
no model, planner or success threshold was changed after the result.

Recorded, direct and true-CEM controls all achieve 100% endpoint success.
The learned-only output's controls_passed=false denotes absent control arms,
not failed controls. White balls pass through one another and no collision cost
is optimized. Collision-free rates are discrete-frame diagnostics; none of
these results establishes active avoidance or Go2 transfer.

The fixed protocol retains the original 303,000-update checkpoints, train-only
calibrated float64/math inference, three history frames, independently encoded
goals and 100-candidate/10-elite/20-iteration CEM. Independent L1 inspection
reproduces every classification, group summary and bootstrap interval; maximum
endpoint discrepancy is 9.615e-6 pixels, within fixed tolerance. Saved singleton
checks pass, weights are unchanged and CUDA was not initialized.

L1 exited normally after 1,441.213 seconds, sampled peak RSS 1.56917 GiB,
with no stop reason, signals or errors and no B003 GPU interval. Science is
reviewed measured_negative; operation and inspection are reviewed passed under
outputs/gated/planning/l1_n6_s0, B003/operations/l1_n6_s0 and
B003/inspection/v2/l1_n6_s0. L0 evidence and earlier reports remain in Git history.

All six original baseline planning cases are now complete; see the
[combined baseline report](FINAL_BASELINE_PLANNING_2026-11-16.md).
Object comparisons, offline P26, further seeds/counts and final packaging remain.
