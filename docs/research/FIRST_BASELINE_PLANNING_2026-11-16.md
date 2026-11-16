# First complete baseline planning result — 16 November 2026

The ordinary L0 baseline with three autonomous balls, seed0, solves nearly all
controlled-ball endpoint tasks. All 3,000 cases have completed and passed
independent saved-trajectory verification and substantive literature review.
This is a positive result for the baseline and a failure to reproduce the
anticipated weak-baseline pattern.

| Horizon | Endpoint success | Episode-bootstrap 95% interval | Collision-free endpoint success |
|---|---:|---:|---:|
| 4 | 97.8% | 96.9–98.7% | 85.0% |
| 6 | 98.2% | 97.40–99.0% | 81.7% |
| 8 | 97.7% | 96.7–98.6% | 77.7% |

Each horizon uses the same fixed 1,000 test episodes. Endpoint success means
the controlled red ball finishes within one pixel of its goal. The five groups
of 200 are evaluation groups, not independent training seeds. Intervals use the
unchanged 2,000 episode bootstrap and omit training-seed uncertainty.

The original C002 reference criterion, L0 H4 success below 10%, is not met.
At 97.8%, the maximum possible O1 improvement on these fixed cases is 2.2
percentage points. The declared 30-point advantage therefore cannot be achieved
for n3 seed0, even by a perfect O1. This arithmetic bound does not fill the missing
O1 comparison cell or establish how O1 will perform. Keep the original criteria
and report this limitation; do not tune the benchmark to make L0 fail.

The [reference paper's sections 4.4–5.2](https://arxiv.org/html/2609.04264v3)
separate latent-goal planning from representation measurements and report a weak
n3 baseline, unlike this local result. Training, architecture, inference and
evaluation adaptations prevent an exact-reproduction claim. Their respective
contributions have not been isolated.

Our [reviewed probes](FINAL_BASELINE_PROBES_2026-11-16.md) already showed precise
red-state readout and weak white-state linear readout for L0n3. High endpoint
success is consistent with that selectivity: white balls pass through the red
ball without affecting its dynamics. This is an interpretation, not a causal
intervention. It shows why success on this endpoint objective cannot establish
good autonomous-object prediction or robot collision avoidance. The lower
collision-free rates are separately observed; collision checks are at discrete
frames and the planner has no additional collision cost.

The evaluation retains the original 100-candidate, 10-elite, 20-iteration CEM,
three-frame history, independently encoded goal and horizons 4/6/8. It uses the
reviewed final 303,000-update checkpoint and train-only calibrated float64/math
inference. No model training, calibration, planner tuning or task selection
occurred. Matching recorded/direct/true-model CEM controls reach 100% at each
horizon. The learned-only summary's `controls_passed=false` flag concerns absent
control arms; matching control results are separately bound and reviewed.

Independent inspection reconstructs all action sequences, checks all endpoint
and collision classifications, verifies task identities against controls and
labels, and reproduces every summary and bootstrap interval. Maximum endpoint
reconstruction discrepancy is 1.065e-5 pixels within the frozen tolerances; all
classifications agree. Saved singleton numerical arrays also pass. The initial
inspector assumed newer fields existed in legacy controls and halted; that
failure is preserved. Reviewed v2 handles the actual legacy schema and checks
goal indices and positions independently. The planning run was not repeated.

The CPU run exited normally after 1,604.78 seconds, with sampled peak RSS
1.567 GiB, no signals or errors and no GPU allocation. Source and input hashes
match; weights remain unchanged. [PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
limits these checks to their tested operations and platform; no cross-device
bitwise or latency superiority is claimed.

Evidence is under `outputs/gated/planning/l0_n3_s0`,
`outputs/gated/B003/operations/l0_n3_s0`, and
`outputs/gated/B003/inspection/v2/l0_n3_s0`, each with its own digest-bound review.
The next fixed comparison is L1n3 seed0. The GPU object encoder continues
unchanged; selected validation, O0/O1 dynamics, binding, P26, additional seeds
and the final evidence bundle remain incomplete. No object-model advantage or
Go2 transfer has been demonstrated.
