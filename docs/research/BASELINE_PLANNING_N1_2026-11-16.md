# Completed one-white-ball planning pair — 16 November 2026

Both final seed0 models completed the same 1,000 test episodes at horizons
4, 6 and 8. Physical auxiliary supervision (L1) has a modest observed endpoint
advantage over the ordinary baseline (L0) at n1. This reverses the
[n3 ranking](BASELINE_PLANNING_N3_2026-11-16.md): the evidence does not support
an unconditional auxiliary benefit or disadvantage. All 6,000 saved cases have
passed independent scoring and substantive literature review.

| Horizon | L0 endpoint success | L1 endpoint success | L1 95% episode interval | L0 collision-free success | L1 collision-free success |
|---|---:|---:|---:|---:|---:|
| 4 | 97.6% | 99.7% | 99.3–100% | 92.8% | 94.9% |
| 6 | 98.5% | 99.7% | 99.3–100% | 92.7% | 93.9% |
| 8 | 98.5% | 99.4% | 98.9–99.8% | 91.2% | 91.9% |

L1's endpoint advantages are 2.1, 1.2 and 0.9 percentage points on matching
tasks and CEM seeds. These are observed differences, without a paired-difference
significance claim. Each model has one training seed; the fixed 2,000 episode
bootstrap omits training-seed uncertainty. Evaluation groups are not additional
trained seeds. L0's intervals remain 96.6–98.5%, 97.7–99.2% and 97.7–99.2%.

L1 has lower mean endpoint error at every horizon, but larger median errors:
0.0722/0.0960/0.1077 pixels versus L0's 0.0463/0.0562/0.0673. Better threshold
success therefore does not mean uniformly better positional precision.
The [completed probes](FINAL_BASELINE_PROBES_2026-11-16.md) show much better
autonomous-ball readout for L1n1. No intervention isolates whether that readout
improvement caused its planning advantage.

Both models plan well despite L0's weak autonomous-ball linear readout. The
autonomous balls do not affect controlled-ball dynamics, and the planner
optimizes terminal latent distance without a collision cost. Collision-free
rates are discrete-frame diagnostics, not evidence of active avoidance or
Go2 transfer. Weak linear readout also does not prove information is absent.

[Sections 4.3–5.2 and appendix A.10 of the reference paper](https://arxiv.org/html/2609.04264v3)
were revisited after completion, with methods and probing revisited again after
independent inspection. Its weak-baseline pattern has not reproduced locally.
Architecture, auxiliary-head update, training, inference calibration and
evaluation differences remain unisolated explanations. The local head receives
predicted-branch gradients, unlike the paper's stability precaution. No model,
planner or test protocol was changed in response to these results.

At H4, even a perfect O1 can improve only 2.4 points over L0n1; the original
30-point target is unreachable on these cases. This ceiling bound does not
populate missing object-model results. Those comparisons remain required.

The fixed protocol uses final 303,000-update checkpoints, train-only calibrated
float64/math inference, three history frames, independently encoded goals and
100-candidate/10-elite/20-iteration CEM. Matched recorded, direct and true-model
controls achieve 100% endpoint success. The learned-only `controls_passed=false`
field concerns absent control arms in that output; controls are separately reviewed.

Independent reconstruction reproduces every L1 trajectory classification, group
summary and confidence interval. Maximum endpoint discrepancy is 1.022e-5 pixels,
within the fixed combined relative/absolute tolerance; minimum classification
margin is 0.0007389. Saved singleton checks pass, weights are unchanged and CUDA
was not initialized. [PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
limits these checks to tested computations, without cross-platform equality claims.

L1 exited normally after 1,603.30 seconds, sampled peak RSS 1.5616 GiB, with no
timeout, signals, errors or GPU interval. Separate science, operation and
inspection reviews are under `outputs/gated/planning/l1_n1_s0`,
`outputs/gated/B003/operations/l1_n1_s0`, and
`outputs/gated/B003/inspection/v2/l1_n1_s0`. Earlier L0 evidence is retained.

Next are unchanged L0n6 and L1n6 planning. RGB encoder training continues
separately. Object dynamics, binding/probes/planning, offline Genesis P26,
additional seeds/counts and the final reproducibility bundle remain required.
