# One-white-ball baseline planning — 16 November 2026

L0n1 seed0 completed all 1,000 test episodes at each horizon. Its endpoint
success is 97.6–98.5%, providing a second count with strong ordinary-baseline
planning. The matched L1n1 result is still pending, so there is no auxiliary
ranking for this count yet.

| Horizon | Endpoint success | 95% episode interval | Collision-free success | Median endpoint error, pixels |
|---|---:|---:|---:|---:|
| 4 | 97.6% | 96.6–98.5% | 92.8% | 0.04630 |
| 6 | 98.5% | 97.7–99.2% | 92.7% | 0.05619 |
| 8 | 98.5% | 97.7–99.2% | 91.2% | 0.06732 |

These are one trained seed and 1,000 episodes, with the frozen 2,000 episode
bootstrap. Evaluation groups are not independent training seeds. The n1 and
n3 cases differ; the two strong baselines do not establish seed robustness.

The [completed probes](FINAL_BASELINE_PROBES_2026-11-16.md) recovered L0n1's
controlled-ball position accurately, while its autonomous-ball linear readout
had correlations near zero. Strong endpoint success can coexist with this
selectivity because autonomous balls do not affect controlled-ball dynamics,
and the planner optimizes terminal latent distance without a collision cost.
The collision-free rates are discrete-frame diagnostics. Neither result
demonstrates active robot avoidance or proves all autonomous-ball information
is absent from the representation.

[Sections 4.3–5.2 and appendix A.10 of the reference paper](https://arxiv.org/html/2609.04264v3)
were revisited after completion and inspection. The reported weak n1 baseline
has not reproduced locally. Architecture, training, inference calibration and
evaluation adaptations remain unisolated explanations. The fixed test tasks,
planner and models were preserved. At H4, even a perfect O1 can gain only
2.4 percentage points, so the original 30-point target is unreachable on these
cases. This arithmetic bound does not fill any missing object-model result.

Independent reconstruction of all 3,000 saved action trajectories reproduces
every endpoint/collision flag, group summary and confidence interval. Maximum
endpoint discrepancy is 7.511e-6 pixels, within the original tolerance; minimum
classification-boundary margin is 0.0005027. Saved singleton numerical checks
pass, reversed grouping agrees, weights are unchanged and CUDA was not
initialized. [PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
limits these checks to the tested computations. The separately reviewed matched
recorded/direct/true-model controls reach 100% endpoint success.

The CPU job exited normally after 1,588.96 seconds with sampled peak RSS
1.5608 GiB and no timeout, signals, errors or GPU interval. Science, operation
and inspection reviews are respectively under `outputs/gated/planning/l0_n1_s0`,
`outputs/gated/B003/operations/l0_n1_s0`, and
`outputs/gated/B003/inspection/v2/l0_n1_s0`.

Next is the unchanged matched L1n1 evaluation, then L0n6 and L1n6. RGB encoder
training continues separately. Object dynamics, binding/probes/planning, P26,
additional seeds and the final reproducibility bundle remain required.
