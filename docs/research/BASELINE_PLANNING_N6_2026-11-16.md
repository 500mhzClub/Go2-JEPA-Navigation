# Six-white-ball baseline planning — 16 November 2026

L0 seed0 completed all1,000 test episodes at horizons4/6/8, and all3,000 saved
cases passed independent inspection and literature review. L1 remains pending.

| Horizon | L0 endpoint success | 95% episode interval | Collision-free endpoint success | Median endpoint error, pixels |
|---|---:|---:|---:|---:|
| 4 | 97.1% | 96.0–98.1% | 71.2% | 0.2632 |
| 6 | 93.5% | 91.8975–95.0025% | 62.6% | 0.3878 |
| 8 | 82.6% | 80.3–84.9% | 52.1% | 0.5029 |

Longer-horizon endpoint performance is weaker in this case, while H4 remains
strong. Goals differ according to the frozen horizon-specific tasks, so this
is descriptive rather than an isolated causal test of horizon. One training
seed and the fixed2,000 episode bootstrap do not establish seed robustness.
L1's poor n6 readout does not determine its still-unmeasured planning result.

The gap between endpoint and collision-free success is substantial. Balls pass
through one another and the planner has no collision cost. These discrete-frame
diagnostics do not establish active avoidance or robot transfer. Recorded,
direct and true-CEM controls all achieve100% endpoint success on matching tasks.

[Sections4.3–5.2 and appendixA.10 of the reference paper](https://arxiv.org/html/2609.04264v3)
were revisited after completion and inspection. Its near-zero baseline planning
pattern has not reproduced locally. Architecture, training, inference calibration
and evaluation differences remain unisolated explanations. Weak linear readout
does not prove information is absent; no model or planner was retuned.

Even perfect O1 performance can improve H4 by only2.9percentage points here.
Together with the reviewed n1/n3 results, the original30-point gain target is
unreachable on all three seed0 test sets. This bound is not an object-model
measurement; O0/O1 evaluations remain necessary.

The original303,000-update checkpoint, train-only calibrated float64/math
inference and fixed100-candidate/10-elite/20-iteration CEM were retained.
Independent reconstruction reproduces every task, classification, group summary
and interval. Maximum endpoint discrepancy is9.5851e-6pixels, within the fixed
tolerance; the smallest classification margin is0.000832974pixels. Saved
singleton checks pass, weights are unchanged and CUDA was not initialized.

The CPU case exited normally after1,393.084seconds, with sampled peak RSS
1.56945GiB and no signals or errors. Concurrent authorized GPU activity changed
the shared ledger; this CPU run created no GPU interval. Separate reviews are
under `outputs/gated/planning/l0_n6_s0`, `outputs/gated/B003/operations/l0_n6_s0`
and `outputs/gated/B003/inspection/v2/l0_n6_s0`.

Next is the unchanged paired L1n6 evaluation, followed by completion of object
binding/dynamics/planning, offline Genesis P26, further seeds/counts and the
final reproducibility bundle, within the approved allowances.
