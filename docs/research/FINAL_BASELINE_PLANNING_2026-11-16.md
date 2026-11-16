# All six final baseline planning results — 16 November 2026

All six original seed0 models finished training, numerical validation, physical
probes and fixed planning evaluation. All 18,000 saved planning cases passed
independent reconstruction. Physical auxiliary supervision gives a modest
observed benefit at n1, trails the ordinary baseline at n3, and fails badly at n6.

Here n counts autonomous white balls in addition to one controlled red ball.
L0 is latent prediction alone; L1 adds the original physical auxiliary losses.

| White balls | Model | H4 endpoint success | H6 endpoint success | H8 endpoint success |
|---|---|---:|---:|---:|
| 1 | L0 | 97.6% | 98.5% | 98.5% |
| 1 | L1 | 99.7% | 99.7% | 99.4% |
| 3 | L0 | 97.8% | 98.2% | 97.7% |
| 3 | L1 | 94.7% | 94.2% | 94.7% |
| 6 | L0 | 97.1% | 93.5% | 82.6% |
| 6 | L1 | 2.9% | 1.8% | 1.0% |

Each row uses 1,000 matching episodes per horizon, with identical tasks and CEM
seeds within each count. All runs completed normally. Detailed intervals,
endpoint errors, collision diagnostics and evidence are in the
[n1](BASELINE_PLANNING_N1_2026-11-16.md),
[n3](BASELINE_PLANNING_N3_2026-11-16.md) and
[n6](BASELINE_PLANNING_N6_2026-11-16.md) reports. These are one training seed per
model/count. Episode confidence intervals omit seed variation, and cross-count
comparisons also change datasets. No new paired-significance analysis is claimed.

The most useful finding is the separation between accessible physical state and
task performance. At n1/n3, L0 plans well despite poor autonomous-ball linear
readout. The task permits balls to pass through each other, and the planner has
no collision cost. Accurate controlled-ball endpoint prediction can therefore
suffice. At n6, the earlier weak controlled-ball readout in L1 accompanies severe
planning failure. These observations do not isolate causality or establish robot
obstacle avoidance. The [probe report](FINAL_BASELINE_PROBES_2026-11-16.md)
preserves the measured readout/prediction tradeoffs.

[The reference paper's methods, results and probing appendix](https://arxiv.org/html/2609.04264v3)
were revisited for each completed result. Its weak-baseline pattern and consistent
auxiliary advantage have not reproduced locally. The documented local gradient
policy differs from its stability precaution; architecture, inference and
protocol differences are also unisolated. Preserve these contrary results and
complete the paired comparisons before proposing a mechanism or changing training.

The preregistered 30-point O1 H4 gain over L0 is mathematically unreachable on
these seed0 tasks: remaining headroom is only 2.4, 2.2 and 2.9 points for n1/n3/n6.
This is a benchmark ceiling, not an O1 result. Object-model physical prediction,
binding and planning remain unmeasured. The full C002 comparison has not yet
been regenerated; its earlier probes-only snapshot remains historical evidence.

The first object encoder and shared cache are complete. O0 completed 30 epochs
and passed the terminal-state audit, but its selected-weight numerical check
found four long-rollout grouping failures. See the
[O0 report](OBJECT_PREDICTOR_O0_2026-11-16.md). Diagnose that gate under a fresh
bounded literature-backed protocol, then complete paired O1, object evaluations,
offline Genesis P26 and approved replication. No Go2 transfer is yet demonstrated.
