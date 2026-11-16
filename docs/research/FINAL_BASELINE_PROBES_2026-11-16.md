# All six final baseline probes — 16 November 2026

All six seed0 L0/L1 physical probes are complete and reviewed. Physical supervision
helps recover autonomous-ball state at n1 and n3, with worse controlled-ball
precision. At n6, L1 is substantially worse for the controlled ball and has no
aggregate white-ball prediction advantage. This is a valid measured negative,
not a failed execution. L0 also retains white-ball information at n6, so its
lower-count selectivity cannot be generalized to every count.

Here n counts autonomous white balls, in addition to one controlled red ball.
L0 is the latent-prediction baseline; L1 adds the original local physical
auxiliary supervision. Every source model finished 303,000 updates and passed
the A015/A003 final checks before these probes.

| n | Model | Current red RMSE | Current white RMSE | Four-step red RMSE | Four-step white RMSE |
|---|---|---:|---:|---:|---:|
| 1 | L0 | 0.833 | 23.957 | 1.011 | 23.842 |
| 1 | L1 | 1.400 | 2.325 | 1.682 | 4.223 |
| 3 | L0 | 0.704 | 20.782 | 0.873 | 20.762 |
| 3 | L1 | 2.682 | 12.718 | 3.019 | 13.638 |
| 6 | L0 | 5.924 | 14.849 | 6.092 | 15.167 |
| 6 | L1 | 20.071 | 14.811 | 21.099 | 15.389 |

Errors are pixels: square root of mean squared Euclidean XY error, averaging
over white balls before taking the square root. White targets are sorted
lexicographically at each time. They represent positional ranks, not persistent
identities; their distributions change with n, so cross-count errors are not
controlled causal comparisons.

At n1, L1 makes both current positions highly accessible: mean coordinate
correlations are 0.9984 red and 0.9952 white. Its future errors of 1.682/4.223
pixels beat holding the current latent at 5.292/7.913 pixels. L0's current and
future white correlations are near zero, with all intervals spanning zero.
At n3 the same broad tradeoff holds, but L1's future white error of 13.638 pixels
only modestly improves on holding the current latent at 14.059 pixels. The
[n3 report](FINAL_PROBES_N3_2026-11-16.md) preserves the detailed earlier pair.

At n6, L0's current/future white mean correlations are 0.6857/0.6632, with all
coordinate intervals above zero. L1's corresponding values are 0.6722/0.6354.
L1's red mean correlations fall to 0.5854 currently and 0.5331 after rollout,
versus L0's 0.9715/0.9700. L1's red rollout error also exceeds holding its own
current latent: 21.099 versus 20.326 pixels. The failure is already visible in
current-state readout, so future prediction alone cannot explain it.

The unchanged C002 comparison audit accepts all six reviewed probe artifacts
and retains all 18 planned model/count/seed rows. Its available reference check
expects n3 L0 red mean Pearson below 0.5; the measured value is 0.999603, so that
reference pattern is **not met**. The other 24 numerical bars remain incomplete.
None of the learned planning, object-model probe or binding evidence is present
yet. In particular, the 0.9 current/0.8 future-white thresholds belong to the
planned n3 O1 comparison; their absence cannot be converted into a pass or failure.

The literature review was performed before every case and revisited after each
result. [Appendix A.10 and sections 4.4–5.1 of the reference paper](https://arxiv.org/html/2609.04264v3)
motivate separate assessments of accessible state, geometry and planning. A weak
linear probe can miss nonlinear information. The paper reports a different
baseline object emphasis and an n6 benefit from supervision, so these local
results must not be described as an exact reproduction or universal L1 advantage.

One candidate explanation deserves preserving without claiming causality:
the paper's section 4.3 excludes predictor-side loss from auxiliary-head updates
as a stability precaution. Original local L1 updates that head from both encoded
and predicted branches (`go2nav/training/balls.py`, `terms`). Predictor width also
differs. Neither deviation was isolated by these experiments. Seed variation,
optimization and probe limitations remain unresolved; no correction or retuning
was made after seeing test results. Any future mechanistic intervention needs
its own prior literature gate and an explicit comparison protocol.

C006 preserves the C001 computation: 4,096 train windows for a standardized
ridge readout with alpha 0.001, then 5,000 test windows from 1,000 episodes,
four-step rollout, future-action reversal and held-current controls. Each pair
has exactly matching sample indices and targets. No future image enters rollout,
and no test label selects weights, coefficients or checkpoints. The paper uses
50,000 probe-training samples and ridge 1e-6. Reversal often preserves net
displacement and remains a weak action-order control.

Per-coordinate intervals use 2,000 whole-episode bootstrap samples; they omit
training-seed uncertainty. No paired-difference significance or temporal
improvement against A017's different validation cases is claimed. CPU float64
execution is an explicit adaptation. All cases pass finite, grouping and
solver checks: maximum grouping discrepancy 1.435e-13 and maximum ridge residual
7.234e-16. Independent raw-case calculations reproduce point correlations and
coordinate errors. Saved input/artifact hashes match and learned weights are
unchanged. [PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
limits these checks to their tested platforms, samples and operations.

The six CPU cases used 136.34 seconds total, peaked below 1.678 GiB sampled RSS,
exited normally without signals and initialized no CUDA context or GPU ledger
interval. Scientific reviews are under `outputs/gated/probes/RUN`; separate
operational reviews are under `outputs/gated/C006/supervision/RUN`. L1n6 is
`measured_negative`; the other five scientific reviews and all six operational
reviews pass. Source and preregistration bindings are retained in C006's software
result and each manifest. The inspected partial C002 audit and its review are
under `outputs/gated/C002/final_baseline_probes`.

Next are the unchanged RGB-slot encoder, selected-checkpoint validation, shared
cache, O0/O1 dynamics, object binding and fixed paired planning. The encoder is
still running; its first epoch is complete and a checkpoint verifies 4,269 updates
of 23,752. P26, additional seeds/counts and the final reproducibility bundle remain
required. There is no established object-model advantage, planning improvement
or Go2 transfer yet.
