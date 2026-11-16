# First object-predictor physical probes — 16 November 2026

O0 n3 seed0 completed the fixed test normally. Independent CPU inspection
reproduces every reported statistic exactly; its scientific review is
`measured_negative`. The current representation contains some linearly accessible
position information, but four-step predictions are worse than holding the
current readout. O1 physical probes and both planning evaluations remain pending.

| Physical readout | Red position RMSE, pixels | White position RMSE, pixels | Red mean Pearson | White mean Pearson |
|---|---:|---:|---:|---:|
| Current | 21.893 | 13.669 | 0.490 | 0.779 |
| Four-step prediction | 78.807 | 24.987 | 0.064 | 0.501 |
| Four-step action reversal | 78.815 | 24.998 | 0.064 | 0.500 |
| Hold current estimate for future | 21.990 | 15.014 | 0.479 | 0.710 |

RMSE is Euclidean position error, pooled across windows and the corresponding
balls. White targets are sorted positional ranks at each frame, not tracked
identities. The fixed readout fits 4,096 training windows and evaluates 5,000
windows from 1,000 test episodes. Its unconstrained predictions can exceed the
64-pixel image extent. Four-step red x/y correlations are 0.0242/0.1045, with
95% episode-bootstrap intervals [-0.0107, 0.0606]/[0.0717, 0.1363].

The [physical-probing appendix](https://arxiv.org/html/2609.04264v3) distinguishes
linear accessibility from nonlinear encoding. Current white readout therefore
qualifies the [mask-binding negative](OBJECT_BINDING_N3_2026-11-16.md): failed
object-aligned masks do not mean the latents contain no scene information.
The local fit size and ridge penalty differ from that paper. No nonlinear probe
or additional fitting has been performed.

The [Causal-JEPA objective and inference method](https://arxiv.org/html/2602.11389v2)
were revisited after the result. O1 adds anchored masking and history recovery
to a predictor sharing O0's frozen encoder/cache. Its 3.89% lower logged unmasked
validation loss has not yet established useful physical prediction. Both models
passed their separate deployed full64 numerical admission checks.

Possible explanations include poor spatial grouping, distributed or nonlinear
state, recursive drift, and the documented full-sequence cached-history versus
fresh-three-frame inference difference. These are hypotheses, not isolated
causes. Action reversal preserves total displacement, so similar scores do not
establish that actions are ignored. Planning and Go2 transfer remain unmeasured.
These results cover one training seed; episode intervals omit seed variation.

The inspector verifies the exact sample/target cohort, all four modes, saved
readout dimensions/finite parameters/checkpoint binding, source/input identity,
closed GPU intervals and all 2,000 episode-bootstrap replicates. Its eleven
software groups cover float32 and float64 targets, all counts, unequal clusters,
analytic cases, corruption refusals and a 5,000-row mixed-dtype fixture. An initial
float64-only coverage gap was preserved and reviewed before correction. The
scientific evaluator, weights, data and tolerances were unchanged. Unsaved training
features prevent independent ridge-fit residual or coefficient reconstruction;
unsaved latent features prevent independent model-output reconstruction.

O0 evaluation took 207.327 seconds; inspection took 5.345 seconds on CPU. All
owned GPU intervals are closed. At completion, 43.235 additional GPU-hours and
23.605 B/C hours remain. Evidence is under `outputs/gated/probes/o0_n3_s0`,
`outputs/gated/C009/v2/runs/probes/o0_n3_s0` and the C009v2 inspector directory.
The operation and inspection reviews pass separately from the scientific negative.

Next: unchanged O1 probes, paired O0/O1 planning, offline Genesis P26, approved
additional seeds/counts, complete comparison and final reproducibility archive.
