# Preliminary checkpoint analysis — 15 November 2026

The checkpoints support a useful preliminary finding: auxiliary-supervised L1 has
latent geometry associated with both the controlled red ball and the moving white
balls. Baseline L0 has a modest red-ball association and little association with
the white-ball set in this diagnostic. This supports further evaluation of L1's
physical representation; it does not establish better prediction or navigation.

The analysis follows [A007](records/A007.json) and [A008](records/A008.json), with
post-result literature reviews beside the exact result digests. It uses preserved
intermediate checkpoints and validation scenes. No original learned weights,
training configuration or test data were changed.

At matched step 40,000, n=3, seed 0:

| Measurement | L0 baseline | L1 auxiliary supervision |
|---|---:|---:|
| Latent-distance / red-position-distance Spearman correlation | 0.242 | 0.308 |
| Latent-distance / matched white-set-distance Spearman correlation | −0.007 | 0.291 |
| Mean per-coordinate latent standard deviation | 0.977 | 0.979 |

These are 500 observations from 100 validation episodes and 2,000 cross-episode
pairs, shared between models. White-ball distance uses optimal permutation
matching. Pairs share observations, so they are not independent replicates; no
significance test or confidence interval is claimed. The representations are not
constant on this sample, but the standard deviations do not establish full rank.
Near-zero geometry correlation does not prove that information is absent or
cannot be decoded nonlinearly.

This CPU reference uses the exact train-calibrated A001 artifacts whose float64
CPU/GPU validity passed A002. Actual batch/single/reversed grouping checks pass;
checkpoint bytes and model state are unchanged; GPU initialization remained off.
Results and plot:
`outputs/gated/A008/geometry/{result.json,review.json,latent_geometry.png,latent_geometry.pdf}`.
The generating script and sampled arrays are retained under `outputs/gated`.

The result differs from the controlled-ball-blind baseline described in
[the reference paper, section 3](https://arxiv.org/html/2609.04264v3#S3).
Our L0 already has measurable red-ball sensitivity. We should therefore test its
planning performance without presuming failure. Auxiliary supervision is a
plausible explanation for L1's broader geometry, but this one checkpoint pair
does not establish a causal advantage. The paper's Appendix A.10 also separates
linear from nonlinear physical-state accessibility.

That distinction matters for this repository: balls pass through one another,
red motion is independent of white motion, and the primary planning metric is
red endpoint success. A model can be useful for that narrow task while poorly
representing environmental objects. Keep endpoint success, collision-free
success and physical-state fidelity separate. None yet establishes suitability
for closed-loop robot navigation.

The learning curves show decreasing training prediction error in all six runs;
all recorded loss/gradient values in the preserved streams are finite. At step
100,000, L1's raw validation/train prediction-error ratios are approximately
16.3, 4.18 and 1.18 for n=1,3,6. The training denominator is the median over
95,000–100,000 updates. These are descriptive gaps, not proof of overfitting.

Earlier A001 measurements reduced validation error by roughly 3–26 times using
train-only BatchNorm calibration with learned weights frozen. Inference moments
are therefore a demonstrated confound. This follows the distinction between
training batch statistics and fixed inference statistics in
[Batch Normalization, section 3.1 and Algorithm 2](https://arxiv.org/pdf/1502.03167v3).
The benefit has not been established at every later checkpoint. Absolute latent
MSE also cannot rank architectures with different learned targets and losses.
The preserved curves are in `outputs/gated/A007/curves/learning_curves.png` and
`.pdf`, with their result and review alongside them.

The source audit found a material reproduction difference: original
`go2nav/training/balls.py:terms` updates the auxiliary head from both encoded- and
predicted-latent losses. The reference paper's section 4.3 and Table 10 freeze
head parameters on the predictor-side branch. Our predictor width also differs.
Existing L1 should be described as a local adaptation. Preserve its experiment;
a faithful-gradient comparison would need a separate preregistration and scope
decision. There is no evidence tying this difference to the runtime stalls.

The additional A007 GPU numerical/probe diagnostic returned no completed case.
It was stopped after 912.8 seconds against its 900-second limit; the 12.8-second
enforcement overrun is recorded and charged. Its review is `budget_incomplete`.
No new linear-probe or action-control scores exist, and this outcome is not a
scientific negative. The unexecuted probe implementation has three passing
helper tests, which do not validate GPU execution.

During the closing runtime check, all four remaining original processes were
live but their logs were stale at 159,500 / 159,500 / 160,000 / 157,500 updates.
The last timestamps are near the A007 launch; causality is unresolved. A009
preserves their latest complete, finite step-155,000 checkpoints and read-only
observations. The earlier two interrupted L1 checkpoints remain preserved at
105,000. The four remaining originals have not been signalled. Treat the older
"four advancing trainers" status as historical, pending recovery evidence.

A010 subsequently restored all four saved model/head, optimizer, scheduler and
CPU RNG states exactly under original source, with GPUs hidden. Isolated recovery
copies and commands are ready in `outputs/gated/A010/continuation_plan.json`.
This verifies CPU restoration, not successful GPU continuation. The user subsequently
approved the exception: all four additional originals exited on SIGTERM and cleanup
passed review. The first serial L1n1 recovery has launched, but no new metrics are
yet observed. A011 records denied stack reads and a request for user stack evidence.

Next, resolve the runtime stall and complete the authorized original runs,
then apply the common final inference checks. The discriminating scientific
tests remain physical-state readouts, factual versus ablated-action prediction,
and paired learned planning under the fixed protocol. Only then compare L0/L1
with O0/O1 and proceed to the frozen-Genesis comparison. Preserve baseline
success if observed, and retain every incomplete or negative outcome.
