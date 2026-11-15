# CPU checkpoint comparison — L1 n1, 105,000 to 220,000 steps

The later checkpoint recovers positions more accurately and improves four-step
physical prediction on the same validation scenes. This supports continued
evaluation of the current L1 experiment. It does not yet establish planning
success: Euclidean latent-distance correlation weakened despite stronger readout.

This is [A017](records/A017.json), requested while GPU training continued. The
comparison uses the preserved 105,000-step checkpoint and one latest complete
snapshot selected by availability at 220,000 steps. Both were copied through open file descriptors;
their source hashes, metric histories and data/source identity are retained under
`outputs/gated/A017/snapshots`. Training was neither stopped nor modified.

The two CPU phases took 37.20 and 8.32 seconds including supervision, used eight
threads and peaked below 1.74 GB RSS. No GPU context was initialized and the shared
GPU ledger was unchanged. Five helper checks passed; actual numerical grouping,
gradient and readout checks also passed. Primary literature was consulted before
implementation and after each result; exact-result reviews are beside the artifacts.

The physical measurements use 4,096 identical training samples to fit separate frozen
linear readouts, and 500 identical validation scenes from 100 episodes. Both models
first receive the same train-only population-statistics calibration. Position RMSE
below is the square root of mean squared Euclidean XY error, in pixels.

| Measurement | 105,000 steps | 220,000 steps |
|---|---:|---:|
| Current red position RMSE | 1.788 | 1.426 |
| Current white position RMSE | 2.537 | 2.440 |
| Four-step red prediction RMSE | 2.316 | 1.847 |
| Four-step white prediction RMSE | 5.942 | 4.537 |
| Current red mean coordinate Pearson | 0.9975 | 0.9984 |
| Current white mean coordinate Pearson | 0.9944 | 0.9948 |
| Four-step red mean coordinate Pearson | 0.9959 | 0.9976 |
| Four-step white mean coordinate Pearson | 0.9697 | 0.9822 |

The four-step RMSE improvement is about 20% for red and 24% for white. Paired
mean squared-position-error improvements are 1.950 pixels² for red, with a 95%
episode-bootstrap interval [1.347, 2.572], and 14.718 pixels² for white,
[11.809, 17.707]. These intervals resample whole episodes, keeping their windows
together. They describe uncertainty conditional on these two checkpoints and this
validation corpus; they do not include training-seed uncertainty or adjust for
multiple comparisons.

The controls support meaningful prediction on these samples. At 220,000 steps, retaining
the current latent without rollout gives red/white future RMSE 5.237/7.998 pixels,
versus 1.847/4.537 for factual rollout. Setting only future actions to zero increases
red error to 5.349 pixels. The paired red squared-error benefit of factual actions is
25.196 pixels² [22.278, 28.430]. White zero-action error is 4.640 pixels; its small
paired difference from factual rollout has an interval crossing zero. That is
consistent with the simulator's autonomous white motion, without proving perfect
separation of physical factors. Reversing actions is a weak control because it can
preserve net displacement away from boundaries; all reverse-control results are
retained in the raw report.

Encoding the actual future image, solely as a readout reference, gives red/white
RMSE 1.452/2.579 pixels at 220,000 steps. Those images never enter rollout. The larger
rollout errors suggest remaining dynamics error, particularly for white motion;
this reference is not a mathematical lower bound or a planning result.

The geometry result is mixed:

| Latent-distance Spearman correlation | 105,000 steps | 220,000 steps |
|---|---:|---:|
| Red position distance | 0.176 | 0.145 |
| White position distance | 0.309 | 0.292 |
| Mean coordinate latent SD | 0.982 | 0.985 |

These are the same 2,000 cross-episode pairs from the 500 scenes. Pairs share
endpoints, so no significance or independent-pair interval is claimed. Strong
linear position readout coexisting with modest, declining distance correlation
shows why metric geometry and accessible information must be kept separate.
The reference paper's [section 3 and Appendix A.10](https://arxiv.org/html/2609.04264v3)
motivate these distinct diagnostics. Here, the unresolved question is whether
the latent Euclidean objective used by CEM is sufficiently aligned with the
physical goal. Only the planned fixed-protocol planning evaluation can answer it.

The optimizer counters verify 115,000 resumed updates: all populated counters are
105,000 in the input and 220,000 in the snapshot, with the expected scheduler cursor
and learning rate. Saved model/head/optimizer state is finite. CPU reference
gradients are finite and all tested grouping differences are below 9e-14. Learned
weights remain unchanged by analysis. This does not establish later-checkpoint GPU
parity or certify every unlogged intermediate gradient.

BatchNorm remains a substantial inference confound:

| Fixed 64-window validation prediction MSE | 105,000 steps | 220,000 steps |
|---|---:|---:|
| Saved statistics | 0.123966 | 0.024362 |
| Train-only calibrated statistics | 0.005860 | 0.004138 |
| Reduction from calibration | 21.16× | 5.89× |

The later calibrated MSE is 29.4% lower in this small fixed diagnostic. Calibration
uses 4,096 training windows, with encoder moments established before predictor
moments, following the frozen-weight inference principle in
[Batch Normalization, section 3.1 and Algorithm 2](https://arxiv.org/pdf/1502.03167v3).
This intervention changes normalization and encoded targets, so latent MSE alone
cannot establish physical prediction quality. The position probes provide the
separate evidence above. The original logged validation/train prediction ratio
also narrows from 16.17 at 100,000 steps to 3.32 at 220,000 steps, but those logs use
different samples from the 64-window diagnostic and do not isolate overfitting.

This remains a single-seed, within-L1 temporal comparison on a previously inspected
validation corpus. It is not a comparison with L0, a blinded test, or evidence that
rebooting improves the model. The readout uses 4,096 samples and ridge 0.001 rather
than the paper's 50,000 and 1e-6. Original L1 also retains its disclosed auxiliary-head
gradient-routing and predictor-width deviations. CPU float64 reference checks do
not replace the later common GPU inference gate, as
[PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
distinguishes platform and grouping behavior.

Continue the unchanged trainer to 303,000 steps, validate the full artifacts, finish the
other five serial continuations, and run A003. Then compare physical probes and
paired CEM planning at final checkpoints. These results give no basis for tuning
the original training or planner from validation scores, or skipping the approved
O0/O1 and frozen-Genesis comparisons.

Artifacts are under `outputs/gated/A017`: `calibration/{result,review}.json`,
`probes/{result,review}.json`, frozen derived checkpoints, fitted readouts and raw
sample/prediction arrays. Three standalone figures are available as PNG and PDF:

- [Physical prediction progress](../../outputs/gated/A017/plots_final/physical_progress.png)
- [Calibration and learning curves](../../outputs/gated/A017/plots_final/calibration_and_curves.png)
- [Latent geometry](../../outputs/gated/A017/plots_final/latent_geometry.png)
