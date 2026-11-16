# First object-mask binding result — 16 November 2026

The selected n3 seed0 RGB encoder has no successful object binding under the
preregistered per-frame mask criterion: none of 20,000 ball appearances achieves
best mask IoU at least 0.5. The C002 scientific review is `measured_negative`.
The evaluation, independent inspection and operational checks completed normally.

| Subset | Ball appearances | Binding share | Mean best IoU | Best slot reused |
|---|---:|---:|---:|---:|
| All | 20,000 | 0% | 0.02238 | 54.05% |
| Controlled red ball | 5,000 | 0% | 0.02215 | 60.96% |
| Autonomous white balls | 15,000 | 0% | 0.02245 | 51.75% |
| At least 90% visible | 19,354 | 0% | 0.02241 | 53.10% |

These are five deterministic three-frame history windows from each of 1,000
test episodes. Each current-frame geometric disc is compared with every hard
decoder mask, including the complete predicted mask area in the IoU union.
No label-dependent mask threshold, checkpoint selection or fitting is performed.
The same selected encoder and immutable feature cache are shared by O0 and O1.

The fully visible red ball also has zero binding, and the mostly visible subset
retains the negative. Occlusion alone therefore cannot account for the absence
of binding. The measured masks do not support treating slots as isolated physical
objects in this encoder. This does not prove that the latent vectors lack all
position information: information could be distributed across slots, or encoded
in ways a mask does not reveal. It also does not measure tracking consistency.

[VideoSAUR section 4 and Appendix A](https://arxiv.org/pdf/2306.04829v2) distinguish
image mask fit from video consistency and include background area in best-overlap
scoring. Appendix B.4 reports non-object groupings under reconstruction in its
mixer/feature setting. That is a possible explanation to investigate, not an
identified mechanism here: this experiment uses an RGB mixture decoder and a
local deterministic SAVi-style recipe. The [Causal-JEPA encoder and inference
description](https://arxiv.org/html/2602.11389v2) was also revisited. Its object
masking rationale depends on the supplied representation; our current evidence
does not justify calling masked slot histories isolated object histories.

All 2,000 episode-bootstrap replicates give a zero observed binding share.
The resulting [0, 0] empirical intervals do not establish zero population success
probability, and they omit training-seed variation. This is one selected encoder.
No causal attribution to architecture, optimization, precision or data is made.

The independent inspector verifies all 20,000 sample/frame/ball identities,
threshold flags, slot reuse, geometric visible fractions, subgroup summaries and
episode-bootstrap intervals. Its eight software groups cover all three object
counts, perfect/merged/occluded masks and deliberate corruptions. Decoder masks
were not retained by the original evaluator, so the inspector does not independently
recompute raw IoUs. This limitation remains in the actual inspection and review.

C009v2 now has 16 passing software groups and a separate C014 review. It admits
the actual selected encoder and both predictors, rejects changed numerical,
inspection, operation, source, cache, data and control evidence, and preserves all
21 original scientific commands except their explicit C014 review argument.
The binding operation took 105.007 seconds without errors or signals; both
owned GPU intervals are closed and union-accounted. After this result, 43.292
additional GPU-hours and 23.662 B/C hours remain.

Evidence is under `outputs/gated/binding/slots_n3_s0` and
`outputs/gated/C009/v2/runs/binding/slots_n3_s0`, with runtime a492dbe. The
independent inspector and its software evidence are under `outputs/gated/C009/v2`.
All numerical evidence, original failures, selected weights and cache remain intact.

Next are the unchanged O0/O1 current-state and four-step physical probes and paired
planning. These discriminate poor mask binding from poor predictive utility. Keep
this negative in the full comparison; do not tune the encoder against the test.
P26 uses its separately frozen camera encoder and remains a distinct experiment.
Approved replications and the final evidence archive remain required. No Go2
transfer has been demonstrated.
