# Deployed object inference validation — 16 November 2026

The selected n3 seed0 encoder, O0 and O1 now pass their complete C014 numerical
suites through the repository full64 loader. Each actual result has a separate
independent inspection, scientific literature review and operational review.
Training weights and the shared float32 cache remain unchanged.

| Selected model | Declared comparisons | Result | Supervised elapsed seconds |
|---|---:|---|---:|
| RGB encoder | 1,104 | All pass | 30.579 |
| O0 | 2,772 | All pass | 55.655 |
| O1 | 4,164 | All pass | 57.725 |

Both predictors were checked on the original 32 train/validation windows and
two 100-candidate, eight-step planning cohorts. O1 includes another 1,392
fixed-mask output, gradient and grouping comparisons. Each predictor's independent
CPU inspection reconstructs 84 summaries from 32 saved arrays and enumerates
1,000 assignment costs. Maximum independent cost differences are 1.137e-13 for
O0 and 1.421e-14 for O1. Raw small-batch outputs and gradients were not retained;
their coverage and reported summaries are checked, without independent tensor
recomputation. The encoder has no retained planning arrays.

Maximum CPU/GPU rollout differences are 2.841e-7 for O0 and 4.489e-8 for O1.
All checks retain rtol 1e-4 and atol 1e-5, with the original reference operand.
Repeated rollouts are exact. Full64 addresses the tested inference inconsistency;
it does not establish physical prediction accuracy. Smaller numerical differences
for O1 do not demonstrate a better learned model. The original C004 failures and
C013 rejected mixed policy remain preserved.

[PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
was revisited for batch/platform differences, together with
[NumPy's asymmetric tolerance](https://numpy.org/doc/2.4/reference/generated/numpy.isclose.html)
and [the assignment objective](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.linear_sum_assignment.html).
The [Causal-JEPA objective, inference and planning appendices](https://arxiv.org/html/2602.11389v2)
support separate masked training and unmasked inference checks. They do not
establish interaction learning in our independent pass-through balls.

Four loader software groups and eight numerical/supervision software groups
have passed and received exact-result reviews. The first numerical software
attempt caught an incorrect expected output shape in a test fixture; it and
its review remain under `software_numerics_attempt_001`. A system-Python
invocation lacking Torch failed before GPU admission and is preserved under
`launch_preflight_o0_system_python`; the corrected invocation uses the research
interpreter for both supervisor and child. Neither required changing the model.

Actual evidence is under `outputs/gated/C014/{slots_n3_s0,o0_n3_s0,o1_n3_s0}`
and `outputs/gated/C014/operations/RUN`, using runtime b5a0d47. The C014 record
and six numerical helper sources remain unchanged throughout all three runs.
At numerical completion all dedicated-device intervals were closed; 43.321 of the original 48 additional
GPU-hours remained, including 23.691 B/C hours. No trainer or evaluator was active.

C009v2 supervision and routing subsequently passed 16 software groups and their
review. The first [object binding result](OBJECT_BINDING_N3_2026-11-16.md) is a
reviewed scientific negative, with passing independent inspection and operation.
Physical probes and planning remain under the unchanged protocols. P26 offline
Genesis, additional seeds/counts, the full comparison and final archive follow.
The object-model advantage and Go2 transfer remain unmeasured.
