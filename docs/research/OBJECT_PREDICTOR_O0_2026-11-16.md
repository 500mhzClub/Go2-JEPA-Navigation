# First object predictor: training complete, precision diagnosis reviewed — 16 November 2026

The n3 seed0 O0 model completed its fixed 30 epochs and 11,370 updates normally.
All 83 Adam parameter states, scheduler counters, original source/data/cache
bindings and strict CPU restoration pass the terminal audit. The first strict
validation minimum is the final trained epoch29; selected and final weights
match exactly. Future latent MSE is 0.0041897866. This is objective optimization
and artifact consistency, not physical prediction or planning evidence.

O0 uses object slots with unmasked history; paired O1 will add anchored history
masking on the identical frozen encoder/cache and seed/order policy.
[Causal-JEPA sections 4.1/4.2 and appendix E](https://arxiv.org/html/2602.11389v2)
were revisited around training and audit. The local RGB adaptation and independent
balls differ from the published environments. Masking benefit is unmeasured.

Training took 1,001.526 seconds and exited0 without timeout, signals or errors.
C008 audit, C001 training and C005v2 operation reviews pass separately.
Historical runtime commit a75523b and immutable checkpoints are retained.

The actual selected-weight C004 check ran next and found a numerical failure:

| Comparison | Count | Outcome |
|---|---:|---|
| Small-batch CPU/GPU outputs | 16 | All pass |
| Parameter gradients | 1,328 | All pass |
| Small-batch repeat/reversal/separate grouping | 48 | All pass |
| 100-candidate, eight-step rollout grouping | 6 | Two exact repeats pass; four regroupings fail |

There are 32 real train/validation windows and 83 gradients per paired batch.
Both candidate reversal and batches-of-two fail on both splits at the unchanged
relative1e-4/absolute1e-5 tolerance. Maximum rollout discrepancy is 7.3284e-5 latent
units; maximum RMS is 2.713e-6. Absolute maxima alone do not identify failing
relative-error elements. Predictor, checkpoint and encoder remain unchanged.

Independent saved-result inspection verifies all 1,398 comparison summaries,
sampling, original source/input identities and closed intervals. It does not
recompute raw CPU/GPU tensors. C004 is reviewed technical_blocked; C007 operation
passes because the child exited normally after5.586seconds without errors or
signals. Exit0 does not override the failed numerical checks.

[PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
was revisited after the result. Different batching can change floating-point
arithmetic. Accumulation across rollout and backend choice are plausible causes,
but no diagnostic has isolated them. The guidance does not justify silently
relaxing the fixed tolerance or reporting valid planning scores.

Next is a fresh bounded precision/backend/accumulation diagnostic on the same
immutable cases, preserving this failure and all model weights. O1 and object
science remain pending. This technical result establishes neither physical
model failure nor lack of masking benefit; no robot-transfer claim follows.

Evidence is under outputs/gated/C008/runs/o0_n3_s0, C001/o0_n3_s0,
C005/v2/runs/o0_n3_s0, C004/o0_n3_s0 and C007/runs/o0_n3_s0.

## Subsequent C012 diagnostic

The bounded diagnostic completed normally in 20.656 seconds. An independent
NumPy inspection reconstructed all 180 overall/per-step summaries from 28 saved
arrays. Inspection, diagnostic and operation have separate exact-result reviews.
The original C004 failure above remains unchanged.

Using the same two train/validation windows, 100 candidates and eight prediction
steps, float32 again fails all four regrouping comparisons. Failures first appear
at step seven; maximum grouping discrepancy is 7.3612e-5 latent units. When each
step instead receives identical reference histories, every float32 grouping
comparison passes, with maximum discrepancy 2.3842e-6.

Promoting the exact saved predictor weights and inputs to float64 makes every
tested grouping and CPU/GPU comparison pass at the original tolerance. Maximum
grouping discrepancy is 8.7805e-14; maximum CPU/GPU discrepancy is 1.1705e-13.
Math attention was already forced in the original model. Precision was the
intervention; no parameters were trained or replaced, and the encoder stayed
float32.

After revisiting [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html),
[math attention precision](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)
and the [NumPy comparison definition](https://numpy.org/doc/2.4/reference/generated/numpy.isclose.html),
the supported inference is recursive amplification of floating-point differences
on these histories. This does not isolate a kernel, validate training precision,
or establish physical accuracy. Common-history conditioning is a diagnostic, not
a physical prediction score. Timing differences are not a fair precision benchmark.

C012 passes as a completed diagnostic, not as inference admission. Full validation
of the explicit float32 encoder/float64 predictor policy, including independently
encoded CPU/GPU inputs, gradients, recursive grouping and downstream dtype
boundaries, requires a new preregistration before implementation. O1, binding,
physical probes, object planning and robot transfer remain unmeasured.

Evidence: outputs/gated/C012/o0_n3_s0 and C012/operations/o0_n3_s0.
Diagnostic review SHA256: 4991565009d55d7cc29ce164f2667a7f5c77d227764fbf53780f2a5d8e4550d0.
