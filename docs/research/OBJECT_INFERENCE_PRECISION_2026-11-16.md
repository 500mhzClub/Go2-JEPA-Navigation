# O0 inference precision validation — 16 November 2026

The selected O0 n3 seed0 model passes the full C013 numerical suite with both
encoder and predictor in float64. Keeping the encoder float32 fails one
long-rollout CPU/GPU comparison. Original learned values, training, cache and
C004 failure are preserved. Downstream evaluator integration remains required.

| Comparisons per policy | Count | Encoder32 / predictor64 | Encoder64 / predictor64 |
|---|---:|---|---|
| Encoder outputs, gradients and grouping | 1,104 | Pass | Pass |
| Predictor outputs, gradients and grouping | 1,392 | Pass | Pass |
| Independent image-to-prediction paths and goal costs | 256 | Pass | Pass |
| Planning-sized rollouts, costs and encoding | 20 | 19 pass, one fails | Pass |
| Total | 2,772 | Does not qualify | Qualifies |

Both policies use the same 32 real train/validation windows and the original
relative 1e-4 / absolute 1e-5 tolerance. The planning cohorts contain 100
candidates over eight steps on one window per split. Independent CPU/GPU image
normalization and encoding are included; goals are encoded separately from
history. The original image normalization still computes in float32 before
conversion to the encoder dtype. The attention backend remains math.

Mixed precision fails on 600 elements of the validation CPU/GPU rollout, all at
steps seven/eight. Maximum difference is 3.7686e-5 latent units. Its goal costs
pass their relative tolerance, which does not override the latent failure.
With both modules in float64, maximum complete-path CPU/GPU rollout difference
is 2.8408e-7; maximum GPU regrouping difference is 1.2140e-13. All checks pass.

An independent NumPy inspector checks coverage, gradient names, sample and input
identities and every saved planning-array comparison: 168 overall/per-step
summaries across 64 arrays. It also enumerates all 120 assignments for each of
2,000 five-slot goal-cost matrices. Maximum difference from the recorded costs
is 1.1369e-13. Small-batch raw outputs and gradients were not retained, so their
values are not independently recomputed by this inspector.

[PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
and [precision conversion semantics](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html#torch.nn.Module.double)
were revisited after the result. C012's shared-history success was insufficient
to validate mixed precision: differences from independently encoded inputs can
grow through recursive prediction. The evidence supports float64 inference for
this selected model under the tested protocol, without identifying a kernel
root cause or establishing physical accuracy. C013 does not validate a different
checkpoint or prove consistency for every possible input.

The first actual attempt stopped after 5.329 seconds because the new validator
added an extra goal-image dimension. That exception, its sources and reviews
remain under C013/o0_n3_s0 and C013/operations/o0_n3_s0. A corrected real-data
indexing fixture passes; the explicit retry completed normally in 75.756 seconds.
The earlier software preprocessing coverage finding is also preserved.

Corrected evidence is under outputs/gated/C013/retry_001/o0_n3_s0 and its sibling
operations/o0_n3_s0. Inspection, numerical policy and operation have separate
reviews. Numerical review SHA256:
65db08474cbb253746ab9d55f59dc6b1f7e0d54870447f606d9dc3fca97bee87.

C014 now preregisters integration of this explicit policy and future selected
checkpoint validation. Its loader and three evaluation boundaries pass four CPU
software groups, including exact real-input equivalence to the C013 full64
wrapper and invalid-evidence refusals. Actual numerical validation and the new
supervised evaluation route remain pending; original C009 must not be bypassed.
[O1 training and its terminal audit are complete](OBJECT_PREDICTOR_O1_2026-11-16.md),
using the same frozen cache, initialization and epoch order from runtime a75523b.
Its own numerical evidence, masking benefit, physical prediction and planning
remain unmeasured.
