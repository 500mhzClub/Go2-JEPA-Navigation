# Paired O1 training complete — 16 November 2026

Both n3 seed0 object predictors have completed their fixed 30 epochs and
11,370 updates. O1 finished normally in 997.851 seconds; its terminal C008
audit, C001 completion and C005 operation have separate passing literature
reviews. No trainer remains active.

| Common logged selection metric | O0 | O1 |
|---|---:|---:|
| Minimum unmasked validation future latent MSE | 0.0041897866 | 0.0040267887 |
| Selected epoch, zero based | 29 | 28 |
| Completed updates | 11,370 | 11,370 |

O1's recorded minimum is 3.89% lower. This is a descriptive training-log
comparison on one seed, not a demonstrated physical prediction or planning
improvement. O1's selected-checkpoint numerical validation remains pending.
Its selected epoch28 artifact and final resume checkpoint are distinct and
both are preserved; all 83 Adam parameter states, schedule, CPU restoration,
source/data/cache bindings and ordered epoch logs pass the terminal audit.

O0 and O1 share the frozen RGB encoder, float32 feature cache, initialization
and epoch order. O1 adds zero to two masked object histories with an earliest
frame identity anchor and a masked-history reconstruction term. Its training
loss therefore includes an objective absent from O0; their common selection
metric is unmasked future prediction. The
[Causal-JEPA objective and predictor appendix](https://arxiv.org/html/2602.11389v2)
were revisited after completion. Masking motivates a test of reliance on
observed context, but our independent pass-through balls differ from the
paper's interacting environments. A masking benefit remains unmeasured.

The [O0 precision result](OBJECT_INFERENCE_PRECISION_2026-11-16.md) qualifies
full64 encoding and prediction on its tested checkpoint. C014 now supplies an
explicit repository inference loader and routes object probes, planning and
binding through checkpoint-specific numerical admission. Four CPU software
groups pass: exact parameter/buffer promotion, real train/validation equivalence
to C013, 23 invalid-evidence refusals and the original full-default CEM/probe
interfaces on a toy world. These checks do not qualify an actual C014 numerical
run. The general numerical validator/supervisor and versioned C009 route still
need implementation and review before actual object evaluations.

Next are the encoder/O0/O1 numerical gates through that deployed policy,
object-mask binding, physical probes and paired planning. Then comes the
approved offline Genesis comparison, additional seeds/counts and final evidence
report. Original failed numerical results and all training checkpoints remain
unchanged. There is no demonstrated Go2 transfer yet.

Evidence: outputs/gated/C001/o1_n3_s0, C008/runs/o1_n3_s0,
C005/v2/runs/o1_n3_s0 and C014/software_loader. Runtime a75523b is preserved.
At completion all dedicated-device intervals are closed, with 43.361 of the
48 additional GPU-hours remaining.
