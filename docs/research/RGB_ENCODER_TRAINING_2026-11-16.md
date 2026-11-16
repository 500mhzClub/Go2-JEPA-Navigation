# First RGB encoder completed — 16 November 2026

The n3 seed0 RGB object encoder completed all eight epochs and exited normally.
The final checkpoint audit and separate training/operation reviews pass. This
establishes complete, consistent training artifacts; object binding and useful
prediction or planning are still unmeasured.

| Check | Observed result |
|---|---:|
| Six-frame training windows per epoch | 95,000 |
| Batches per epoch | 2,969 |
| Total updates | 23,752 |
| Adam parameter states at that update count | 57 |
| Selected epoch, zero-indexed | 7 |
| Final validation reconstruction loss | 0.0775776363 |
| Dedicated GPU time, nested intervals counted once | 3.72282 hours |

Validation loss was 0.981318 after epoch0, 0.079238 after epoch1 and 0.077578
at completion. These are reconstruction measurements on the fixed validation
subset, not binding accuracy, endpoint success or comparable L0/L1 latent losses.
The small late changes do not establish that the slots identify distinct objects.

The CPU audit restores model, optimizer and scheduler exactly, verifies every
update counter, finite moments, final learning rate and cursor, and checks CPU
RNG restoration. GPU RNG is inspected only as bytes. Epoch7 is the first strict
validation minimum, and its selected weights exactly match the final state.
Original378 source files were verified in the still-pinned runtime checkout,
alongside executable audit source and the original six train/validation files.
No forward pass, parameter update or CUDA initialization occurred in the audit.

The [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
was revisited before and after the audit. Optimizer restoration follows the
[scheduler ordering documented for Adam](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Adam.html).
[Causal-JEPA appendices D.2 and E.2/E.3](https://arxiv.org/html/2602.11389v2)
were also revisited: its stochastic SAVi and our deterministic RGB adaptation
have different objectives and settings. This is not an exact reproduction.

The exact trainer and supervisor are absent. Both GPU accounting intervals are
closed, with no timeout, signals or errors. This successful run does not diagnose
the earlier device hangs or prove permanent runtime stability.

Evidence and separate reviews are under `outputs/gated/C008/runs/slots_n3_s0`,
`outputs/gated/C001/slots_n3_s0` and `outputs/gated/C005/runs/slots_n3_s0`.
The historical runtime commit is81171a2; its original source/config/data bindings
remain recorded. After the actual completion reviews, the gated checkout advanced
to6925f47 for selected-checkpoint validation and cache extraction.

Selected-checkpoint C004 validation also passed and received separate science
and C007 operation reviews. On32 real train/validation windows, all1104 declared
comparisons passed:48 outputs,912 parameter gradients and144 grouping comparisons.
Maximum absolute discrepancies were5.484e-6,4.082e-6 and1.431e-6 respectively,
at unchanged relative1e-4/absolute1e-5 tolerances. The checkpoint and learned
weights remained unchanged. Independent inspection checked the recorded coverage,
sampling, finite summaries and source/input identities; it did not rerun GPU
arithmetic. The encoder has no predictor candidate-rollout test.

[PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)
and Causal-JEPA D.2/E.2/E.3 were revisited after these actual results. This clears
sampled numerical validity, with no object-binding or planning claim. C007 exited
normally after7.238seconds, without timeout, signals or errors; owned intervals
are closed. Evidence is under `outputs/gated/C004/slots_n3_s0` and
`outputs/gated/C007/runs/slots_n3_s0`.

Shared frozen-feature extraction completed normally at06:20UTC through C005v2,
using the same selected encoder and original train/validation data. Full cache
inspection and separate science/operation reviews pass. Train/validation arrays
have shapes1000/100 by100 by5 by128; all70.4million float32 values are finite.
Payload sizes, index/array hashes, all396 source entries and original six data
files match. Encoder bytes are unchanged, and no partial arrays remain.

The inspection uses read-only NumPy memory maps, without Torch or re-encoding.
Sequential causal carry across10-frame chunks and resets between episodes follow
the bound source. Full-episode cache context differs from independently reset
short histories and goals; their numerical and scientific evaluations remain
required. Causal-JEPA D.2/E.1–E.3 and the
[NumPy loading documentation](https://numpy.org/doc/2.4/reference/generated/numpy.load.html)
were revisited after extraction and inspection. Valid cache structure does not
establish useful object identities or dynamics.

Extraction took101.700seconds with no timeout, signals or errors. Both exact
processes are absent and owned GPU intervals are closed. Evidence, the inspection
script and review are under `outputs/gated/C001/extract_n3_s0`; operations are
under `outputs/gated/C005/v2/runs/extract_n3_s0`. The historical extraction source
is6925f47. Next is O0, followed by paired O1 using the same frozen cache and
initialization/sample-order policy, with actual completion and numerical reviews
between stages.
