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
remain recorded. Next is actual selected-weight numerical validation through
C007/C004, followed by the shared cache and O0/O1 comparisons if that gate passes.
