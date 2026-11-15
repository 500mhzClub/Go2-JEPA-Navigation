# GPU startup failure and recovery handoff

After the authorized reboot, the unchanged default-environment primitive passed
in2.025seconds and the first A014 L1n1 continuation advanced from105000 to a fresh
105500. Final training and numerical validity remain pending. All six original
recovery inputs and the failed preboot attempts are preserved.

The first run has subsequently passed143000 while remaining the only GPU trainer.
Use `outputs/gated/A014/runs/l1_n1_s0/monitor_latest.json` for the latest observation;
the startup pass is not final completion. The CPU-only A015 completion checker has
passed and received literature review for21 software checks across all six preserved
schemas, including deliberate corruption and refusal of the current live run.
Its actual final audits remain pending; commands and review requirements are in
[the runbook](RUNBOOK.md).

Before reboot, two fresh processes both stalled at their first16KiB CPU-to-GPU
copy, without using a model, dataset, checkpoint or optimizer. This history
motivated the host-state intervention described below.

The user-supplied A011 stack places the first serial continuation at
`Module.to -> Tensor.to`, original `build` line125 / `train` line180. Checkpoint
loading starts later in that function. This attempt therefore did not read its
checkpoint or perform a resumed update. A012 stopped it with SIGTERM; its
checkpoint, configuration, metrics, lineage and empty log hashes remained unchanged.
The failed attempt and its literature review remain under
`outputs/gated/A010/runs/l1_n1_s0`.

A012 tested exactly 4,096 float32 elements with an external deadline:

| Diagnostic | Completed | Last operation | Elapsed / outcome |
|---|---|---|---|
| Original runtime settings | Import, device query, allocation | First blocking host-to-device copy | 45.071 s, SIGTERM, `technical_blocked` |
| Only `HSA_ENABLE_SDMA=0` changed | Import, device query, allocation | Same copy | 45.071 s, SIGTERM, `technical_blocked` |

Both internal 10-second tracebacks locate the same call. Neither test reached
arithmetic, device-to-host transfer or equality validation. Both contexts exited;
the observed kernel windows contained no GPU errors. This does not identify a
specific PyTorch, HIP, driver, firmware or hardware defect. The original training
environment remains unchanged. [PyTorch synchronization semantics](https://docs.pytorch.org/docs/2.12/notes/cuda.html#asynchronous-execution)
explain the host boundary; [AMD's HIP debugging guidance](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/debugging.html#hsa-related-environment-variables-linux)
motivated the second copy-path test. Its failure supplies no working workaround.

Including the postboot primitive, the additional-device ledger is closed at about
**0.292 hours**, leaving about **47.708 hours**. Original training/continuation
time is excluded as approved.

## Preserved continuation inputs

| Runs | Full checkpoint step |
|---|---:|
| L1 n1/n3 seed0 | 105,000 |
| L0 n1/n3/n6 and L1 n6 seed0 | 155,000 |

All six hashes still match their reviewed CPU restoration inputs. The original
trainer checkout remains pinned to `7dfc83722da0a9fb805ccc879fe7e5459709a2fd`.
The handoff stores checkpoint/configuration/metric hashes, source hashes, boot ID,
device mapping, desktop clients and kernel/SMI observations under
`outputs/gated/A013`. Existing source, data, checkpoints and failed evidence are
retained. None of the six final step303000 results exists yet.

## Next action and gate

An orderly workstation reboot is proposed. GPU0 maps to BDF `0000:03:00.0` /
DRM card1, is the boot VGA device, drives connected DP3 and DP4, and has live
desktop clients. A live GPU reset or driver reload would affect that session.
The user subsequently authorized the proposed orderly reboot. Systemd accepted
and confirmed a schedule for2026-11-15 at19:06:50GMT, with a one-minute delay
to save desktop work. This confirms scheduling, not completed reboot or GPU
recovery; see `outputs/gated/A013/host_action/review.json`. No forced shutdown,
driver upgrade or runtime-setting change is part of this action.

After the host returns, the agent will:

1. Require a changed boot UUID and unchanged recovery/source hashes.
2. From the validation checkout, run
   `python3 outputs/gated/A013/run_postboot.py` once. This uses exactly the A012
   baseline probe and original environment, with the same bounded supervision.
3. Consult the literature and review the exact result. A passing primitive is
   only a prerequisite for a distinct original-training recovery attempt.
4. If valid, preserve the failed A010 attempt, prepare a separate first recovery
   output directory, and resume the already-approved six-run serial plan. Require
   new metric/checkpoint progress, then the unchanged final A003 checks.

If the postboot probe fails, stop retries and prepare a runtime issue reproducer.
Do not change the model to address an unlocalized infrastructure failure.

The postboot supervisor has passed syntax and same-boot refusal checks without
creating a GPU context or ledger interval. The boot guard uses Linux's
[documented boot UUID](https://docs.kernel.org/admin-guide/sysctl/kernel.html#random).
The postboot test subsequently passed all phases with exact equality and exit0;
its digest-bound review is `outputs/gated/A013/postboot_primitive/review.json`.
The new boot UUID differs from the recorded preboot UUID. Boot logs include an
iGPU display REG_WAIT timeout before the probe; this is a different BDF from the
dedicated R9700 and does not justify claiming a wholly error-free boot.

A014 preserves the original runtime/recipe and uses reviewed full-state clones
at `outputs/balls/recovery_A014`. Its first continuation has fresh training
progress. The external supervisor records identity and progress every60seconds,
with bounded no-progress/wall limits; each final result requires review before
the next child. There is no automatic restart or self-approved successor.

The successful small probe supports a state-dependent runtime/device issue;
it does not identify the faulty component or establish sustained training
stability. The scientific comparison remains incomplete, and the preliminary
geometry findings are unchanged.

A014 follow-up at121seconds confirms5,000 fresh updates through110000 and a new
checkpoint write. Its startup review is `outputs/gated/A014/startup/review.json`;
this limited pass does not unlock a successor or replace final-state checks.
