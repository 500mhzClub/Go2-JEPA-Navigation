# GPU startup failure and recovery handoff

The current blocker reproduces without a model, dataset, checkpoint or optimizer:
two fresh processes both stalled at their first 16 KiB CPU-to-GPU copy. No trainer
or diagnostic child remains running. All six recovery checkpoints are preserved.

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

The additional-device ledger is closed at **0.290958 hours**, leaving about
**47.709 hours**. Original training/continuation time is excluded as approved.

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
The user must coordinate the reboot after saving desktop work; approval for
trainer cleanup does not authorize restarting the workstation. No reboot or
device reset is scheduled. Use the normal desktop restart action; no forced
shutdown, driver upgrade or runtime-setting change is part of this proposal.

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
It has **not** run a postboot GPU test yet. The scientific comparison remains
incomplete; this technical result does not change the preliminary geometry
findings or establish a model-performance negative.
