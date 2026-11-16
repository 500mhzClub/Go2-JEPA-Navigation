# Matched physical forecasting comparison

The next experiment asks whether the model predicts where obstacles go, including
how their paths change when the robot chooses a different action. Learned physical
heads are part of the solution. Success is measured in physical trajectories and
responses, with a path to Go2's observation and action interface.

[F004](records/F004.json) freezes the comparison before implementation. It follows
the [F002 evidence audit](FORECAST_EVIDENCE_AUDIT_2026-11-16.md) and
[F003 branch audit](BRANCH_DATA_AUDIT_2026-11-16.md). M5 already predicts some useful
responses, but its false responses outweigh those gains overall on the saved
hold/retreat cases. That is the specific weakness this experiment targets.

## Comparison

| Arm | Training futures | Objective |
|---|---|---|
| Factual | Recorded continuation, duplicated to match trajectory/compute budget | Physical trajectory error |
| Branch | Two distinct archived alternatives from the same initial scene | Physical trajectory error |
| Paired | Exactly the same alternatives as Branch | Trajectory error plus error in the difference between their physical futures |

All three share initial scenes, observed inputs, physical target labels, model
capacity, initialization, update count and slot permutations. The existing
deterministic ObjectJEPA displacement head is used with residual velocity and
dropout zero. Each arm receives 20,000 updates with 128 pairs per batch, processed
in microbatches of 16 pairs, and a 16-step autoregressive horizon (6.4 seconds).
The final checkpoint is selected; validation does not choose a checkpoint.

This is a new supervised recipe, not an exact M5 reproduction. The factual arm
deliberately has fewer distinct continuations. That difference is the treatment
tested by Branch versus Factual. Paired versus Branch isolates the added loss.
The experiment does not attribute the result to a latent representation separately
from its head or claim superiority over a capacity-matched direct forecaster.

## What must improve

Evaluate all seven futures at every retained validation scene, against current M5,
position persistence and straight-line constant velocity. Report physical ADE/FDE,
tails and errors at 1.6, 3.2 and 6.4 seconds, including waiting and behavior subsets.

For the 15 pairs of alternative actions, measure the direction and magnitude of
the predicted obstacle response. Include scenes where the correct response is no
change. The Paired arm must improve mean response-vector error by at least 20%
over Branch at both 3.2 and 6.4 seconds, overall and on genuinely responding cases,
with paired layout-bootstrap intervals excluding zero improvement. False-response
rates must not increase, and factual trajectory accuracy must remain within 5% of
the Factual arm. These are preregistered engineering bars, not universal thresholds
derived from a paper. The original P22 slope criterion remains separately reported.

## Implemented and verified

The data adapter, observable feature construction, physical objectives, shared
sampler and checkpoint recovery primitives are implemented. Eight CPU software
groups passed, including label/input separation, future-prefix causality, contact
censoring, matched sampling, masked-loss gradients and exact optimizer/RNG recovery
on a small software fixture. The inherited model contains BatchNorm; the loss
normalization check does not assert that model training is invariant to microbatch
size. Actual trained CPU/GPU and inference grouping checks remain required.

The archived data export and independent verification are complete:

| Coverage | Train | Validation |
|---|---:|---:|
| Original branch points | 123,215 | 12,653 |
| Excluded: incomplete factual future | 18,545 | 2,038 |
| Excluded: no initial match, with full future | 80 | 14 |
| Retained shared scenes | 104,590 | 10,601 |
| Retained layout IDs | 1,136 | 98 |

The verifier checked all 15,000 raw-file hashes and all 40 exported-file hashes,
independently reconstructed initial matching and eligibility for every point, and
reconstructed targets and inputs for the first and last retained point of every
eligible episode: 7,744 points across the 5,000 source episodes. The cache occupies
approximately 1.54 GB. No simulation was run and no new data was collected.

| Evidence | SHA-256 |
|---|---|
| F004 protocol | `bb97f211155fdb8bec204f44774b4702bc83c85f4698918fe4041720b4a7f69e` |
| Software result | `0bec8221d4c1206ca23bc85b6d9cf720c81cf880da614dbb6af9157ef11ae2eb` |
| Export result | `86c72b97b2ab60b5f2986a933fb1de1d45eb888532c8cbd4caca85a3916dd02b` |
| Independent verification | `4e611f42c7a0e18f3886695115bc9e82d98c463205b00d648195ab13e47bdc7f` |

Exact-result literature reviews are saved beside the software and data results in
`outputs/gated/F004`. These establish software/data validity; they are not evidence
of improved prediction. The GPU execution supervisor now passes seven additional
CPU groups, including real process ownership/stall checks and elapsed-only serial
admission. Its review is under `execution_software`. The final physical evaluator
and trained-model numerical checks remain required.

The factual seed0 arm started on the dedicated GPU at15:16UTC on16November from
the isolated `Go2-JEPA-Navigation-forecast-training` checkout, pinned at
`f053c265a75cbc31971cc04a0ca1893d251c1391`. Its owned child was confirmed live and
completed its first20updates. This is operational progress, not an accuracy result.
The source checkout stays unchanged while the development checkout prepares the
remaining evaluation work. No branch or paired arm has started.

## Transfer limits and next action

The model sees tracked observations, requested actions, a full map and a specified
robot path. Simulator truth is used only for targets. The recorded robot path makes
this a conditional forecast test; it does not yet establish command-only prediction
or robustness to robot execution error. Initial detection/matching coverage and
contact exclusions must accompany every accuracy claim. Requiring a complete
factual future selects encounters that survive that horizon; failures after contact
are outside this dataset's valid target coverage.

The current balls environment has no obstacle response to the controlled ball, so
it can test independent motion and nonresponse controls. The existing ring branches
contain the reactive phenomenon needed here. A useful result can then move to the
offline Go2 comparison, with perception, occlusion, action-response accuracy and
inference cost tested explicitly. Reused validation and simulator behavior remain
limits on generalization.

[O0 is complete and reviewed](FULL_O0_PLANNING_2026-11-16.md). While the factual arm
runs, prepare the fixed physical evaluator and checkpoint verification; review
actual training and operations before the next arm. The runner starts fresh runs only; a restart needs
a separately reviewed invocation preserving the earlier attempt. Science artifacts
are under `runs/ARM_s0/science`, operations under `runs/ARM_s0/operations`.
O1 planning and the old incomplete matrix remain preserved and do not advance
automatically.

The literature gate remains in force. [Trajectron++, Sections 4–5](https://arxiv.org/pdf/2001.03093)
supports explicit robot-plan conditioning and physical trajectory evaluation.
[Causal Confusion, Section 3.1](https://proceedings.neurips.cc/paper_files/paper/2019/file/947018640bf36a2bb609d3557a285329-Paper.pdf)
motivates distinguishing observed correlations from interventions; its
imitation-learning results do not guarantee this proposed dynamics loss will work.
