# Existing branch data can support the physical forecasting comparison

[F003](records/F003.json) checked every archived ring branch and its parent data.
All5,000 branch files passed the declared structural checks. The data can support
the comparison selected after [F002](FORECAST_EVIDENCE_AUDIT_2026-11-16.md), without
new collection. Target quality and coverage still need explicit treatment in the
training protocol.

The audit and independent verification ran on the CPU. No model was trained or
evaluated and no simulator was replayed. The live O0 evaluation was unchanged.

| Property | Training | Validation |
|---|---:|---:|
| Episode files |4,500|500|
| Branches |739,290|75,918|
| Starting scenes, six branches each |123,215|12,653|
| Distinct layout IDs |1,180|100|
| Initially matched visible nearby walkers |174,994 /187,435|18,027 /19,304|

All15,000 branch/parent/metadata file digests were frozen before the audit and
checked again independently. Training and validation share no layout IDs or
recorded walker placement seeds. The original branch/hold/contact counts match.

For each six-way group, all nine history frames agree, including poses, track
identities, observations and truth labels; all eight past command rows agree.
The history matches its parent episode. Future contact frames and subsequent
frames have empty objects/labels, later requests are zero and the recorded robot
pose remains fixed. Every identical-action sibling pair has identical saved
outcomes. None of these checks reran a world or reconstructed a hidden RNG state.

## Tracker-derived paired targets are accurate on average, with tails

At the common initial scene, visible walkers within3m are matched to observed
slots using mutual nearest neighbors within0.3m, excluding ambiguous ties.
Physical identity is fixed there; the analysis does not rematch using future
truth. It considers all15 sibling pairs and excludes a pair at a horizon if either
branch has already contacted something. Tracker targets additionally require
both future observations and unchanged track IDs.

The error below compares the **difference between tracker positions under two
actions** with the difference between the corresponding physical positions.

| Validation horizon | Valid physical pairs | Pairs with track targets | Mean target error | RMSE | Maximum |
|---|---:|---:|---:|---:|---:|
|1.6s|245,175|226,181 (92.3%)|4.9mm|15.0mm|0.79m|
|3.2s|216,003|182,756 (84.6%)|6.6mm|30.7mm|1.29m|
|6.4s|180,910|128,120 (70.8%)|15.2mm|87.5mm|2.41m|

Among physically zero-effect validation pairs that have valid tracker targets,
the tracker nevertheless produces a contrast above5cm in0.66%,1.25% and2.51%
at the three horizons. Training means are5.0/6.7/14.9mm, with similarly increasing
tails. The audit identifies target errors; it does not determine whether each
large error arose from an identity swap, localization error or another cause.

These sibling pairs and nearby starting scenes are correlated. Counts describe
available examples, not independent sample sizes. Their action distribution
differs from F002's hold/retreat evaluation, so these numbers cannot be subtracted
from F002 model errors or used to claim a measured explanation of those errors.

## Consequences for the next experiment

The next protocol should compare a fixed deterministic physical forecaster under
ordinary/factual data, alternative-action data, and the same alternative-action
data plus a paired action-effect loss. Learned displacement heads remain valid.

Give every arm the same target-label access. Using simulator truth only for the
paired loss would mix two changes: cleaner supervision and the paired objective.
A clean comparison can use the archived physical targets consistently across
arms, while keeping truth out of the model inputs. This must be explicitly
preregistered as a new supervised recipe, rather than described as an exact
reproduction of the historical tracker-target M5 training.

Match starting-scene distributions where possible. Otherwise, ordinary versus
branched training also changes how often the model sees close encounters. Keep
updates, sampled trajectories, initialization and model capacity matched. Preserve
the split by source episode/layout; never split sibling branches across training
and evaluation.

Report missing tracks and contact censoring beside accuracy. The valid-track
coverage falls to70.8% by6.4s, so evaluating only surviving tracks would conceal
an important part of the deployment problem. Ordinary trajectory accuracy,
genuine action effects and no-effect controls all remain required. Hidden walker
state, limited history and stochastic futures remain possible limits.

[De Haan et al., Sections3.1 and4.2](https://proceedings.neurips.cc/paper_files/paper/2019/file/947018640bf36a2bb609d3557a285329-Paper.pdf)
distinguish observational associations from interventional queries. That supports
using matched action alternatives; their imitation-learning method is not a
guarantee for this forecasting loss. [Trajectron++, Sections3–4](https://arxiv.org/pdf/2001.03093)
supports making the candidate robot path an explicit input. Recorded future
poses still define a conditional test, not a complete Go2 deployment contract.

F003 passes as a data audit. It does not qualify a new training run. F004 must
freeze the target contract, data adapter, losses, selections and evaluation bars
before implementation. P26 and the unfinished historical P25/P26 comparisons
remain pending; O1 planning is not the automatic next job.

## Evidence and reproduction

Sources: [audit](../../tools/research/f003_branch_audit.py) and
[independent verifier](../../tools/research/f003_verify.py).

Local evidence under `outputs/gated/F003` includes:

- `inputs.json`:15,000 files,7,134,897,422 bytes;
  SHA256 `cd7a487de368044c7d25f7cca86c91a0be59eb3a51d07cd3eaa1802f56bfe86d`.
- `per_file.jsonl`: all5,000 complete file outcomes.
- `result.json`:
  SHA256 `5ab94bbb5db36c0c26d9fcee2b0d0e5aeff3159ab84d82fbacaef8c652d73aa1`.
- `verification.json` and `review.json`: exact-result verification and substantive
  post-result literature review.

The independent verifier rehashes all inputs, checks468 aggregate values and
reconstructs90 target counters/statistics using scalar calculations over two
whole episode files, one per split. It does not claim a second independent
structural scan of every file.

Five software groups pass, including inclusive contact-at-horizon censoring.
The initial absent-slot arithmetic warnings and an independent-verifier failure
on empty strata are preserved with source and reviews. Finite masked subtraction
and explicit empty-stratum defaults corrected those issues; the actual dataset
result was not modified or selectively rerun.

Historical files do not include exact generator source/hidden-state digests.
Current byte identities, inspected generator semantics and outcome consistency
are the evidence available. Disjoint layout IDs are not a separate proof that
every geometric topology is unique across the split.
