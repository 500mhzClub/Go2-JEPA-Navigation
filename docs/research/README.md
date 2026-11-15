# P25/P26 research gate

Approved scope: P25 balls and P26 offline Genesis, with 48 additional dedicated GPU-hours. Existing six L0/L1 jobs are excluded from that allowance and remain on their pinned checkout. No new missions, planner tuning, value models, hierarchy, stochastic variants, or dense-feature baseline in this stage. A missing dense-feature comparison precludes a claim that slots are superior to dense features.

Every implementation and experiment needs a preregistration in `records/`. It must identify the question, primary sources and sections actually read, differences from this repository, contrary evidence, frozen protocol, success/stop rules, resource limit and verification. Every positive, negative or inconclusive result needs a subsequent literature review before interpretation or a dependent experiment. A source may be revisited; novelty of the citation is not a requirement. Software correctness questions use official documentation/source/issues. Abstract-only reviews do not pass.

The executable gate checks record completeness and hashes. It cannot certify that a human has understood a paper. Result reviews bind to the exact result digest and identify the next discriminating test. Changing the protocol invalidates its digest and requires a new record/version. Seeds under an unchanged preregistration and its declared verification suite may share a record. Never silently select a new protocol from test results.

## Order and decisions

1. A: snapshot/check all six L0/L1 runs; resolve saved versus batch BatchNorm discrepancy; CPU/GPU forward/backward and batch-invariant inference. Train-only calibration is a derived artifact, never an overwrite. Resume must reject incompatible configuration before modifying logs. Unresolved numerical validity blocks learned-model interpretation.
2. B: untouched world and training/validation data; new test split, 1,000 episodes per n=1,3,6, five groups of 200. Three history frames, one goal, horizons 4/6/8, same tasks and CEM randomness across arms. Recorded/direct controls must reach 100%; true-model CEM at least 95% at each horizon. Report endpoint and collision-free success separately: white balls have no contact dynamics.
3. C: RGB SAVi-style adaptation and C-JEPA-style predictor with O0 masking ablation. Three-ball seed 0 first, P26 next, then extra seeds/counts. Frozen slots are shared by O0/O1 for each seed. No labels in either trainer. Keep published protocols and local adaptations explicit.
4. D: pinned camera-slot code and existing s0, no retraining; cached DINOv2, 600 training and 100 validation units. Export sequential tracks; compare engineered M5, M5 on slots, S1 on slots and CV/persistence on identical cases, including detection/identity/occlusion coverage. This is conditional on simulator pose/full map and a previously used validation corpus.

Allocation: A 4 hours, B/C 28, D 8, reserve 8. Record elapsed dedicated-device intervals (overlaps counted once) and each run's duration. Diagnostics may coexist with old jobs below 85% memory; other new GPU jobs wait. Training never falls back to CPU. Budget exhaustion is `budget_incomplete`, not evidence against a hypothesis.

## Evidence index

| Record | Question | Current status |
|---|---|---|
| [A001](records/A001.json) | Are saved inference statistics valid; can train-only calibration repair them? | Six intermediate snapshots reviewed; L1 float32 validity blocked, followed by A002 |
| [B001](records/B001.json) | Does the fixed balls evaluator solve its controls without future leakage? | Controls pass at every n/horizon; learned planning pending |
| [A002](records/A002.json) | Does precision explain the remaining L1 discrepancy? | All six pass float64 math inference; reviewed |
| [A003](records/A003.json) | Do final checkpoints pass the selected common inference policy? | Waiting for existing training |
| [A004](records/A004.json) | Why are two live original trainers no longer logging steps? | User-supplied stacks locate a GPU scalar synchronization boundary; root cause unresolved |
| [A005](records/A005.json) | Can full checkpoint state be restored for the interrupted experiments? | Both step105000 checkpoints pass CPU preflight; first GPU recovery stalled before checkpoint load and was retired |
| [A006](records/A006.json) | Can the two hung processes be retired immediately with all evidence preserved? | Both exited on SIGTERM; all four peers continued advancing, no kernel errors observed; reviewed |
| [A007](records/A007.json) | What do intermediate learning curves and frozen probes establish? | Curves/source audit reviewed; GPU diagnostic budget incomplete, no probe scores |
| [A008](records/A008.json) | Does matched intermediate latent geometry track physical positions? | n3 step40000 CPU reference reviewed: L1 geometry associates with red and white positions; preliminary, no planning claim |
| [A009](records/A009.json) | Have the four remaining originals stopped making observable progress? | Live identities with stale logs near A007 launch; step155000 checkpoints preserved and finite; no signals to originals |
| [A010](records/A010.json) | Can the four additional originals be restored for sequential continuation? | User approved; cleanup passed; first serial L1n1 attempt retired without progress, reviewed technical_blocked |
| [A011](records/A011.json) | Where is the first serial recovery waiting before new metrics appear? | User stack locates initial Module.to before checkpoint loading; reviewed technical_blocked |
| [A012](records/A012.json) | Does a fresh primitive GPU copy complete? | Both default and SDMA-disabled16KiB copies timed out; children exited, checkpoints unchanged; reviewed technical_blocked |
| [A013](records/A013.json) | Does the unchanged primitive complete after an orderly reboot? | New boot confirmed; exact copy/arithmetic round trip passed in2.025s and reviewed |
| [A014](records/A014.json) | Can the unchanged original experiments finish after reboot? | Six inputs cloned with verified source/data lineage; first L1n1 continuation has new step105500, final303000 pending |
| [A015](records/A015.json) | Do recovered final files preserve the complete original training state? | CPU checker reviewed: six structural fixtures,14 rejected faults and live-run refusal; actual final audits pending |
| [C001](records/C001.json) | Do RGB slots and anchored masking support physical prediction/planning? | Small-batch numerical checks pass; training pending |
| [C002](records/C002.json) | Do slot masks bind objects; do paired reviewed results satisfy the declared bars? | Evaluators verified on synthetic cases; trained-model measurements pending |
| [C003](records/C003.json) | Does optimizer recovery preserve subsequent updates and updated-weight numerics? | Both default n3 models pass the small GPU recovery/forward/gradient/grouping diagnostic; reviewed |
| [C004](records/C004.json) | Are selected learned slot checkpoints numerically valid on real inputs and planning-sized batches? | Validator and failure-detection tests implemented; learned-checkpoint results pending |
| [D001](records/D001.json) | Can frozen camera slots replace engineered tracks in Genesis? | Matched 600/100 units prepared; model comparison pending |
| [D002](records/D002.json) | Does the full historical M5 offline summary still reproduce? | All 400 units replayed; all 144 reported numeric fields exactly match; reviewed |
| [E001](records/E001.json) | Does the repeat-based value estimate establish impossibility? | Documentation corrected; V2 remains a measured negative |
| [E002](records/E002.json) | Does the evidence archive preserve source/artifact identity and distinguish unreviewed results? | Fault-injection tests pass; actual interim archive independently verifies1859 payload files and historical source identity; reviewed |

Results and review digests live beside run artifacts. This index must be updated when a result is interpreted. Status vocabulary: `passed`, `measured_negative`, `technical_blocked`, `budget_incomplete`. Separate implementation validity, representation, prediction, planning and generalization evidence.

See [commands and remaining gates](RUNBOOK.md) and [measured implementation evidence](IMPLEMENTATION_2026-11-15.md).

See the [preliminary checkpoint analysis](CHECKPOINT_ANALYSIS_2026-11-15.md) for
the matched-step measurements, literature interpretation and current runtime limitation.
See the [runtime recovery handoff](RUNTIME_RECOVERY_2026-11-15.md) for the current
infrastructure evidence, preserved checkpoints and postboot gate.
