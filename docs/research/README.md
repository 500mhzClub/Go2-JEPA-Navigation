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
| [B001](records/B001.json) | Does the fixed balls evaluator solve its controls without future leakage? | Controls pass at every n/horizon; full n3 L0/L1 CPU planning reviewed through B003; remaining learned arms pending |
| [B002](records/B002.json) | Is unchanged baseline planning numerically valid and practical on spare CPUs? | Six software check groups and actual L0n3 preflight reviewed; sampled cost discrepancy at most5.116e-13, repeated actions exact; rough full-model estimate27.39minutes, no success-rate claim |
| [B003](records/B003.json) | What are the six final baselines' complete fixed-task planning results under CPU execution? | Three cases reviewed: n3 L0 H4/6/8 endpoint97.8/98.2/97.7%, L1 94.7/94.2/94.7%; n1 L0 97.6/98.5/98.5%. L1n1 and n6 pair pending; all completed raw cases independently checked, initial inspector schema failure preserved |
| [A002](records/A002.json) | Does precision explain the remaining L1 discrepancy? | All six pass float64 math inference; reviewed |
| [A003](records/A003.json) | Do final checkpoints pass the selected common inference policy? | All six final303000 cases reviewed passed; 1,620 declared comparisons pass in float64/math, train-only calibrated artifacts preserved |
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
| [A014](records/A014.json) | Can the unchanged original experiments finish after reboot? | All six completed 303,000 steps and passed actual completion reviews; no original trainer remains active |
| [A015](records/A015.json) | Do recovered final files preserve the complete original training state? | All six actual audits reviewed: exact restoration, complete finite logs, 198,000 resumed updates for L1 n1/n3 and 148,000 for the other four |
| [A016](records/A016.json) | Can unchanged final evaluations be supervised within the remaining budget? | 15 CPU checks and all six actual normal completions reviewed; closed union accounting, no timeout or signals |
| [A017](records/A017.json) | What changed from 105,000 to 220,000 steps in L1 n1? | CPU calibration and physical probes reviewed: readout/rollout improve, distance geometry weakens; validation-only, no planning claim |
| [A018](records/A018.json) | Do fresh n3 L0/L1 seeds1/2 contain complete original-recipe training state? | CPU audit and nine focused software groups reviewed: four default-state fixtures,29 corruptions and14 terminal refusals; actual extra-seed audits await training |
| [A019](records/A019.json) | Do additional n3 baseline seeds pass unchanged train-only calibration and numerical validity? | Scope preregistered after literature consultation; implementation and software checks pending, no actual extra-seed evaluation |
| [C001](records/C001.json) | Do RGB slots and anchored masking support physical prediction/planning? | n3 seed0 RGB encoder training under C005; at 04:51 UTC checkpoint confirms 15,845 of 23,752 updates and five completed epochs; cache/O0/O1 and science remain pending |
| [C002](records/C002.json) | Do slot masks bind objects; do paired reviewed results satisfy the declared bars? | Reviewed partial audit includes six final baseline probes across all18 planned rows; n3 L0 red-readout reference pattern not met;24 other bars incomplete, with binding/planning pending |
| [C003](records/C003.json) | Does optimizer recovery preserve subsequent updates and updated-weight numerics? | Both default n3 models pass the small GPU recovery/forward/gradient/grouping diagnostic; reviewed |
| [C004](records/C004.json) | Are selected learned slot checkpoints numerically valid on real inputs and planning-sized batches? | Validator and failure-detection tests implemented; learned-checkpoint results pending |
| [C005](records/C005.json) | Can unchanged slot stages respect both B/C and shared time allocations if GPU progress stalls? | v1's 18 CPU checks reviewed and live n3 trainer supervised; isolated v2 corrects future C004 reserve classification with eight reviewed checks; terminal training result pending |
| [C006](records/C006.json) | What physical state and four-step dynamics do the six final baselines preserve under the frozen probes? | All six actual CPU cases reviewed; L1 helps white readout at n1/n3 with red precision costs; n6 is a measured negative for L1 advantage; numerical/operational checks pass |
| [C007](records/C007.json) | Can unchanged selected-checkpoint validation respect per-call, reserve and global time limits if GPU progress stalls? | External supervisor implemented; thirteen focused CPU check groups inspected and literature-reviewed, with unchanged scientific source; actual C004 cases await terminal training reviews |
| [C008](records/C008.json) | Do completed slot/predictor artifacts contain full training state and the logged validation-selected model? | Twelve focused CPU groups reviewed, including deliberate state/selection/operation corruption and pinned-source routing; actual terminal audits pending |
| [C009](records/C009.json) | Can unchanged object evaluations respect existing GPU budgets during a stalled operation? | External supervisor and ten focused CPU groups reviewed; original-data binding gap preserved and corrected; actual planning/probes/binding await selected-model parents |
| [C010](records/C010.json) | Can the four approved additional n3 baseline seeds run within existing budgets after core/P26? | External supervisor and fifteen focused CPU groups reviewed; original training computation/defaults retained, actual admission correctly refuses missing P26 evidence; no extra-seed training yet |
| [D001](records/D001.json) | Can frozen camera slots replace engineered tracks in Genesis? | Matched 600/100 units prepared; model comparison pending |
| [D002](records/D002.json) | Does the full historical M5 offline summary still reproduce? | All 400 units replayed; all 144 reported numeric fields exactly match; reviewed |
| [D003](records/D003.json) | Can unchanged P26 stages obey the existing eight-hour D and global budgets during GPU stalls? | External supervisor and eleven focused CPU check groups reviewed; initial malformed-input fixture failure preserved; actual export/S1/comparison await the reviewed n3 core |
| [E001](records/E001.json) | Does the repeat-based value estimate establish impossibility? | Documentation corrected; V2 remains a measured negative |
| [E002](records/E002.json) | Does the evidence archive preserve source/artifact identity and distinguish unreviewed results? | Fault-injection tests pass; latest interim archive independently verifies 2,103 payload files, 58 reviewed result rows and historical source identity; pending experiments remain incomplete |

Results and review digests live beside run artifacts. This index must be updated when a result is interpreted. Status vocabulary: `passed`, `measured_negative`, `technical_blocked`, `budget_incomplete`. Separate implementation validity, representation, prediction, planning and generalization evidence.

See [commands and remaining gates](RUNBOOK.md) and [measured implementation evidence](IMPLEMENTATION_2026-11-15.md).

See the [final inference report](FINAL_INFERENCE_2026-11-16.md) for all-six
completion, numerical coverage, calibration measurements and remaining science.
See the [final n3 physical probes](FINAL_PROBES_N3_2026-11-16.md) for the matched
L0/L1 red/white tradeoff on 5,000 test windows and its limits.
See the [all-six final baseline probes](FINAL_BASELINE_PROBES_2026-11-16.md) for
the completed comparisons, n6 negative and partial C002 reference-pattern audit.
See the [first complete baseline planning result](FIRST_BASELINE_PLANNING_2026-11-16.md)
for strong L0n3 endpoint success, collision-free rates and the comparison ceiling.
See the [completed n3 baseline planning pair](BASELINE_PLANNING_N3_2026-11-16.md)
for L1's lower observed endpoint and collision-free success on identical cases.
See the [n1 baseline planning result](BASELINE_PLANNING_N1_2026-11-16.md)
for another strong ordinary baseline and the pending matched L1 comparison.
See the [baseline checkpoint report](BASELINE_PROGRESS_2026-11-15.md) for the
completed L1 n1/n3 logs, the frozen L0 n1 snapshot and what their comparisons test.
See the [CPU checkpoint progress report](CHECKPOINT_PROGRESS_2026-11-15.md) for
the reviewed temporal comparison, physical prediction controls and figures.
See the [preliminary checkpoint analysis](CHECKPOINT_ANALYSIS_2026-11-15.md) for
the matched-step measurements, literature interpretation and current runtime limitation.
See the [runtime recovery handoff](RUNTIME_RECOVERY_2026-11-15.md) for the current
infrastructure evidence, preserved checkpoints and postboot gate.
