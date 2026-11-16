# P25/P26 research gate

## Current user authorization — GPU time limits removed

On 16 November 2026 the user explicitly removed GPU-hour limits for the approved
P25/P26 plan. [U001](records/U001.json) supersedes all historical global, stage
and per-run GPU-time caps. Keep elapsed-time accounting, serial GPU execution,
literature review and owned-process stall handling. Do not optimize merely to
fit an old time allocation. Historical budget statements below describe earlier
runs and must not be used to stop current progressing work.

The immediate execution route runs the original full64 GPU planner unchanged:

```sh
"$PY" outputs/gated/U001/run.py --run o0_n3_s0 --python "$PY" \
  --preconsultation outputs/gated/U001/pre_o0_n3_s0.json
```

Fresh results go to `outputs/gated/planning_full/RUN`; operations are under
`outputs/gated/U001/operations/RUN`. Preserve the original partial O0 directory.
Review O0 fully before the paired O1 run, then proceed to offline P26 and the
remaining approved seeds/counts. Later stage supervisors must use this override
instead of their historical enforced caps. The CPU feasibility diversion is
cancelled; no CPU diagnostic or C015 planner optimization is being adopted.


Approved scope: P25 balls and P26 offline Genesis, with no GPU-hour limit under U001. All six original L0/L1 jobs are complete; preserve their evidence and pinned source. No new missions, planner tuning, value models, hierarchy, stochastic variants, or dense-feature baseline in this stage. A missing dense-feature comparison precludes a claim that slots are superior to dense features.

Every implementation and experiment needs a preregistration in `records/`. It must identify the question, primary sources and sections actually read, differences from this repository, contrary evidence, frozen protocol, success/stop rules, resource limit and verification. Every positive, negative or inconclusive result needs a subsequent literature review before interpretation or a dependent experiment. A source may be revisited; novelty of the citation is not a requirement. Software correctness questions use official documentation/source/issues. Abstract-only reviews do not pass.

The executable gate checks record completeness and hashes. It cannot certify that a human has understood a paper. Result reviews bind to the exact result digest and identify the next discriminating test. Changing the protocol invalidates its digest and requires a new record/version. Seeds under an unchanged preregistration and its declared verification suite may share a record. Never silently select a new protocol from test results.

## Order and decisions

1. A: snapshot/check all six L0/L1 runs; resolve saved versus batch BatchNorm discrepancy; CPU/GPU forward/backward and batch-invariant inference. Train-only calibration is a derived artifact, never an overwrite. Resume must reject incompatible configuration before modifying logs. Unresolved numerical validity blocks learned-model interpretation.
2. B: untouched world and training/validation data; new test split, 1,000 episodes per n=1,3,6, five groups of 200. Three history frames, one goal, horizons 4/6/8, same tasks and CEM randomness across arms. Recorded/direct controls must reach 100%; true-model CEM at least 95% at each horizon. Report endpoint and collision-free success separately: white balls have no contact dynamics.
3. C: RGB SAVi-style adaptation and C-JEPA-style predictor with O0 masking ablation. Three-ball seed 0 first, P26 next, then extra seeds/counts. Frozen slots are shared by O0/O1 for each seed. No labels in either trainer. Keep published protocols and local adaptations explicit.
4. D: pinned camera-slot code and existing s0, no retraining; cached DINOv2, 600 training and 100 validation units. Export sequential tracks; compare engineered M5, M5 on slots, S1 on slots and CV/persistence on identical cases, including detection/identity/occlusion coverage. This is conditional on simulator pose/full map and a previously used validation corpus.

Record elapsed dedicated-device intervals (overlaps counted once) and each run's duration. The historical A4/B-C28/D8/reserve8 allocations are superseded by U001. Keep serial full GPU work and the memory ceiling; training never falls back to CPU. Preserve historical `budget_incomplete` results as unfinished evidence.

## Evidence index

| Record | Question | Current status |
|---|---|---|
| [U003](records/U003.json) | Can unchanged P26 use unlimited time and complete fresh planning evidence? | Six CPU groups reviewed; original stages/configs/models retained,15 core reviews required, actual P26 pending |
| [U002](records/U002.json) | Can the unchanged independent planner scoring inspect U001 full outputs? | Six CPU adapter groups reviewed; original thirteen-group scoring retained. Complete actual O0/O1 results remain pending |
| [A001](records/A001.json) | Are saved inference statistics valid; can train-only calibration repair them? | Six intermediate snapshots reviewed; L1 float32 validity blocked, followed by A002 |
| [B001](records/B001.json) | Does the fixed balls evaluator solve its controls without future leakage? | Controls pass at every n/horizon; all six original baseline planning cases reviewed through B003; object-model arms remain |
| [B002](records/B002.json) | Is unchanged baseline planning numerically valid and practical on spare CPUs? | Six software check groups and actual L0n3 preflight reviewed; sampled cost discrepancy at most5.116e-13, repeated actions exact; rough full-model estimate27.39minutes, no success-rate claim |
| [B003](records/B003.json) | What are the six final baselines' complete fixed-task planning results under CPU execution? | All six complete;18,000 saved cases independently checked. L1 helps modestly at n1, trails at n3, and is a measured negative at n6:2.9/1.8/1.0% endpoint success versus L0 97.1/93.5/82.6%. All operations pass |
| [B004](records/B004.json) | Do the additional n3 baseline seeds solve the same fixed planning tasks? | Thirteen software groups reviewed with unchanged B003 computation and independent scoring; initial missing output-parent finding preserved and corrected. Actual additional-seed planning awaits training, numerics and probes |
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
| [A019](records/A019.json) | Do additional n3 baseline seeds pass unchanged train-only calibration and numerical validity? | External supervisor and CPU artifact verifier implemented;13 focused CPU groups reviewed, including four default exports and18 corruptions. Original numerical computation retained; no actual extra-seed evaluation |
| [C001](records/C001.json) | Do RGB slots and anchored masking support physical prediction/planning? | First encoder/cache and bothO0/O1training complete and reviewed. O1logged validation MSE0.00402679 versusO0 0.00418979. Selected encoder/O0/O1 deployed full64 numerical gates pass under C014. Paired O0/O1 probes reviewed negative: red four-step RMSE78.807/60.673px versus held-current21.990px; O1 improves error23% but misses physical bars. Paired planning pending |
| [C002](records/C002.json) | Do slot masks bind objects; do paired reviewed results satisfy the declared bars? | First n3 seed0 binding is a reviewed negative: 0/20,000 cases reach IoU0.5, including the mostly visible subset. Independent summaries and operation pass. Full comparison still pending; historical six-baseline probe audit retained |
| [C003](records/C003.json) | Does optimizer recovery preserve subsequent updates and updated-weight numerics? | Both default n3 models pass the small GPU recovery/forward/gradient/grouping diagnostic; reviewed |
| [C004](records/C004.json) | Are selected learned slot checkpoints numerically valid on real inputs and planning-sized batches? | Encoder1104 comparisons pass. O0:1392 small-batch checks and2 exact rollout repeats pass,4 regrouped rollouts fail; originalreview remains technical_blocked. C012diagnosis and C013full64validation complete; downstream integration must explicitly bind new evidence |
| [C005](records/C005.json) | Can unchanged slot stages respect both B/C and shared time allocations if GPU progress stalls? | First encoder, extraction and bothO0/O1 completed normally and separately reviewed. O1took997.851seconds; all intervals closed, originalv1 evidence preserved |
| [C006](records/C006.json) | What physical state and four-step dynamics do the six final baselines preserve under the frozen probes? | All six actual CPU cases reviewed; L1 helps white readout at n1/n3 with red precision costs; n6 is a measured negative for L1 advantage; numerical/operational checks pass |
| [C007](records/C007.json) | Can unchanged selected-checkpoint validation respect per-call, reserve and global time limits if GPU progress stalls? | Encoder and O0 operations exited normally and are reviewed; O0 C004 numerical failure remains separately blocked despite exit0. No automatic successor |
| [C008](records/C008.json) | Do completed slot/predictor artifacts contain full training state and the logged validation-selected model? | First encoder and bothO0/O1 actual terminal audits reviewed. Bothpredictors have83Adam states/11370updates; O1selectedepoch28 differs from final resume state, as expected |
| [C009](records/C009.json) | Can unchanged object evaluations respect existing GPU budgets during a stalled operation? | C014-compatible v2 passes 16 software groups and all 21 command routes. First binding completed normally in105.007seconds with separate operational review; scientific negative preserved. O0/O1 probes completed normally in207.327/207.227seconds with separate passing operations/inspections; both scientific negatives preserved. Planning remains |
| [C010](records/C010.json) | Can the four approved additional n3 baseline seeds run within existing budgets after core/P26? | External supervisor and fifteen focused CPU groups reviewed; original training computation/defaults retained, actual admission correctly refuses missing P26 evidence; no extra-seed training yet |
| [C011](records/C011.json) | What physical state and prediction do the additional n3 baselines preserve under unchanged CPU probes? | Eighteen software groups reviewed, including unchanged C006 computation, complete parent/data bindings and owned CPU process checks. Initial test-variable failure preserved; actual extra-seed probes await training and numerical reviews |
| [C012](records/C012.json) | Does predictor precision explain O0 regrouping failures, and does recursive feedback amplify them? | Actual diagnostic and operation reviewed; independent saved-array inspection verifies180 summaries. Float32 failures recur at steps7/8; common-history float32 and recursive float64 comparisons pass. Full precision-policy validation remains required; C004 stays technical_blocked |
| [C013](records/C013.json) | Which explicit precision policy passes complete independently encoded O0 inference? | Corrected actual run, independent inspection and operation reviewed: encoder64/predictor64 passes2772checks; mixed precision fails one validation rollout.168raw-array summaries and2000assignment costs independently reconstructed. Initial shape exception and preprocessing coverage finding preserved; downstream integration remains |
| [C014](records/C014.json) | Can full64 object inference require exact selected-checkpoint numerical evidence? | Four loader, eight numerical/supervision and 16 C009v2 groups reviewed. Actual n3 seed0 encoder/O0/O1 pass 1,104/2,772/4,164 comparisons. Eight binding-inspector groups and actual independent reconstruction pass; scientific binding negative is separate. Eleven probe-inspector groups and both actual O0/O1 reconstructions pass; initial dtype-coverage gap preserved. Thirteen planning-inspector groups reviewed; actual planning pending |
| [C015](records/C015.json) | Can consolidated assignment transfers preserve decisions and improve object-planning throughput? | Original55-case partial run preserved. Both sampled transfer variants pass independent trace checks; fastest observed time reduction3.02%. User then removed GPU-hour limits. Original planner now runs through U001; neither optimization is adopted and CPU feasibility was cancelled |
| [D001](records/D001.json) | Can frozen camera slots replace engineered tracks in Genesis? | Matched 600/100 units prepared; model comparison pending |
| [D002](records/D002.json) | Does the full historical M5 offline summary still reproduce? | All 400 units replayed; all 144 reported numeric fields exactly match; reviewed |
| [D003](records/D003.json) | Can unchanged P26 stages obey the existing eight-hour D and global budgets during GPU stalls? | External supervisor and eleven focused CPU check groups reviewed; initial malformed-input fixture failure preserved; actual export/S1/comparison await the reviewed n3 core |
| [E001](records/E001.json) | Does the repeat-based value estimate establish impossibility? | Documentation corrected; V2 remains a measured negative |
| [E002](records/E002.json) | Does the evidence archive preserve source/artifact identity and distinguish unreviewed results? | Fault-injection tests pass; latest interim archive independently verifies 2,103 payload files, 58 reviewed result rows and historical source identity; pending experiments remain incomplete |
| [E003](records/E003.json) | Can the complete final evidence fit a larger bounded archive without weakening E002 integrity? | Six software groups reviewed, including all nine inherited cases and an actual 280 MiB synthetic capture; 64 MiB per-file and 512 MiB total limits enforced. Final actual archive remains pending |

[Completion inventory](COMPLETION_INVENTORY_2026-11-16.md) lists the remaining18 comparison rows, P26 stages and later launcher corrections. [U004](records/U004.json) qualifies the final reporter without the obsolete48-hour denominator; four groups including nine inherited cases pass. Final archive remains pending.

[U005](records/U005.json) qualifies the additional-baseline launcher under U001
with original training and complete full-planning/P26/seed-order prerequisites.
Seven CPU groups pass. [U006](records/U006.json) now qualifies A018 metadata
integration with six groups,23 metadata and14 terminal corruption refusals;
its full state computation is unchanged. No additional seed or actual audit has
run.

[U007](records/U007.json) qualifies the additional-baseline numerical launcher
and artifact verifier under U001. Eight integration groups pass; numerical
source and artifact-check computation are unchanged. Actual extra-seed numerical
results remain pending. C011/B004 source inventories match the preserved gated
checkout; complete actual parent admission still needs checking before use.

Results and review digests live beside run artifacts. This index must be updated when a result is interpreted. Status vocabulary: `passed`, `measured_negative`, `technical_blocked`, `budget_incomplete`. Separate implementation validity, representation, prediction, planning and generalization evidence.

See the [first completed RGB encoder](RGB_ENCODER_TRAINING_2026-11-16.md) for audited training, numerical checks and the reviewed shared cache.

See the [first n3 mask-binding result](OBJECT_BINDING_N3_2026-11-16.md): the
selected encoder has zero binding under the declared criterion. This supersedes
earlier pending-binding status; physical probes and planning are separate tests.

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
for the modest L1 advantage at n1, contrasting its lower n3 planning scores.
See the [n6 baseline planning result](BASELINE_PLANNING_N6_2026-11-16.md)
for the severe L1 planning negative against L0 on matching tasks.
See the [all-six planning synthesis](FINAL_BASELINE_PLANNING_2026-11-16.md) and
[completed O0 training and numerical diagnostic history](OBJECT_PREDICTOR_O0_2026-11-16.md).
See the subsequent [complete O0 precision validation](OBJECT_INFERENCE_PRECISION_2026-11-16.md)
for the qualifying full64 policy, rejected mixed policy and pending evaluator integration.
See the [baseline checkpoint report](BASELINE_PROGRESS_2026-11-15.md) for the
completed L1 n1/n3 logs, the frozen L0 n1 snapshot and what their comparisons test.
See the [CPU checkpoint progress report](CHECKPOINT_PROGRESS_2026-11-15.md) for
the reviewed temporal comparison, physical prediction controls and figures.
See the [preliminary checkpoint analysis](CHECKPOINT_ANALYSIS_2026-11-15.md) for
the matched-step measurements, literature interpretation and current runtime limitation.
See the [runtime recovery handoff](RUNTIME_RECOVERY_2026-11-15.md) for the current
infrastructure evidence, preserved checkpoints and postboot gate.

See the [preliminary O0 planning assessment](PRELIMINARY_OBJECT_PLANNING_2026-11-16.md): the independently checked495case prefix has2O0 successes versus486L0 and469L1 on identical tasks. Full O0 and all O1 planning remain pending; [U008](records/U008.json) records this interim-only readout.

[U009](records/U009.json) qualifies accounting-only execution of the14 remaining
encoder/cache/predictor stages. Seven CPU integration groups pass; original
training computation and commands are retained. Initial fixture failure is
preserved. [U010](records/U010.json) now qualifies C008 metadata integration:
seven CPU groups pass, including 31 metadata and 15 terminal corruption refusals.
The eight original state/terminal/source functions remain byte-identical. Actual
later checkpoint audits remain pending; no additional object stage has run.

[U011](records/U011.json) now qualifies the later C014 numerical route under
U001: eight CPU groups and 46 metadata corruption refusals pass. Both outer and
internal GPU deadlines are removed; the numerical loop and five summary/inspection
functions remain byte-identical. The initial lexical fixture failure is preserved.
No actual additional-checkpoint numerical result exists yet.

[U012](records/U012.json) qualifies the 16 remaining object measurements and their
three independent inspectors under U001. Eight CPU groups and 44 metadata
corruption refusals pass. Original scientific commands and inspection calculations
are unchanged; complete reviewed negative predecessors remain admissible. The
preparation and fixture failures are preserved. Actual later binding, probes and
planning still await their fullcore/P26, training, audit and numerical prerequisites.

[U013](records/U013.json) corrects the later baseline CPU launcher to check and
collect `planning/RUN`. Its original `probes/RUN` path refused every required
completed probe. Six CPU groups pass, including four complete launcher fixtures
and additive U005/U006/U007 metadata compatibility. Actual additional-seed
planning remains pending; use the isolated CPU checkout and updated runbook.
