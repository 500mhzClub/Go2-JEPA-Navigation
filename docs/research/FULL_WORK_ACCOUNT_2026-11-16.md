# Full work account — 15–16 November 2026

**Status: experimentation stopped at the user's request.** The last trainer was
stopped at 16:03 UTC on 16 November. Its supervisor and child have exited, all
GPU-accounting intervals are closed, and AMD's process report shows no compute
processes on either GPU. No further training or evaluation is being started.

This account covers the repository review, implementation, completed experiments,
operational recovery, subsequent analyses and the stopped forecasting experiment.
Historical Go2 results are identified as pre-existing evidence. Prepared software
is distinguished from experiments actually run. Earlier dated reports remain
available; their then-current instructions to continue are superseded by this stop.

The main outcome is mixed. We completed and verified the original toy baselines
and part of the slot-model comparison, obtained several useful negative results,
and clarified an existing supervised forecaster's response errors. We did not
demonstrate a JEPA with a learned physical head accurately forecasting reactive
obstacles for Go2. The final experiment moved into improving a direct supervised
forecaster. That was a change of research direction that I should have made
explicit, and it was the wrong priority for the question you were asking.

## The intended question and where the work drifted

The intended practical capability is to predict where dynamic obstacles will go,
both independently and in response to the robot's possible actions. You explicitly
accepted learned heads and supplementary supervision to interpret predicted
latents. You also accepted toy environments when they make relevant experiments
faster. Neither clarification required latent-only planning, nor did it imply
that another direct supervised baseline should become the main experiment.

There are three different things in this repository:

| System | Training and output | What it can test |
|---|---|---|
| L0 global ball model | Predict learned representations using the LeWM-style latent objective | Information retained in latents and the behavior of the fixed latent planner |
| L1 global ball model | The same latent model with supervised physical auxiliary losses | Whether those losses change retained state, prediction and planning |
| RGB slot encoder | Reconstruction-trained perception front end | Whether its slots/masks organize useful visual state; this encoder alone is not a JEPA objective |
| O0/O1 slot predictors | Predict frozen learned slot representations; O1 adds anchored masking/recovery | Whether the latent-prediction intervention helps on the controlled toy task |
| Existing M5 and new F004 arms | Predict physical object displacement/position with direct trajectory losses | Supervised object forecasting, including action-response accuracy |

M5 and F004 use a class named `ObjectJEPA`, but that name does not make their
objective joint-embedding prediction. F004 has no learned target-embedding loss.
A JEPA-plus-head experiment instead predicts a future representation and decodes
that representation into physical quantities with a learned head. This distinction
is consistent with the representation-space objective in
[I-JEPA, Sections 2–3](https://arxiv.org/html/2301.08243v3).

The original repository review already made this distinction about M5. I failed
to preserve it when translating your later clarification into the next experiment.
I treated permission to use physical heads as permission to move the primary
comparison to direct supervised dynamics. You were right to challenge that.

F004 was not an exact repetition of M5: it aimed to isolate alternative-action
data from an explicit paired-response loss, with matched scenes and labels. That
is a narrower supervised-learning question. It could refine a baseline, but it
does not establish the missing JEPA contribution. Existing Go2 work had already
tested supervised readouts and direct dynamics, so this was insufficient reason
to make another such comparison the next main line of work.

## What was already present when this work started

The [initial repository review](../REVIEW_2026-11-15_repository.md) inspected the
goals, history, principal implementation paths, branches, artifacts and running
processes. Six original ball trainers were already running; I did not initiate
those original launches as part of this follow-through.

The preceding project had progressed from static navigation through moving
obstacles, sensing/memory and planner changes, to isolated object dynamics. Its
important pre-existing observations were:

- Improved offline representation measurements had not consistently produced
  better missions than command-based motion prediction.
- Go2 had already used ridge, MLP and spatial readouts with additional mover
  supervision. Low latent loss and readable current state were already known to
  be insufficient evidence of useful physical future prediction.
- M5 was a useful direct object-dynamics reference. In the recorded four-seed
  ring comparison it averaged 16.75/40 completions and 0.5 contact stops, versus
  16.25 completions and 8.25 contact stops for constant velocity. Its timeouts
  were higher: 22.75 versus 15.5.
- Its historical Genesis median position errors at 1.6/3.2/6.4 seconds were
  0.0279/0.0736/0.2509 m, versus 0.0466/0.1915/0.6712 m for constant velocity.
  Waiting walkers remained weaker than position persistence.

Those M5 numbers were not new achievements from this session. They were existing
evidence I reviewed. They were also conditional on the historical observation,
map, robot-pose and future-path interfaces, not demonstrations of an onboard,
closed-loop JEPA navigator.

## Review, planning and research controls

After the review, I implemented the approved P25/P26 research route and your
literature gate. Records identify hypotheses, primary sources, protocol differences,
success criteria and verification. Subsequent reviews bind to exact result hashes.
Failures and inconclusive outcomes were retained rather than overwritten.

The implementation added task/control generation, physical probes, planning
evaluation, slot training and extraction, P26 adapters, recovery checks, source
and data binding, independent inspectors, process supervision and reporting tools.
The initial broader suite reported 321 passed, 9 skipped and 6 deselected tests;
later targeted suites overlap that coverage and must not be added as independent
test totals. Details are in the [implementation record](IMPLEMENTATION_2026-11-15.md).

This improved auditability. It did not itself produce a scientific result. The
amount of supporting machinery also became disproportionate to the remaining
decisive comparison. A literature citation and a passing software gate cannot
substitute for checking whether the experiment still answers your question.

## Inference problems, GPU hangs and recovery

The first substantive issue was a training/inference discrepancy in the running
ball models. Saved BatchNorm statistics produced much larger prediction errors
than the diagnostic using batch statistics. Train-only recalibration improved
the inspected errors without changing learned weights. Original checkpoints
were preserved; derived inference artifacts were separate.

Float32 inference then showed grouping/device discrepancies for L1. Selecting
math attention alone did not resolve them. The declared common float64/math
inference policy passed the checks for both baseline families. This established
validity for the tested inference route, not that every training operation was
numerically identical or that the model was useful.

Two original trainers stopped progressing. Your supplied stack dumps located
their waits at a GPU scalar synchronization boundary. Later the other four also
stopped progressing. Checkpoints were preserved, the specifically authorized
processes were retired, and CPU restoration checks passed. Recovery attempts
then stalled at initial GPU transfer, including two small isolated transfer
tests. After the authorized workstation reboot, the unchanged primitive test
completed and the six trainers were resumed sequentially.

All six completed **303,000 updates**. Final-state checks covered model/head,
optimizer, scheduler, counters, logs and saved random state; all six final
inference checks passed. The underlying GPU-hang cause was not established.
See [runtime recovery](RUNTIME_RECOVERY_2026-11-15.md) and
[final inference](FINAL_INFERENCE_2026-11-16.md).

While the GPU problem was being resolved, CPU checkpoint analyses examined
latent geometry and physical readouts. One matched L1 n1 comparison from
105,000 to 220,000 updates improved four-step red/white RMSE from 2.316/5.942
to 1.847/4.537 pixels. Latent-distance correlations declined despite those
readout gains. These were single-seed validation diagnostics, not final planning
or transfer evidence. The [checkpoint progress report](CHECKPOINT_PROGRESS_2026-11-15.md)
retains their precise cohort and limitations.

## Completed global toy-model results

The toy contains one controlled red ball and n autonomous white balls. White
balls do not react to the controlled ball, and balls pass through one another.
The fixed planning objective measures the controlled ball's endpoint, with no
collision cost. This is an important limitation: high endpoint success does not
demonstrate dynamic-obstacle avoidance or action-induced obstacle responses.

All six seed0 models completed physical probes and planning. Recorded/direct/
true-model planning controls reached 100% in their declared cells. All 18,000
learned planning cases were independently reconstructed.

| White balls | Model | H4 endpoint success | H6 | H8 |
|---|---|---:|---:|---:|
| 1 | L0 | 97.6% | 98.5% | 98.5% |
| 1 | L1 | 99.7% | 99.7% | 99.4% |
| 3 | L0 | 97.8% | 98.2% | 97.7% |
| 3 | L1 | 94.7% | 94.2% | 94.7% |
| 6 | L0 | 97.1% | 93.5% | 82.6% |
| 6 | L1 | 2.9% | 1.8% | 1.0% |

Each cell uses 1,000 episodes. These are single training seeds; episode intervals
do not measure seed variability. The expected weak-L0 reference-paper behavior
did not reproduce. With L0 already near 98% at H4, the original proposed
30-percentage-point improvement criterion is unreachable on these seed0 tasks.
We retained the mismatch instead of modifying the task to manufacture a weak
baseline. [Full baseline planning report](FINAL_BASELINE_PLANNING_2026-11-16.md).

The physical probes showed a more qualified result:

| n | Model | Current red RMSE | Current white RMSE | Four-step red RMSE | Four-step white RMSE |
|---|---|---:|---:|---:|---:|
| 1 | L0 | 0.833 | 23.957 | 1.011 | 23.842 |
| 1 | L1 | 1.400 | 2.325 | 1.682 | 4.223 |
| 3 | L0 | 0.704 | 20.782 | 0.873 | 20.762 |
| 3 | L1 | 2.682 | 12.718 | 3.019 | 13.638 |
| 6 | L0 | 5.924 | 14.849 | 6.092 | 15.167 |
| 6 | L1 | 20.071 | 14.811 | 21.099 | 15.389 |

Errors are pixels. The readout is a train-fitted ridge probe; white-ball targets
are sorted positional ranks, not persistent tracked identities. L1 substantially
improved access to autonomous-ball state at n1 and n3, while trading away red-ball
precision. At n6 it had no aggregate white prediction advantage and its red
readout and planning were poor. The comparison did not isolate the mechanism.
Known differences from the reference paper included the auxiliary-gradient policy,
predictor width and probing recipe. No post-result retuning isolated those changes.
[Full baseline probe report](FINAL_BASELINE_PROBES_2026-11-16.md).

These results add configuration-specific evidence. They do not newly establish
the general proposition that prediction loss and planning quality can disagree;
that was already known from the Go2 work.

## Completed slot-model work and its negative results

The first n3 seed0 RGB slot encoder completed eight epochs/23,752 updates.
The shared feature cache was exported and verified. O0 and O1 each completed
30 epochs/11,370 updates, with their full-state audits and selected-checkpoint
numerical checks. O1 used the same frozen encoder/cache and added anchored
masking/recovery. The masking anchor was checked against the literature before
training. Neither predictor used physical position labels in its latent objective.

Initial long-rollout float32 checks failed for the selected O0 checkpoint.
Subsequent precision diagnostics separated common-history from recursive effects.
The complete float64 encoder/predictor route passed; a mixed-precision route did
not. The qualified deployed policy was used for the subsequent object evaluations.
This was a numerical correction, not evidence of learned object identity.

The actual downstream findings were poor:

- **Mask binding:** 0 of 20,000 ball appearances reached IoU 0.5, including the
  mostly visible subset. The intended object grouping was not demonstrated.
- **Latent validation loss:** O1's selected logged loss was 0.00402679 versus
  O0's 0.00418979, about 3.89% lower. That did not establish physical usefulness.
- **Physical prediction:** O1 improved on O0, but both lost to retaining the
  current position estimate.

| Four-step position RMSE | O0 | O1 | Hold current estimate |
|---|---:|---:|---:|
| Red, pixels | 78.807 | 60.673 | 21.990 |
| White, pixels | 24.987 | 22.757 | 15.014 |

O1's relative reductions were 23.01% for red and 8.93% for white. Its future-white
correlation remained around 0.501 and missed the declared 0.8 bar. Current/held
readouts were identical across the pair, as expected from the shared encoder.
This is a valid negative for the tested pipeline. It does not prove that all
information is absent or that a nonlinear readout cannot help. No new nonlinear
physical-head experiment was completed for these slot predictors.

The full **O0 planning** run then completed all 3,000 cases:

| Horizon | Success | 95% episode-bootstrap interval | Median endpoint error |
|---|---:|---:|---:|
| H4 | 1.5% | 0.8–2.3% | 7.8291 px |
| H6 | 1.1% | 0.5–1.8% | 9.5546 px |
| H8 | 1.2% | 0.6–1.9% | 11.1945 px |

Independent scoring and operations checks passed. The separately stored controls
passed; the learned-only summary's `controls_passed: false` reflected absent
control rows in that file, not a failed control experiment. O0 used latent-distance
planning, without a physical head at inference. **O1 planning was not run.**

These failures did not isolate the encoder, predictor, cached-history interface
or goal cost as the cause. Poor binding, recursive drift and distributed state
remain possible explanations rather than established mechanisms. Sources:
[binding](OBJECT_BINDING_N3_2026-11-16.md),
[paired probes](OBJECT_PROBES_N3_2026-11-16.md),
[full O0 planning](FULL_O0_PLANNING_2026-11-16.md).

## P26 and the Go2 bridge: prepared, not completed

The Genesis adapter and matched evaluation path were implemented. Existing
feature-cache selections were converted into **600 train/100 validation units**;
all 80 overlapping old validation units retained identical arrays. This used
existing recorded data, with no new Genesis collection.

The legacy **400-unit M5 reference replay** completed on CPU. All 144 reported
numeric fields matched the historical summary at its published precision. That
was a compatibility result for the adapter, not a new model improvement.

The frozen-s0 camera-track export, original 20,000-step S1 training and matched
five-arm offline P26 comparison were **not executed**. Consequently the central
observation-to-forecast transfer experiment remains unfinished. No new real-robot
or closed-loop Genesis improvement was demonstrated by this work.

Even that planned offline comparison supplies recorded future robot poses. It
does not by itself establish command-only prediction, robustness to execution
error, onboard pose/map estimation or real-time operation.

## The later audit of existing reaction evidence

After your clarification, I inspected the existing Go2 and M5/M6/N4 evidence
instead of immediately collecting more data. F002 added an analysis of saved
paired reaction vectors; it did not retrain models or replay simulation.

All three models had the same 511 accepted hold/retreat windows from 338 episodes.
For M5, mean error in the predicted difference between the two obstacle futures
was worse overall than predicting no action-induced difference:

| Horizon | M5 response-vector error | Zero-response error | False response on nonresponsive windows |
|---|---:|---:|---:|
| 1.6 s | 0.0408 m | 0.0323 m | 13.6% |
| 3.2 s | 0.1077 m | 0.0928 m | 30.5% |
| 6.4 s | 0.2508 m | 0.2104 m | 47.0% |

On genuinely responsive windows, M5 did beat the zero-response baseline at all
three horizons. Its useful response predictions were outweighed by errors on
little/no-response cases. Zero response here means no difference caused by the
alternative action; it does not mean a stationary obstacle.

This sharpened the description of an existing model's weakness. It did not
demonstrate a new forecasting capability. The validation windows had been reused,
the old accepted-case files omitted rejection reasons, and their original exact
checkpoint/source bindings were incomplete. M6/N4 had worse saved pooled errors,
but their different recipes prevent attributing that difference to one factor.
[F002 audit and uncertainty](FORECAST_EVIDENCE_AUDIT_2026-11-16.md).

F003 then audited all 5,000 archived branch files and 15,000 branch/parent/metadata
hashes. It checked shared histories, action alignment, matching, contact censoring
and separate layout IDs. Tracker-derived paired targets were accurate on average
but had long tails. Validation target availability fell from 92.3% at 1.6 seconds
to 70.8% at 6.4 seconds; maximum paired-label errors reached 2.41 m. These findings
support explicit label/coverage controls, not an explanation of all M5 errors.
[F003 audit](BRANCH_DATA_AUDIT_2026-11-16.md).

## F004: what I built, what ran, and why it was stopped

F004 proposed three directly supervised deterministic forecasters:

1. Factual: the recorded continuation from each selected starting scene.
2. Branch: alternative recorded robot-action branches from those same scenes.
3. Paired: the same branches, with an additional loss on the difference between
   their physical obstacle futures.

The architecture, initialization, starting scenes, physical truth labels and
update budget were matched. Each arm was specified for 20,000 updates, 128 pairs
per batch and a 16-step/6.4-second autoregressive rollout. Factual continuations
were duplicated to match the trajectory budget. All arms used physical truth as
targets, whereas historical M5 training used tracker-derived targets. This was
therefore a new supervised recipe, not an exact M5 reproduction or a JEPA-head test.

The cache/export and independent verification completed: **104,590 training and
10,601 validation scenes**, with seven futures each, occupying about 1.54 GB.
Initial matching and eligibility were independently reconstructed for all points;
target/input reconstruction covered the first and last retained point per eligible
episode, 7,744 points. Selecting complete factual futures introduces survivor
bias; contact exclusions and reused validation remain limits.

I implemented the data adapter, objectives, paired sampler, complete checkpoints,
training supervisor, terminal inspector, physical metrics, inference/artifact
checks and evaluation/scoring runner. The six software suites reported 8, 7, 7,
6, 7 and 6 passing groups respectively. These cover primitives, training execution,
metrics, terminal inspection, inference/artifacts and evaluation execution. They
are software checks, not 41 independent scientific validations. The evaluation
runner's GPU dispatch was tested with mocks; no selected F004 model completed
actual CPU/GPU numerical evaluation or full forecast scoring.

Only the factual seed0 arm actually trained. It started at 15:16 UTC and was
stopped at your instruction at 16:03 UTC:

| Item | Preserved state |
|---|---|
| Last completed update | 2,361 of 20,000 |
| Latest saved checkpoint | Update 2,000 |
| Other saved states | Initialization and update 1,000 |
| Dedicated-job interval | 2,824.82 seconds, about 47.08 minutes |
| Branch arm | Never started |
| Paired arm | Never started |
| Final selected checkpoint | Does not exist |
| Full physical forecast comparison | Never run |

The supervisor was interrupted after verifying PID, creation time, command and
checkout. Its existing cleanup sent SIGTERM to the owned trainer. It closed the
interval and verified unchanged inputs. The raw supervisor result retains
`technical_blocked`, `KeyboardInterrupt()` and return code `-15`; the separate
review classifies the event as **user_stopped**, not a GPU stall or scientific
negative. No accuracy conclusion is drawn from the interrupted run.

The experiment's intended novelty was the matched branch/paired-loss separation.
That did not justify moving away from JEPA-predicted futures plus a learned head.
The useful reusable parts are the physical reaction/nonreaction measurements and
the audited branch data. Their existence does not recover the missed research
question or turn this stopped run into a successful contribution.

## Engineering, publication and resource use

Work was kept in separate checkouts so active runs retained their source hashes.
The recovery, inference, evaluation, inspection and reporting tools were published
on the corresponding feature branches. The current development branch is
`feature/p25-followthrough`; the main working checkout remains on its original
`feature/second-world` revision. These changes have not been merged into main.

The source archive contains **513 files**, including historical and failed helper
variants previously stored beneath ignored `outputs/` directories. New F002–F004
tools are tracked in the normal source tree. Large datasets/checkpoints remain
local and are not uploaded as code. A prior verified interim archive contains
2,103 payload files and 58 evidence rows, but covers only the earlier A017-era
snapshot. It is not a complete final archive of this programme.

The shared ledger records **11.343 union device-hours** at shutdown. This is
accounting for the intervals in that ledger, not total workstation or development
time; original baseline training is excluded from the additional-work allowance.
Full O0 planning alone took 5.483 hours, and stopped F004 took about 0.785 hours.
Overlapping outer/inner intervals are merged rather than counted twice.

You removed the GPU-hour cap. I retained elapsed accounting and serial GPU use,
and implemented U001–U013 adapters to remove old enforced limits from later
launchers without changing their scientific computation. Most of those later
experiments never ran. The CPU planner diversion/optimization was not adopted
for the completed full O0 result. This operational work consumed considerable
attention and should not be presented as equivalent to answering the research
question more quickly.

## Complete status of the workstreams

| Workstream / records | Actual outcome at stop |
|---|---|
| Initial review, plan and literature gates | Completed; original goals and implementation deficiencies documented |
| A001–A003 inference/calibration | Diagnostics and all-six final checks completed; preserved derived artifacts |
| A004–A016 recovery/custody/supervision | Hangs investigated; authorized cleanup/reboot; six original trainings recovered and completed; root cause unresolved |
| A007/A008/A017 checkpoint analyses | Completed bounded CPU/intermediate diagnostics; not final transfer evidence |
| A018/A019 additional-seed state/numerical tools | Prepared and software-tested; no new seed1/seed2 models |
| B001–B003 controls and original baseline planning | Controls and all six 3,000-case evaluations completed |
| B004 later-seed planning | Prepared; no actual later-seed planning |
| C001/C003–C014 first object pipeline | First encoder/cache/O0/O1 training, state/numerical checks and both probes completed; binding negative; only O0 full planning completed |
| C002 comparison | Binding and partial/input-preparation audits completed; full 18-row comparison not produced |
| C010/C011 later training/probes | Prepared; actual additional seeds absent |
| C015 planner performance work | Diagnostic/implementation work retained; not the adopted full O0 scientific route |
| D001–D003 / P26 | Adapters, prepared corpus and legacy regression completed; new slot export/S1/five-arm comparison absent |
| E001 value-result correction | Unsupported impossibility interpretation corrected; existing V2 negative retained |
| E002–E004 reporting/archive | Interim archives and final-builder capacity/evidence handling prepared; complete final research archive absent |
| U001–U013 overrides/integration | Accounting-only launchers and inspectors prepared/tested; U001/U002 used for completed O0; most later routes unused |
| F001 priority clarification | Documented practical forecasting/head acceptance; subsequently overextended into the wrong primary comparison |
| F002/F003 | Existing reaction analysis and branch-data audit completed |
| F004 | Data and software prepared; factual training interrupted by user; other arms/evaluations never started |

The original additional n3 seeds, n1/n6 object runs, O1 planning, P26 experiment,
full comparison and final archive remain incomplete. Their absence is not a
scientific failure. The research goal has not been achieved and is being paused,
not marked complete.

## Assessment of what the work achieved

The strongest completed contribution is reliable, inspectable evidence about the
tested configurations: recovered full baselines; a strong local L0 result that
contradicts the expected reference pattern; mixed L1 benefits; failed slot binding;
O1's limited probe improvement; very weak O0 latent planning; and a more specific
description of M5's false reactions. The tools and preserved artifacts make those
findings easier to audit and reuse.

What remains missing is the intended positive chain: a JEPA representation that
preserves the needed obstacle state, predicts its independent and action-dependent
futures, exposes those futures through an adequate learned head, and transfers
that benefit to the Go2 observation/action setting. This work has not demonstrated
that chain, or established that it is impossible.

The process failure was allowing valid local checks and adjacent experiments to
replace the decisive question. The toy task's limited interaction content was
known at the initial review. The earlier Go2 head/baseline work was also known.
After the slot negatives, I should have kept the distinction between testing
JEPA future representations with a useful head and improving an existing direct
forecaster explicit. Instead, I made F004 the primary next experiment. Your stop
ends that work; this account does not authorize a replacement experiment.

## Evidence locations and custody

The linked reports contain the detailed measurements, uncertainty, literature
reviews and deviations. The companion
[`FULL_WORK_ACCOUNT_EVIDENCE_2026-11-16.json`](FULL_WORK_ACCOUNT_EVIDENCE_2026-11-16.json)
binds the source reports, research records and selected exact result/review pairs
used for this account. It is a documentation inventory, not a new scientific run
or a completed final reproducibility release.

The stopped run is under `outputs/gated/F004/runs/factual_s0`. Its pinned source
remains commit `f053c265a75cbc31971cc04a0ca1893d251c1391` in the separate
`Go2-JEPA-Navigation-forecast-training` checkout. Stop request/verification are
`outputs/gated/F004/user_stop_request.json` and `user_stop_verification.json`.

| Preserved checkpoint | SHA-256 |
|---|---|
| `initial.pt` | `7a256514796bedb23b47a3a6bb4e98d5303363f0a6f77ffc47a16299cad013ec` |
| `checkpoint_01000.pt` | `7b8c37a15eed50f9e8c198a2595e25c7dbaf0c77f1b6b802a2c105f61ba981f2` |
| `checkpoint_02000.pt` | `3f32ff3d2b673425e011d662c3606f2994808dbd90fa9b091e1c9717ec2fa9af` |

The latest saved state is update 2,000, not the last observed update 2,361.
Nothing in this write-up implies that the stopped run completed its scientific
protocol. No further experiments will run under the old continuation instructions.
