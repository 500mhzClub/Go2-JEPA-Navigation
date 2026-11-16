# Repository review, 15 November 2026

Review of the project's goals, implementation, development history, results and running work. Snapshot: branch
`feature/second-world`, commit `7dfc837`; local `main` at `1958c24`. Runtime observations approximately
14:31–14:36 UTC. This review changes no implementation, experiment configuration or running job.

The repository is a strong experimental platform with useful negative results and one convincing positive result
for learned object dynamics. It does not yet demonstrate a general self-supervised JEPA navigator. The next work
should establish that missing result under controlled conditions, rather than enlarge or retune the navigation stack.

**Scope of inspection.** Surveyed the full tracked package and test inventory, history, branches and worktrees;
read the goals, plans, reviews and result records; traced the principal model, training, tracking, planning,
simulation and evaluation paths; checked selected saved results against the documents; inspected the live processes,
logs and checkpoints. This is a repository-level scientific and engineering review, not a claim that every line
has been formally audited or every historical experiment rerun. The package contains 210 Python files, about
47,000 lines; tests contain another 4,840 lines.

## The question and the evidence needed

The README asks when JEPA representations carry decision-relevant information beyond command-based motion
prediction, reactive control and explicit memory. The later reviews add a stronger ambition: a reusable recipe for
self-supervised world models that plan in dynamic, unfamiliar environments, with few hand-written decisions.

These are related but different claims. The first can be answered by a carefully scoped negative result. The second
needs a positive chain of evidence:

1. Observations and memory contain the information needed to decide.
2. The representation preserves the relevant state, including small moving objects.
3. Its predicted futures respond correctly to alternative robot actions, beyond current-state and motion baselines.
4. A fixed planner converts that difference into better decisions and closed-loop outcomes.
5. The effect survives fresh environments, training seeds and an explicit compute budget.

The project has learnt a great deal about the first four links. It has not yet connected all five in one system.
Low latent loss, readable object position, good trajectory error, and mission completion are distinct measurements.
None alone establishes the complete claim.

## Development narrative

**Before October, inherited work.** The migration records describe a substantial preceding LeWMQuad-v3 programme,
including the May corpus and later frozen-V-JEPA predictors. This checkout's own history starts on 6 October;
earlier development is documented provenance rather than something independently reconstructed here.

**6–14 October: make the platform reproducible.** Maze generation, locomotion, data collection, learned models,
tracking, planning and evaluation were extracted into a clean package. Replay and equivalence gates checked
decisions, physics traces, generated data and fitted models against the original. This was valuable foundational
work. It established confidence that later differences could be attributed to changes rather than an uncontrolled
port. It established equivalence, not the scientific adequacy of the original measurements.

**16–23 October: improve the world models, discover the static task's limited headroom.** LeWM and then H-JEPA
improved representation/readout measurements, but mission results remained close to command-only prediction:
approximately 72/100 for C1, 71 for C5/C6 and 70 for C7/C8. Decoder protocols mattered, and better offline gates
did not consistently predict better missions. Only H-JEPA's first level drove the navigation harness; these
results do not test the paper's full top-down hierarchical planning procedure.

**23 October–2 November: make moving-obstacle navigation functional.** Continuous locomotion, a candidate lattice,
map clearing, mover trails and close-up handling improved outcomes. The held-out progression of 81 to 112 to 125
round trips out of 140 was primarily attributable to action search and hand-built navigation changes. Learned
walker positions reached parity, but did not become the decisive component.

**6–9 November: expose sensing, interface and measurement failures.** Memory could improve offline and harm
missions. Pose oracles offered little mission headroom. Camera blind zones and held-robot drift confounded the
interaction task; range sensing and the solver's no-slip pass made that task more interpretable. Phase C then
revealed a representation dominated by command history, and several apparently promising gates could be matched
without predicting the future. This was an important methodological correction.

**10–11 November: isolate planning and dynamics.** The true-model planner and the fast ring world made
counterfactuals affordable. The cost-based Genesis controller replaced much of the staged selection logic, though
integration still required terminal, frontier and dispatch handling. The ring JEPA failed, motivating the split
between engineered object tracks with learned dynamics and camera-derived self-supervised slots.

**12–15 November: find M5, then overextend the experiment loop.** M5 learned a correction to constant velocity
and passed the offline trajectory bar. Several-future models and planner adjustments consumed substantial work
before the decisive M5 comparison was completed. M5 then showed much fewer contact stops with approximately
unchanged completion. The learned value failed its decision-ranking test. The Genesis offline transfer was
positive overall but failed its waiting-walker stratum. The subsequent planner port spent four smoke rounds on
interface compatibility and was stopped before its comparison suites. That stop was justified.

**Now: the second-world experiment has actually started.** Six baseline and supervised-reference models are
training on bouncing balls. This corrects the previous tendency to defer the thesis-facing experiment behind
another round of planner work.

The recurring development weakness is locally sensible fixes without a stable end-to-end experimental question.
The recurring strength is unusually candid diagnosis and retention of negative results. The next improvement
should make the question and its deciding experiment stable enough that another review is not needed to restore them.

## What is established

**The infrastructure is valuable.** Exact request replay, seeded data generation, matched controllers, component
oracles, recorded failure causes, pinned worktrees and a fast surrogate world make this a useful research instrument.
The original source migration is particularly careful. Separating sensing, dynamics and planning has already
prevented several incorrect explanations of failures.

**M5 is a real positive result, within its scope.** Under the frozen ring-world rule, four seeds of the four-walker
suite average 16.75/40 completed round trips and 0.5 contact stops for M5, versus 16.25 and 8.25 for constant-velocity
tracking. Timeouts rise from 15.5 to 22.75. This supports substantially fewer contact stops at similar completion
on that suite; it is not evidence of universally superior navigation or a general safety guarantee.

The saved Genesis report reproduces the documented median errors:

| Horizon | M5 | Constant velocity |
|---|---:|---:|
| 1.6 s | 0.0279 m | 0.0466 m |
| 3.2 s | 0.0736 m | 0.1915 m |
| 6.4 s | 0.2509 m | 0.6712 m |

M5 uses observation-derived tracks as targets, so simulator walker labels are not required for its dynamics
training. Nevertheless, it predicts explicit positions with a trajectory loss; the class name `ObjectJEPA` does
not turn this into latent joint-embedding prediction. It is a useful object-dynamics reference for the JEPA claim.

**The negative JEPA findings are useful but specific.** Static ego-motion leaves little room over commands;
from-scratch scene latents have failed to preserve or predict the relevant interactions; the measured reaction
slopes remain weak. This rejects those configurations and training/evaluation combinations. It does not reject
JEPA generally or establish that the proposed slot recipe will work.

**Learned slots are promising perception, not yet dependable tracks.** The reported best binding is 84.9%, below
the 90% gate. The slot implementation is on `feature/camera-slots`, not in this checkout. Its measure matches a
walker to the best slot independently at each frame using image columns. That does not establish stable identity,
low false-positive rates, or retention through occlusion. P26 must measure those properties as well as position error.

## Findings that need attention

**1. The running ball models have a large training/inference discrepancy.** At step 20,000, logged validation
losses for L0 are 0.596, 0.252 and 14.132 for one, three and six white balls; corresponding recent training losses
are roughly 0.1. All three L1 models also show large discrepancies.

On isolated CPU copies of L0's step-20,000 checkpoints, one fixed validation batch of 64 windows gives:

| White balls | Prediction MSE with stored BatchNorm statistics | Prediction MSE with batch statistics |
|---|---:|---:|
| 1 | 0.1949 | 0.0127 |
| 3 | 0.0282 | 0.0119 |
| 6 | 5.3744 | 0.0292 |

The model remained in eval mode except for its BatchNorm1d modules; dropout remained disabled. The same pattern
occurs on training windows, so this is not simply validation-scene overfitting. No source checkpoint was changed.
Sampling used NumPy seed 42, 64 random episodes and starts in 0–96; SIGReg used a Torch generator seeded 1.

This establishes sensitivity to BatchNorm statistics, not the ultimate cause or the final trained result. Check
running statistics, train/eval parity and a training-only recalibration diagnostic. Do not simply plan in training
mode: batch-dependent normalization could couple a candidate's forecast to the other candidates. The scientific
comparison needs stable inference first. See `training/balls.py:154` and `models/lewm.py:392`.

**2. The Genesis result is conditional dynamics transfer.** `data/genesis_units.py:50` takes pose from the simulator
record. `eval/object_measures.py:50` builds the full layout geometry; lines 83–89 supply the tracker with pose and
the predictor with the recorded future robot path. Lines 93–103 skip unmatched walkers and future targets that
are no longer visible or nearby. These are legitimate controls for isolating dynamics, but are easier conditions
than onboard state estimation, partial mapping and occluded prediction. Add detection coverage, tail errors,
initially-visible walkers after occlusion, and estimated-pose/map evaluations before broadening the claim.

Waiting walkers also need the persistence baseline: at 3.2 s, keeping their position gives 0.0089 m median error
against M5's 0.0575 m. Average improvement over constant velocity is not sufficient for the interaction cases.

**3. The value experiment does not prove its bar impossible.** `eval/value_noise.py` chooses the lowest estimated
mean among six tapes using just three fresh futures each, then scores that choice on another realized future.
Its 0.582 regret ratio is a noisy Monte Carlo policy result, not the Bayes-optimal lower bound. A 10,000-replicate
bootstrap of the existing 582 moments, clustered by their 300 source missions, gives an approximate 95% interval
of 0.378–0.807 for the ratio (seed 15). This does not account for every source of uncertainty, but already includes
0.5. More future samples and a separate evaluation ensemble are needed to bound attainable performance.

V2's measured failure stands: regret 30.24 s versus the rule's 30.54 s. Similarly, matching `dice:8` is evidence
that the tested planner is a bottleneck, not a proof that no predictive model could do better with that planner.
Finite sampling, model errors and optimization interact.

**4. P25 answers representation retention before interactive navigation.** The ball world makes white-ball motion
independent of the robot and allows balls to pass through one another (`sim/balls.py:49`). The primary goal tests
only the red ball's final position. A method can succeed without learning reactions or using white-ball forecasts
to avoid collision. Keep this controlled test, but describe it as a representation/planning diagnostic. Include
a red-position-plus-kinematics control, hold/random controls, and the white-ball prediction probes already planned.
Collision-aware navigation would be a separately specified follow-on, not an implicit claim of this benchmark.

Reproducing L0's published failure is a replication check. If a sound L0 succeeds, do not tune the world until it
fails; investigate the implementation differences and retain the result. The current predictor width differs
from the paper, and data-generation choices were reconstructed, as the plan acknowledges.

**5. The strongest relevant baseline remains absent.** The plan calls for a DINO-WM-style dense-patch predictor
beside slots. It is absent from the inspected current and slot branches. Without an appropriate dense-feature
comparison, a future slot result cannot identify whether gains came from pretraining, object grouping, masking,
or simply better training. A matched no-masking object predictor is also necessary for attributing a masking gain.

**6. External reproducibility and current-state documentation lag internal reproducibility.** The README still
describes the October RGB-D controller comparison. `configs/working.yaml` names the earlier hand-planner setup;
later cost/range/no-slip experiments override it. Main, second-world, learned-value and camera-slots contain
different pieces. Artifact download URLs remain unset. The ball trainer does not use the existing lineage utility
or reject mismatched settings when resuming; it overwrites config.json from the new arguments after loading state.
No checked-in CI workflow was found. These are fixable maintainability gaps, not reasons to rewrite the platform.

**7. Simulation success is not a deployment timing result.** `eval/mission.py:265` runs planning synchronously
before the next physics step and logs wall time. Expensive inference does not automatically consume simulated
reaction time. Any real-robot claim needs latency distributions, stale-observation tests and deadline-aware
execution, alongside sensor and dynamics transfer. These are later deployment requirements, not prerequisites
for the current representation experiment.

## Running work

All six jobs are `go2nav.training.balls train`, launched at 14:03:51 UTC from the detached `-p25` worktree pinned
to `7dfc837`. They share the main checkout's outputs. L0 is LeWM; L1 adds supervised Fourier targets. Each runs
seed 0 for n = 1, 3 and 6, with 1,000 training and 100 validation episodes of 100 frames per n.

At the last progress check they were at 20,500–21,000 of 303,000 optimizer steps, approximately 7% complete,
at 10.4–11.2 steps/s per process. Extrapolation gives about 7–7.6 hours remaining if throughput holds. The dedicated
32-GB GPU reported 100% utilization and about 16 GiB VRAM used. Checkpoints and metrics are being written; the
inference discrepancy above means process health must not be confused with model validity.

No ongoing Genesis mission campaign, slot training, O1 ball-object training, or P26 pipeline was found. The ball
world, dataset and L0/L1 trainers exist; ball CEM evaluation, probes and O1 remain planned in this checkout.

## Recommended sequence

1. **Validate inference on current checkpoints now.** Diagnose the BatchNorm discrepancy, check deterministic
   frozen inference and candidate-batch independence, preserve the original run identities. Do not interpret
   low training loss or a failed planner as evidence for representation laziness until these checks pass.
2. **Finish one controlled P25 comparison.** Build and verify the evaluator on the true model and simple controls;
   run L0/L1 before interpreting O1. Hold out final episodes, preserve planned seeds, and report parameter/data/
   compute differences. Do not add several new model lines in response to the first failure.
3. **Complete P26 offline.** Replace engineered tracks with s0-derived tracks on exactly the same units, compare
   frozen M5, slot-trained S1 and constant velocity, and report detection, false positives, identity continuity,
   occlusion and trajectory tails. Establish whether the representation-to-dynamics connection works before
   another planner port. Retain M5 on engineered tracks as the reference, not the JEPA contribution.
4. **Specify a single causal comparison panel.** One observation contract, candidate bank, planner, costs and
   inference budget; compare persistence, constant velocity, learned direct trajectory prediction and the latent
   predictor, with no-rollout/action-shuffle controls. Use a mixed task suite, with non-yielding tree-maze deadlocks
   as a stress stratum. Report completion, contact stops, timeouts and time together, with uncertainty over
   independent missions/layouts and training seeds. Retire extensively tuned validation sets from final testing.
5. **Return to Genesis missions only after a predictive contribution is established.** A forecast-only insertion
   through P18's cost can answer a narrow short-horizon question. Its 0.8-s horizon cannot settle whether
   multi-second interaction prediction helps. If that longer-horizon experiment is needed, specify its execution,
   reflex, mapping and arrival contracts before implementation and treat it as a separate planner experiment.
6. **Consolidate the research record.** Add one current-state manifest with commit/config/model/data identities,
   publish a minimal reproducible artifact bundle, update the README and current baseline entry point, add CI
   for independent tests, and replace narrative impossibility claims with appropriately qualified measurements.

Keep the simulator, locomotion, replay tools, engineered memory, fast ring world, M5 and negative-result record.
Defer further planner heuristics, stochastic-model variants, value redesign and hierarchy work until an experiment
requires them. The next decisive result is an observation-trained representation whose predicted futures improve
decisions under a fixed, independently validated planner.

## Verification and external references

- Ran the current checkout's unit tests with GPU visibility disabled and CPU thread limits:
  `python -m pytest tests/unit -q -m 'not slow'`: **306 passed, 5 skipped, 6 deselected**, 156.38 s.
  The two warnings concern multiprocessing fork in a multithreaded process. This did not rerun the historical
  equivalence campaigns or establish GPU numerical correctness.
- Checked the saved Genesis trajectory reports, robot-path report, V2 ranking and noise summaries against STATUS.
- Ran the CPU BatchNorm diagnostic and the clustered bootstrap described above, without changing their artifacts.
- Re-read the primary sources supporting the current direction. Zhu et al. support the distinction between
  non-collapse and physically useful representation; their balls test and this reconstruction are not an
  interactive navigation benchmark. C-JEPA supports testing object masking, not presuming that masking proves
  causal understanding. DINO-WM motivates the missing dense-feature comparison; H-JEPA uses top-down planning,
  which the C7/C8 mission comparison does not implement.

Primary sources: [Zhu et al.](https://arxiv.org/html/2609.04264v3),
[C-JEPA](https://arxiv.org/html/2602.11389v2),
[DINO-WM](https://arxiv.org/html/2411.04983),
[H-JEPA](https://arxiv.org/abs/2610.06805).

Repository evidence: README; MIGRATION_STATUS; lewm and hjepa plans/status; dynamic PLAN/STATUS; the reviews of
7, 9, 11, 14 and 15 November; `outputs/genesis/m5_genesis_offline.json`; `outputs/genesis/robot_path.json`;
`outputs/value/v2/ranking.json`; `outputs/value/noise/`; `outputs/balls/runs/`.
