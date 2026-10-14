# Migration status

This repository consolidates the navigation-capability programme of LeWMQuad-v3
(`github.com/500mhzClub/LeWMQuad-v3`, branch `jepa-spatial-world-model-nav`). The plan, with every decision, is
`docs/go2_research_artifact_migration_plan_2026-10-06.md` in that repository. This file records progress and is the place
to resume from.

## Decisions (Andrew, 6 October 2026)

- **Port all of it:**
  - maze generation;
  - RL locomotion training;
  - data generation;
  - training of every learned component;
  - inference for each controller type (C0–C4, plus the C1A and C1R development variants);
  - every experiment that compares the controllers.
- **Public repository,** named Go2-JEPA-Navigation. Pushed to `github.com/500mhzClub/Go2-JEPA-Navigation` with its
  full history on Andrew's instruction. Still for Andrew's review before it is announced: licences (V-JEPA 2.1
  weights, Go2 URDF) and the legacy role label (see "Release").
- **Ship checkpoints with their lineage, and the code to retrain each component from scratch with a seed.** Multi-seed
  full-chain retraining comes later, for publication.
- **ROCm** (7.2, torch 2.12, Genesis 0.4.6).
- **Ship the development extras for later review.**

## The oracle: frozen snapshot A

`LeWMQuad-v3-snapshot-2026-10-06` (local; fresh git history; `SNAPSHOT_MANIFEST.json`):
- 2,751 files (45 MB), exported by explicit, SHA-256-checked manifest from LeWMQuad-v3 commit `935f923a`;
- the static closure of every pipeline entry point, including runtime-bound docs and configs, plus the locomotion
  checkpoint, textures and results documents;
- sealed material excluded by name, never read.

**Check (passed, 6 October):** a C1 mission (prelim maze 30, recovery off) run from the snapshot reproduces the same mission run from the original repository: identical physics trace (51,010 samples), dispatch requests, poses, model calls and outcome; `planning.json` differs only in one wall-clock timing field (`added_routing_s`).

## Gates (every port must pass its gate before it counts as done)

| Gate | Check | Tolerance |
|---|---|---|
| G1 component | new module vs snapshot module on recorded inputs | bit-exact on CPU paths |
| G2 mission replay | the new runner replays logged missions: identical decisions, dispatches, native traces | exact |
| G3 offline | the new cache and fits reproduce the logged fit records and the stage-2 tables | exact |
| G4 fresh closed loop | a small cohort run in snapshot and in the new repository | identical (CPU controllers); C3 within batch-numerics tolerance |
| G5 retraining | each component retrained with its original seed and data order reproduces its logged metrics | exact where deterministic |

## Work list

### Infrastructure

| Item | Status |
|---|---|
| repository skeleton, environment lock (`env/`), custody test | done |
| porting map from the P1 trace (`docs/PORTING_MAP.md`, `tools/porting/porting_map.py`) | done (first pass; 70 files to classify) |
| artefact manifest (`artifacts/manifest.yaml`): checkpoints with SHA-256 (sources as `<root>/path` provenance) | done |
| equivalence tests against the snapshot (`tests/equivalence/`, `tests/unit` with GO2NAV_SNAPSHOT) and the porting gates (`tools/porting`) | done |

### Runtime core (P1 trace: 1,126 functions in 350 files; see PORTING_MAP.md)

| Module | Files | Executed lines | Status |
|---|---:|---:|---|
| `go2nav.mazes` | 10 | 38 | **done**: regeneration reproduces capability_v1 (90 layouts, 180 episodes), c3v2, c3v3 and the stage-2 sets (with their strip-coverage qualification) byte for byte (`tests/unit/test_mazes.py`) |
| `go2nav.sim` | 34 | 1,123 | **done**: scene, appearance, sensors, contacts, mission session, snapshots (C0), dynamics (uniform friction, stage-2 strips v3 with markers and the 0.40-m/s guard, recording stall stop); driven with a recorded C1 mission's commands it reproduces all 51,010 physics samples and 1,006 sensor frames exactly (`tools/porting/sim_check.py`) |
| `go2nav.locomotion` | 1 | 15 | **done**: training env (Genesis Go2 example port with the primitive bank), PPO training CLI with lineage, run-time policy (bit-identical actions to the original wrapper), tracking benchmark; config equals the shipped cfgs.pkl (`tests/unit/test_locomotion.py`) |
| `go2nav.perception` | 103 | 2,200 | **done**: visual odometry (with its degraded modes: measured-increment bridge, cross-camera check, direct and chained corner-flow re-association, joint-camera chained fit, plane/image consensus), floor registration, depth obstacles, routing map; every worker output identical on 11 recorded missions (C0–C4, C1A, C1R, three sensitivity hooks; `tools/porting/replay_check.py`). Added 12 October, each checked on a fresh mission against its original: floor patches too narrow for a plane (height along a known normal), floor re-acquisition after a rejected floor measurement, the visual-support view recovery |
| `go2nav.planning` | 64 | 1,880 | **done** (with the C2 reactive selector, hold-relative heading recovery, and the forecast-degradation, calibrated-margin and pessimistic-unknown variants) |
| `go2nav.models` | 21 | 583 | **done** (see the progress log) |
| `go2nav.controllers` | 6 | 180 | **done** for C0 (oracle), C1, C1A, C1R, C2, C3, C4: replays of recorded missions reproduce every 50 Hz request and mission row (G2), and fresh closed-loop missions reproduce the original runs exactly (G4, below), including the sensitivity hooks (forecast degradation, calibrated margin, look-around unknown cells) |
| `go2nav.data`, `go2nav.training` | 10 | 299 | **done** (see "Outside the runtime trace" and "Retrained chain") |
| `go2nav.eval` | 24 | 719 | mission runner, episode reader and cohort runner **done**: the reader reproduces 15 original evaluations exactly (every arm, contact stops, technical failures); R1 and R3-R6 rerun against the original cohorts (see "Cohort reruns"); experiment reports: todo |
| experiments | 7 | 513 | **done**: one config per comparison in `configs/experiments` (R1, R3-R6, the retrained-chain and retrained-locomotion checks, the decoder data sources); paper tables: todo |
| unclassified | 70 | 917 | classify |

### Outside the runtime trace (ported from reading, gated by G5 or a dedicated check)

| Component | Original entry points | Status |
|---|---|---|
| maze generation | `generate_go2_navigation_capability_sets_development.py`, `register_go2_{c3v2,c3v3,stage2_patch}_sets_development.py` | **done**: every registered set regenerates byte for byte, the stage-2 sets with their qualification (strips over at least 20% of the route; 100 files, 7,853 candidates); shipped in `data/mazes` with exclusion registries |
| May-corpus scene generation | `lewm_worlds` (T1 lineage only) | **done** (`go2nav.corpus.build`, plan `data/corpus/may_2026_plan.json`): all 4,350 scene files of the May corpus (1,450 scenes) are byte-identical to the original build |
| locomotion RL | `train_genesis_go2_locomotion_contract.{py,sh}`, upstream Genesis go2 example, `check_genesis_go2_policy_contract.{py,sh}` | **ported** (`go2nav.locomotion`); retraining against the shipped policy: see below |
| May-corpus data generation | `datagen_all_resumable.sh` → `run_mass_datagen.sh` → bulk rollout, convert, labels, render replay | **done** (`go2nav.corpus.datagen`: resumable chunked rollouts, render plans and labels on the CPU, then the render; the rollout writes the recorded messages directly instead of a ROS bag): one lineage scene per family (all eight) reproduces the May messages line for line (the only difference is the scene-id integer, which the original took from Python's randomised `hash()` and the port from SHA-256), and through the driver a lineage scene's render plan (again up to that integer) and labels are identical. **Rendered frames are not bit-reproducible**, in the original either: on the current Vulkan stack two runs of the original render script differ in 265 of 1,500 frames (at most 115 of 50,176 pixels, by up to 24 levels); the port is within that spread (22 of 1,500 against one original run), and the original and the port differ from the retained May frames alike (4 of 13). `--frames-list` renders only the frames a training set uses (each frame renders independently: a subset equals the sequential render) |
| T1 training rows | the v03 temporal, two-step, proprio, cache-map and factorial row builders | **done** (`go2nav.data.t1_rows`): from the May rollouts the deriver reproduces `data/training/t1_temporal/rows.jsonl` byte for byte (actions and command history recomputed by the rate limiter and checked against every logged block). The row *selection* ships as data: the original chose it through earlier paired-navigation and supervision datasets whose scene-role assignment applied exclusion sets from benchmark roles, which are not reproduced |
| T1 temporal predictor | `run_dev_proprio_factorial_driver_v1.py` + v03 temporal-JEPA cache builders | **done** (`go2nav.training.t1_temporal`); features reproduce the cache to f16 precision; retrained on its record (see "Retrained chain") |
| T2–T5 predictor and readout training | `train_go2_{frozen_vjepa_native_adaptation,balanced_start_predictor,horizon_dense_predictor,full_heading_readout,maze_view_readout,dense_visual_motion_readout}_development.py` | **done** (`go2nav.training.native_predictor`, `go2nav.training.readouts`); schedules equal the originals; retrained on their records |
| training data collection | balanced-start (+horizon), moving-action switch family, geometry-progress family, short pulse, full heading, maze-view training and transfer, C3-v2 rest/turn, C3-v3 on-policy round | **done** (`go2nav.data.collect`, `go2nav.data.corridor`, `go2nav.data.rerender`; plans in `data/collections`): checked cases reproduce the original physics trace, every RGB frame and native depth exactly (full heading, maze-view training, C3-v2 rest/turn, balanced start, balanced-start horizon, geometry family, switch family, including contact-stopped trials); C3-v3 on-policy frames come from verified mission replay. **Known limitation:** short-pulse physics is exact but its frames are not (the floor renders differently in the original, recorded 16 September; not a device or cache effect), so its shipped frames are the record |
| training samples (data/training) | the samples each original training stage saved (exported by `tools/porting/export_*`) | **done** (`go2nav.data.samples`, `go2nav.data.decoder_contexts`): `python -m go2nav.data.samples all` regenerates every file of data/training except T1's from the recordings and mission runs (with their verified replays); on the original recordings all 11 files are byte-identical to the shipped exports (the decoder contexts now name the rest/turn recording relatively instead of by an absolute path) |
| C1 fit | `fit_go2_local_motion_controls_development.py`, `fit_go2_short_pulse_command_control_development.py` | **done** (`go2nav.training.command_model`): the refit reproduces the shipped C1 |
| C3/C4 fits | feature caches, `fit_go2_dev_decoder_development.py`, stage-2 fits | **done** (`go2nav.training.decoder`): on the original cache the ported fit reproduces the deployed decoder-fix fit (8.509 mm) and the stage-2 fit (9.868 mm) exactly (`docs/validation/decoder_fit_port_2026-10-06.md`); on the retrained chain see "Retrained chain" below |
| verified replay, re-render, videos | replay, unmarked re-render, prelim, hero and capability videos | verified replay and re-render **done** (`go2nav.data.rerender`); mission videos **done** (`go2nav.eval.video`: egocentric frames from the hash-checked replay, a chase camera from a second physics-checked replay, route map and status panel; the prelim and capability videos differ only in labels); hero video **done** (`go2nav.eval.hero_video`, forecast fan from the new `forecast_xy_m` planning field; recording it leaves C3 prelim 30 identical in requests, trace, frames, mission rows and all 266 actions); demonstration surfaces **done** (`--demo-textures`, `go2nav.sim.demo_appearance`: meshes identical to the original's) |

### Release

| Item | Status |
|---|---|
| artefact hosting (Hugging Face Hub: checkpoints, maze sets, training data, result summaries) | todo |
| paper tables and figures from result files | todo |
| licences and pre-publication review (Andrew) | todo; draft `THIRD_PARTY_NOTICES.md` (Genesis-derived locomotion environment, rsl-rl, V-JEPA 2.1 weights terms to confirm, CC0 textures); no checkpoint contains encoder weights (largest: 17.2 M parameters) |
| machine paths | none in tracked files (custody test); the git history before commit 8fde314 has absolute paths and the username, and was pushed as it is (Andrew's choice) |
| legacy role label (for Andrew's review) | the 181 files of `data/mazes/prelim_test_v1` carry `role`/`data_role` `"sealed_test"` from before the 1 October declassification, and episode records of runs on them inherit it; the files are byte-bound to the maze-regeneration gate, so the choice is to keep the label with a note or relabel to `prelim_test` and re-bind | open |

## Gate G4 (7 October 2026): fresh closed-loop missions, clean repository vs original runs

Each mission run with `python -m go2nav.eval.mission` reproduces the original run exactly: every 50 Hz request and
executed command, the full physics trace, frame times, mission rows and outcome (`tools/porting/mission_check.py`).

| Arm | Episode | Options | Steps |
|---|---|---|---:|
| C0 (oracle) | dev_tune 0 | recovery on | 5,406 |
| C1 | prelim 30 | recovery on | 5,026 |
| C1A | prelim 30 | recovery off, uniform friction 0.3 | 7,226 |
| C1R | stage-2 eval 30 | recovery off, marked strips 0.3 | 12,961 |
| C2 | prelim 31 | recovery on | 8,846 |
| C3 | prelim 30 | recovery off | 5,501 |
| C4 | prelim 45 | recovery on (full 480-s budget) | 24,000 |
| C1, forecast turn scale 2.0 | prelim 31 | recovery off | 11,581 |
| C1, look-around unknown cells p95 | prelim 30 | recovery off | 6,096 |
| C1, calibrated margin p95 | prelim 31 | recovery off (full budget) | 24,000 |
| C1, forecast noise 40 mm | prelim 30 | recovery off; the tracker's direct-flow fallback once (frame 2596); contact stop as the original | 16,116 |
| C1R, marked strips 0.3 | stage-2 eval 38 | recovery off; pose lost at frame 1305 as the original, after every fallback pass | 6,525 |

## R4 rerun (7 October 2026): reserve-exit v2 cohort, clean repository vs original cohort

`python -m go2nav.eval.cohort configs/experiments/r4_reserve_exit_v2.yaml --workers 8`, checked with
`tools/porting/cohort_check.py --traces` (outcome fields, every request and the physics trace): **100 of 100 missions
identical** (rexit2_ref 60: C1, C2, C4 on prelim mazes 30-49; gates 20 + 20: C0 and C1 on dev-tune). The first pass
gave 95; the other five were three missions started during a broken edit of the live tree and two that reached
unported branches (tracker plane/image consensus, hold-relative heading recovery). Both branches were ported and the
five reruns are identical; earlier attempts are kept as `runs/<assignment>.attempt1`.

Preliminary R4 table (recovery off): C1 20/20 round trips, 0 contacts, minimum clearance 0.075 m; C4 19/20, 0
contacts, 0.029 m; C2 15/20, 1 contact sample, 18 hard-violation samples, minimum clearance 0.000 m.

The default harness is the final one (reserve-exit v2 with the coverage fix). The experiments recorded under the
earlier v4 harness (R1, R3 and the decoder's source missions) rerun under `--harness v4` (next section).

## v4 harness and the remaining verification (7 October 2026, evening)

The experiments recorded before the final harness (R1: the 2 October preliminary comparison; R3: forecast sensitivity
and margins; the decoder's data-source missions) ran on the earlier v4 harness. It is now a mode of the port:
`--harness v4` drops the three reserve-exit v2 changes (the reserve exit, the in-place turn exit and C2's
dispatched-path check, restoring C2's footprint rule), and `--coverage-fix off` restores the translation coverage
rule before 2 October (observed floor only). v4-era C3 can also read motion with a plain readout (the C3-v2 readout,
an artefact now). Fresh missions on it reproduce the original v4-era runs exactly (requests, physics trace, frames,
mission rows, outcome):

| Gate | Original run | Options |
|---|---|---|
| R1, C1 recovery off | dev_prelim_off_C1 prelim 30 | v4, coverage fix off |
| R1, C1 recovery on (all six fixes) | dev_prelim_on_C1 prelim 33 | v4, coverage fix off |
| decoder source, C3-v3 round | c3v3_onpolicy_C1 m00 | v4, no fixes |
| R3, C1 forecast scale 0.5 | dev_sens_scale0p5_C1 prelim 30 | v4, coverage fix on |

Configs: `r1_preliminary_comparison.yaml` (332 missions) and `r3_sensitivity_and_margins.yaml` (500), from the
original cohort plans; `decoder_sources.yaml` (the C3-v3 round and the C3-v2 fresh checks).

Also new: `python -m go2nav.data.recordings` regenerates every case of every collection (428) in the original
layout, checked trial by trial against the originals by `tools/porting/collection_check.py`;
`python -m go2nav.data.mission_sources` arranges regenerated decoder-source runs with their verified replays; missions
can run on another locomotion policy (`models: go2_locomotion_policy`; cohort `retrained_locomotion.yaml`); the datagen
rollout takes a scene list (`--scenes`, e.g. T1's 80 scenes with their chunk seeds).

## Regeneration and reruns (12 October 2026)

What has been regenerated or rerun from this repository and compared with the original programme's records.

**Data**

| Stage | Regenerated | Result |
|---|---|---|
| mazes | every registered set, the stage-2 sets with their qualification | byte-identical |
| training collections | all 428 trials of the nine collections (`go2nav.data.recordings`), compared trial by trial (`tools/porting/collection_check.py`: physics trace, command histories, every RGB frame, native depth) | 392 identical; the 36 short-pulse trials have identical physics and different frames (the known limitation above) |
| training samples | T2, T3, T4, the three T5 sets and C1's contexts, rebuilt from the regenerated recordings | all seven byte-identical to the shipped files |
| May corpus for T1 | all 80 scenes T1 uses: rollouts with the per-scene seeds (`data/corpus/may_2026_rollout_seeds.json`), rows, and the 18,101 frames rendered | rows byte-identical to the shipped ones. The seeds file is needed: two scenes were replaced after the May first pass, which moved 36 others one place in the name order the chunk seed rule uses |
| decoder source missions | the C3-v3 on-policy round (22 missions), the C3-v2 fresh checks (10) and the stage-2 recordings (30) | every mission identical to its original |
| decoder contexts | rebuilt from those 62 regenerated missions with their verified replays and the regenerated rest/turn recordings (43,265 rows) | byte-identical to the shipped file |

**v4 harness gates.** With the four in the table above, fresh missions also reproduce exactly: C2 (dev_prelim_off_C2
prelim 30, the full 480-s budget), C3 (dev_prelim_off_C3 prelim 30), C4 (dev_prelim_off_C4 prelim 30) and the first
C3-v2 fresh check (C3 with the C3-v2 readout, full budget). Every arm is now checked on the v4 harness.

**Branches ported after the cohorts reached them** (each checked on a fresh mission against its original: requests,
physics trace, frames, mission rows, outcome):
- obstacles from the downward camera alone, when the forward camera has no returns (ported 7 October);
- a floor patch too narrow for a plane normal: its height along a known normal, for the obstacle observer and for
  the floor registration (sens_fwdscale0p25 C1 prelim 40; sens_noise160 C1 prelim 33);
- view recovery when visual support drops low (s2eval_cpu C1R stage-2 eval 47, episode 1: the same 20 recovery
  plans at the same frames, all 1,086 plan actions equal);
- floor re-acquisition after a rejected floor measurement (s2eval_cpu C1 stage-2 eval 49, episode 0: 12 rejected
  frames, 38 held frames and 190 held requests, as in the original);
- the recovery fixes that R1's recovery-on condition uses beyond those already ported: the terminal position burst
  (C1 prelim 38), the scripted backup (C1 prelim 59 and 79, C4 prelim 44, C0 prelim 32, which ends in the original's
  oracle prefix-check failure), the forecasting controllers' deadlock escape with its relaxed dispatch (C1 prelim 61
  and 55, the full budget: 10 escapes, 10 backup steps, 5 relaxed dispatches). Two parts never occur in the original
  R1 runs and are ported from reading: the frontier exclusion after a repeated stall, and the backup's relaxed dispatch.

Still stopping with `UnportedBranch` (no rerun mission reaches them): the start-up recovery when the first frame
has no floor; tracker frames or references without a floor plane, and its measured-plane height constraint without
a gyro-coherent floor constraint; a floor plane rejected for another reason than too few points or too narrow a patch;
floor and transport corrections beyond their bounds (a pipeline fault in the original too); and a veto view recovery
across a mission phase change.

**Cohort reruns** (`tools/porting/cohort_check.py --traces`: evaluation outcome, every request, the physics trace).
Missions that the original did not evaluate (physical stops) are compared by their recorded ending and traces.

| Experiment | Missions | Identical to the original |
|---|---:|---|
| R1 preliminary comparison (v4 harness; recovery on and off) | 332 | running, from the commit with the recovery fixes (e6817fd); the 201 finished by midday 13 October are all identical |
| R3 forecast sensitivity and margins (v4 harness) | 500 | 500 |
| R4 reserve-exit v2 | 100 | 100 |
| R5 dynamics stage 1 | 120 | 120 |
| R6 dynamics stage 2 | 150 | 150 |
| decoder sources (the C3-v3 round and the C3-v2 fresh checks) | 32 | 32 |

40 of these missions first stopped at a branch not yet ported and were rerun after its port (earlier attempts are kept
as `runs/<assignment>.attempt<N>`). The original `margin_c4_nominal` cohort of R3 was never run (it was relaunched as
`margin_c4_nominal_b`), so the R3 config has 23 cohorts.

**End-to-end run** (regenerated data, then the chain, then the maze test). Every training file regenerated from
this repository alone (T1's rows, the seven sample files and the decoder contexts) is byte-identical to the shipped
one. The chain retrained from the regenerated frames (13 October; new encoder features, original seeds) lands on the
records like the first retrain:

| Stage | Original record | First retrain (original frames) | End-to-end (regenerated frames) |
|---|---|---|---|
| T1 e1 | about 0.324 | 0.326 | 0.326 |
| T2 / T4 final L1 | 0.228 / about 0.240 | 0.2279 / 0.2405 | 0.2253 / 0.2465 |
| T7 C3, 3 seeds (mm) | 9.014 / **8.509** / 7.990 | 11.812 / **8.936** / 8.896 | 9.980 / **8.565** / 7.615 |
| T7 C4 (median seed, mm) | 5.023 | 5.023 (bit-identical) | 4.834 |
| T8 stage-2 C3 (mm) | 9.868 | 9.821 | 10.370 |

The same median seed is chosen each time. C4 differs end to end (its training frames include the re-rendered
short-pulse frames). Maze test on the end-to-end models (prelim 30-49, recovery off, reserve-exit v2 harness, one
seed; `configs/experiments/e2e_chain.yaml`):

| Arm | Shipped models: round trips, contact samples, hard-clearance samples, minimum clearance, median outbound SPL | End-to-end models |
|---|---|---|
| C4 | 19/20, 0, 0, 0.029 m, 0.811 (R4 rexit2_ref) | 20/20, 0, 0, 0.061 m, 0.813 |
| C3 | 20/20, 0, 0, 0.091 m, 0.835 (c3_shipped_v2) | running: 10/10 so far, 0 contacts, 0 hard, 0.081 m |

**Interruption (14 October, 05:03).** A memory-heavy comparison started from this session's tool shell (two whole
feature caches read into memory, 78 GB) was killed by the kernel, and systemd then stopped that shell's scope, which
held the R1 runners and the end-to-end maze test. Nothing finished was lost; both were resumed with
`python -m go2nav.eval.cohort ... --resume` (new: keeps finished missions, sets aside unfinished ones as attempts),
now as their own systemd user services so a tool-shell failure cannot stop them.

## Retrained chain (7 October 2026): T1-T8 from scratch with the original seeds

`python -m go2nav.training.chain OUT --render-root ... --recordings-root ...` retrains every stage in order (resumable;
`--seed-offset K` for other seed chains) and writes `OUT/models.yaml` for `go2nav.eval.mission --models` and the cohort
option `models`. Every stage lands on its original record:

| Stage | Original record | Retrained |
|---|---|---|
| T1 temporal predictor (seed 2026080901, epoch 21) | rgb_rollout final e1 about 0.324 | 0.326 |
| T2 native adaptation | final L1 0.228 | 0.2279 |
| T4 horizon predictor | final L1 about 0.240 | 0.2405 |
| T7 C3 decoder, 3 seeds (eval_onpolicy 800 ms, mm) | 9.014 / **8.509** / 7.990 | 11.812 / **8.936** / 8.896 (same median seed, 2026093011) |
| T7 C4 | 5.463 / 5.023 / 5.468 | identical (the C4 head is bit-identical) |
| T8 stage-2 refit (C3 / C4) | 9.868 / 6.781 | 9.821 / 6.781 (C4 bit-identical) |
| T6 C1 command model | shipped | equal to within 6e-13 (summation order) |

So only C3 changes with the retrained chain.

**Closed loop on the retrained components** (preliminary; one seed, prelim mazes 30-49, recovery off, v2 harness;
`configs/experiments/retrained_chain.yaml` and `retrained_locomotion.yaml`):

| Arm | Component retrained | Shipped: round trips, contact samples, hard-clearance samples, minimum clearance, median outbound SPL | Retrained |
|---|---|---|---|
| C3 | the learned chain (T1-T8) | 20/20, 0, 0, 0.091 m, 0.835 | 18/20, 0, 0, 0.083 m, 0.829 (mazes 39 and 40 used up the step budget holding) |
| C1 | locomotion policy (seed 11) | 20/20, 0, 0, 0.075 m, 0.832 | 17/20, 1, 1,969, -0.005 m, 0.786 (maze 32 stopped on a contact) |
| C2 | locomotion policy | 15/20, 1, 18, -0.000 m, 0.788 | 20/20, 0, 0, 0.017 m, 0.768 |
| C4 | locomotion policy | 19/20, 0, 0, 0.029 m, 0.811 | 19/20, 1, 28, -0.000 m, 0.802 (maze 44 stopped on a contact) |

The shipped C1 command model and the learned arms' training data describe the shipped policy's response; with another
policy they are used unchanged here. Retraining them on the new policy's response is the full seed chain, not yet run.

## Progress log (6 October 2026, evening)

**Done and verified:**
- `go2nav.mazes`: regeneration reproduces the registered sets byte for byte.
- `go2nav.locomotion`: the retrained policy (seed 11) was accepted by Andrew; the comparison is in `docs/validation/`.
- `go2nav.models`: the shipped T4, T5b, C3 decoder, C4 and C1 checkpoints load into the ported classes and match the original modules exactly; seeded initialisation matches the original.
- `go2nav.training`:
  - **T1:** the consolidated dataset reproduces the cached features to f16 precision.
  - **T2–T4:** the regenerated schedules equal the originals.
  - **T5:** the T5b schedule equals the original, and the T5-0 targets match the recorded hash.
  - **T6:** the C1 refit reproduces the shipped model.
- Training datasets are exported to `data/training/*` (hosted artefact, not in git) by `tools/porting/export_*`.

**Running (background; logs in `outputs/training/`):**
- `outputs/training/run_t1_t4.sh` builds the T1 and native features, then retrains T1→T4 with the original seeds.
- `run_t5.sh` follows: readout features, then T5-0 → T5a → T5b.

**Done since (6 October, late afternoon):**
- **T7/T8** (`go2nav.training.decoder`): the cache builder, the matched C3-decoder/C4 fit, evaluation and median-seed
  choice. On the original cache, both the decoder-fix recipe and the stage-2 recipe reproduce their recorded fits exactly.

**Next:**
- When T4 and T5a are retrained: `decoder cache --predictor <retrained T4>`, fit three seeds from the retrained T5a,
  choose the median, and compare with the deployed fit.
- Compare each retrained stage's training-loss history with the original records:
  - T1 rgb_rollout final e1 about 0.324;
  - T2 final L1 0.228;
  - T4 final L1 about 0.240.
- The controller and harness runtime port (sim, perception, planning, controllers, mission runner, evaluation), then
  the maze benchmark rerun:
  - first with the original checkpoints (G2/G4);
  - then with retrained components, one at a time.
