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
- **Public repository,** named Go2-JEPA-Navigation. **Nothing is pushed until Andrew's pre-publication review:** licences
  (V-JEPA 2.1 weights, Go2 URDF), secrets, absolute paths, personal data, sealed material.
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
| artefact manifest (`artifacts/manifest.yaml`): 14 checkpoints with SHA-256 (source paths are provenance; strip before publication) | done (checkpoints) |
| equivalence-test harness against the snapshot (`tests/equivalence/`) | todo |

### Runtime core (P1 trace: 1,126 functions in 350 files; see PORTING_MAP.md)

| Module | Files | Executed lines | Status |
|---|---:|---:|---|
| `go2nav.mazes` | 10 | 38 | **done**: regeneration reproduces capability_v1 (90 layouts, 180 episodes), c3v2 and c3v3 sets byte for byte (`tests/unit/test_mazes.py`); stage-2 qualification waits for `go2nav.sim` patches |
| `go2nav.sim` | 34 | 1,123 | **done**: scene, appearance, sensors, contacts, mission session, snapshots (C0), dynamics (uniform friction, stage-2 strips v3 with markers and the 0.40-m/s guard, recording stall stop); driven with a recorded C1 mission's commands it reproduces all 51,010 physics samples and 1,006 sensor frames exactly (`tools/porting/sim_check.py`) |
| `go2nav.locomotion` | 1 | 15 | **done**: training env (Genesis Go2 example port with the primitive bank), PPO training CLI with lineage, run-time policy (bit-identical actions to the original wrapper), tracking benchmark; config equals the shipped cfgs.pkl (`tests/unit/test_locomotion.py`) |
| `go2nav.perception` | 103 | 2,200 | **done**: visual odometry (with its degraded modes: measured-increment bridge, cross-camera check, direct and chained corner-flow re-association, joint-camera chained fit, plane/image consensus), floor registration, depth obstacles, routing map; every worker output identical on 11 recorded missions (C0–C4, C1A, C1R, three sensitivity hooks; `tools/porting/replay_check.py`) |
| `go2nav.planning` | 64 | 1,880 | **done** (with the C2 reactive selector, hold-relative heading recovery, and the forecast-degradation, calibrated-margin and pessimistic-unknown variants) |
| `go2nav.models` | 21 | 583 | **done** (see the progress log) |
| `go2nav.controllers` | 6 | 180 | **done** for C0 (oracle), C1, C1A, C1R, C2, C3, C4: replays of recorded missions reproduce every 50 Hz request and mission row (G2), and fresh closed-loop missions reproduce the original runs exactly (G4, below), including the sensitivity hooks (forecast degradation, calibrated margin, look-around unknown cells) |
| `go2nav.data`, `go2nav.training` | 10 | 299 | todo |
| `go2nav.eval` | 24 | 719 | mission runner, episode reader and cohort runner **done**: the reader reproduces 15 original evaluations exactly (every arm, contact stops, technical failures); R4 rerun reproduces the original cohort exactly (below); R5/R6 running; experiment reports: todo |
| experiments and reports | 7 | 513 | todo |
| unclassified | 70 | 917 | classify |

### Outside the runtime trace (ported from reading, gated by G5 or a dedicated check)

| Component | Original entry points | Status |
|---|---|---|
| maze generation | `generate_go2_navigation_capability_sets_development.py`, `register_go2_{c3v2,c3v3,stage2_patch}_sets_development.py` | **done** except the stage-2 qualification hook; registered sets shipped in `data/mazes` (3.9 MB) with exclusion registries (307 built graphs) |
| May-corpus scene generation | `lewm_worlds` (T1 lineage only) | todo |
| locomotion RL | `train_genesis_go2_locomotion_contract.{py,sh}`, upstream Genesis go2 example, `check_genesis_go2_policy_contract.{py,sh}` | **ported** (`go2nav.locomotion`); retraining against the shipped policy: see below |
| May-corpus data generation | `datagen_all_resumable.sh` → `run_mass_datagen.sh` → bulk rollout, convert, labels, render replay | todo |
| T1 temporal predictor | `run_dev_proprio_factorial_driver_v1.py` + v03 temporal-JEPA cache builders | **ported** (`go2nav.training.t1_temporal`); features reproduce the cache to f16 precision; retrain running |
| T2–T5 predictor and readout training | `train_go2_{frozen_vjepa_native_adaptation,balanced_start_predictor,horizon_dense_predictor,full_heading_readout,maze_view_readout,dense_visual_motion_readout}_development.py` | **ported** (`go2nav.training.native_predictor`, `go2nav.training.readouts`); schedules equal the originals; retrain running |
| training data collection | balanced-start (+horizon), moving-action switch family, geometry-progress family, short pulse, full heading, maze-view training and transfer, C3-v2 rest/turn, C3-v3 on-policy round | **done** (`go2nav.data.collect`, `go2nav.data.corridor`, `go2nav.data.rerender`; plans in `data/collections`): checked cases reproduce the original physics trace, every RGB frame and native depth exactly (full heading, maze-view training, C3-v2 rest/turn, balanced start, balanced-start horizon, geometry family, switch family, including contact-stopped trials); C3-v3 on-policy frames come from verified mission replay. **Known limitation:** short-pulse physics is exact but its frames are not (the floor renders differently in the original, recorded 16 September; not a device or cache effect), so its shipped frames are the record |
| training samples (data/training) | the samples each original training stage saved (exported by `tools/porting/export_*`) | **done** (`go2nav.data.samples`, `go2nav.data.decoder_contexts`): `python -m go2nav.data.samples all` regenerates every file of data/training except T1's from the recordings and mission runs (with their verified replays); on the original recordings all 11 files are byte-identical to the shipped exports (the decoder contexts now name the rest/turn recording relatively instead of by an absolute path) |
| C1 fit | `fit_go2_local_motion_controls_development.py`, `fit_go2_short_pulse_command_control_development.py` | **done** (`go2nav.training.command_model`): the refit reproduces the shipped C1 |
| C3/C4 fits | feature caches, `fit_go2_dev_decoder_development.py`, stage-2 fits | **done** (`go2nav.training.decoder`): on the original cache the ported fit reproduces the deployed decoder-fix fit (8.509 mm) and the stage-2 fit (9.868 mm) exactly (`docs/validation/decoder_fit_port_2026-10-06.md`); rerun on the retrained T4/T5a pending |
| verified replay, re-render, videos | replay, unmarked re-render, prelim, hero and capability videos | verified replay and re-render **done** (`go2nav.data.rerender`); videos todo |

### Release

| Item | Status |
|---|---|
| artefact hosting (Hugging Face Hub: checkpoints, maze sets, training data, result summaries) | todo |
| paper tables and figures from result files | todo |
| licences and pre-publication review (Andrew) | todo |

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

Scope note: the clean repository implements the final harness (reserve-exit v2 with the coverage fix). Results recorded
under the earlier v4 harness (R1, R3: the 2 October preliminary comparison, sensitivity and margins) are shipped as
data; rerunning them here gives the v2 numbers (R4 documents the v4 to v2 change).

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
