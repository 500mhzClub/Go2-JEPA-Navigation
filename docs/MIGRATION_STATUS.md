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
| `go2nav.sim` | 34 | 1,123 | todo |
| `go2nav.locomotion` | 1 | 15 | **done**: training env (Genesis Go2 example port with the primitive bank), PPO training CLI with lineage, run-time policy (bit-identical actions to the original wrapper), tracking benchmark; config equals the shipped cfgs.pkl (`tests/unit/test_locomotion.py`) |
| `go2nav.perception` | 103 | 2,200 | todo |
| `go2nav.planning` | 64 | 1,880 | todo |
| `go2nav.models` | 21 | 583 | todo |
| `go2nav.controllers` | 6 | 180 | todo |
| `go2nav.data`, `go2nav.training` | 10 | 299 | todo |
| `go2nav.eval` | 24 | 719 | todo |
| experiments and reports | 7 | 513 | todo |
| unclassified | 70 | 917 | classify |

### Outside the runtime trace (ported from reading, gated by G5 or a dedicated check)

| Component | Original entry points | Status |
|---|---|---|
| maze generation | `generate_go2_navigation_capability_sets_development.py`, `register_go2_{c3v2,c3v3,stage2_patch}_sets_development.py` | **done** except the stage-2 qualification hook; registered sets shipped in `data/mazes` (3.9 MB) with exclusion registries (307 built graphs) |
| May-corpus scene generation | `lewm_worlds` (T1 lineage only) | todo |
| locomotion RL | `train_genesis_go2_locomotion_contract.{py,sh}`, upstream Genesis go2 example, `check_genesis_go2_policy_contract.{py,sh}` | **ported** (`go2nav.locomotion`); retraining against the shipped policy: see below |
| May-corpus data generation | `datagen_all_resumable.sh` → `run_mass_datagen.sh` → bulk rollout, convert, labels, render replay | todo |
| T1 temporal predictor | `run_dev_proprio_factorial_driver_v1.py` + v03 temporal-JEPA cache builders | todo |
| T2–T5 predictor and readout training | `train_go2_{frozen_vjepa_native_adaptation,balanced_start_predictor,horizon_dense_predictor,full_heading_readout,maze_view_readout,dense_visual_motion_readout}_development.py` | todo |
| training data collection | balanced-start (+horizon), moving-action switch family, geometry-progress family, short pulse, full heading, maze-view training and transfer, C3-v2 rest/turn, C3-v3 on-policy round | todo |
| C1 fit | `fit_go2_local_motion_controls_development.py`, `fit_go2_short_pulse_command_control_development.py` | todo |
| C3/C4 fits | feature caches, `fit_go2_dev_decoder_development.py`, stage-2 fits | todo |
| verified replay, re-render, videos | replay, unmarked re-render, prelim, hero and capability videos | todo |

### Release

| Item | Status |
|---|---|
| artefact hosting (Hugging Face Hub: checkpoints, maze sets, training data, result summaries) | todo |
| paper tables and figures from result files | todo |
| licences and pre-publication review (Andrew) | todo |

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

**Next:**
- **T7 (C3 decoder + C4).** Port `fit_go2_dev_decoder_development.py` and its feature cache (`build_go2_dev_c3_feature_cache_development.py`):
  - the cache's predicted features depend on T4, so rebuild them with each retrained T4;
  - the decoder initialises from **T5a** (full-heading `mixed_data`), not T5b;
  - C4 is fresh, with AdamW lr 3e-4, weight decay 1e-4, clip 1.0; its target normalisation comes from the prereg C4 head (key name to resolve);
  - export the items from `stage2_feature_cache_v1/items.json`.
- **T8:** the stage-2 refits.
- **Then:** the controller and harness runtime port (planning, perception, sim) and the maze benchmark rerun.
- **Comparison per stage:** retrained vs original training-loss histories, then downstream C3/C4 forecast accuracy on the evaluation sets.
