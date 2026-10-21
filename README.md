# Go2-JEPA-Navigation

[![JEPA navigation demo: a simulated Unitree Go2 reaches the goal in a textured maze](assets/demo/jepa-navigation-preview.webp)](assets/demo/jepa-navigation.mp4)

**[Full navigation demo — 1080p, 30 fps, 52 seconds](assets/demo/jepa-navigation.mp4)** · Reach the goal, then return home.

When do JEPA world-model representations carry decision-relevant information for robot navigation, beyond kinematic
prediction, reactive control and explicit memory?

This repository is a simulated Unitree Go2 navigation testbed in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis).
It compares seven controller types on procedurally generated mazes: each must reach a beacon, given as a position in
its starting frame, and return home, from onboard RGB-D only.

| Controller | What forecasts the motion of each candidate action |
|---|---|
| **C0** | oracle (privileged geometry and pose) |
| **C1** | command history: a ridge model on past and prospective commands (no images) |
| **C2** | reactive: no motion forecast |
| **C3** | **JEPA**: a frozen V-JEPA 2.1 ViT-L encoder and a learned action-conditioned predictor; a decoder reads motion from the predicted future features |
| **C4** | supervised: the same frozen features, commands and candidate tape, mapped straight to motion |
| **C5** | **LeWM**: a LeWorldModel (arXiv:2603.19312) trained end to end from Go2 pixels with SIGReg and inverse dynamics; a decoder reads motion from its rolled-out latents |
| **C6** | supervised on LeWM's own encoder: its latents, commands and candidate tape, mapped straight to motion |

All controllers share one harness, the reserve-exit-v2 navigation stack: RGB-D tracking, an occupancy map, route
planning, safety vetoes, and an optional recovery layer. Only the motion forecast differs.

> **Status:** the LeWMQuad-v3 development programme is consolidated here as a clean, reproducible artefact. Every
> port was gated against a frozen snapshot of the original code: rerun missions and experiments reproduce the original
> runs exactly, and every component retrains from scratch with its seed. See
> [docs/MIGRATION_STATUS.md](docs/MIGRATION_STATUS.md). All results so far are **preliminary** (development mode).

## What is here

| Path | Contents |
|---|---|
| `go2nav/mazes` | maze and scene generation, set registration with exclusion of earlier graphs |
| `go2nav/sim` | Genesis scene from a maze spec, cameras and sensor packets, depth noise, dynamics perturbations |
| `go2nav/locomotion` | the Go2 PPO locomotion policy: training (rsl-rl) and the command contract |
| `go2nav/perception` | RGB-D visual tracking, floor registration, occupancy mapping, obstacle evidence |
| `go2nav/planning` | candidate primitives, route planning, the reserve-exit harness, safety vetoes, recovery |
| `go2nav/models` | V-JEPA encoder wrapper, the horizon-conditioned predictor, readouts, the C3 decoder, C4, C1, the LeWorldModel and its C5/C6 heads |
| `go2nav/controllers` | C0–C6 and the development variants C1A (adaptive) and C1R (refit); the arm registry |
| `go2nav/corpus` | the May bulk-rollout corpus behind T1: scene generation, resumable bulk rollouts with the data collectors, render plans, labels, the render |
| `go2nav/data` | data collection, verified deterministic replay, training samples, feature caches |
| `go2nav/training` | every learned component, retrainable from scratch with a seed and a lineage manifest |
| `go2nav/eval` | mission and cohort runners, episode evaluation, mission and demonstration videos |
| `configs/experiments` | one config per comparison, run by `python -m go2nav.eval.cohort`; the development variants (forecast degradation, clearance margins, look-around, adaptive and refit C1, marked strips) are mission options |
| `data` | registered maze sets, collection plans, the May corpus plan; training samples (hosted, see `artifacts`) |
| `assets` | CC0 surface textures (corpus render and demonstration videos) |
| `tests` | unit tests and equivalence tests against recorded originals |
| `artifacts` | manifest of checkpoints and data (URLs, SHA-256, licences) and a fetch script |
| `env` | the locked ROCm environment and the corpus render environment |
| `tools/porting` | the porting gates against the original programme (replay, mission, evaluation and cohort checks) |

## Getting started

The pipeline runs in five stages: mazes → locomotion policy → training data → learned models → navigation benchmarks.
Commands run from the repository root, in the main environment, with `PYTHONPATH=.`. Long stages resume where they
stopped. [docs/REPRODUCE.md](docs/REPRODUCE.md) has the detail of each stage and how it was checked against the
original programme.

### 0. Set up

```bash
git clone https://github.com/500mhzClub/Go2-JEPA-Navigation.git && cd Go2-JEPA-Navigation
env/setup_rocm_env.sh .venv && source .venv/bin/activate   # ROCm 7.2, torch 2.12, Genesis 0.4.6 (CPU physics)
env/setup_render_env.sh .venv-render                         # only for rendering T1's corpus (Genesis 0.3.14, Vulkan)
export PYTHONPATH=.
python artifacts/fetch.py vjepa                              # V-JEPA 2.1 ViT-L weights (5.2 GB) and source
```

> **Artefacts.** The pipeline reads trained checkpoints from the go2nav cache (`$GO2NAV_CACHE`, default
> `~/.cache/go2nav`), and training lists and samples from `data/training/`. Both are listed in `artifacts/manifest.yaml`.
> Their hosted download is not published yet. Until it is:
> - a fresh clone can generate the mazes, train the locomotion policy, and run C0 and C2 missions on that policy;
> - the data collectors drive the robot with the shipped policy (`go2_locomotion_policy` in the cache);
> - the decoder-source missions (step 3) run shipped models, including the earlier C3-v2 decoder;
> - T1 trains on a fixed scene and row selection (`data/training/t1_temporal/`).

### 1. Mazes

```bash
python -m go2nav.mazes.cli configs/mazes/capability_v1.yaml --out outputs/mazes   # dev_tune, validation, prelim_test_v1
python -m go2nav.mazes.cli configs/mazes/stage2_sets_v1.yaml --out outputs/mazes  # also c3v2_sets_v1, c3v3_sets_v1
```

The registered sets that missions use ship in `data/mazes`, and regenerating them reproduces their files byte for byte.
A config for a new set lists the maze graphs it must not repeat (`exclude:`); include
`data/mazes/registries/built_graphs_2026-10-06.json` and any set built since.

### 2. Locomotion policy (PPO, rsl-rl)

```bash
python -m go2nav.locomotion.train --out outputs/locomotion/seed11 --seed 11       # about 8 min on one GPU
python -m go2nav.locomotion.benchmark outputs/locomotion/seed11/model_500.pt --out outputs/locomotion/seed11.json
python -m go2nav.locomotion.benchmark ~/.cache/go2nav/go2_locomotion_policy/model_500.pt --out outputs/locomotion/shipped.json
python -m go2nav.locomotion.benchmark --compare outputs/locomotion/shipped.json outputs/locomotion/seed11.json
```

The benchmark measures how well a policy tracks each velocity command. The training defaults reproduce the shipped
policy's run (seed 11, 4,096 environments, 501 iterations); other seeds give a seed chain. Missions take the new policy
through a models file, a YAML map from artefact name to checkpoint (step 5):
`go2_locomotion_policy: outputs/locomotion/seed11/model_500.pt`.

### 3. Training data

The recordings drive the robot through fixed command tapes in the training mazes. Some of these stages take hours;
see the resource notes at the end.

```bash
# Recordings for T2-T8 and C1: 428 trials in nine collections
python -m go2nav.data.recordings outputs/recordings --workers 8

# Missions whose replays also train the decoder: the C3-v3 on-policy round, the C3-v2 checks, the stage-2 recordings
python -m go2nav.eval.cohort configs/experiments/decoder_sources.yaml --out outputs/cohorts
python -m go2nav.eval.cohort configs/experiments/r6_dynamics_stage2.yaml --cohort s2rec2 --out outputs/cohorts
python -m go2nav.data.mission_sources outputs/recordings --cohorts outputs/cohorts

# Training samples (every file of data/training except T1's)
python -m go2nav.data.samples all --recordings-root outputs/recordings --out data/training

# T1's data, from the May bulk-rollout corpus: its 80 scenes and 18,101 rendered frames
python -m go2nav.corpus.build outputs/corpus
python -m go2nav.corpus.datagen rollout outputs/corpus outputs/datagen \
    --scenes data/training/t1_temporal/frames.txt --seeds data/corpus/may_2026_rollout_seeds.json
python -m go2nav.data.t1_rows outputs/datagen --out outputs/t1_rows.jsonl   # equals data/training/t1_temporal/rows.jsonl
python -m go2nav.corpus.datagen render outputs/corpus outputs/datagen --render-python .venv-render/bin/python \
    --textures --frames-list data/training/t1_temporal/frames.txt
```

`--egl-device` picks the render GPU (default 1). Without `--scenes` and `--frames-list`, `datagen` builds the whole
corpus: 1,450 scenes × 48 robots, 69.6 M frames, about 400 GB.

### 4. Learned models

One command retrains every learned component in order, resumably, and writes `outputs/chain/models.yaml`:

```bash
python -m go2nav.training.chain outputs/chain --render-root outputs/datagen/render \
    --recordings-root outputs/recordings [--seed-offset K]
```

| Stage | Module | Trains | Used by |
|---|---|---|---|
| T1 | `t1_temporal` | temporal action-conditioned predictor over frozen V-JEPA features (May corpus) | starts T2 |
| T2 → T3 → T4 | `native_predictor` | adaptation to the Go2's own camera, balanced starts, the horizon-conditioned predictor | **C3** (T4) |
| T5-0 → T5a → T5b | `readouts` | the motion readout: body motion between a frame and a future frame (T5a adds full-heading turns, T5b maze views) | C3, C4 (T5b) |
| T7 | `decoder` | the C3 decoder (reads motion from predicted features) and the matched C4 head (features straight to motion); three seeds, the median kept | **C3**, **C4** |
| T8 | `decoder` | the stage-2 refit of T7 with decisions near low-friction strips | not deployed (it failed its no-loss check away from the strips); `--decoder-fit t8_stage2_c3_c4_refit` loads it |
| T6 | `command_model` | the C1 command-history ridge model (closed form) | **C1** |

C0 (oracle) and C2 (reactive) have nothing to train. With `--seed-offset 0` every stage uses its original seed, and
other offsets give independent seed chains. Each stage also runs on its own
(`python -m go2nav.training.<module> --help`), and each writes a lineage file with its seed, parents and data.

### 5. Navigation benchmarks and evaluation

One mission: the robot must reach a beacon, given as a position in its starting frame, then return home.

```bash
python -m go2nav.eval.mission --arm C3 --set prelim_test_v1 --maze 30 --out outputs/runs/C3_30 \
    [--recovery on] [--models outputs/chain/models.yaml]
python -m go2nav.eval.episode outputs/runs/C3_30       # writes episode_evaluation.json
python -m go2nav.eval.video outputs/runs/C3_30 outputs/videos/C3_30
```

The episode evaluation gives round-trip, beacon and home success, SPL, contacts, clearance violations, minimum clearance
and a stall and failure taxonomy. `--arm` takes C0–C6 or the development variants C1A and C1R (C5 and C6 need
`--models`, below). Other mission options
are degraded forecasts (`--degrade`), low friction (`--friction`, `--strips`), clearance margins and the earlier v4
harness; `--help` lists them all.

Each comparison is one experiment config. Running it evaluates every mission and writes `result.json` and
`summary.md` per cohort:

```bash
python -m go2nav.eval.cohort configs/experiments/r4_reserve_exit_v2.yaml --workers 8 --out outputs/cohorts
python -m go2nav.eval.cohort --summarise outputs/cohorts/rexit2_ref    # the tables again
```

| Config | Missions | Comparison |
|---|---:|---|
| `r1_preliminary_comparison` | 332 | all controllers on the preliminary test mazes, recovery on and off (v4 harness) |
| `r3_sensitivity_and_margins` | 500 | degraded forecasts and calibrated clearance margins (v4 harness) |
| `r4_reserve_exit_v2` | 100 | the controllers under the final reserve-exit v2 harness |
| `r5_dynamics_stage1` | 120 | uniform low floor friction, zero-shot |
| `r6_dynamics_stage2` | 150 | marked low-friction strips |
| `retrained_chain`, `retrained_locomotion`, `e2e_chain` | 40-60 | shipped vs retrained models or policy |
| `lewm_benchmarks` | 310 | C5 and C6 (idm100 world model) on R4's, R5's and R6's episodes, plus a visual-transfer cohort (not run) |
| `lewm_benchmarks_span8w`, `_span8wc`, `_protocol` | 100-200 | C5 and C6 on the same episodes: the span8w world model with T7's fit, the C5 baseline (corpus readout), and the shared readout protocol's heads |

A cohort runs on retrained models with `models: outputs/chain/models.yaml`. `--resume` continues an interrupted
experiment, and `--retry ASSIGNMENT` reruns single missions.

**Resource notes:**
- Physics runs on the CPU and is deterministic: rerun with the same models, a mission reproduces its earlier run.
- C3 and C4 use the GPU for the encoder. A C3 mission takes about 30 minutes, and `--c3-lanes` caps how many run at
  once (keep it at 4 or fewer on one GPU).
- The full chain takes about a day on one AMD R9700 GPU, and its T7 cache needs about 63 GB of disk.
- C5 and C6 missions use the GPU for LeWM; `--lewm-lanes` caps them (6 on one R9700: 9 or more concurrent missions
  livelocked the GPU). Run every GPU arm's cohorts from one cohort process, since lanes are not shared across processes.
- Rendering uses EGL device `EGL_DEVICE_ID` (default 0). On the development machine device 0 is the integrated GPU
  and device 1 the R9700; frames render bit-identically on both, so `EGL_DEVICE_ID=1` only makes many parallel
  missions faster.

### 6. The LeWM arms (C5, C6)

C5 and C6 read a LeWorldModel trained here from Go2 pixels (design, gates and results:
[docs/lewm/PLAN.md](docs/lewm/PLAN.md), [docs/lewm/STATUS.md](docs/lewm/STATUS.md)). Each step is resumable.

```bash
# The corpus: lewm_sets_v1, 1,200 training and 100 validation mazes on a 5 x 4 grid (no evaluation maze is 5 x 4)
python -m go2nav.data.lewm_corpus collect $LEWM_ROOT [--egl-devices 1]
python -m go2nav.data.lewm_corpus audit $LEWM_ROOT                     # the data gate (L0)

# The world model, its T7-style decoder fits and their model set (outputs/lewm/chain/models.yaml)
python -m go2nav.training.lewm_chain outputs/lewm/chain --corpus $LEWM_ROOT --recordings-root outputs/recordings \
    --config configs/lewm/go2_idm_span8w.yaml

# The C5 baseline: the corpus readout alone as C5's decoder (no T7 fit), C6 from the chain's chosen fit
python -m go2nav.training.lewm_readout corpus --world-model outputs/lewm/chain/world/lewm.pt --corpus $LEWM_ROOT \
    --out outputs/lewm/readout
python -m go2nav.training.lewm_readout package --readout outputs/lewm/readout/readout.pt \
    --c6-from CHOSEN_FIT.pt --out outputs/lewm/c5_baseline     # CHOSEN_FIT: the fit outputs/lewm/chain/choice.json names
```

| Step | Module | What |
|---|---|---|
| corpus | `data.lewm_corpus` | explore (C1 with random actions), mission (C1) and scripted free-driving units under random looks; frames, executed commands, poses |
| world model | `training.lewm` | ViT-Tiny encoder and action-conditioned predictor, SIGReg and inverse dynamics over 100- and 800-ms spans (`go2_idm_span8w`); gates L1 (health) and L2 (action sensitivity) |
| decoders | `training.lewm_decoder` | C5's decoder and C6 fitted like C3's decoder and C4 (T7's contexts and recipe); gate L3 (probe) and L4 (T7 score) |
| readouts | `training.lewm_readout` | a decoder trained on corpus windows labelled by the simulator; `package` makes it C5's decoder |
| protocol | `training.readout_protocol` | one fixed readout recipe and data set (`readout_v1`) for every latent arm: C5/C6 and, from full-resolution re-collections (`data.lewm_fullres`), C3/C4 |
| offline checks | `eval.lewm_turn_gate`, `eval.latent_measures`, `eval.lewm_corpus_probe`, `eval.vjepa_probe` | in-place turn forecasts near walls, readout-free measures (linear probe, candidate ranking, rollout error against baselines), screening probes |
| deadlocks | `eval.stuck_replay` | a cohort's stuck decisions replayed through other forecasters, and the simulator's ground truth there |

Missions and cohorts take the model set with `--models` (or the cohort option `models`): `lewm_world_model` and
`lewm_c5_decoder_and_c6`. The benchmarked baseline is the span8w world model with the packaged corpus readout as C5
(71/100 round trips, level with C1, C4 and C6).

## Platform

AMD ROCm 7.2, torch 2.12, Genesis 0.4.6, Python 3.12. See [env/](env/). CPU physics is deterministic, and logged
missions replay bit-exactly.

## Licence

Apache License 2.0: [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party components (Genesis, the Go2 model, rsl-rl,
the V-JEPA 2.1 encoder, CC0 textures) are listed with their terms in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
