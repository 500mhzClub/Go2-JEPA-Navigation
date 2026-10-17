# LeWM status

The place to resume from. Plan and decisions: [PLAN.md](PLAN.md). Branch `feature/lewm`.

## Done (16-17 October 2026)

| Item | Check |
|---|---|
| One arm registry (`go2nav.controllers.registry`): mission CLI, cohort memory, GPU lanes (`--lewm-lanes`) | C1 prelim 30 rerun identical to R4's record (requests, trace, frames, mission rows, outcome) |
| Maze shapes (`grid.Shape`), parallel episode generation, `lewm_sets_v1` (5 x 4; registry committed) | every registered set regenerates byte for byte; the 1,300-layout set builds in 47 s |
| LeWorldModel (`go2nav.models.lewm`), C5/C6 heads | unit tests: sizes, AdaLN-zero identity, causality, rollout alignment, SIGReg against the official statistic, gradients to the encoder |
| Corpus driver (`go2nav.data.lewm_corpus`), randomised looks (`go2nav.sim.random_appearance`) | smoke corpus (20 validation mazes): no failures or contacts, lag 0, renders identical on both GPUs |
| Training (`go2nav.training.lewm`), window batches (`go2nav.data.lewm_dataset`), configs | smoke run killed after a checkpoint and resumed to the end |
| C5/C6 fit (`go2nav.training.lewm_decoder`) with the L3 probe, forecasters (`go2nav.controllers.lewm`), arms | unit tests; C5 and C6 plan inside the harness with smoke models |
| Chain driver (`go2nav.training.lewm_chain`), benchmark config (`configs/experiments/lewm_benchmarks.yaml`) | |
| Corpus (16-17 October, 10 h, 28 workers) | gate L0 passed: 3,999 units, 3.26M windows of 12 frames; fewest of a movement type 28,926 (start from rest); contact stops 14 of 3,999 (0.35%); yaw command-to-rate correlation peaks at lag 0 (0.99) |

The 4,000th unit, `lewm_train/0604_mission`, fails at its first frame on every attempt (the visual odometry finds too
little floor in the start view to fix the plane); the maze's explore and free units are complete. It is left out.
`LEWM_ROOT` below is the corpus root: on the development machine, `go2nav_lewm` on the workspace NVMe (346 GB).

## Go2 chain result (17 October): C5 fails L3 and L4; C6 matches C4

`outputs/lewm/chain` (60,000 steps; interrupted at 52,199 by a GPU fault, resumed from the 52,000 checkpoint after a
power cycle). Gates:

| Gate | Result |
|---|---|
| L1 health | pass: SIGReg 3.0, z std >= 0.81, effective rank 37.8 (validation mazes) |
| L2 action sensitivity | pass: true / shuffled actions 0.077 at 300 ms, 0.073 at 800 ms; true / persistence 0.091 at 800 ms |
| L3 probe | **fail**: median XY 24.5 mm, RMS yaw 3.25 degrees (gate 10 mm, 1.5 degrees) |
| L4 forecast | **fail**: C5's T7 score 21.5 mm (seeds 21.2 / 21.5 / 22.3; gate 15). C6: 5.0 mm (C4: 5.02) |

At 800 ms on T7's sets (median XY, mm): eval_onpolicy C3 12.1, C5 26.1, C4 6.3, C6 6.6; eval_offline C3 5.7, C5
14.7, C4 4.4, C6 4.4. C5's decoded displacements are short by about 17% (median ratio 0.83).

**Diagnosis** (a probe fitted on 246k corpus windows from 300 training units, about 70 times T7's data, evaluated on 60
validation-maze units; 800 ms, median XY on windows moving more than 50 mm): the true future z reads 11.6 mm on
training mazes but 29.7 on validation mazes; the rolled-out z 7.8 and 18.7; the commands alone 4.2; z with commands
5.5. More decoder data does not close the gap, and the probe at 10k / 30k / 60k steps (37.1 / 31.0 / 29.7 mm;
rollout 18.5 / 18.1 / 18.7) shows training longer would not either. The latents hold where a view is in a known maze
much more than how the camera moved, in a form that does not carry to new mazes; what C5 reads comes from the actions
the predictor writes into its rollout. On the held-out look the probe recovers 23% of the displacement.

## Next

1. Decide the response to the L3/L4 failure (below the plan's fallbacks; awaiting Andrew). The chain command:
   `python -m go2nav.training.lewm_chain outputs/lewm/chain --corpus $LEWM_ROOT --recordings-root outputs/recordings`.
2. The paper-faithful comparison: `--config configs/lewm/paper.yaml` into another folder. Gates L1-L4 from
   `world/evaluation.json`, `choice.json` and the fits' `evaluation.json`.
3. If the gates pass: `python -m go2nav.eval.cohort configs/experiments/lewm_benchmarks.yaml --workers 8`.
4. Seed chains (`--seed-offset`), artefact manifest entries, results, README tables.
