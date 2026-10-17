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

## Running

**The Go2 chain**, started 17 October 02:07 as the user service `lewm-chain-go2` from the frozen worktree
`../Go2-JEPA-Navigation-lewm-chain` (commit 7fd71d1), into `outputs/lewm/chain` (about 8 h: 3,190 images/s):

```bash
systemctl --user status lewm-chain-go2
tail -f outputs/lewm/chain/logs/world.log    # and outputs/lewm/chain/world/metrics.jsonl
```

It restarts itself on failure and resumes from its last checkpoint. The worktree links `data/training/decoder`
(untracked) from this checkout. The first launch failed at compile: systemd's soft limit of 1,024 open files is below
the corpus's one descriptor per memory-mapped unit; `Corpus` now raises it.

## Next

1. Gates L1-L4 of the Go2 chain. The chain command, for reruns:
   `python -m go2nav.training.lewm_chain outputs/lewm/chain --corpus $LEWM_ROOT --recordings-root outputs/recordings`.
2. The paper-faithful comparison: `--config configs/lewm/paper.yaml` into another folder. Gates L1-L4 from
   `world/evaluation.json`, `choice.json` and the fits' `evaluation.json`.
3. If the gates pass: `python -m go2nav.eval.cohort configs/experiments/lewm_benchmarks.yaml --workers 8`.
4. Seed chains (`--seed-offset`), artefact manifest entries, results, README tables.
