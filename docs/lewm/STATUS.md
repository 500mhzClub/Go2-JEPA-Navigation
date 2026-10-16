# LeWM status

The place to resume from. Plan and decisions: [PLAN.md](PLAN.md). Branch `feature/lewm`.

## Done (16 October 2026)

| Item | Check |
|---|---|
| One arm registry (`go2nav.controllers.registry`): mission CLI, cohort memory, GPU lanes (`--lewm-lanes`) | C1 prelim 30 rerun identical to R4's record (requests, trace, frames, mission rows, outcome) |
| Maze shapes (`grid.Shape`), parallel episode generation, `lewm_sets_v1` (5 x 4; registry committed) | every registered set regenerates byte for byte; the 1,300-layout set builds in 47 s |
| LeWorldModel (`go2nav.models.lewm`), C5/C6 heads | unit tests: sizes, AdaLN-zero identity, causality, rollout alignment, SIGReg against the official statistic, gradients to the encoder |
| Corpus driver (`go2nav.data.lewm_corpus`), randomised looks (`go2nav.sim.random_appearance`) | smoke corpus (20 validation mazes): no failures or contacts, lag 0, renders identical on both GPUs |
| Training (`go2nav.training.lewm`), window batches (`go2nav.data.lewm_dataset`), configs | smoke run killed after a checkpoint and resumed to the end |
| C5/C6 fit (`go2nav.training.lewm_decoder`) with the L3 probe, forecasters (`go2nav.controllers.lewm`), arms | unit tests; C5 and C6 plan inside the harness with smoke models |
| Chain driver (`go2nav.training.lewm_chain`), benchmark config (`configs/experiments/lewm_benchmarks.yaml`) | |

## Running

**Corpus collection**, started 16 October 16:04 as the user service `lewm-corpus` from the frozen worktree
`../Go2-JEPA-Navigation-lewm-data` (commit 64544a0), 28 workers, rendering on EGL device 1:

```bash
systemctl --user status lewm-corpus
tail -f $LEWM_ROOT/collect.log        # one line per unit, with the ETA
```

`LEWM_ROOT` is the corpus root: on the development machine, `go2nav_lewm` on the workspace NVMe (mazes/, units/,
logs/; about 380 GB when complete). If it stops, start the same command again: finished
units are kept, partial ones redone, failed ones retried.

## Next

1. When collection ends: `python -m go2nav.data.lewm_corpus audit $LEWM_ROOT` (gate L0).
2. The chain on the Go2 configuration, as a user service (it resumes itself):
   `python -m go2nav.training.lewm_chain outputs/lewm/chain --corpus $LEWM_ROOT --recordings-root outputs/recordings`.
   The paper-faithful comparison: `--config configs/lewm/paper.yaml` into another folder. Gates L1-L4 from
   `world/evaluation.json`, `choice.json` and the fits' `evaluation.json`.
3. If the gates pass: `python -m go2nav.eval.cohort configs/experiments/lewm_benchmarks.yaml --workers 8`.
4. Seed chains (`--seed-offset`), artefact manifest entries, results, README tables.
