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

## C6 benchmarks (17 October): level with C4 and C1

Round trips on the recorded cohorts' episodes (`outputs/cohorts/lewm_*_c6`, 100 missions, all with exit 0):

| Condition | C6 | C4 | C1 | C3 | Recorded |
|---|---|---|---|---|---|
| R4 | 20/20 | 19/20 | 20/20 | 20/20 | `rexit2_ref`, `c3_shipped_v2` |
| R5, friction 0.3 | 13/20 | 15/20 | 16/20 | 13/20 | `dyn1v2_mu030_cpu`, `dyn1v2_mu030_c3` |
| R5, friction 0.25 | 0/20 | 1/20 | 1/20 | - | `dyn1v2_mu025_cpu` |
| R6 reference | 19/20 | 20/20 | 20/20 | 17/20 | `s2ref_cpu`, `s2ref_c3` |
| R6 marked strips | 15/20 | - | 15/20 | - | `s2eval_cpu` (C1A 16, C1R 14) |

Paired with C4 over the 80 shared missions, the outcomes differ in 5: C6 alone succeeds in 1, C4 alone in 4 (sign test
p = 0.38). C6 missions take about 5 minutes of wall time. C6 adds nothing over the commands, though: in the corpus
probe the commands alone read 800-ms motion on validation mazes to 3.8 mm and with the latents 4.5 mm, so its parity
with C4 is no evidence for the LeWM encoder.

## Screening (17 October)

| Variant (15,000 steps) | `c5` val@800 ms, median XY all / moving (mm) | true-future probe (mm) | Verdict |
|---|---|---|---|
| Go2 chain at 10k / 60k (reference) | 14.2 / 18.5, 13.6 / 18.7 | 22.7 / 37.1, 17.8 / 29.7 | - |
| `inverse` | 14.1 / 19.5 | 21.6 / 34.8 | no gain (L2 passes, rank 26.9; the inverse head's error stays near 0.23) |
| `batch4` (4 mazes x 32 windows) | 14.1 / 19.9 | 23.2 / 38.3 | no gain (L2 passes, rank 21.7); on the held-out look the `c5` moving median improves, 36.4 against 59.2 mm |
| `paper` (teacher-forced, i.i.d. batches of 5-frame windows) | 23.0 / 33.2 | 28.9 / 47.4 | worse (L2 passes, rank 53.4; its 8-step rollout drifts, true / shuffled 0.43 at 800 ms) |
| `idm100` (inverse dynamics 100, SIGReg 0.08) | **8.0** / 15.8 | 12.2 / 30.3 | **passes** (L2 passes: true / persistence 0.23 at 800 ms; rank 8.8, SIGReg 6.6; held-out look `c5` 9.6 mm against 17.9; C6 3.2 mm, below the commands' 3.8) |
| `idm10` (inverse dynamics 10, SIGReg 0.08) | 9.1 / 15.5 | 13.1 / 28.6 | passes (L2 passes; rank 16.2, SIGReg 4.3; step share 0.07; held-out look 11.6 mm; yaw 1.51 against 1.82 degrees) |

The H-JEPA paper (arXiv 2610.06805, LeWM at level 1) reports the same failure on DROID: without inverse-dynamics supervision a purely predictive objective keeps scene identity and drops the agent, and SIGReg, acting only on the marginal distribution, cannot prevent it. Its level-1 weights are IDM 100 with SIGReg 0.08, to be scaled together; `inverse` used 0.1.

Variance shares of z_t (`lewm_corpus_probe --updates 0`, 100 training units; train / validation / transfer): within-unit
0.77 / 0.68 / 0.64 for the Go2 chain (inverse 0.72 / 0.62 / 0.67, batch4 0.80 / 0.73 / 0.65, paper 0.74 / 0.64 / 0.61),
so no DROID-style collapse onto the episode; but the step share (mean squared 100-ms change over twice the variance) is
0.0077 / 0.0066 / 0.0032 (inverse 0.0091, batch4 0.0104, paper 0.0161 on training units): z moves less than 1% of its
spread per step, so it varies with place over many steps, not with each step's motion. `idm100` raises it to 0.21 / 0.23 / 0.20 (within-unit
0.82 / 0.85 / 0.85).

Round 2 (Andrew, 17 October: run it before the visual-transfer cohort): `idm100` (`go2_idm100.yaml`) and `idm10` (the
same at inverse weight 10), 15,000 steps each (`outputs/lewm/screen/run2.sh`). The visual-transfer cohort is cancelled until a LeWM
passes its gates (Andrew, 17 October: the basics first); an earlier attempt ran out of GPU memory beside a training run
and is set aside as `outputs/cohorts/lewm_visual_transfer.oom_attempt1`.


## idm100 chain result (18 October): L3 and L4 still fail; C6 beats C4

`outputs/lewm/chain_idm` (60,000 steps, paused at 20,000 overnight and resumed). L1 passes (SIGReg 4.1, rank 17.3),
L2 passes. L3 fails: 26.0 mm, 3.52 degrees (Go2 chain 24.5, 3.25). L4 fails: C5's T7 score 20.8 mm (seeds 19.9 / 20.8
/ 20.9; Go2 chain 21.5). C6: 4.5 mm (C4 5.02, Go2 chain's C6 5.0).

eval_onpolicy at 800 ms, C5 median XY by movement type (Go2 chain -> idm100): turn 17.2 -> 7.9, hold 3.2 -> 4.6,
cruise 31.9 -> 41.1 (ratio 0.73), arc_steady 34.7 -> 23.0, switch 31.3 -> 45.9 (ratio 0.59); eval_fresh_c3 hold 27.0
-> 3.7. Inverse dynamics fixed rotation and standing still; forward translation is under-read, by the true-future probe
too (cruise 58.8 mm), so in the encoder. C6 on cruise: 4.5 mm (Go2 chain 6.8).

The screen misled: the corpus is mostly slow (median 13 mm at 800 ms, 34% of windows over 50 mm) where T7's on-policy
decisions mostly translate (median 123 mm), so the median over all windows rewarded the rotation and stillness gains.
The probe now also reports translation bins (still < 20, slow < 80, walk < 160, fast mm) and their mean; but that does
not reproduce T7 either (`c5` bin mean: Go2 chain 16.4, idm100 chain 13.1, walk bin 18.0 -> 14.7 at ratio 0.92), and the
true-future probe reads corpus walking at 24 mm but T7 cruising at 59. The latents of T7's frames are not collapsed
(rank 17.6, z std 0.90, as on the corpus). So the corpus probe is not a reliable screen for T7: forward motion in the
decoder contexts is read worse than in corpus windows of the same speed.

## Running (17 October)

| Service | Worktree (commit) | What | Output |
|---|---|---|---|
| `lewm-screen2` | `../Go2-JEPA-Navigation-lewm-chain` (718ed62) | screening round 2, `idm100` then `idm10`, each then probed (`outputs/lewm/screen/run2.sh`; rerunning resumes) | `outputs/lewm/screen` |
| `lewm-chain-idm` | `../Go2-JEPA-Navigation-lewm-chain` (718ed62) | paused 18 October 01:30 at the step-20,000 checkpoint, resumed there at 07:56; from 17 October 22:44: the full chain (60,000 steps, decoder cache, three fits) for `idm100` (lower on `c5` than `idm10`) (`outputs/lewm/chain_idm.sh`; rerunning resumes) | `outputs/lewm/chain_idm` |

The bench worktree links `data/training/*`, `tools/porting/original_roots.json` and `outputs` from this checkout.

## Next

1. **Screening** (Andrew, 17 October: screen variants before any further full chain). Three 15,000-step runs into
   `outputs/lewm/screen/<name>`, each scored with `go2nav.eval.lewm_corpus_probe` (the `c5` head on validation mazes at
   800 ms; the Go2 chain reads 13.6 mm, and a variant needs 9.5 mm or less, on track for L4, to earn a full chain):
   `inverse` (`go2_inverse.yaml`), `batch4` (4 mazes x 32 windows) and `paper` (`paper.yaml`). The chain command:
   `python -m go2nav.training.lewm_chain outputs/lewm/chain --corpus $LEWM_ROOT --recordings-root outputs/recordings`.
2. The paper-faithful comparison: `--config configs/lewm/paper.yaml` into another folder. Gates L1-L4 from
   `world/evaluation.json`, `choice.json` and the fits' `evaluation.json`.
3. If the gates pass: `python -m go2nav.eval.cohort configs/experiments/lewm_benchmarks.yaml --workers 8`.
4. Seed chains (`--seed-offset`), artefact manifest entries, results, README tables.
