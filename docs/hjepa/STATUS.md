# H-JEPA status

The place to resume from. Design: [PLAN.md](PLAN.md); timescales: [TIMESCALES.md](TIMESCALES.md). Branch
`feature/hjepa`.

## Screen run (21-22 October): every gate passes; level 1 is level with span8w; online C7/C8 are level with C5/C6

`outputs/hjepa/screen_3level` (`configs/hjepa/go2_3level.yaml` with `--steps 15000`, cosine to zero at 15,000; 0.83
s/step, 3.5 h). Three levels stepping 100 ms, 1.6 s and 12.8 s, trained jointly from scratch. The evaluation queue
(`outputs/hjepa/eval_queue.sh`, log `outputs/hjepa/eval_queue.log`) ran every offline measure the LeWM arm has, then
`eval.hjepa_levels`, then 200 missions. The results are in `outputs/hjepa/screen_3level/eval`.

### Training gates (validation mazes, step 14,999)

| Level | Health (effective rank) | H2: true / shuffled macro-actions | One-step prediction |
|---|---|---|---|
| 1 (100 ms) | 16.7 | pass | rollout loss 0.074 (span8w at step 15,000 of its 60,000: 0.086) |
| 2 (1.6 s) | 21.8 | pass | 0.136 |
| 3 (12.8 s) | 21.9 | pass | 0.408 |

### What each level captured (`eval.hjepa_levels`, `levels.json`)

Linear probes are fitted on 60 validation mazes and scored on the other 40. The figures are NMSE: 1 means the probe
does no better than the mean.

| Level | Maze position | Heading | Body (roll, pitch, rates) | Motion | Within-maze variance (chi) |
|---|---|---|---|---|---|
| 1 | 1.02 | 0.48 | 0.54 | 0.30 | 0.85 |
| 2 | 1.00 | 0.49 | 0.54 | 0.29 | 0.85 |
| 3 | 1.02 | 0.51 | 0.55 | 0.30 | 0.84 |

- The three levels hold the same information. The upper levels do not discard fast body state, so this probe shows no
  slow/fast separation.
  - This is expected from the design: each upper state is an MLP of one level-1 z (window 1). An upper state can
    only lose information, and the objective does not push it to lose fast state.
- No level encodes absolute maze position (NMSE about 1.0). Frames are egocentric and mazes differ, so absolute position
  is the wrong target. The displacement test below replaces it.

The upper levels' rollouts were decoded with the heading probe. The table gives the median heading error.

| Horizon | Predicted | Probe floor (encoded truth) | Persistence | Shuffled macro-actions |
|---|---|---|---|---|
| L2, 1.6 s (1 step) | **12.0 deg** | 10.1 | 27.6 | 33.3 |
| L2, 3.2 s (2 steps) | **12.6** | 11.0 | 41.6 | - |
| L2, 4.8 s (3 steps) | **13.5** | 10.9 | 53.7 | - |
| L3, 12.8 s (1 step) | **18.6** | 9.9 | 75.0 | 88.2 |

The upper levels predict where the robot will face seconds ahead, to within a few degrees of what the probe can read
from the true future frame. They do this from the macro-actions: with shuffled macro-actions the predictions are worse
than persistence. Position predictions sit at chance for every source (1.7-1.9 m), because the position probe fails.

**Displacement** (`levels_displacement.json`). A probe reads, from a pair of consecutive states [S_t, S_(t+1) - S_t],
the motion between them in the frame of S_t. It is fitted on true pairs from the fit mazes and decodes each predicted
transition, composed over the L2 rollout. Dead reckoning is the same probe fitted on the transition's raw command
blocks alone. The table gives median position error in mm and median heading error in degrees.

| Horizon | Predicted | Encoded truth (ceiling) | Dead reckoning | No motion | Shuffled macro-actions |
|---|---|---|---|---|---|
| L2, 1.6 s | 26 mm, 2.1 deg | 27, 2.1 | **7, 0.3** | 57, 21.9 | 74, 26.8 |
| L2, 3.2 s | 42, 3.6 | 46, 3.1 | **10, 0.4** | 119, 40.8 | - |
| L2, 4.8 s | 58, 5.1 | 64, 3.8 | **13, 0.6** | 199, 52.1 | - |
| L3, 12.8 s | 639, 23.1 | 654, 27.4 | **262, 12.8** | 969, 73.4 | 810, 82.0 |

- The predicted transitions are as good as encoding the true future frames. The predictors are not the limit; the
  states are, because they carry metric motion only to a few centimetres.
- Dead reckoning from the commands is 2-4 times better than either. On these units the Go2's controller tracks its
  commands closely, so the commands nearly determine the motion. The upper levels' predictions add nothing over
  integrating them.
- The world model can only beat dead reckoning where the commands stop determining the motion: walls, slip and
  friction, and (next on the roadmap) moving obstacles. The validation units seldom contain these.

Cost geometry: the Spearman correlation of latent distance with distance between the two places is weak at every
level.

| Level | All pairs: along the maze | All pairs: straight line | Detour pairs: along the maze | Detour pairs: straight line |
|---|---|---|---|---|
| 1 | 0.16 | 0.20 | 0.004 | -0.006 |
| 2 | 0.19 | 0.22 | 0.017 | -0.011 |
| 3 | 0.21 | 0.24 | 0.036 | 0.001 |

Over all pairs the correlation rises with level. Detour pairs are those where the route through the maze is much longer
than the straight line. On those, only the distance along the maze rises with level. The trend points the right way,
but the values are negligible.

### Level 1 against span8w (span8w at 60,000 steps; H-JEPA at 15,000)

| Measure | H-JEPA level 1 | span8w |
|---|---|---|
| L4 (T7 score) of the C5-style corpus readout | 5.74 mm | 5.0 (corpus), 5.82 (protocol) |
| L4 of the C6 protocol head | 2.48 mm | 2.43 |
| Turn gate from rest within 0.5 m, C5 head (median / p90) | 6.6 / 15.7 mm | 6.2 / 14.2 (fixed C5) |
| Turn gate from rest within 0.5 m, C6 head (median / p90) | 4.5 / 9.0 mm | 4.4 / 8.8 |
| Deadlock states escaped, of 24 (truth 20) | C5 20, C6 20 | C5 19, C6 21 |
| Linear probe at 800 ms, readout-free | 39.7 mm | 35-39 |
| Candidate ranking, top-1 (on-policy) | 0.95 | 0.97-0.99 |
| Rollout error relative to persistence (on-policy, all) | 0.39 | 0.24 |

- Through the readouts, H-JEPA's level 1 already matches span8w, at a quarter of the steps and with gradients from two
  upper levels flowing into its ViT.
- The readout-free rollout error is the one measure that lags. span8w gained most of its quality between 15,000 and
  60,000 steps (L4 9.5 to 7.3 mm under its own readout).

### Missions (`lewm_benchmarks_hjepa.yaml`, `outputs/cohorts/lewm_*_{c7,c8}_hjepa`, finished 06:47)

The arms ran on the same 100 episodes as C5 and C6, with 6 LeWM lanes and rendering on the R9700. The table counts
round trips.

| Condition | C7 | C8 | C5 (fixed) | C6 (protocol) |
|---|---|---|---|---|
| R4 | 20/20 | 19/20 | 20 | 20 |
| R5 friction 0.30 | 14/20 | 14/20 | 17 | 17 |
| R5 friction 0.25 | 0/20 | 1/20 | 0 | 0 |
| R6 reference | 20/20 | 20/20 | 19 | 20 |
| R6 marked strips | 14/20 | 16/20 | 15 | 14 |
| Total | **68/100** | **70/100** | 71/100 | 71/100 |

Pooled discordant pairs (exact sign test):

| Pair | This arm alone succeeded / the other alone succeeded | p |
|---|---|---|
| C7 vs fixed C5 | 4/7 | 0.55 |
| C8 vs protocol C6 | 4/5 | 1.0 |
| C7 vs protocol C5 | 7/6 | 1.0 |

Failures with more than 100 blocked decisions: C7 12, C8 6, fixed C5 9, protocol C6 7. Both new arms are level with
their baselines. The harness reads only level 1, so the upper levels cannot change these numbers. They show only that
joint training left level 1 intact.

## Next

- The full run (`hjepa-full`, 60,000 steps, `outputs/hjepa/full_3level`, log `outputs/hjepa/full_3level.log`)
  started at 08:05 on 22 October. Rerun the evaluation queue and the displacement test on it.
- Split the displacement test by condition (contact, low friction, near walls). There dead reckoning fails, and a
  world model has room to beat it.
- The upper levels show value online only once a controller consumes them, for example as a subgoal or cost for the
  planner. That belongs to the roadmap step that hands harness components to the world model.
