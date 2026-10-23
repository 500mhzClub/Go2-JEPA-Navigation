# H-JEPA status

The place to resume from. Design: [PLAN.md](PLAN.md); timescales: [TIMESCALES.md](TIMESCALES.md). Branch
`feature/hjepa`.

## The H-JEPA baseline: full run (22-23 October)

C7 and C8 are level with C5 and C6 at 70/100 each. Level 1's decoders read best of all LeWM models. The upper levels
predict heading to the probe's limit at 12.8 s, but show no abstraction and give no planning cost yet.

`outputs/hjepa/full_3level` (`configs/hjepa/go2_3level.yaml`, 60,000 steps; 08:07 to 22:29 on 22 October, 0.83
s/step). Level 1 trains on exactly span8w's batches and learning-rate schedule, so the two compare step for step. The
evaluation queue (`outputs/hjepa/eval_queue_full.sh`, log `outputs/hjepa/eval_queue_full.log`) ran the screen's
measures plus the displacement test. Results are in `outputs/hjepa/full_3level/eval`; missions in
`outputs/cohorts/lewm_*_{c7,c8}_hjepa_full` (`lewm_benchmarks_hjepa_full.yaml`, finished 02:45 on 23 October). This
is the frozen H-JEPA baseline.

### Training (validation mazes)

Every gate passes at every level and every validation. The level-1 rollout ratio is the 8-step rollout error divided
by the error of persistence.

| | Full, step 60,000 | Screen, step 15,000 | span8w, step 60,000 |
|---|---|---|---|
| L2 / L3 prediction error | **0.091 / 0.292** | 0.136 / 0.408 | - |
| Effective rank, levels 1 / 2 / 3 | 28.0 / 34.5 / 38.0 | 16.7 / 21.8 / 21.9 | 23.0 |
| Level-1 8-step rollout / persistence | 0.110 | 0.176 | **0.084** |
| Level-1 8-step rollout, held-out look | 0.092 | - | 0.090 |

- Against span8w on the same batches, level 1 led at 15,000 steps (0.173 against 0.198).
- span8w then improved faster and finished about 30% ahead in distribution. On the held-out look they are level.
- The likely cost of the joint objective: level 1's gradient is about 3x span8w's before clipping, and its latent
  spreads over more dimensions (rank 28 against 23).

### Level 1 against span8w and the screen

| Measure | Full | Screen | span8w |
|---|---|---|---|
| L4 (T7 score), C5-style corpus readout | **3.84 mm** | 5.74 | 5.0 (corpus), 5.82 (protocol) |
| L4, C6 protocol head | **2.40 mm** | 2.48 | 2.43 |
| Turn gate from rest within 0.5 m, C5 head (median / p90) | **5.5 / 12.5 mm** | 6.6 / 15.7 | 6.2 / 14.2 (fixed C5) |
| Turn gate from rest within 0.5 m, C6 head (median / p90) | **4.2 / 8.6 mm** | 4.5 / 9.0 | 4.4 / 8.8 |
| Deadlock states escaped, C5 / C6 (truth 20 of 24) | 17 / 20 | 20 / 20 | 19 / 21 |
| Rollout error / persistence, readout-free (on-policy) | 0.29 | 0.39 | 0.24 |
| Linear probe at 800 ms | 36.9 mm | 39.7 | 35-39 |
| Candidate ranking, top-1 | 0.99 | 0.95 | 0.97-0.99 |

- The readouts more than recover the rollout gap: both heads score their best L4 and turn gate so far.
- The exception is the C5 head on the deadlock states (17 against 19). It finds no escape at maze 44's state 153 in
  either R6 condition (truth: either turn; span8w: left). The missions failed at exactly that place (below).

### What each level captured (`levels.json`)

Probe NMSE on the 40 scoring mazes (1 = the mean):

| Level | Maze position | Heading | Body | Motion | chi |
|---|---|---|---|---|---|
| 1 | 0.99 | 0.42 | 0.42 | 0.22 | 0.87 |
| 2 | 0.98 | 0.43 | 0.43 | 0.22 | 0.87 |
| 3 | 0.99 | 0.43 | 0.47 | 0.23 | 0.86 |

Every probe improved on the screen. Level 3 sheds a little body state (0.47 against 0.42), but there is still no real
slow/fast separation. This is the design limit noted for the screen.

Upper-level predictions, decoded (median heading error in degrees; displacement as position mm, heading deg):

| Horizon | Heading: predicted | Heading: encoded truth | Heading: persistence | Displacement: predicted | Displacement: encoded truth | Displacement: dead reckoning | Displacement: no motion |
|---|---|---|---|---|---|---|---|
| L2, 1.6 s | **5.6** | 6.6 | 25.8 | 19, 1.3 | 20, 1.2 | **7, 0.3** | 57, 21.9 |
| L2, 3.2 s | **5.6** | 6.6 | 41.6 | 28, 2.2 | 31, 1.8 | **10, 0.4** | 119, 40.8 |
| L2, 4.8 s | **5.5** | 6.7 | 53.6 | 38, 3.1 | 41, 2.2 | **13, 0.6** | 199, 52.1 |
| L3, 12.8 s | **7.0** | 5.9 | 74.6 | 474, 18.0 | 510, 18.8 | **224, 12.1** | 969, 73.4 |

- Heading errors halved against the screen (12.0 to 5.6 degrees at 1.6 s; 18.6 to 7.0 at 12.8 s). Predictions sit at
  the probe's ceiling.
- Shuffled macro-actions are worse than persistence (29.4 and 85.7 degrees).
- Displacement improved by about a quarter, but dead reckoning from the commands still wins by 2-3x. As at the screen,
  the predictors are not the limit; the states are, and the commands nearly determine the motion on these units.

Cost geometry (Spearman correlation of latent distance with maze-path / straight-line distance):

| Level | All pairs | Detour pairs |
|---|---|---|
| 1 | 0.17 / 0.21 | -0.04 / -0.03 |
| 2 | 0.23 / 0.26 | -0.06 / -0.05 |
| 3 | 0.30 / 0.34 | -0.02 / -0.05 |

Latent distance tracks distance more at each level up, but along the line of sight, not around walls. Planning with
the upper levels needs goal-aware training first.

### Missions

On the same 100 episodes as C5 and C6, with 6 LeWM lanes, rendering on the R9700. Round trips:

| Condition | C7 | C8 | C5 (fixed) | C6 (protocol) | C7 screen | C8 screen |
|---|---|---|---|---|---|---|
| R4 | 20 | 20 | 20 | 20 | 20 | 19 |
| R5 friction 0.30 | 16 | 15 | 17 | 17 | 14 | 14 |
| R5 friction 0.25 | 0 | 0 | 0 | 0 | 0 | 1 |
| R6 reference | 19 | 20 | 19 | 20 | 20 | 20 |
| R6 marked strips | 15 | 15 | 15 | 14 | 14 | 16 |
| Total | **70/100** | **70/100** | 71 | 71 | 68 | 70 |

Pooled discordant pairs (exact sign test):

| Pair | This arm alone / the other alone | p |
|---|---|---|
| C7 vs fixed C5 | 3/4 | 1.0 |
| C7 vs protocol C5 | 6/3 | 0.51 |
| C7 vs C7 screen | 7/5 | 0.77 |
| C8 vs protocol C6 | 2/3 | 1.0 |
| C8 vs C8 screen | 5/5 | 1.0 |

Failures:

| Arm | Missions with contact | Long deadlocks (more than 100 blocked decisions) |
|---|---|---|
| C7 | 22 | 9 |
| C8 | 24 | 4 |
| Fixed C5 | 20 | 9 |
| Protocol C6 | 17 | 7 |

- Most contacts are at friction 0.25, which every arm fails.
- Two of C7's failures are one episode, R6 maze 44 episode 0, in both looks. C7 deadlocked for 115 decisions at the
  state the deadlock replay flags. Floor registration then raised `UnportedBranch('partial floor height correction
  out of bounds')`: a branch of the original harness that was never ported, where the original re-acquires the floor.
  Every other arm completes that episode.
- So the offline deadlock test predicted a mission failure. The unported branch is a harness gap for the failure
  analysis.

The harness reads only level 1, so the missions show that joint training keeps level 1 at least as useful as
span8w's. They cannot show the upper levels' value.

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

## Next (the dynamic-environments branch)

- Dynamic environments and the continuous action contract together (`docs/ACTION_CONTRACT.md`). Explore CEM and other
  planners for the JEPA arms, and a continuous-command solution for the other controllers.
- First H-JEPA experiment there: offline hierarchical planning. CEM over L3/L2 macro-actions from a start frame to a
  goal frame 12.8-25 s later, checked against the macro-actions taken. If static-maze goals are out of reach, the upper
  levels need goal-aware training (the cost geometry above) before they enter the harness.
- Split the displacement test by condition (contact, low friction, near walls, moving obstacles). Dead reckoning fails
  there, and a world model has room to beat it.
- Optional ablation: a separate learning rate or loss weight for level 1, if the rollout gap to span8w starts to
  matter.
- Failure analysis: the unported floor-registration branch (C7, R6 maze 44) joins the harness gaps.
