# Dynamic environments and continuous commands: status

The place to resume from. Plan: [PLAN.md](PLAN.md). Branch `feature/dynamic-continuous`.

## P1: locomotion policy on command ranges (23 October)

`outputs/locomotion/continuous_seed11`:
- `go2nav.locomotion.train --config configs/locomotion/continuous_ppo.yaml`, seed 11, 4,096 robots, 1,500 iterations,
  about 22 min on the R9700.
- Commands are drawn from a mixture: 50% uniform over forward -0.25 to 0.45 m/s, lateral ±0.20 m/s and yaw ±0.8
  rad/s; 30% the nine primitives; 20% hold.

Benchmark (`go2nav.locomotion.benchmark`; `outputs/locomotion/benchmark_{original_lattice,continuous_seed11}{,_mu0.3}.json`).
The lattice is 90 commands (6 forward x 3 lateral x 5 yaw), 8 robots each. Errors are RMS in (forward m/s, lateral
m/s, yaw rad/s). The shipped policy is measured on the same grid.

| | Shipped policy | Continuous policy |
|---|---|---|
| Lattice within M1 (0.05 m/s, 0.1 rad/s, no resets), nominal floor | 17/90 | **87/90** |
| Lattice within M1, friction 0.3 | 13/90 | 80/90 |
| Lattice median RMS, nominal | 0.034, 0.188, 0.066 | **0.019, 0.022, 0.027** |
| Lattice worst RMS, nominal | 0.144, 0.288, 0.173 | 0.043, 0.062, 0.074 |
| Continuous switching, settled error, nominal | 0.032, 0.105, 0.045 | 0.016, 0.016, 0.019 |
| Continuous switching resets, friction 0.3 | 15 | 5 |

- **M1 (the range): 87/90.**
  - The three misses are the extreme corners: full reverse, full lateral and yaw 0.4-0.8 rad/s.
  - The worst lateral RMS there is 0.062 against the 0.05 threshold. Noted, not retrained.
  - The shipped policy does not track lateral motion at all.
- **M2 (the nine primitives): pass.** They track as well or better on nominal floor, with two small regressions:
  - hold drifts in yaw (mean -0.008 rad/s against -0.002; RMS 0.017 against 0.003);
  - backward undershoots (-0.175 m/s against -0.189).
- **Friction 0.3: far better.** The shipped policy reaches only 0.166 m/s on the 0.30 m/s command and drifts sideways
  at 0.095 m/s; the new policy reaches 0.322 with no drift. R5 at friction 0.30 may improve for every arm, so the
  static baselines may not be "level" on the new policy. M3 measures this.
- **M3:** C1 on the continuous policy, on the 100 benchmark episodes (`loco_continuous_c1.yaml`, cohorts
  `loco_cont_*_c1`), paired with C1 on the shipped policy.

| Condition | Shipped policy | Continuous policy | Wins / losses |
|---|---|---|---|
| R4 | 20 | 18 | 0 / 2 |
| R5, friction 0.30 | 16 | 18 | 2 / 0 |
| R5, friction 0.25 | 1 | **16** | 15 / 0 |
| R6 reference | 20 | 18 | 0 / 2 |
| R6, low-friction strips | 15 | 19 | 5 / 1 |
| Total | 72 | **89** | 22 / 5, sign test p = 0.0015 |

- Low friction gains a great deal, as the benchmark predicted.
- **M3 is not passed yet.** The static gate (R4 and R6 reference) is 36/40 against 40/40: four losses, no wins
  (p = 0.125).
  - Two reached the beacon, then took about twice the shortest path home and ran out of time (R4 37, R6 31 ep0).
  - Two never reached the beacon (R4 49, R6 41 ep1).
  - To diagnose: C1's command model was fitted on the shipped policy's pulse response; the new policy drifts slightly
    in yaw while holding.

## P2 spike: kinematic obstacles in Genesis (23 October)

A cylinder (0.35 m across, 0.9 m tall) is added as a fixed entity and moved with `set_pos(..., zero_velocity=True)`
every physics step, on the CPU backend. It walks at 0.3 m/s into a Go2 holding position under the policy.

- **Deterministic:** two runs give the same robot and obstacle trace hash and pixel-identical frames.
- **Contacts:** reported in `robot.get_contacts` as robot link against obstacle link: first contact at 3.29 s, 6.8 N.
  The kinematic body then shoves the robot aside, so the mission's contact stop must fire first.
- **Rendering and depth follow the moved pose:** depth along the centre row tracks the obstacle from 1.0 m down to
  0.24 m.
- A free (non-fixed) body moved the same way also works, but with larger contact forces (47 N). Fixed is the choice.
- **Needed in `sim/scene.py`:** `fix_raster_order` checks that the visible population is exactly floor, walls and
  robot, so it must admit the obstacles. The contact topology must name obstacle links for attribution.

## P2: moving obstacles in missions (23 October)

`go2nav/sim/obstacles.py`:
- **Placement:** seeded by set, maze, episode and count. Each obstacle starts at least three passages from home, never
  in the beacon cell.
- **Walk:** cell centre to cell centre along passages at 0.15-0.35 m/s, with random pauses.
- **Polite:** an obstacle stops while the robot is within 0.8 m ahead of it, and turns back after 3 s blocked.
- **Simulation:**
  - The cylinders are fixed entities, moved before each physics step after settling, and drawn after the walls.
  - Their walk state is part of the C0 oracle's snapshots, so its what-if branches see the true future.
  - Missions and cohorts take `--obstacles N` / `obstacles: N`.
- **Evaluation:** the run's `native/obstacle_trace.npz` gives the closest approach from the base to an obstacle
  surface. Each obstacle contact is attributed to whichever body was closing faster.

Checks:
- **Static scenes are untouched:** C1 on R4 maze 30 rerun with this code matches its `rexit2_ref` record for all
  51,010 physics steps.
- **Unit tests** (`tests/unit/test_obstacles.py`): seeded placement, passage-following walks, yield and turn-back.
- **Smoke mission:** C1 on R4 maze 30 with two obstacles.
  - No contact; the closest approach was 0.47 m.
  - The robot never reached the beacon. 83% of its plans were holds, 988 of them `LATCHED_RECOVERY_TURN_BLOCKED`: a
    latched turn that the clearance check keeps refusing.
  - This fits the predicted ghost-trail failure (the map remembers every cell an obstacle once occupied). R7 measures
    how common it is.

R7 (`configs/experiments/r7_moving_obstacles.yaml`, cohorts `r7_obst{1,2}_{c0,c1,c5,c7}`):
- R4's 20 episodes, run as-is on the shipped policy with the six primitives.
- One and two obstacles; C0 as the forecast-perfect reference; C1, C5 and C7 paired with their R4 records.
- D2 needs rewording. C0 shares the harness, so it measures the harness with perfect forecasts, not whether an
  episode can be solved.

## R7 results (23 October)

| Arm | 1 obstacle: round trips (beacon) | 2 obstacles: round trips (beacon) | Failures ending with no route on the map |
|---|---|---|---|
| C1 | 2 (4) | 0 (0) | 35 of 38 |
| C5 | 2 (3) | 0 (0) | 36 of 38 |
| C7 | 3 (4) | 0 (0) | 36 of 37 |

- **No contacts.** The closest approach to an obstacle was 0.43 m: the polite obstacles and the dispatch check work.
- **Ghost trails dominate.**
  - Almost every failure ends in `VIEW_BUDGET_EXHAUSTED`, first reached after a median 60-90 s.
  - The map shows no route and no frontier, the robot scans every heading, then holds until the budget runs out.
  - The evaluator called this "budget exhaustion despite progress". It now has its own rows: "no route on the map
    (scans exhausted)" and "budget exhaustion with no route on the map".
- **C0 crashed in all 40 of its missions** (`IndexError`).
  - The oracle empties the session's sample trace before each what-if branch, and the obstacle update read the robot
    position from it.
  - Fixed in 4d6ee96: the session keeps the base position as its own state, captured in snapshots.
  - A C0 mission with the fix (maze 30, one obstacle) ran the full 480 s and failed like the others: 1,041 of 1,199
    plan frames had no route. With perfect forecasts the harness still deadlocks on ghost trails.

## Ghost-trail fix: clearing seen-free cells (4d6ee96)

`go2nav/perception/mapping.py`, `--map-decay on` / `map_decay: 'on'` (off by default, so static records reproduce):
- An occupied cell that a map update sees as floor (its whole square visible floor) and that holds no sample above the
  floor in either image becomes free again, with its fine cells that hold no sample.
  - A cell an obstacle stands in, or a wall in view, is never seen as floor, so it is never cleared.
- When cells clear after the robot ran out of scan headings, its scan restarts.

Test missions (C1, clearing on), each a failure in R7:

| Mission | R7 | With clearing |
|---|---|---|
| Maze 31, 1 obstacle | No beacon | Round trip in 254 s (observed) |
| Maze 30, 2 obstacles | No beacon | Round trip in 115 s (observed) |
| Maze 34, 2 obstacles | No beacon | Round trip in 237 s (observed) |

"Observed": the mission ended on the robot's own confirmed arrival home; the evaluator's physical check is pending.

**R7b** (`configs/experiments/r7b_map_decay.yaml`, cohorts `r7b_*`): R7 and static R4 with clearing on. Stopped after
the static part, because clearing alone leaves a deadlock:
- **Static R4 with clearing: 60/60** (C1 20, C5 20, C7 20; R4 records 20, 18, 17).
- **C0, maze 30, one obstacle, with clearing:** stuck from 40 s, turning in place 3.7 m from the beacon while the
  obstacle walked elsewhere. The trail cells round the corner were never in view, so never seen free; the scan
  restart kept it turning.

**Second rule (80ba129):** when no route is left after a full look around, the map forgets occupied cells not seen
occupied in the last 30 s (`FORGET_UNSEEN_NS`) and the scan restarts. Walls come back as soon as they are seen; the
dispatch check on fresh depth still guards contact.

| Test mission (both rules) | Result |
|---|---|
| C0, maze 30, 1 obstacle | Round trip in 251 s (observed); was stuck with clearing alone |
| C1, maze 30, 1 obstacle | Round trip in 204 s (observed) |
| C1, maze 35, 2 obstacles | No beacon in 480 s, 3.1 m short; nine no-route events, 1,790 cells forgotten |
| C1, maze 40, 2 obstacles | Pose lost at 159 s (visual odometry), obstacles 2.9 m and more behind it |

- **Pose loss is a second failure type.** It also ended the one C0 R7b mission with an obstacle (maze 36), with the
  obstacle 0.8 m away and possibly in view. Visual odometry assumes a static scene. To be measured in R7c.

**R7c** (`configs/experiments/r7c_map_decay_forget.yaml`, cohorts `r7c_*`, log `outputs/dynamic/r7c.log`), running
since 19:47: R7 with both rules for C0, C1, C5 and C7 (160 missions), and static R4 with both rules for C1, C5 and C7
(60).
