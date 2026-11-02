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

**R7c** (`configs/experiments/r7c_map_decay_forget.yaml`, cohorts `r7c_*`): R7 and static R4 with both rules, all
220 missions (finished 24 October, 05:18).

| Arm | Static R4 | 1 obstacle | 2 obstacles | vs R7 (wins / losses) | vs R4, 1 + 2 obstacles (wins / losses) |
|---|---|---|---|---|---|
| C0 | not run | 14 | 8 | 22 / 0 | no R4 record |
| C1 | 20 | 13 | 11 | 22 / 0 | 0 / 16 |
| C5 | 20 | 10 | 8 | 16 / 0 | 0 / 22 |
| C7 | 20 | 11 | 9 | 17 / 0 | 0 / 20 |

- **Static: 60/60**, the same as the R4 records (all 20/20).
- **No contacts** in any mission.
- Round trips take a median 1.5-2.2 times their R4 time.
- C0 counts five missions that came home but failed the oracle's executed-prefix check (an override hold at tick 0
  matches no branch); four more C0 missions failed that check without a round trip.
- **Failures:**
  - Most end with no route after repeated forgetting: failed missions see 6-14 no-route events and forget
    1,400-3,000 cells, against 1-5 and 400-800 in round trips. Forgetting drops real walls, so the robot replans
    through them and finds them again. Obstacle proximity is the same in failures and round trips.
  - With one obstacle, eight no-beacon failures hold on `LATCHED_RECOVERY_TURN_BLOCKED` for most of the mission.
  - **Pose loss** in 8 missions. Maze 36 with one obstacle fails the same way in every arm (26.6 s, opening scan,
    obstacle 0.75 m to the side, in view). The others have the obstacles 2-4 m away, mostly out of view.

## Round 1 rules and the R7d probe (24 October)

Three changes (77096bd):
- **`--map-decay floor`:** with no route left, forget only stale cells once seen as floor (a passage an obstacle
  walked through; a wall's cell is never seen as floor). Forget every stale cell only when none of those is left.
- **`--deadlock-escape on`:** the latch timeout and the deadlock escape, without the other recovery fixes.
  - R7c's latched-turn holds are real walls, not trails. The robot sits 0.49-0.59 m from a true wall (map clearance
    0.44-0.48 m), just inside the 0.48 m a turn needs, so both turns are refused for the rest of the mission.
  - The existing escape never fired, because an active latch disables it.
- **C0 audit:** the executed-prefix check skips plans whose first command was overridden at dispatch. All nine of
  R7c's check failures were this.

**Probe** (`configs/experiments/r7d_probe.yaml`, cohorts `r7dp_*`): 26 R7c missions on the shipped policy (17
failures of every type, 6 round trips, 3 static). Paired with R7c:
- **4 wins, 1 loss** (C0, two obstacles, maze 32: beacon, not home). No contacts; static missions identical.
- Forgetting falls from 1,100-3,200 cells to 0-260 in the one-obstacle missions. With two obstacles the precise rule
  often finds nothing and falls back to forgetting every stale cell (C0: 11-16 no-route events, 1,000-1,600 cells).
- **Four failures changed type, from time-out to pose loss** (C1 one obstacle 37; C1 two obstacles 36 and 39; C5 two
  obstacles 42). Robots that the escape gets moving again now lose visual odometry instead of holding.
- These missions were diagnosed on the test mazes, so the probe checks the changes; it is not a tuning round under
  P2b.

## C1 refit on the continuous policy (24 October): rejected

M3's four static losses all end the same way: the robot about 4 cm from a waypoint (arrival radius 2 cm), choosing
700-900 left turns while the waypoint stays about 48° to the left. On the continuous policy a turn in place moves the
base about 8 mm per 800 ms, so the base circles the waypoint.

The C1 command model was refitted on that policy:
- Corridor recordings (geometry family, switch family, short pulse; 276 trials) re-recorded with `--policy`
  (4f36ff7), giving 4,660 contexts. The refit lowers the 800-ms error on turns in place from 9.3 to 1.6 mm forward
  (in-sample).
- **M3 with the refit** (`loco_continuous_c1_refit.yaml`, cohorts `loco_refit_*`): **71/100**, against 89 for M3
  (6 wins, 24 losses). R4 16, R5 at friction 0.30 14, R5 at 0.25 8, R6 reference 16, R6 strips 17.
- Its failures spend about 1,000 of 1,200 plans within 10 cm of a waypoint. The better model sees the few millimetres
  a turn moves the base as position progress and keeps turning.

The limit is the terminal approach, not C1's forecast: a 2 cm arrival with six primitives on a policy that does not
turn on the spot. M3 keeps the shipped C1 fit. The terminal approach is round 2, developed on mazes 00-29.

## Development rounds dv0-dv2 (24 October): the stopping rule fires

C1 on the continuous policy, recovery off, validation mazes 10-29 episodes 0 and 1, static and with one and two
obstacles (120 missions a round; `configs/experiments/dv{0,1,2}_development.yaml`, cohorts `dv*_c1`).
- **dv0:** the R7c rules (clearing, forgetting every stale cell).
- **dv1 (round 1):** dv0 plus floor-first forgetting, the latch timeout and the deadlock escape.
- **dv2 (round 2):** dv1 plus the terminal burst (`--terminal-burst`) for the waypoint orbit found in M3.

| Round | Static | One obstacle | Two obstacles | Total | Against the previous round | p |
|---|---|---|---|---|---|---|
| dv0 | 29 | 13 | 4 | 46 | | |
| dv1 | 30 | 14 | 7 | 51 | 9 wins, 4 losses (+5) | 0.27 |
| dv2 | 32 | 16 | 7 | 55 | 5 wins, 1 loss (+4) | 0.22 |

dv2 against dv0: 14 wins, 5 losses (+9), p = 0.064. No contacts in any round.

- **Stopping rule:** two rounds in a row with p > 0.1 and under 3 net missions per 40. Rule tuning stops at dv2.
  Constant-velocity obstacle prediction is still added (a required part of the baseline, P2b point 1) as the last
  round; the baseline freezes after it whatever its result.
- **Density:** 80% static, 40% with one obstacle, 18% with two. Most obstacle failures are time-outs without
  reaching the beacon and without contact: the robot waits and replans.
- **Pose loss** (one/two obstacles): dv0 3/1, dv1 4/9, dv2 4/8; none static. As in the R7d probe, the escape turns
  time-outs into visual-odometry losses. To be explained before the freeze (P2b point 4).
- **dv2's 8 static failures are all time-outs.** Three are dominated by motion-dependent footprint rejections (14 ep1,
  15 ep0, 26 ep1); four show little blocking and run out of time (20, 23, 24, 28 ep0).
- **Defect:** maze 26 episode 0 with one obstacle fails at frame 0 in every round (floor plane unavailable): the
  obstacle spawns in front of the start camera. Placement needs a keep-out around the start pose.

## Round 3, dv3, and the freeze (24 October)

dv3: dv2 plus constant-velocity obstacle prediction (`--obstacle-prediction`: tracks of compact clusters standing on
once-seen floor, predicted only once confirmed moving) and footprint clearing (`--footprint-clearing`: map cells
inside the body's outline dropped). Same 120 missions (`configs/experiments/dv3_development.yaml`, cohorts `dv3_*_c1`).

| | Static | One obstacle | Two obstacles | Total |
|---|---|---|---|---|
| dv2 | 32 | 16 | 7 | 55 |
| dv3 | 32 | 13 | 9 | 54 |
| Wins / losses | 0 / 0 | 0 / 3 | 4 / 2 | 4 / 5 (net -1, p = 1) |

No contacts; pose losses 11 (dv2 12).
- **Tracking works:** on the probe, 75-100% of predicted tracks lie on a true obstacle, with a median velocity error
  of 0.02-0.09 m/s. The first detector also tracked static wall fragments, whose centroid drifts with the view; tracks
  are now predicted only after two updates in which they entered floor seen free or left their cells free.
- **Footprint clearing removed the probe's only contact.** Phantom cells inside the body outline (an obstacle that
  passed close, or pose error) blocked every translation. The turn exit watched the nearest (phantom) cell while turns
  in place on the continuous policy crept the base 0.35 m into a real wall.
- **Prediction does not unfreeze the robot.** Knowing where an obstacle will be does not tell the six-primitive
  harness what to do about it.

**The six-primitive baseline is frozen at dv3** (P2b): rounds 1 and 2 each gained under 3 net per 40 with p > 0.1,
round 3 trades failures, and the budget of three rounds is spent. Development record of the frozen harness on
validation 10-29: static 32/40, one obstacle 13/40, two obstacles 9/40.

Other findings of the round:
- **Maze 26 episode 0** fails at frame 0 with obstacles in every round: the start faces a wall 25 cm away, the
  frame-0 floor plane is marginal, and an extra entity shifts the settled pose enough to tip it. A start-pose
  perception edge case, not an obstacle placement fault (the obstacle is 3.9 m away, behind).
- **Pose loss with obstacles** ends long stretches of turning in place or holding while stuck (the obstacle is
  usually out of view): a symptom of the freeze, classified as such.

Next: the frozen baseline's evaluation (`configs/experiments/frozen_baseline_eval.yaml`): static parity on R4 and the
R6 reference, held-out static episode 1, and held-out obstacles (placement series heldout-v1) for C1 and C0.

## The frozen baseline's evaluation (25 October)

`configs/experiments/frozen_baseline_eval.yaml`, the dv3 harness, C1 on the continuous policy (cohorts `fb_*`).

**Static parity** (paired with the records):

| | Frozen dv3 | M3 (continuous policy, R7c-era rules) | Record (six-primitive policy) |
|---|---|---|---|
| R4, prelim_test 30-49 ep 0 | 18/20 | 18/20 (1 win, 1 loss) | 20/20 (`rexit2_ref`) |
| R6 reference, stage2_sets 30-49 | 18/20 | 18/20 (1 win, 1 loss) | 20/20 (`s2ref_cpu`) |

The dv rules cost nothing on static mazes; the gap to the records is the policy's, as M3 found. (The R6 reference
first ran ten missions with an episode the set lacks; they were rerun with the records' episodes.)

**Held-out** (prelim_test 30-49, placement series heldout-v1, never used in development):

| | Development (validation 10-29) | Held-out |
|---|---|---|
| Static | 32/40 | 15/20 (episode 1) |
| One obstacle | 13/40 | 20/40 |
| Two obstacles | 9/40 | 10/40 |

- **No sign of overfitting** the rules to the development mazes: held-out matches or beats development.
- **Pose loss is the largest held-out failure with obstacles** (10 and 11 of 40, development 4 and 7): visual
  odometry loses both cameras at the end of long stretches of turning or holding while stuck.
- **One contact in 120 held-out missions** (one obstacle, maze 48 ep0): a rear calf swept a wall during a primitive
  turn in place.

**C0 on the continuous policy** (its first run there; development never used it) exposed two harness faults:
- **What-if contacts ended the mission.** A C0 branch that touched an obstacle raised the mission's contact stop: 14
  of 80 missions were ended by a contact that never happened. Fixed (03efbec): a branch that stops physically holds
  its last pose. The 14 are rerun.
- **The executed-prefix check failed 27 missions** that were fine: every failing plan started within 400 ms of an
  escape dispatch (the escape sends a vetoed turn while the ledger keeps the zero request, so the next plan's
  committed prefix is zero while the robot still turns). Wherever commands matched, branch and execution agreed to
  0 mm. The check now skips those plans; outcomes come from the arrivals and stand.
- **Result, C0 against C1 on held-out obstacles:** one obstacle 18 against 20 (C1 10 wins, 8 losses, p 0.82), two
  obstacles 8 against 10 (7 wins, 5 losses, p 0.77). One real contact for C0 (two obstacles, maze 47 ep0; its rerun
  also exposed a check crash on a stop-interrupted step, fixed in 8ef7a87). **Perfect forecasts buy the six-primitive
  harness nothing with moving obstacles:** its rules, not C1's forecasts, limit it.

## The continuous-command baseline: lattice rounds (25 October)

The dv3 harness with C1 scoring a command lattice instead of the six primitives (`--candidates lattice`: forward
-0.10 to 0.45 m/s, lateral ±0.15 m/s, yaw in five steps), the continuous safety block, the dynamic-window fallback,
and C1's command model refitted on the command-range collection (held-out median error at 800 ms 11.3 mm, against
24.3 mm for the shipped model). Same 120 development missions as dv3, paired with it.

**lv0** (yaw to ±0.6 rad/s; `configs/experiments/lv0_development.yaml`):

| | Static | One obstacle | Two obstacles | Total |
|---|---|---|---|---|
| dv3 (six primitives) | 32 | 13 | 9 | 54 |
| lv0 | 38 | 25 | 13 | 76 |
| Wins / losses | 6 / 0 (p 0.03) | 15 / 3 (p 0.008) | 9 / 5 (p 0.42) | 30 / 8 |

- **Faster and more successful,** but **seven missions ended in contact** (dv3: none). Six were turning at 0.6 rad/s,
  in place or with lateral 0.15 m/s, and a calf swept past the 0.45-m footprint into a wall or an obstacle; one was
  an 8 N hip touch while holding. 36% of lv0's plans were turns in place at 0.6 rad/s.
- **lv1** caps the lattice's yaw at the primitives' 0.45 rad/s, the rate the footprint was validated at (7debb4f).

**lv1** (yaw capped at 0.45 rad/s): static 38, one obstacle 24, two obstacles 13; 75 in all (lv0 76; 9 wins, 10
losses), and **12 contacts** (lv0 7). The yaw rate was not the cause. Two mechanisms account for all 19 contacts of
lv0 and lv1:
- **Sideways into unwatched space (10, all with obstacles):** a command with a lateral component (in lv1 always
  lateral and yaw of one sign, a crab into the turn), a front calf meeting an obstacle the map had never shown (0.5-0.65 m
  predicted clearance, no track). Both cameras look ahead; the six primitives never step sideways.
- **Creep while held (9, walls):** the continuous policy creeps about 1 mm/s on a zero command. The robot held for
  minutes with every candidate blocked while its base crept from 0.45 m to 0.26-0.32 m of a wall (maze 13 ep0 with
  two obstacles: 0.33 m in 300 s, 98% zero commands), until a calf, hip or the base touched it.
- **lv2** (`configs/experiments/lv2_development.yaml`): no lateral axis (25 candidates), and a back-off from inside
  the footprint (the translation never closer than hold that ends furthest out, at least 5 mm beyond hold; the
  stopping projection, which a pose inside the footprint always fails, does not veto it). A first lv2 launch was
  stopped a third of the way in: the back-off never ran while the planner scanned for a view (scan utilities leave out
  every translation, and the reserve exit had already marked the reversals clear), so a robot crept into a wall as in
  lv1. The back-off now comes first when inside the footprint (its partial cohorts are kept as `lv2pre_*`). A second
  launch was stopped too (`lv2pre2_*`): the floor-coverage rule then vetoed every back-off (reversing sweeps floor the
  forward cameras never saw). A back-off reaching at most 0.10 m is now exempt: a few centimetres into space the body
  has just left is less risk than creeping on. Smoke: maze 10 ep0 (one obstacle), a wall contact at 417 s in both
  stopped launches, completes its round trip in 302 s after three back-offs; maze 13 ep0 (two obstacles), lv1's creep
  contact, stays at 0.40 m or more and times out without contact.

**lv2** (no lateral, back-off; commit d5099d0):

| | Static | One obstacle | Two obstacles | Total | Contacts |
|---|---|---|---|---|---|
| dv3 (six primitives) | 32 | 13 | 9 | 54 | 0 |
| lv0 | 38 | 25 | 13 | 76 | 7 |
| lv1 | 38 | 24 | 13 | 75 | 12 |
| lv2 | 38 | 31 | 18 | 87 | 0 |
| lv2 against lv1 | 0 / 0 | 10 / 3 (p 0.09) | 8 / 3 (p 0.23) | 18 / 6 (p 0.02) | |
| lv2 against dv3 | 7 / 1 (p 0.07) | 18 / 0 (p < 0.001) | 11 / 2 (p 0.02) | 36 / 3 | |

- **No contacts** (lv1: 12). 491 back-off plans in 47 missions, every one executed (none overridden).
- **The remaining failures are one mechanism, shared with dv3.** In 27 of lv2's 33 failures, all with obstacles, over
  40% of plans find `OBSERVED_COMPONENT_HAS_NO_FRONTIER`: the floor reachable on the map has no route to the goal and
  no frontier. The robot then turns in place through its five scan headings every ~19 s and
  forgets the few cells not seen for a while (1-14 each time), while the movers walk past. In maze 22 episode 1 with two
  obstacles it stays within 0.1 m of one spot for 120 s, the obstacles passing 0.8-2.6 m away. Cells a mover left
  where no camera sees them as whole floor seal a passage: neither clearing (whole square seen as floor) nor
  forgetting (not seen lately) reaches them. dv3 has the same failure (17 of its 58 obstacle failures, and 13 of 30 in
  the held-out run with two obstacles); its other failures (holds, latched turns) are what the lattice removed.
  Static missions never show it.
- The other six: static 14 ep1 (time-out) and 17 ep0 (pose loss after long holds); with one obstacle 17 ep0 (beacon
  only) and 26 ep0 (the frame-0 floor plane, as in every round); with two, 18 ep1 (pose loss at 85 s) and 29 ep0
  (time-out under repeated translation vetoes).

**The lattice baseline is frozen at lv2.** The rule had not fired: lv2 gained 4 net per 40 (p 0.02), and one round of
the budget was left. It is frozen anyway, a deviation from P2b recorded here: the only change the failures point to
is a map rule (how a mover's trail is cleared), which belongs to both baselines. Adding it to the lattice alone
would change what the gap between the two baselines measures, the action set alone. It goes to the failure analysis,
for both arms. Held-out evaluation: `configs/experiments/lattice_baseline_eval.yaml` (cohorts `lb_*`), the frozen
six-primitive evaluation's C1 missions with the lattice flags, paired with them.

## The lattice baseline's evaluation (25 October)

`configs/experiments/lattice_baseline_eval.yaml` (cohorts `lb_*`, commit 71d3f54): the frozen six-primitive
evaluation's C1 missions with the lattice frozen at lv2, paired with them.

| | Six primitives (dv3) | Lattice (lv2) | Wins / losses | p | Lattice, development |
|---|---|---|---|---|---|
| R4, prelim_test 30-49 ep 0 | 18/20 | 20/20 | 2 / 0 | 0.50 | |
| R6 reference, stage2_sets 30-49 | 18/20 | 20/20 | 2 / 0 | 0.50 | |
| Held-out static (episode 1) | 15/20 | 19/20 | 5 / 1 | 0.22 | 38/40 |
| Held-out, one obstacle | 20/40 | 32/40 | 14 / 2 | 0.004 | 31/40 |
| Held-out, two obstacles | 10/40 | 21/40 | 15 / 4 | 0.02 | 18/40 |
| All 140 | 81 | 112 | 38 / 7 | < 0.001 | |

- **Static parity restored:** 20/20 on R4 and on the R6 reference, the records' result (no wins or losses against
  `rexit2_ref` and `s2ref_cpu`). The continuous policy's two static losses on each set (M3) were the six primitives',
  not the policy's.
- **No contacts** in 140 missions (six primitives: one).
- **No sign of overfitting:** held-out matches development at every level.
- **Faster where both succeed on static mazes** (median mission 114-137 s against 136-176 s), at a slightly lower
  outbound SPL (0.65-0.76 against 0.72-0.85): the lattice drives faster on wider arcs.
- **Pose loss nearly gone** with obstacles (3 and 1 of 40, against 10 and 11): it ended long stretches of turning or
  holding while stuck, which the lattice has far fewer of.
- **The residual is the map:** 23 of the 27 obstacle failures spend over 40% of their plans with no route and no
  frontier on the map (the sealed passages of lv2). Two obstacles also leave nine missions that reach the beacon
  and do not get home.

**What the action set is worth:** with the same forecaster (C1), harness rules and policy, scoring a 25-command lattice
instead of six primitives turns 81 of 140 held-out missions into 112 (38 wins, 7 losses). The gap includes what the
lattice needed to be safe (C1's refit on the command range, the continuous safety block, the back-off from inside the
footprint). Both baselines are frozen; the failure analysis starts from the map's sealed passages, common to both.

## Failure analysis on the lattice baseline (25 October)

Branch `feature/failure-analysis`. The lattice baseline frozen at lv2 (C1, the non-learned forecaster; development
mazes 10-29, episodes 0 and 1), paired with lv2's missions on the same obstacle placements.

### Density curve

`configs/experiments/density_curve.yaml` (cohorts `dn_*`, commit 0c9faa4). Round trips of 40:

| Obstacles | 0.15-0.35 m/s (lv2) | 0.35-0.55 m/s | Wins / losses (fast against lv2) | p |
|---|---|---|---|---|
| 0 | 38 | | | |
| 1 | 31 | 28 | 4 / 7 | 0.55 |
| 2 | 18 | 11 | 2 / 9 | 0.07 |
| 4 | 5 | 4 | | |

- **The sealed passage dominates at every point** (failures with over 40% of plans finding no route and no frontier):
  26 of 35 failures at four obstacles, 20 of 29 at two fast ones. At four fast obstacles pose loss joins it (15 of
  36 failures).
- **First lattice contacts** (no contacts in lv2's 260 missions): four runs, at the fast or four-obstacle settings, all
  with an obstacle and all attributed to the robot. Two came while backing up and turning (rear calf, 129-146 N), two
  while turning in place or holding (base 58 N, front hip 2 N). The cameras look forward, and the back-off's safety
  check cannot see what is behind the robot.

### Module oracles

`configs/experiments/module_oracles.yaml` (cohorts `or_*`, commit f63aa82; `go2nav/controllers/truth.py`). One
module at a time is given the simulator's ground truth, placed through the estimated pose; 80 missions each (one and
two obstacles, 40 each).

| Oracle | One obstacle | Two obstacles | All 80 | Wins / losses (against lv2) | p |
|---|---|---|---|---|---|
| None (lv2) | 31 | 18 | 49 | | |
| Map, `full` (true walls, floor and obstacles; movers block routes as perceived ones do) | 36 | 23 | 59 | 14 / 4 | 0.03 |
| Map, `static` (as `full`, movers kept out of the 5-cm route grid) | 33 | 24 | 57 | 17 / 9 | 0.17 |
| Obstacle prediction (true futures in the clearance check) | 29 | 20 | 49 | 4 / 4 | 1.0 |
| Pose (true pose for odometry and floor registration) | 26 | 17 | 43 | 8 / 14 | 0.29 |

- **The map is the only module with headroom:** the true map turns 10 of the 31 failures into round trips (p 0.03).
  It removes the ghost trails and the need to explore. Under it, 16 of the 21 failures are still sealed passages, now
  sealed by the true mover on the maze's only path, routed as a wall. Pose loss rises to 9 failures (from 3).
- **Routing through movers trades sealed passages for standoffs.** With movers out of the route grid, no failure is a
  sealed passage. Instead, every failure that ran (22 of 23; one floor-registration error) has a still mover and the
  robot within 1 m of each other for over 60 s, about 0.7 m apart. The mover is parked on the beacon (16) or on home
  (6). No other cohort, lv2 included, has a standoff that long.
  - **Why:** a polite walker stops while the robot is within 0.8 m ahead of it, and turns back after 3 s. A mover on
    a goal at the end of a dead end turns back into the dead end, can only come out towards the robot, and stops
    again. The robot, routed through it, waits on the clearance check.
  - Routing movers as walls keeps the robot back, so the mover walks out. Routing through it chases it into the dead
    end and traps it.
  - The two map oracles end level (11 / 9 head to head).
- **Prediction is not the limit:** true futures change nothing; the failure mix is unchanged.
- **Pose is not the limit either.** The deficit is within noise (p 0.29), and the failures are lv2's kind (31 of 36
  sealed passages, none from the oracle itself). One loss traced (maze 11, episode 0) runs identical to lv2 for 60 s,
  then diverges on a mover's timing and runs out of time on the return.
- **Noise:** missions that differ in anything diverge after the first encounter with a mover, so a few missions in 40
  either way is noise. Module comparisons need the pooled 80.

**What it says:** among polite walkers in perfect mazes, the limit is a decision, not perception. The robot has to
choose between holding back, backing off to let a mover out, and taking another way. Perfect perception of any one
module recovers at most 10 of lv2's 31 obstacle failures. The hand fix the oracles point to, routing through movers,
works only with a give-way rule (retreat beyond the yield distance when a still mover holds the goal or the route
ahead); without one it trades sealed passages for standoffs. Moving obstacles that react to the robot are what a
learned world model can predict and a hand rule has to special-case.

### The obstacles map oracle and the trail probe (26 October)

**Obstacles map oracle** (`map_oracle: obstacles`, cohorts `or_map_obst_*`, commit 72fdbbc). This is the perceived map
with the obstacles' pixels removed and the true obstacles added at each update, so it has no trails. It separates the
true map's gain into trails and prior knowledge of the maze.

| | One obstacle | Two obstacles | All 80 | Wins / losses (against lv2) | p |
|---|---|---|---|---|---|
| lv2 | 31 | 18 | 49 | | |
| Map, `obstacles` | 33 | 23 | 56 | 12 / 5 | 0.14 |
| Map, `full` | 36 | 23 | 59 | 14 / 4 | 0.03 |

Trails account for 7 of the true map's 10. No `obstacles` failure is a standoff, but 12 of 24 are still sealed
passages, sealed by the true mover where it stands.

**Trail probe** (`configs/experiments/trail_probe.yaml`, cohorts `tp_*`, commit 6692bf0): the deployable rule.
- **The rule** (`mover_trails`, `perception.mapping`): a confirmed track's cells stay in the routing map only while an
  update sees them occupied. Out of view they revert to floor; in view they block routes as before.
- **The setup:** both frozen baselines with the rule on, on development mazes 10-29, paired with dv3 and lv2.

| | Six primitives (dv3 → probe) | Wins / losses | Lattice (lv2 → probe) | Wins / losses |
|---|---|---|---|---|
| Static | 32 → 32 | 1 / 1 | 38 → 37 | 0 / 1 |
| One obstacle | 13 → 15 | 8 / 6 | 31 → 31 | 5 / 5 |
| Two obstacles | 9 → 10 | 6 / 5 | 18 → 23 | 6 / 1 |
| Obstacles, all 80 | 22 → 25 | 14 / 11 (p 0.69) | 49 → 54 | 11 / 6 (p 0.33) |

- **No regression and no contacts** (240 missions); no standoffs.
- **The lattice reaches the oracle's level:** 54 against the `obstacles` oracle's 56 (5 / 5 head to head). The rule
  recovers what removing trails can give; what is left of the sealed passages (22 of 26 failures) is the mover in
  view on the only path.
- **The six primitives trade:** sealed passages fall from 18 of 58 failures to 4 of 55, but pose loss rises (22
  against 11, mostly while turning for views). Their limit is elsewhere (the primitives' turning, `ADDITIONAL_VIEW`).
- **Stopping rule:** neither baseline passes. Both trade failures (losses reach half the wins), and neither is
  significant.

**Decision:** the baselines stay frozen at dv3 and lv2. The probe is the hand-rule ceiling on the map's headroom:
the best deployable trail rule matches the oracle for trails (+5 of 80 on the lattice, not significant). The
remaining headroom is the decision a mover in view forces, which is where learning goes next.

## P3: corpus v2 (26 October)

Branch `feature/dynamic-world-models`. Collected on the continuous policy with 0-4 movers (674 units with a
non-polite fifth), lateral commands kept in the blocks.

- **Size:** 4,000 units, 3.23 M action blocks, 2.93 M frames (about 340 GB). Units by mover count: 589 / 1,259 /
  1,144 / 615 / 393 for 0-4.
- **Stops:** 3,847 clean, 121 pose loss, 25 disallowed contacts (contact stops under 5% in every kind).
- **Command statistics:** `data/training/control_norm_stats_corpus_v2.json`; yaw command to rate correlation 0.99 at
  lag 0.

Three of the audit's gates missed. I accepted each with its reason (`gates_accepted.json` beside the audit) rather
than collect again:

| Gate | Result | Why accepted |
|---|---|---|
| Each movement type at least 20k windows | Starts from rest 17.4k (87%) | The v1 primitive classifier undercounts starts from rest under continuous commands; every other type passes |
| Lateral at least 10% of blocks | 8.9% (289k blocks) | Mission units carry none by design (the lattice harness has no lateral) and are 39% of blocks; free 19.9%, explore 10.9% |
| Mover in view at least 20% of frames | 17.4% (478k frames) | 3-4 mover units reach 23-29%; a top-up of free units would move it under one point |

## P4: LeWM v2 and the mover gate (27 October)

**LeWM v2** (`outputs/lewm/go2_v2`): span8w's settings on corpus v2 (ViT-Tiny, inverse dynamics on spans 1 and 8,
15-value action blocks), 60k steps at 0.49 s/step.
- Validation rollout error 0.032 at the first step to 0.092 at the eighth, against shuffled actions 0.83, zero
  command 0.61 and persistence 0.68; transfer 0.050 to about 0.13. L2 action sensitivity holds; effective rank 20.3.
- Gates L1-L4 (the LeWM acceptance gates) still to run on it: the GPU is busy with H-JEPA v2.

**Two fixes on the way** (both committed):
- The trainer held 30.4 GB of the R9700's 34 GB; GPU jobs are now capped at 85% (f434a1d), since that card also
  drives the desktop.
- Validation crashed at step 4,999: the zero-command baseline assumed two command components (bda8b42).

**R9700 bus drop** (27 October, 05:17): "device lost from bus" in plain training, with no warning before it. Resumed
from the step-11,999 checkpoint with matching losses. A telemetry service now logs temperatures, power, clocks and
fan every 15 s (memory runs at 92-94 °C against a critical 108; power at the 300 W cap with spikes to 420 W).

### The mover gate

The gate (PLAN P4) was set before the final model was scored. "Ceiling" is the better per measure of two supervised
CNNs trained on the four history frames themselves (`go2nav.eval.mover_ceiling`, plain and regularised: mirror and
photometric augmentation, dropout, weight decay, epoch selection on held-out training units).

| Measure | Pass mark | LeWM v2 | Patch tokens | Ceiling |
|---|---|---|---|---|
| Localisation, h = 0 (median error) | 1.25 x ceiling = 0.164 m | 0.216 m ✗ | 0.152 m ✓ | 0.131 m |
| Velocity skill at 0.8 s | half the ceiling = 0.126 | 0.08 (0.02 through rollouts) ✗ | 0.08 ✗ | 0.251 |
| Response at 1.6 / 3.2 s | gain and 95% CI above 0.12, R2 above 0 | gain 0.16 / 0.17, CI from 0.121 / 0.141, R2 -0.48 / -0.24 ✗ | | |

- **Patch tokens:** the same frozen LeWM v2 encoder, but its 12 x 16 final-layer patch tokens instead of the [CLS]
  summary, read by a small convolutional head trained exactly as the ceiling (`mover_ceiling --patches`, abea321).
- **The response is not real.** The rollouts' predicted displacement is as large where no response should happen
  (spurious 0.04 / 0.09 m against predicted 0.045 / 0.095 m), so R2 stays negative.
- **Event skill** (reported, not gated): LeWM 0.02-0.07, the ceiling 0.07-0.17. The plain ceiling's 0 or below
  was overfitting; the regularised one does see some of the walkers' turns and pauses.
- **The patch head** selected epoch 2 (it overfits fast and has no augmentation), so its velocity is weaker
  evidence than its localisation.

**Reading.** The encoder holds the movers: its patch tokens localise within 16% of a CNN trained for the job. The
[CLS] bottleneck the world model predicts in loses most of it (65% worse). H-JEPA's upper levels are built from the
same [CLS] latent, so they cannot recover it; they are still probed when H-JEPA v2 finishes, as planned.

**Decision (per the gate):** a representation change is proposed (PLAN P4b, a patch-grid LeWM) and P5's hybrid
is built meanwhile.

## H-JEPA v2, the C5/C6 heads and the hybrid (28 October)

**H-JEPA v2** (`outputs/hjepa/go2_3level_v2`, three levels trained jointly on corpus v2) fails the mover gate as LeWM
v2 does. Mover probe, validation:

| | Localisation | Velocity skill at 0.8 s | Response R2 at 1.6 / 3.2 s |
|---|---|---|---|
| Level 1, step 40k | 0.199 m | 0.074 | |
| Level 1, final | 0.193 m | 0.087 | -0.78 / -0.45 |
| Level 2, final | 0.245 m | | about -1.0 |
| Pass mark | 0.164 m | 0.126 | above 0 |

The response gains rise (0.20-0.28) but the spurious response is as large as the predicted one: the robot's own
motion is still read as the mover's.

**C5/C6 heads v2** (`training.lewm_heads_v2 --rescore`, de4fed3): the overall medians are flattered by rests and turns in
place, so the evaluation now also reports the moving and translating windows. On translating windows at 0.8 s, C6
(the direct head) is off by a median 4 mm (validation) and 6 mm (transfer); C5 (the decoder on rollouts) 21 and 47 mm.
For comparison, span8w's C5 and C6 were 5.8 and 2.4 mm, H-JEPA v1's 3.8 mm. C5's gap is the rollout, not the decoder.

**P5 hybrid** (`hybrid_development`, lv2's harness with C5 or C6 as the forecaster over the 25-tape lattice;
validation mazes 10-29, two episodes each), round trips of 40, paired with lv2:

| Setting | lv2 | C5 | C6 |
|---|---|---|---|
| Static | 38 | 40 | 38 |
| One mover | 31 | 31 | 31 |
| Two movers | 18 | 20 | 13 |
| Wins / losses against lv2 | | 13 / 9 (sign test p = 0.52) | 7 / 12 (p = 0.36) |

Neither differs from lv2. One contact: C6, maze 15 episode 1, reversing (forward -0.10 m/s, yaw -0.45 rad/s) into a
walker on its blind right side about 0.58 m away (the rear-right calf).

## Two PyTorch faults on the R9700 (28 October)

- **LayerNorm gradients** (torch 2.12.0+rocm7.2, gfx1201): the fused LayerNorm and RMSNorm backward never writes the
  odd-indexed weight and bias gradient elements from 128 rows (pytorch/pytorch#199265, fixed in 2.13). Inductor's
  kernels are right, so the compiled LeWM and H-JEPA runs were unaffected (no odd/even difference in their
  checkpoints); the eagerly trained T1-T4 predictors' final norm was (odd gains four times as far from 1). `models.norm`
  computes the gradients itself; checkpoints load unchanged (cd67933, ee377ec). LeWM v2 is not retrained.
- **Compiled grid backward:** compiling the whole grid loss corrupts gradients from batch 64 (the projectors' 1e5-1e6
  against about 1 eagerly; the first grid run diverged and was discarded). The compiled grid predictor blocks alone
  reproduce it; not the LayerNorm, masked attention, dropout or checkpointing. `runtime.compile_scope: encoder`
  compiles only the ViT blocks (532aa33): eager gradients, 0.89 s/step against 0.71 compiled.

## P4b: the patch-grid LeWM (29 October)

**Training** (`outputs/lewm/go2_v2_grid`, 60k steps, 14:05-05:05): as a world model it is slightly better than LeWM v2.
Validation rollout error 0.030 at the first step to 0.080 at the eighth (v2 0.032 to 0.092), transfer 0.046 to about
0.12 (v2 0.050 to 0.13); the grid tokens beat persistence at every step on both splits. Effective rank 17.7 (v2 20.3).

**Mover gate: fails, as LeWM v2.**

| Validation | Pass mark | Grid | LeWM v2 |
|---|---|---|---|
| Localisation (MLP head) | 0.164 m | 0.215 m | 0.216 m |
| Velocity skill at 0.8 s | 0.126 | 0.083 | 0.080 |
| Response R2 at 1.6 / 3.2 s | above 0 | -0.75 / -0.54 | -0.48 / -0.24 |

**Where the mover is lost.** The patch-token probe's convolutional head (`mover_ceiling --patches` settings: 20 epochs,
dropout 0.3, weight decay 0.05, epoch selection) on 3 x 4 token stacks; "cell spread" is the standard deviation across
a frame's 12 cells:

| Frozen input, validation | Localisation | Velocity skill at 0.8 s | Cell spread |
|---|---|---|---|
| LeWM v2 encoder, 12 x 16 patch tokens | 0.152 m | 0.079 | |
| LeWM v2 encoder, pooled to 3 x 4 | 0.132 m | 0.100 | 0.47 |
| Grid encoder, pooled to 3 x 4 (before the grid projector), final | 0.156 m | 0.082 | 0.37 |
| Grid tokens (after the grid projector), final | 0.228 m | 0.042 | 0.12 |

- Pooling to 3 x 4 keeps the mover; the grid model's own encoder still holds it (step 20k: 0.161 m).
- The grid projector loses it: its cells collapse towards one frame-wide signal. SIGReg is already applied to every
  grid cell (per cell and time step, across the batch), but it does not stop the cells carrying the same content.
- Velocity is short of the mark in every row, even v2's raw patch tokens: the frozen per-frame encoder does not hold
  the walkers' motion well enough for this head (the pixel ceiling reaches 0.251).

**Decision pending:** a grid objective that keeps the cells distinct (no grid projector, or SIGReg on each cell's
difference from the frame mean) would likely bring localisation to the mark, not velocity.

## P4f: the give-way rule in missions (30 October)

The committed rule (`give_way_commit.yaml`, cohorts `gwcf_*`, PLAN P4f) on 80 validation missions with mover trails, paired
with the trail probe's lattice cohorts (`tp_lat_*`). **It does not beat them, so it stays off;** lv2 with mover trails is
the baseline.

| Round trips | Mover trails | Trails and committed rule | Gained / lost |
|---|---|---|---|
| One mover | 31 / 40 | 32 / 40 | 3 / 2 |
| Two movers | 23 / 40 | 18 / 40 | 4 / 9 |
| Both | 54 / 80 | 50 / 80 | 7 / 11 (sign test p = 0.48) |

- **It now backs off:** 258 triggers in 65 missions, a median of 32 cm (the first rule 3 mm), 39% the full 40 cm. 151 ended
  on a blocked reverse. No mission ended in a collision.
- **It does not change the encounter.** Within 6 s of a back-off's end the mover had walked 0.3 m in 80% of full back-offs
  and 87% of shorter ones, against 85% after the first rule's millimetres: polite walkers turn back on their own. With a
  still walker ahead and the robot held up, the median time until the robot is 1 m away is 22 s with the rule and 21-24 s
  without. After a full back-off the robot reached the walker's old spot within 15 s in 27% of cases.
- **The two-mover losses are not explained.** Eight of the nine ran out the 480 s on missions mover trails finished in
  130-310 s, a slowdown well beyond the 14-69 s spent backing off; one lost its pose. Pose loss overall: 4 missions with
  the rule, 1 without (79 of 80 pairs compared; one pair has an empty trace).

## P4d: grid LeWM with mover labels (31 October)

`outputs/lewm/go2_v2_grid_movers`, 60k steps, probed on `eval.mover_probe` and `eval.grid_readout`. **It fails the give-way
gate on all three readouts, and adds nothing to the frozen-target model's (P4c) forecasts.**

| Retreat contrast: AUC [95%], pass above 0.5 | Validation 1.6 s | Validation 3.2 s | Transfer 1.6 s | Transfer 3.2 s |
|---|---|---|---|---|
| Labels, ridge | 0.47 [0.39, 0.56] | 0.49 [0.41, 0.57] | 0.55 [0.46, 0.64] | 0.55 [0.45, 0.64] |
| Labels, MLP | 0.34 [0.27, 0.43] | 0.36 [0.30, 0.43] | 0.49 [0.41, 0.57] | 0.44 [0.38, 0.51] |
| Labels, patch | 0.25 [0.20, 0.32] | 0.24 [0.19, 0.31] | 0.46 [0.35, 0.56] | 0.36 [0.31, 0.42] |
| Frozen target, best readout | 0.41 [0.34, 0.49] | 0.39 [0.32, 0.47] | 0.60 [0.50, 0.68] | 0.57 [0.48, 0.65] |
| Still-mover rule | 0.80 | | 0.77 | |

- The contrast's R2 is about 0 on every readout (at most 0.05). Models' rankings are still inverted on validation.
- **Velocity skill at 0.8 s through the rollouts (mark 0.126):** 0.111 MLP, 0.108 patch on validation (frozen target 0.100,
  0.103); 0.135 patch on transfer (0.130). On the true state 0.214 (0.230): the encoder holds the motion and the
  predictor loses about half of it.
- **Response R2 stays below 0:** -0.09 to -1.5 for the patch readout, -1.1 to -10 for ridge and MLP.
- **Localisation** 0.093 m validation, 0.150 m transfer (frozen target 0.096, 0.128). The labels supervise this, so it is
  reported and is not evidence; the transfer split does not gain.

## Machine faults (30-31 October)

**System drive, 30 October 01:43.** The Samsung 990 PRO stopped answering, its reset failed, the kernel disabled it and
PID 1 died. Models, corpus and repository are on the other drive; training resumed from step 32,000. NVMe health is clean.

**R9700 ring timeouts, 31 October 02:18-02:43.** 46 `gfx_0.0.0` timeouts, the first 56 s after the mover probe finished
extracting features on the GPU beside ten collection renderers, each aborting one give-way unit (`exit -6`). Every ring
reset succeeded and the bus stayed up. The cause is not confirmed; rendering beside training ran three hours without
one. The 46 units were collected again with the GPU otherwise idle, without a timeout: 1,200 of 1,200, and 31.5% of the
139,087 waiting windows precede a retreat (data gate 25%; corpus v2 4.3%).

## P4f: the frozen-target model on the give-way top-up (31 October)

`outputs/lewm/go2_v2_frozen_giveway`: P4c's seed, frozen encoder and schedule (the predictor trained from scratch, 60k steps,
6.4 h) on corpus v2 plus the 1,200 give-way units, about 20% of the training windows. **The data do not help: it fails the
give-way gate on all three readouts, and the rankings P4c had inverted are more inverted, or at chance.**

| Retreat contrast: AUC [95%], pass above 0.5 | Validation 1.6 s | Validation 3.2 s | Transfer 1.6 s | Transfer 3.2 s |
|---|---|---|---|---|
| Top-up, ridge | 0.41 [0.36, 0.47] | 0.53 [0.46, 0.59] | 0.47 [0.40, 0.54] | 0.52 [0.46, 0.58] |
| P4c, ridge | 0.41 [0.34, 0.49] | 0.39 [0.32, 0.47] | 0.60 [0.50, 0.68] | 0.57 [0.48, 0.65] |
| Top-up, MLP | 0.25 [0.20, 0.31] | 0.25 [0.20, 0.32] | 0.36 [0.28, 0.43] | 0.39 [0.33, 0.47] |
| P4c, MLP | 0.32 [0.27, 0.39] | 0.30 [0.26, 0.36] | 0.50 [0.41, 0.57] | 0.40 [0.34, 0.48] |
| Top-up, patch | 0.19 [0.15, 0.25] | 0.28 [0.24, 0.34] | 0.30 [0.22, 0.38] | 0.34 [0.27, 0.40] |
| P4c, patch | 0.25 [0.20, 0.32] | 0.31 [0.25, 0.37] | 0.34 [0.28, 0.39] | 0.39 [0.33, 0.45] |

- The contrast's R2 is at most 0.05 on every readout.
- **Velocity skill at 0.8 s through the rollouts (mark 0.126):** patch 0.077 validation, 0.089 transfer (P4c 0.103, 0.130); MLP
  0.110, 0.104 (P4c 0.100, 0.118). On the true state 0.210 (0.230): the encoder is unchanged by construction.
- The plan's reading of an inverted ranking: the predictor's prior of carrying motion on wins even with the data.

**Where the give-way path stands.** Three models fail the gate, and a rule scoring AUC 0.80 / 0.77 on it gains nothing in
missions (above). The gate measures the ranking of responses, not whether acting on that ranking helps a mission.

## What seals a route: the mover-trails failures sorted (31 October)

The mazes are trees (16 cells, 15 corridors), so one route joins any two places, and a walker in a corridor cuts it. Each
plan of the 80 mover-trails missions (79 with traces) is compared with the true walkers on that route, with static and
map-oracle missions as controls. Sealed means `OBSERVED_COMPONENT_HAS_NO_FRONTIER`.

| Sealed time, s per mission | All | Failed | Succeeded |
|---|---|---|---|
| Static, mover trails | 0 | 0 | 0 |
| Walkers, mover trails | 114 | 244 | 53 |
| Walkers, obstacles oracle (true walkers, perceived walls) | 85 | 172 | 49 |
| Walkers, full oracle (true walkers, true walls) | 118 | 262 | 69 |

- **The map is not the fault.** Static missions never seal, so noise and thick walls do not do it, and a perfect map seals
  as long as the perceived one. This corrects the trail probe's reading above: what remains is not only a mover in view,
  and a perfect map would not clear it.
- **Failures are many short stops:** a median of 22 sealed stretches of about 10 s each in the 24 failures that timed out,
  moving in about 6% of sealed time. The robot scans where the route was lost.
- **The walker is often not near:** in failed missions' sealed time the nearest walker on the route was within 2 m for 46%,
  2-5 m for 32%, further for 9%, and not on it for 13%. With the route open the shares within 2 m are 29% in failed missions
  and 20% over all missions, so nearness is raised, not universal. The cells behind the not-on-route share are not
  recorded: stored cells of unconfirmed tracks are a candidate, and no map snapshots exist to tell.
- **It could account for many failures, by an optimistic estimate.** Of 25 failures, 1 lost its pose and 24 timed out (14
  outbound, 10 returning). Those 24 stood sealed with more than 2 m clear for a median of 127 s and needed about 141 s to
  finish at the successful missions' pace; in 11 of 24 the first covers the second. Walkers could still have blocked them.
- **Wrong-way walking:** with the route open, failed missions go backward along the route for 54 s (18 s in successes, 2 s
  static), since a cut leaves only frontiers elsewhere.

**Decision:** the close-up rule (PLAN P4g).

## P4g: the close-up probe (31 October)

`configs/experiments/close_up_probe.yaml`, cohorts `cu_obst*`, commit dc3b886, 80 validation missions with mover trails.
**It meets the bar set before the run, exactly: no collisions and twice as many wins as losses, so the rule is adopted as
the working configuration.** Losses reach half the wins, which the P2b freeze rule treats as trading failures.

| Round trips | Mover trails | Close-up | Gained / lost |
|---|---|---|---|
| One walker | 31 / 40 | 32 / 40 | 3 / 2 |
| Two walkers | 23 / 40 | 28 / 40 | 9 / 4 |
| Both | 54 / 80 | 60 / 80 | 12 / 6 (p 0.24) |

| Close-up against | Score | Gained / lost | p |
|---|---|---|---|
| Frozen lv2 | 60 against 49 | 17 / 6 | 0.035 |
| Full map oracle | 60 against 59 | 10 / 9 | 1.0 |

- **Sealed time** falls from 112 to 52 s a mission on the same missions, with 40 s walking to a cut: about 92 s against 112 s.
- **The cut route makes progress when missions succeed:** of 20 s a mission, 14 s is followed by progress along the true
  route within 10 s, against 40 of 105 s in failures, where 52 s is flat and 10 s backward.
- **A wall blob switches the rule off and on.** The rule keeps its target 1 m from every track, including short-lived ones
  that never showed motion. In failed missions, 74% of the switches from cut to sealed had only such tracks within 1 m of
  the target (52% in successes), against 7% with a confirmed walker. The robot then scans, loses the blob after 1.2 s,
  gets its target back and turns to it, about every 10 s. Within 0.4 m of one spot, this took 11 of 20 failures at least
  60 s (4 of 60 successes) and 24% of failed-mission time. Half the tracks in these cycles were over 1 m from any walker
  and half lay on or past a wall line; the three worst missions' blobs sit at the same wall lines. An early count of
  target jumps pointed to an unstable target, but per minute of cut time they are rarer in failures (11 against 15), and
  stalling is less likely just after a jump (34% against 69%). A steadier target is therefore not the fix.
- **Remaining 20 failures:** 18 timed out, 1 lost its pose, 1 stopped at frame 0 (floor plane unavailable, as in the
  baseline). None is a collision. Trapping a walker in a dead end was not separately measured.

## P4g: the fix check (31 October)

`configs/experiments/close_up_fix.yaml`, cohorts `cuf_obst*`, commit db59928: only confirmed tracks hold the robot off a cut.
**It misses the bar set before the run, so the rule goes back to its first version.** It wins 8 missions and loses 5 where
the bar asked for twice as many wins as losses. The other two parts pass: no collisions, and 5 missions of 80 in the
turning loop for 60 s or more, against 15.

| Same 80 missions | First rule | With the fix | Gained / lost |
|---|---|---|---|
| One walker | 32 / 40 | 33 / 40 | 4 / 3 |
| Two walkers | 28 / 40 | 30 / 40 | 4 / 2 |
| Both | 60 / 80 | 63 / 80 | 8 / 5 (p 0.58) |

| The fix against | Score | Gained / lost | p |
|---|---|---|---|
| Mover trails | 63 against 54 | 15 / 6 | 0.08 |
| Frozen lv2 | 63 against 49 | 19 / 5 | 0.007 |
| Full map oracle | 63 against 59 | 13 / 9 | 0.52 |

- **The diagnosis holds:** switches from cut to sealed with only unconfirmed tracks near the target fall from 552 to 128,
  seconds in the loop from 36 to 12 a mission, and stuck seconds from 83 to 54.
- **Failures:** 17 (20 before): 14 timeouts, 2 pose losses (1 before), 1 frame-0 floor-plane stop. The closest walker centre
  is 0.47 m (0.45 m).
- **Trapping, first rule against mover trails:** a walker stayed within 2 m of the robot and inside a 1-m span for 60 s or
  more in 5 of 79 missions, against 1 of 79 (6 against 2 s a mission). Closest walker centre 0.45 m against 0.55 m.
- **Throughput:** non-learned missions use the GPU only to render, and 14 at once ran at the same 0.54x real time as 6.

**The fix is kept after all, by decision (PLAN P4g).** Mission by mission, 4 of the 8 gains are missions whose loop went
away (0 s against 98-353 s) and that then finished, while 1 of the 5 losses is an early pose loss and 1 of the gains is
the reverse. The other 4 losses are timeouts, one with 143 s in the loop.

**Four walkers, first rule** (`configs/experiments/close_up_four_walkers.yaml`, cohort `cu_obst4_c1`, for information): 18 of
40 against the frozen planner's 5 (14 gained, 1 lost, p 0.001). One collision, the frozen planner also having one: a rear
calf hit a walker while reversing at 0.10 m/s on a route to a frontier (maze 11 episode 0, 110 s), the blind-side reverse
of C6's contact, not a walk to a cut.

## P4g: the held-out evaluation (1 November)

`configs/experiments/close_up_fix_heldout.yaml`, cohorts `cufh_*`, run once on the rule with the fix, paired with lv2 (cohorts
`lb_*`, the same walkers). **125 of 140 against lv2's 112: 17 gained, 4 lost (sign test p 0.007), no collisions.**

| Round trips | lv2 | Close-up with the fix | Gained / lost |
|---|---|---|---|
| R4 reference, 20 | 20 | 20 | 0 / 0 |
| R6 reference, 20 | 20 | 20 | 0 / 0 |
| Held-out, static, 20 | 19 | 18 | 0 / 1 |
| Held-out, one walker, 40 | 32 | 35 | 4 / 1 |
| Held-out, two walkers, 40 | 21 | 32 | 13 / 2 |
| All, 140 | 112 | 125 | 17 / 4 |

- **The gain is where the rule acts:** two walkers, 11 of the 13 net. The references are unchanged.
- **The static loss** is maze 43 episode 1, a pose loss; maze 42 episode 1 lost its pose in both. Nothing here says the rule
  caused it, and I have not looked further.
- **Failures:** 15, of which 12 timed out and 2 lost their pose. One stopped on a code path never ported to this planner
  (`UnportedBranch`, a view recovery across a mission phase change, maze 46 episode 1 with two walkers; lv2 failed it too).
- **The mazes have now been used for this rule:** a further hand rule cannot be judged on them without a fresh set.

Close-up with the fix and mover trails is the working planner and the reference for handing parts to the world model.

## P6: the walkers in shadow (1 November)

The world model's walkers beat the hand tracker's on what the bar measures, and fall behind on velocity, which the bar does
not measure. 120 validation missions, the hand tracker driving, the existing mover head (go2_v2_grid_movers) logged beside it,
scored plan by plan against the true walkers within 2 m in the primary camera's view (8,232 walker-plans):

| | Hand tracker | World model | Bar |
| --- | --- | --- | --- |
| Walkers found | 69% (95% interval 67-72%) | 86% (85-88%) | at least 90% of hand's: passed |
| False alarms per 100 plans | 0.68 | 0.63 | no more than hand's: passed |
| Median position error | 0.151 m | 0.090 m | at most 1.25 times hand's: passed |
| Median velocity error | 0.066 m/s | 0.158 m/s | not in the bar |

**Reference for the live test.** The same missions are C5 with the close-up rule: 40 of 40 with no walkers, 37 of 40 with one,
30 of 40 with two; 107 of 120. So the live bar is at least 64 of 80. One two-walker mission hit a wall or walker
(maze 14 ep1, `DISALLOWED_CONTACT`); three failures were pose losses and one a floor-plane start failure.

**Why velocity is worse.** Most of it is young tracks. The model finds walkers the hand tracker does not, and a
track has no velocity for its first 0.7 s, so 24% of its plans on moving walkers read zero. Tracks with 3-7 detections
err by 0.24 m/s, about a walker's whole speed (median 0.235 m/s); tracks past 20 detections err by 0.10 m/s, still half as
much again as the hand tracker's 0.065. On the 5,587 walker-plans both found, the error is 0.146 against 0.065 m/s. The model calls 79%
of moving walkers moving where hand calls 91%.

**The tracker's settings, tuned before the live test.** The missions log every fourth frame, so the head was replayed at 10 Hz
on 93 validation walker units (true poses, scored at every frame). The best setting (a 0.25 m gate, one walker seen by two columns
merged within 0.3 m, a 0.5 s minimum span, a 1.6 s window) leaves found (91.6% against 91.4%) and false alarms
(0.47 against 0.45 per 100 plans) about where they were, and it cuts tracks per unit from 18.2 to 13.2 and zero-velocity
plans on moving walkers from 28% to 11%, and raises moving walkers called moving from 69% to 85%. The velocity error only
falls from 0.180 to 0.157 m/s, so position noise from the head remains the limit: tuning a straight-line fit will not close it,
and the model's own forecast is what should.

**A leak closed.** In the live path the hand tracker's trails were still being marked on the map. Live, only the model's are.

## P6: the combined model, offline (1 November)

`go2_v2_walkers` trained for 30,000 steps from `go2_v2_grid_movers` with no faults. Validation rollout error at 0.8 s is 0.069,
against 0.075 for its starting model and 0.092 for go2_v2 (0.044 against 0.053 on the grid tokens), and it did not collapse.

**It drives the live test.** The rule was set before reading it. On the 93 validation units through the tuned tracker it found 92.7%
against 91.6% with 0.36 false alarms per 100 plans against 0.47; velocity error 0.147 against 0.157 m/s.

| 12,000 windows, walkers | go2_v2_grid_movers | go2_v2_walkers |
| --- | --- | --- |
| Now, validation: found, error | 80%, 0.107 m | 83%, 0.102 m |
| Now, transfer | 69%, 0.129 m | 78%, 0.115 m |
| Rolled out 0.8 s, validation | 71%, 0.181 m | 88%, 0.126 m |
| Rolled out 0.8 s, transfer | 57%, 0.190 m | 75%, 0.150 m |

**The forecast carries where a walker is, not where it is going.** On the same 93 units, on walkers moving when the forecast
starts (median 0.18 m walked in 0.8 s; both methods within 0.75 m of the truth):

| Median error at 0.8 s | Rollout | Tracker's straight line | Tracker's position held |
| --- | --- | --- | --- |
| Moving, go2_v2_grid_movers | 0.268 m | 0.212 m | 0.212 m |
| Moving, go2_v2_walkers | 0.236 m | 0.221 m | 0.221 m |
| Standing, go2_v2_walkers | 0.052 m | 0.086 m | 0.071 m |

Training took 0.03 m off the moving walkers' error and halved the standing walkers', but no method beats holding the last
position on a moving walker, so the model's tracker gets nothing from its straight line at this range, and the rollout is
0.015 m worse than it (95% interval 0.002 to 0.028 m). Walkers that leave the view are the rollout's worst case: within 0.5 m
19% of the time against the tracker's 75%. So the tracker's straight line stays and the rollout does not replace it. The
model takes 0.3 s of frames, over which a walker moves about 7 cm, near its 0.10 m reading error, so the motion is
hard for it to see. This is a limit of its input.

**Free space: good in view, a guess out of it.** Over all rays the free-space head misses by 0.18 m and reads 65% of near
rays as more than 0.15 m clearer than they are, which sounds unusable and is mostly the rays no camera sees:

| Bearing from straight ahead | Median error | Near rays read too clear | Correlation with the truth |
| --- | --- | --- | --- |
| 0 degrees | 0.034 m | 7% | 0.98 |
| 22 degrees (inside the 39-degree half view) | 0.04 m | 11-25% | 0.94-0.95 |
| 45 degrees | 0.11 m | 44-55% | 0.75-0.78 |
| 90 degrees and behind | 0.26-0.28 m | 90-98% | 0.42-0.48 |

Rollouts barely degrade it (0.185 against 0.181 m over all rays), so the forecast keeps what the state holds. Both limits, the
walkers' motion and the unseen rays, come from the same thing: the model has no memory. That is what steps 4 to 6 need too.

## P6: the live test failed, and why (1 November)

The model's tracks drove the planner on the 80 walker missions: **56 of 80 round trips against the reference's 67 and the bar's 64,
3 gained and 14 lost, with hard contact in 15 missions against 1.** One walker 33 of 40 (reference 37), two walkers 23 of 40
(reference 30). The bar failed on both counts, so the hand tracker stays in charge of this version. Failures: 12 contact stops, 8
timeouts, 3 `UnportedBranch`, 1 pose loss.

**The contacts are walls, not walkers.** All 12 stopped missions' first contact was a rear calf on a wall, usually while
reversing (43-100% of the last 3 s in 11 of 12; 18 ep0 of one walker, 29%). The reference's one contact was a walker. A walker stands
about 0.8 m ahead at the time, because walkers stop for the robot at that range.

**The cause is a false speed on a standing walker while the robot turns.** In two missions read plan by plan, the true walker stood
still and its track read 0.19-0.33 m/s, and the planner chose to reverse where `unpredicted_check_action` shows it would have held
or turned. On the 93 replayed units, a standing walker's fitted speed is above 0.15 m/s on 36% of readings while the robot turns
faster than 0.2 rad/s and 4% otherwise. The head's position drifts with the turn and the fit reads the drift as walking. The shadow's bar
measured none of this: found, false alarms and position error all passed. **Lesson: a bar on a part must cover every value the
planner uses from it.**

**Ruled out.** Mover trails were suspected of erasing walls, because a model track's trail is a drawn disk rather than perceived
cells and marked cells leave the map when unseen. The tracks stayed 0.4-1.0 m from every contact point and under 0.2% of trail
cells were inside walls, so nothing was changed there. Fitting only non-turning frames removes the false speed (3% of
readings) but also the real one (moving walkers called moving 85% to 8%).

**The rerun.** The straight line gave no gain over a held position at 0.8 s, so `walker_mode: held` keeps the fit for the logs and gives the
planner positions only (`configs/experiments/walker_held.yaml`, cohorts `wh_*`): the same 80 missions and the same bar. The missions have
now been looked at, so a pass here is weaker evidence than a first pass.

**The decoder.** `go2_v2_walkers_heads` forecasts the robot a little better than `go2_v2_heads`: at 0.8 s 8.7 against 9.0 mm on validation and
14.4 against 15.2 mm on transfer. Whether one model can do both jobs in missions waits on the rerun.

## P6: the positions-only rerun passed (1 November)

The model's tracks drove the planner with their speeds withheld, on the same 80 missions: **68 of 80 round trips against the
reference's 67 and the bar's 64, 6 gained and 5 lost (sign test p = 1.0), with no contact of any kind in any mission** (reference: 1
hard contact, with a walker; the failed live test: 15). One walker 36 of 40 (reference 37), two walkers 32 of 40 (reference 30). Against
the failed live test it gained 15 missions and lost 3, so the speeds were the fault. **The bar passed: the model now tells the
planner where walkers are, and the planner gets no speeds from it.** The missions had been looked at before this run, so this pass
is weaker evidence than a first pass, and the held-out mazes have not been used.

| | Round trips | Hard contact | Tightest clearance, median of missions |
|---|---|---|---|
| Reference (hand tracker) | 67 of 80 | 1 mission | 0.091 m |
| Live test (model, positions and speeds) | 56 of 80 | 15 missions | |
| Rerun (model, positions only) | 68 of 80 | 0 missions | 0.090 m |

The 12 failed missions: 8 timeouts, most with no route on the map; 3 pose failures; 1 technical failure. None was a contact.

**Walkers while driving.** `eval.walker_compare` on the rerun: the model's tracks found 85% of walkers in reach (95% interval 83-87%),
0.65 false alarms per 100 plans, 0.098 m position error, close to their figures in shadow (86%, 0.63, 0.090 m, the earlier model).
The report's `hand` column reads the planner's tracks, which here are the model's, so its verdict compares the model with itself and
is not evidence. Velocity errors are not read: the rows hold zero speed.

**Still with the hand rules:** the speed fit and straight-line forecast of a walker are no longer used from the model, so a walker in the
planner stands still at its last position. The hand depth stop stays in place.

## P6: one model for both jobs (1 November)

go2_v2_walkers forecast the robot through `go2_v2_walkers_heads` and found the walkers, positions only, on the same 80 missions
(`configs/experiments/walker_one.yaml`, cohorts `wo_*`): **68 of 80 round trips against the rerun's 68 and the bar's 65, 5 gained and 5
lost (sign test p = 1.0), with no contact in any mission.** One walker 35 of 40 (rerun 36), two walkers 33 of 40 (rerun 32). The 12 failed
missions: 9 timeouts, 2 pose failures, 1 technical failure. Tightest clearance, median of missions: 0.092 m against 0.090 m.

**The bar passed, so one model can do both jobs at parity.** This is a measurement: the merged configuration still names the two
models, and a switch is its own piece of work. These missions had been looked at too, though nothing was tuned on this run. Two
copies of the model run in each mission, so nothing is shared yet, and the two jobs are not read from one pass.

## P6: where this handover stands (1 November)

**Handed to the model:** finding walkers and where they are (`walker_mode: held`), 68 of 80 round trips against the hand tracker's 67,
with no contacts. **Not handed over:** walker speed and forecasting. The model's fitted speed caused the failed live test, and neither
its rollout nor a straight line beats a held position on a moving walker at 0.8 s. The clearance checks were measured only. One model
for both jobs is shown at parity and not yet adopted.

**Next, one piece at a time:** adopt the single model, with its own before-and-after. Memory for the model is deferred until then.

## P6: four walkers (1 November)

`configs/experiments/walker_four.yaml`, cohorts `wf_*`, 40 missions paired, for information only. **The model's positions make 21 of 40
round trips against the hand tracker's 20, 6 gained and 5 lost (sign test p = 1.0).**

| Four walkers, 40 missions | Round trips | Stopped on contact | Failures |
|---|---|---|---|
| Frozen planner (`dn_obst4_c1`) | 5 | | |
| Close-up rule, first version (`cu_obst4_c1`) | 18 | | |
| Current planner, hand tracker (`wf_ref_c5`) | 20 | 1 | 14 timeouts, 4 pose, 1 contact, 1 other |
| Current planner, model's positions (`wf_held_c5`) | 21 | 2 | 10 timeouts, 6 pose, 2 contacts, 1 other |

The hand tracker also had one mission at 0.002 m from a walker without contact (maze 27 episode 1). The planner's forecaster differs
between the earlier rows and these, as does the fix, so the gain over the first version is not one change.

**Finding, with the hand tracker driving and the model watching:** the model found 77% of walkers in reach (95% interval 73-80%)
against 57% (53-61%), with 1.38 false alarms per 100 plans against 2.0 and position error 0.10 m against 0.15 m. Both find fewer than at
two walkers, and the model's lead holds.

**The contacts are the robot's blind rear, not the withheld speeds.** All three missions stopped by a contact (two with the model, one
with the hand tracker) were a rear calf on a walker outside the camera's view for at least 3 s, with no track on that walker at
contact. The model's two were maze 25 episode 0, reversing at 0.10 m/s while turning into a walker that had walked out of view, and
maze 13 episode 1, turning with a walker that had stood still for 2 s at about 0.45 m on its side. The taxonomy labels the second
"obstacle moving into the other"; the trace shows the walker standing, so the label is wrong. The hand tracker's was maze 13 episode 0, reversing
while turning. A speed would not have helped: there was no track to carry one. This is the same class as `cu_obst4_c1`'s collision
and C6's contact, and a walker's position behind the robot needs either memory or a limit on blind reversing, both outside this handover.

## P7: one model in the working setup (2 November)

**Reproduction: passed.** Six `wo_*` missions, from 104 s to a 480-s timeout, rerun with one model loaded once, one encoding and no hand tracker
(`configs/working.yaml`, cohorts `smc_repro_*`, commit ed143a7): `planning.json`, `requests.json`, `walker_shadow.json`, `mission.json`, `poses.json`,
`acquisitions.json`, the physics trace and the obstacle trace all equal their records exactly. So `wo_*` stands as the measure of this setup and the 80 missions
were not rerun. A plan frame takes 248 ms against 273 ms (medians of the same six missions, six running at once).

**Crash fixes: both missions now finish.** Rerun alone, because a fix changes nothing but the path that crashed (`smc_fix_obst1`):

| Mission, one walker | `wo_*` | With the fix |
|---|---|---|
| Maze 18 episode 1, a view recovery across a phase change | Crashed at 130 s | Round trip |
| Maze 26 episode 0, no floor plane at frame 0 | Crashed at frame 0 | Round trip |

**The fixed record: 70 of 80, no contacts** (68 before; 2 gained, none lost). The 10 failures are 9 timeouts and 1 pose loss. Every later handover is measured
against this record, with a running total (PLAN P7).

**A correction.** The earlier note on maze 26 calls its first-frame floor marginal. It is not: 5,770 of 10,920 candidates were good floor, and
`dominant_height_cluster` dropped them all because pruning was still removing points just over 3 mm at its eighth round. Any small numerical change could
tip this, which is why it came and went with the number of walkers.
