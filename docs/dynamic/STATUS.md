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
