# Dynamic environments and a continuous action contract (plan, draft for review)

Branch `feature/dynamic-continuous`. This follows the frozen static-maze baselines (round trips of 100: C1 72, C5 71,
C6 71, C7 70, C8 70; C3 50/60, C4 55/80) and `docs/ACTION_CONTRACT.md`.

## Questions

1. With moving obstacles, which parts of the navigation harness break, and which can a world model take over?
2. How much does the six-action contract cost once the scene moves? What do continuous commands, with a planner that
   searches them (CEM, MPPI, hierarchical), buy back?
3. Does H-JEPA's long horizon start to matter? A static maze could not show it, and dead reckoning wins there.

As before, the aim is functional baselines for each arm, not 1:1 rigour. We measure what breaks before replacing it.

## Starting point: what the static harness assumes

| Assumption | Where | What happens with moving obstacles |
|---|---|---|
| Six actions, 400-ms commit after a 300-ms prefix | `planning/actions.py`, `planning/dispatch.py` | No slowing, side-stepping or backing away. A 0.3 m/s obstacle moves 0.21 m during the 700 ms before a plan ends |
| The locomotion policy knows nine fixed commands | `configs/locomotion/contract_ppo.yaml` | Commands off those points are untested |
| Safety limits: forward ±0.3 m/s, lateral 0, yaw ±0.5 rad/s; guard stops a mission above 0.3 m/s | `locomotion/safety.py`, `sim/contacts.py` | These cap any wider command range |
| The map accumulates occupied cells and never clears them | `perception/mapping.py` | A moving obstacle leaves a trail of permanent walls, which blocks routes and frontiers |
| The dispatch check sees only fresh depth within 1 m | `planning/dispatch.py` | This still works: the reactive layer sees current obstacles |
| Any contact other than the feet on the floor stops the mission | `sim/contacts.py` | An obstacle walking into a stationary robot counts as the robot's failure |
| Selection, latches, terminal and deadlock rules are keyed on action names | `planning/selection.py`, `planning/planner.py` | Continuous commands need a category (turn in place, translate, arc, hold) for these rules |

## Decisions proposed (to confirm)

1. **Command range for the new locomotion policy.**
   - Forward -0.25 to 0.45 m/s, lateral ±0.20 m/s, yaw ±0.8 rad/s.
   - The safety block and guard rise to match (guard 0.5 m/s).
   - This contains every current primitive. It stays moderate because perception (10-Hz visual odometry, 2.5-Hz
     mapping) was tuned for 0.2 m/s; faster driving is checked rather than assumed.
2. **Obstacles are scripted, kinematic and polite in the first condition.**
   - Cylinders about 0.35 m across and 0.9 m tall, textured, so the depth layer (3-65 cm) sees them.
   - They walk the maze's passage graph from cell centre to cell centre at 0.15-0.35 m/s, with pauses.
   - An obstacle stops when the robot is within 0.8 m ahead on its path, and turns back after 3 s blocked.
   - Every episode stays solvable; the difficulty is timing, prediction and ghost trails. Obstacles that do not yield
     come later, as a harder condition.
3. **Contact attribution.**
   - Any contact still ends the mission, as now.
   - The evaluator also records whether the robot was commanded to move and which body was moving into the other. The
     failure analysis can then separate unavoidable hits from controller faults.
4. **One new corpus,** recorded after both the policy and the obstacles exist: moving obstacles plus commands across
   the whole range. The static corpus is not re-recorded on its own.
5. **C3 and C4 stay on the six primitives** as static references. Their V-JEPA forecasters cost about 30 minutes a
   mission with six candidates, and retraining them on a new corpus is out of scope. C1, C2 and C5-C8 move to
   continuous commands.

## Phases and gates

### P1. Locomotion policy on command ranges (about a day; independent of P2)

- **Training:** command sampling becomes a mixture, which keeps the old points well tracked while covering the range:
  - 50% uniform over the range;
  - 30% the primitive bank;
  - 20% hold.
  - More iterations if needed (501 now, about 8 min).
  - Resampling stays every 4 s, and switches mid-episode are also trained.
- **Safety limits and guard:** follow the new range (decision 1).
- **Benchmark:** `locomotion.benchmark` gains a lattice over the range (forward x lateral x yaw) and continuous random
  switching, alongside the nine primitives.

| Gate | Test |
|---|---|
| M1 | Tracking RMS across the lattice within 0.05 m/s and 0.1 rad/s, no resets |
| M2 | The nine primitives track no worse than the current policy (benchmark `--compare`), on normal floor and friction 0.3 |
| M3 | C1 on R4 and R6 reference (40 missions) on the new policy is level with its static record (sign test), with C1's command model refitted if needed |

M3 keeps the static baselines meaningful on the new policy.

### P2. Moving obstacles in the simulator (2-3 days; independent of P1)

- **Spike:** move kinematic Genesis entities every physics step on the CPU backend. Check determinism (a rerun matches
  bit for bit), contact reporting against the robot, and rendering in both cameras.
- **`sim/obstacles.py`:** paths seeded from the episode, polite behaviour (decision 2), and the trace recorded for
  replay and evaluation.
- **Evaluation:** contacts attributed per object; minimum clearance to obstacles kept separate from walls; waiting
  time; a taxonomy row for each new failure type.
- **R7 benchmark (`r7_moving_obstacles`):**
  - R4's episodes with one and then two obstacles (40 missions per arm per level), paired with R4.
  - Run as-is on the current harness and primitives with C1, C5 and C7 (C6 and C8 if cheap). This is the
    failure-analysis input.

| Gate | Test |
|---|---|
| D1 | Determinism: a mission rerun reproduces its trace exactly |
| D2 | Solvability: the oracle C0 completes at least 95% of R7 |
| D3 | The taxonomy covers every R7 failure: no "unclassified" |

### P3. Corpus v2 (about 10 h of collection)

- Recorded on the new policy, with obstacles.
- **Explore units:** continuous perturbations (random commands across the range) replace the six-action
  substitutions.
- **Free units:** the scripted driver samples the whole range, including lateral moves and reversing.
- **Mission units:** C1, as before.
- Lateral is kept in the stored command blocks.

| Gate | Test |
|---|---|
| L0 | As for the static corpus, plus coverage of the command range |
| L0 dynamic | Share of frames with an obstacle in view, and with an obstacle moving relative to the robot |

### P4. World models (about 14 h each)

- **LeWM** with span8w's settings and an action block of 15 values (forward, lateral and yaw over five ticks), then
  **H-JEPA** with the merged configuration. Same gates as before (L1-L4, H1-H4).
- **The displacement test split by condition:** contact, low friction, near walls, obstacle in view. Here a world
  model can beat dead reckoning.
- **New: obstacle forecasting.**
  - Probe obstacle position relative to the robot from level-1 rollouts and from upper-level predictions, against
    obstacle persistence.
  - This is the first place H-JEPA's 1.6-12.8 s horizon can matter: whether a corridor will be free in 3 s.

### P5. Planning over continuous commands

1. **Command representation.** A command tape with a derived category replaces the action names, so the harness rules
   still apply. The six primitives become six points of the space. Gate: with only those six candidates, every static
   arm reproduces its R4 missions exactly.
2. **Richer candidate sets.**
   - A fixed lattice (for example forward 5 x lateral 3 x yaw 5) scored by each arm's forecaster through the existing
     selection score and clearance checks.
   - C1's ridge model already takes commands. C5-C8 batch the candidates on the GPU.
   - C2 needs a continuous reactive rule (for example sampled commands checked against fresh depth).
3. **CEM and MPPI** for the JEPA arms: optimise the command tape through the world model, rolled out and decoded,
   against the same objective (waypoint progress plus clearance).
   - This includes timing: an 800-ms tape replanned every 400 ms, and shorter commits once dispatch allows it.
   - Compare against the lattice at equal compute.
4. **Hierarchical planning with H-JEPA.**
   - Offline first: CEM over L3/L2 macro-actions from a start frame to a goal frame 12.8-25 s later, checked against
     the macro-actions taken.
   - Then online: subgoals for level 1. Goal-aware training comes in if the cost geometry is still line-of-sight
     (docs/hjepa/STATUS.md).

Each step runs on R4 (static), R6 reference and R7, so a gain in dynamic scenes cannot hide a loss in static ones.

## Compute and order

| Phase | Compute |
|---|---|
| P1 | About a day, mostly CPU missions |
| P2 | Development, plus about 240 R7 missions (about 5 h) |
| P3 | About 10 h on the CPU |
| P4 | About 14 h per model on the GPU |
| P5 | Mostly missions; CEM costs scale with candidates times rollout |

- P1 and P2 run in parallel. P3 waits for both, and P4 waits for P3.
- P5 step 1 (command representation) can start any time, since it changes no behaviour.
- Nothing here approaches 3 GPU-days at once. P4's two models are about 28 h.

## Risks

- **Ghost trails could dominate R7.**
  - If the accumulating map makes most R7 failures the same, that is the first harness part to hand over or fix.
  - The minimal functional fix is to decay occupied cells that are seen empty again.
  - The question is whether to fix it in the harness for every arm or let the world model own it. Decide after the R7
    taxonomy.
- **Perception at higher speeds.** Visual odometry and mapping were validated at 0.2 m/s. M3 and the first
  continuous missions check it, and the range shrinks if pose loss rises.
- **Kinematic obstacles in Genesis** may not report contacts or stay deterministic as walls do. The P2 spike settles
  this before anything else is built on it.
- **One policy for every arm.** Once P1 lands, every new run uses the new policy. Static comparisons against the
  frozen baselines rely on M2 and M3.
