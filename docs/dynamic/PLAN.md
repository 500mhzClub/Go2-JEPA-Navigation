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

### P2b. Freezing the non-learned baseline (approved 24 October)

**Why a rule.** Hand-built planners that treat moving agents as obstacles, or predict them independently, hesitate
and freeze as density rises: the "freezing robot problem" (Trautman and Krause, 2010). Learned crowd-navigation
policies beat classical baselines such as ORCA mainly in dense scenes (Chen et al., 2019). In static scenes,
classical stacks stay competitive with learned ones (Mishkin et al., 2019), which matches the static 60/60.

So harness rules will reach diminishing returns as the scene gets busier. The comparison needs the baseline frozen at
a point set in advance, not whenever ideas run out. (References from memory; check them before any write-up.)

**A defensible baseline has:**
1. The standard non-learned parts for moving obstacles, and no more:
   - an occupancy map that clears and forgets transient cells (R7c);
   - a deadlock escape (R7d probe);
   - a constant-velocity track of observed obstacles in the clearance check.
     - This is the usual non-learned motion-prediction baseline (Schöller et al., 2020).
     - No arm predicts obstacle motion now. Without it, the comparison with a world model trained on moving
       obstacles could be dismissed as "the baseline ignores motion".
2. **Static parity:** R4 and R6 reference level with their records (paired, no losses).
3. **Tuned on development mazes, evaluated on held-out ones.**
   - Rules are tuned only on dev_tune and validation (mazes 00-29) with obstacles.
   - The frozen baseline is then run once on prelim_test 30-49, episodes 0 and 1, with obstacle seeds not used before.
   - The rules so far (clearing, forgetting, the escape) were diagnosed on 30-49 episode 0. The held-out run is what
     makes the final numbers defensible.
4. **No defect-class failures left.** Every remaining failure is classified, and none is a crash, an audit artefact
   or an implementation bug.

**Stop iterating on rules at the first of these** (paired on the development mazes against the previous round):
- **Diminishing returns:** two rounds in a row each gain fewer than 3 net missions of 40, or have sign-test p > 0.1.
- **Trading failures:** a change's losses reach half its wins.
- **Budget:** three rounds after R7c (round 1 is the R7d probe's rules; constant-velocity tracking is round 2).

**Before handing anything over: a continuous-command baseline (agreed 24 October).** The frozen baseline chooses
among six primitives, so part of its freezing may be the action set (no slowing, side-stepping or backing away), not
perception or prediction. Analysing its failures as they stand could hand a module to the world model when the limit
was the action set. So, before the failure analysis:
1. P5 step 1 (command representation; no behaviour change).
2. A continuous baseline: C1 scoring a command lattice, as the dynamic window approach does (Fox, Burgard and Thrun,
   1997), with C1's command model refitted on recordings across the whole range. Tuned and frozen under the same rule,
   then the same held-out evaluation. The gap between the two baselines is what the action set alone is worth.
3. The density curve and oracle substitution below run on the continuous baseline (the stronger one, which the
   world model must beat); the six-primitive baseline stays as the reference.

**C0 is dropped from the continuous work (agreed 24 October).** Its what-if branches cost about 12 times more per plan
with a 75-command lattice, and far more under tape search. It runs once more on six primitives in the frozen
baseline's held-out evaluation, then not in the continuous baseline's rounds, the density curve or P5. The oracle
substitution's "own motion forecast" row uses the six-primitive C0-C1 gap. C0 comes back for one final evaluation before
a paper draft, as an upper bound.

**Then hand parts over by evidence, not by intuition:**
- **Density curve:** the frozen baseline at 0, 1, 2 and 4 obstacles, at two speeds. The slope is the harness's
  structural limit, and the point where it freezes.
- **Oracle substitution, one module at a time:**
  - the map: true walls and obstacle positions;
  - the pose: the true pose instead of visual odometry;
  - obstacle prediction: true future obstacle positions in the clearance check;
  - the robot's own motion forecast: C0, already.
  - Each oracle's gain bounds what learning that module can give. The largest gain is the first part to hand to the
    world model.
- **The C0-C1 gap** is the share of failures that better motion forecasting alone can fix.

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
- **New: forecasts of switching command tapes.** The planners in P5 search tapes whose command changes every 100 ms,
  while the static corpus held mostly one primitive per tape. Gate: the displacement test on tapes with several
  switches (forward, then an arc, then the other arc), not only constant commands, and with lateral commands.
- **New: obstacle forecasting.**
  - Probe obstacle position relative to the robot from level-1 rollouts and from upper-level predictions, against
    obstacle persistence.
  - This is the first place H-JEPA's 1.6-12.8 s horizon can matter: whether a corridor will be free in 3 s.
- **Mover gate (set 27 October, before the final models were scored).** Probe: `go2nav.eval.mover_probe`;
  supervised ceiling: `go2nav.eval.mover_ceiling`, a CNN trained on the frames themselves (the ceiling below is
  the better of its plain and regularised runs, per measure).
  - What the ceiling showed: the frames localise a mover to 0.156 m (median; the world models' latents read about
    0.24 m), but even supervised, nothing forecasts the walkers' turns and pauses beyond persistence; they are random
    by construction. Event skill is therefore reported, not gated.
  - **Response (the gate):** on the validation mazes, through the rollouts under the hold and advance tapes, the
    ridge head's response gain above 0.12 at 1.6 s and 3.2 s, with its 95% interval above 0.12 and response R2 above
    0 (the predicted responses carry more signal than noise). This is the give-way decision the failure analysis
    found to be the limit, and only a command-conditioned model can supply it.
  - **Localisation:** the h = 0 readout within 1.25 x the ceiling's median error.
  - **Velocity:** velocity skill (1 - error / persistence's error on windows where the mover walks straight, so
    constant velocity is exact) at 0.8 s at least half the ceiling's.
  - Reported for LeWM's encoder and rollouts, and for H-JEPA's level 1 and its 1.6 s level.
  - **If no model passes the response gate:** propose a representation change (patch-level prediction, or the mover
    positions as an auxiliary training target, never an input) and its compute, and meanwhile build P5's hybrid
    (tracked movers for mover state, the world model for the robot's own motion, give-way kept as a rule).

### P4b. Patch-grid LeWM (proposed 27 October, after the mover gate failed)

**Why.** LeWM v2 failed the mover gate on all three measures. Its encoder's patch tokens localise movers within 16%
of the supervised ceiling, but the [CLS] latent the predictor works on is 65% worse (STATUS, P4). The information is
in the encoder and lost at the bottleneck. So the state keeps a coarse grid of patch tokens, and the predictor
forecasts the grid as well as [CLS]: predicting where each region's content goes next is also a direct training
signal for motion, which the velocity measure needs.

**Model** (`grid` in the config; everything not listed is go2_v2's):
- **State per frame:** [CLS]'s z (192) as now, plus G = 12 grid tokens. The 12 x 16 patch tokens are average-pooled
  over 4 x 4 blocks to 3 x 4, and each goes through a shared BatchNorm projector to 192 (the [CLS] projector's shape,
  its own weights).
- **Predictor:** the same transformer (6 blocks, width 192) over the history's 4 x 13 tokens, with a spatial position
  embedding (13) added to the temporal one. Attention is block-causal: a token sees every token of its own frame
  and of earlier frames. Every token of a frame is conditioned on that frame's action (AdaLN, as now). Each position
  predicts the matching token of the next frame; rollouts are autoregressive as now.
- **Loss:** the teacher-forced and 8-step rollout MSE on [CLS] plus the same on the grid (mean over tokens), weight
  1 each. SIGReg on [CLS] and on each grid position (the batch as samples), averaged. Inverse dynamics on [CLS] only,
  as go2_v2 (spans 1 and 8, weight 100), so the robot's own motion stays where C5-C8 read it.
- **Not chosen first:** the movers' positions as an auxiliary target. It uses privileged labels, and it would teach
  [CLS] where movers are without giving the predictor a reason to move them. It stays the fallback if the grid
  model localises but its response test fails.

**Compute.**
- Per sample, go2_v2's encoder does about 30 times the predictor's work (12 frames x 193 tokens through 5.5M
  parameters, against about 36 predicted [CLS] tokens through 10.8M).
- Thirteen tokens per frame raise the predictor to about 40% of the encoder. That gives about 1.4 x go2_v2's
  0.49 s/step: about 0.7 s/step, **about 12 h** for 60k steps.
- A 6 x 8 grid would be about 2.5 x (about 20 h).
- A 30-minute smoke run first measures throughput and memory under the 85% cap; encoder block checkpointing is
  available if memory runs short.
- The R9700 cannot hold it beside H-JEPA v2 (25 GB), so it starts when H-JEPA v2 finishes.

**Probe and gates.**
- `mover_probe` reads grid models from [CLS] and the 12 grid tokens: the ridge and MLP heads on the flattened
  tokens, and the patch head on the 3 x 4 grid, at h = 0 and through the rollouts.
- The mover gate is unchanged. L1-L4 are scored on [CLS] and must not fall below go2_v2's, so the robot's own motion
  does not regress.
- If 3 x 4 misses localisation, 6 x 8 is next (the patch tokens passed at 12 x 16).

**Planning cost.** A rollout pushes 13 tokens per frame through the predictor instead of one, about 10 times
go2_v2's cost per candidate (the encoder runs once per decision). This is measured before P5 sets candidate counts.

**H-JEPA** stays on [CLS] until the grid model passes; then its upper levels can read the pooled grid.

### P4c. Frozen-target grid LeWM (approved 29 October, after the grid model failed the mover gate)

**Why.** The grid LeWM fails the gate as go2_v2 does (STATUS, P4b). Its encoder still holds the mover (its patch
tokens pooled to 3 x 4 localise at 0.156 m), but its grid projector writes one frame-wide signal into every cell
(spread across cells 0.37 before it, 0.12 after), and the cells lose the mover (0.228 m). This is the failure
Sobal et al. (2022, arXiv 2211.10831) predicted for JEPAs with variance-style regularisers: features that differ
between episodes and are predictable over time (layout, texture, the robot's own motion) fill SIGReg's quota at no
prediction cost, so a small, partly random walker only costs error and is dropped. SIGReg is per cell, so nothing
stops the cells sharing that code. Any loss on a learned target can be met by removing the information; the remedies
that worked in the literature use a target the model cannot change (pixels or depth in object masks, frozen
features as in DINO-WM, arXiv 2411.04983), a discriminative constraint, or object slots.

**Model** (`configs/lewm/go2_v2_frozen_grid.yaml`; everything not listed is go2_v2_grid's):
- **Encoder:** go2_v2's ViT and [CLS] projector, loaded from its checkpoint and frozen (eval mode, no gradients):
  the state's [CLS] is go2_v2's z exactly.
- **Grid:** go2_v2's patch tokens average-pooled to 3 x 4, with no grid projector (`grid_projector: false`). Frozen,
  they localise the mover at 0.132 m with the patch head (STATUS, P4b), so the target holds it by construction.
- **Loss:** the teacher-forced and 8-step rollout MSE on [CLS] and on the grid, weight 1 each. No SIGReg and no
  inverse dynamics: nothing upstream of the predictor trains, so neither can act. Only the predictor, the action
  encoder and the two output projectors learn.
- Same seed, data stream and schedule as go2_v2_grid.

**Compute.** No encoder backward pass and no encoder checkpointing: a 30-minute smoke run measures throughput and
memory first; expected about 0.4-0.5 s/step, **6-8 h** for 60k steps.

**Probe and gates** (set before the model is scored):
- The mover gate and its pass marks are unchanged. The ridge and MLP heads on the flattened state stay as they are.
- **Grid readout (new, alongside):** the patch head (`mover_ceiling`'s convolutional head and training) on the 3 x 4
  grid tokens of frames t-3..t plus the state at t+h, true or rolled out under the executed, hold and advance tapes,
  predicting the mover's position at t (h = 0) and its displacement to t+h. It is scored as the probe's heads,
  including the response test. A grid model passes a measure if either readout passes it; both are reported.
- The same readout is scored on go2_v2_grid (final) for comparison.
- L1-L4 on [CLS] cannot regress: [CLS] is go2_v2's encoder; the rollouts are scored as usual.

**What this does not fix.** Velocity: no frozen per-frame representation reaches the mark (go2_v2's raw 12 x 16
patch tokens 0.079 against 0.126). If the rollouts still miss velocity and the response test, the next steps are,
in order:
1. The movers' positions as a training-only auxiliary target per grid cell, from corpus v2's `dynamic.npz` (the
   probe then partly measures that supervision; the transfer split and the response test carry the evidence).
2. Motion in the encoder's input (two-frame tubelets or a frame-difference channel).

Depth supervision (arXiv 2607.16314, on LeWM) would need depth images re-rendered for corpus v2, which stored RGB only.

### P4d. Grid LeWM with training-only mover supervision (approved 29 October)

**Why.** The frozen target (P4c) keeps the mover in the state but caps it at go2_v2's encoder: from four frames its
pooled tokens reach velocity skill 0.100 against 0.126, and the encoder cannot improve. Training jointly lets the
encoder sharpen its mover features; a fixed label target closes the trap P4c describes, since no latent can meet it
by dropping the walker. Precedent: object or position supervision on JEPA world models' latents (arXiv 2609.30214,
which reports SIGReg alone losing pose information that the supervision keeps, against LeWM; 2609.04264; and the
theory of 2509.12249: when the dynamics and auxiliary losses reach zero, frames that differ in the auxiliary target
cannot share a latent); privileged training-only information in world models (Scaffolder, arXiv 2405.14853; informed
POMDPs, 2306.11488; MILE's BEV supervision, 2210.07729).

**Labels** (`go2nav.data.mover_labels`, from corpus v2's records; never a model input): per frame and per image column
of the 3 x 4 grid, whether a walker is in view there (the mover probe's view test: within 5 m, inside the field of
view, centre not hidden by the walls) and the nearest such walker's position in the body frame. A column is the
walker centre's bearing projected through the primary camera (78.3 degrees across, four equal image columns). Frames
with fewer than 200 obstacle depth pixels have none.

**Model** (`configs/lewm/go2_v2_grid_movers.yaml`): go2_v2_grid (trainable encoder, grid projector, SIGReg, inverse
dynamics) plus a mover head, an MLP shared by the four columns that reads a column's three grid tokens of the
encoder's state (and the column's index): a presence logit and the position (scaled by 3 m). Loss `mover`: binary
cross-entropy on presence plus the squared position error where a walker is present, weight 1, on every frame of the
window. **Only the encoder's state is supervised:** the predictor and its rollouts see no labels.

**Evidence** (set before training):
- Localisation at h = 0 is now partly supervised: reported, not evidence.
- Evidence: the forecasts through the rollouts (velocity skill at 0.8 s, displacements to 3.2 s), the response test,
  and the transfer split, read by both the probe's heads and the grid readout, against the same pass marks.
- The final verdict is P5's missions against lv2.
- L1-L4 on [CLS] must not fall below go2_v2's.

**Compute:** labels on the CPU (all 4,000 units, in a process pool); training as go2_v2_grid, about 15 h.

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
   - Compare against the lattice at equal compute, and against **C1 with the same tape search** (CEM through C1's
     command model, which also forecasts any tape). Otherwise a win measures the search, not the world model.
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
| P4b | About 12 h on the GPU (the 3 x 4 grid), after H-JEPA v2 |
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
