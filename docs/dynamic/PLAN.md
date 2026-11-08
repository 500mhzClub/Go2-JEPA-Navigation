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

**Stopping an evaluation once its answer is clear (decided 6 November).** Mission rounds are checked every 20 missions,
and a cohort, or the rest of a round, stops as soon as one of these holds:
- **Bar decided.** A bar set beforehand can no longer change: a count bar ("at most k") is already exceeded, or a floor
  ("at least k") cannot be reached even if every remaining mission succeeds. Only a fail can be called early; a pass
  needs every mission. Once one bar of a step fails, the step's other cohorts are not run unless they answer a different
  question, said before they are skipped.
- **Question answered.** A diagnostic or comparison run states its deciding count before it starts (for example, a
  failure seen in one arm and never in the other); it stops when the count is met on matched missions, or when a one-sided
  exact test on the matched pairs gives p < 0.01.
- **Broken.** A failure mode the bars never anticipated (a new kind of contact, a crash path) in at least 5 of the first
  20 missions: stop, diagnose, and report before running more.
A stopped run is reported as stopped, with how many missions ran and which rule stopped it.

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

### P4e. The give-way gate (declared 29 October, before P4c's and P4d's models are scored)

**Why.** The mover gate checks that a representation holds the movers; it is not a navigation requirement. Its marks
are relative to the supervised ceiling (velocity skill 0.126 is a median gain of under 2 cm over persistence at
0.8 s), it pools medians over every distance to 5 m, and its baselines are not what lv2 already has. The module
oracles (STATUS, failure analysis) found that true futures of every mover change nothing for lv2 (49 against 49,
4 / 4), since lv2 already predicts tracks at constant velocity; the limit is the give-way decision (hold, back off,
or take another way) with a mover near and ahead. A world model can add only what constant velocity cannot: how a
mover responds to the robot's choice.

**Test** (`mover_probe.give_way`, both readouts' rollout heads, the probe's evaluation windows):
- **Windows:** the mover at t within 1.5 m, ahead (x > 0) and within 0.75 m of the path (|y|), with a
  counterfactual.
- **Tapes:** hold, advance (0.3 m/s) and a new retreat tape (-0.10 m/s, the lattice's reverse), through the same
  rate limits; the truth from `branch` (the walkers against a kinematic robot), as the response test.
- **Contrasts:** retreat minus hold, and advance minus hold, of the mover's position at 1.6 s and 3.2 s; the model's
  from its rollouts under each tape, read by the rollout heads.
- **Scores:** the AUC of the predicted contrast's size for whether the true one exceeds 5 cm (the mover responds to
  the choice), with a 95% interval over units; and the contrast's R2. lv2's constant velocity has no contrast: AUC
  0.5, R2 0.

**Pass** (beats lv2's assumption where the decision is made): for the retreat contrast at 1.6 s and at 3.2 s, the
AUC's 95% interval above 0.5 and R2 above 0, on the validation and the transfer windows, by either readout. The
advance contrast is reported. Missions stay the final verdict; this gate chooses which models earn them.

The validation labels are regenerated with the retreat tape (the windows and every other label unchanged); models
already probed are re-scored on their evaluation latents with the new tape.

**Ceiling** (added 29 October, before P4c's MLP and patch rollout readouts were read; its ridge readout had been).
Walkers look alike, and whether one responds depends on state the camera shows only through its motion (stopped for
the robot or pausing at a cell, its heading while stopped, its politeness), so the gate's attainable score depends on
the history a predictor sees. `eval.give_way_ceiling` fits supervised heads on the give-way windows from privileged
inputs: the mover's and the robot's true tracks over 0.3 s (a world model's 4 frames), over 3 s, and the 3-s tracks
plus the walker's internal state (an oracle). The pass mark above is unchanged. A model scores also as a share of
the 0.3-s ceiling. If the 0.3-s ceiling itself fails the pass mark, a 4-frame model's failure is uninformative about
its predictor, and the next model change is the history (a longer context or a recurrent state), not the objective;
if the 3-s ceiling passes and the 0.3-s one does not, that is the measured case for it.

**Ceiling result and the bar** (29 October). The 0.3-s ceiling passes (retreat AUC 0.87 [0.81, 0.92] validation,
0.85 transfer at 1.6 s; R2 0.61 / 0.32), and 3 s adds little: history is not the limit. A mover stopped for the robot
responds to a retreat in 85-97% of windows, a moving one in 15-42%, so a one-line rule (the mover near and ahead is
still) scores AUC 0.80 / 0.77. Raising the bar to the rule was considered and not adopted (approved 29 October): a
lower bound above 0.80 is about the privileged ceiling's own, and the rule has no magnitude or timing. The bar stays
at 0.5; each model's gate report carries the rule's AUC beside it, and the rule is tested where it matters, in
missions (P4f): if lv2 with a give-way rule beats lv2, it becomes the baseline a world model must beat there.

### P4f. The give-way rule in missions, and a give-way top-up of the corpus (approved 29 October)

**Why.** P4c (the frozen-target grid LeWM) keeps the mover (localisation 0.096 m) but fails the give-way gate with
an inverted ranking (retreat AUC 0.25-0.40 on validation): its rollouts carry movers' current motion on and leave
still ones still. In corpus v2, when a mover waits for the robot near and ahead (24.6k training windows), the robot
holds or turns in place in 93% of the next 1.6 s, backs off in 4.3% and advances in 2.3%: the predictor has rarely
seen a waiting mover answer a retreat. Two follow-ups, one per side of the comparison:

1. **lv2 with a give-way rule** (`configs/experiments/give_way_probe.yaml`, `planner.give_way`). A confirmed track
   standing still within 1.5 m ahead (within 0.75 m of the heading) while the selection does not advance makes the
   robot back off at 0.10 m/s, with the scripted backup's checks (observed floor, 0.40 m of stored clearance), until
   it is 1.3 m away, moves or is lost (at most 8 steps; then 10 plans without the rule; 3 back-offs per track).
   Validation mazes 10-29, episodes 0 and 1: static and one and two movers, paired with lv2; and with mover trails on,
   one and two movers, paired with the trail probe's lattice cohorts (their residual is the mover in view on the only
   path). 200 missions. One probe round, like the trail probe: it asks whether the decision the module oracles named
   yields to a hand rule. If it wins, lv2 with the rule is frozen as the mission baseline (held-out run as P2b) before
   any world model is compared in missions.
2. **A give-way top-up of corpus v2** (`lewm_corpus_v2`, kind `giveway`): one free unit per training maze (1,200,
   600 frames each), with two to four movers (half the units non-polite), driven by `GiveWayDriver`: encounters
   weighted to 65%, each two to four decisions while the mover stays in sight (approach to 0.6-1.1 m, wait 0.3-1.2 s,
   then retreat at -0.10 m/s 50%, advance 20%, hold 20%, turn away 10%), under the free driver's stand-off rule. A
   six-unit pilot at a first setting (wait to 2 s, retreat 40%, a fifth non-polite) put 14% of one unit's waiting
   windows before a retreat; at this one, 25.7% (advance 12.4%) over 638 waiting windows (pilot units discarded). Then P4c retrained on corpus v2
   plus the top-up (`go2_v2_frozen_giveway`: the same seed, frozen encoder, model and schedule; only
   `data.train_kinds` differs), probed and read out as P4c, and scored on the mover and give-way gates.
   - **Gate on the data first:** among the top-up's give-way windows with a mover waiting, the share followed by a
     retreat within 1.6 s at least 25% (against 4.3%).
   - **Compute:** collection about 5 h on the CPU (rendering on the R9700, after P4d's training), training about 6 h
     on the GPU, probe and readout about 4 h on the CPU.

**What the outcomes mean.** If the retrained model passes the give-way gate, the predictor learns the reaction once
the data contain it: the path is data that exercises the decision, not a new objective. If it still fails with an
inverted ranking, the predictor's prior (carry motion on) wins even with the data, and the next step is supervising
the predicted states (P4d's labels on rollouts). If the rule wins in missions and the model passes too, the model must
beat lv2 with the rule there.

**The first rule did not back off (30 October).** It scored 38/40, 31/40 and 15/40 against lv2's 38, 31 and 18, which
says little about backing off: of its 255 triggers, 109 took no reverse step, six moved the robot 10 cm or more, none 30
cm, and the median was 3 mm. 97 ended on the scripted backup's rule against passing more than 1 cm closer to a wall than
the robot stands (94 of them had the 0.40 m of clearance), and 130 when the track moved or was lost. Against the true
positions, the mover was standing still at 40% of the triggers, and 216 of the 255 had walked 0.3 m within 6 s, polite
walkers turning back on their own; the robot had then moved 0.3 m in 14% of those.

So the rule now commits (`configs/experiments/give_way_commit.yaml`): once triggered the robot backs off 0.40 m whatever
the track does, at most 25 steps, and a step needs only observed floor and the footprint's 0.45 m of clearance. (At the
scripted backup's 0.40 m, inside the planner's footprint, a rear calf hit a wall in 2 of the first 7 missions, ending
both, 4 and 17 frames after a back-off stopped at that floor; those runs are `gwct_*`.) Mover trails on, one and
two movers, paired with the trail probe's lattice cohorts (`tp_lat_obst*`): their residual is the mover in view on the
only path, and without trails a mover that leaves still seals the passage with its ghost trail. 80 missions, replacing
the runs of the first rule with mover trails, stopped at 9 of 80 because the rule they tested changes the robot's motion
by millimetres. The first rule's cohorts (`gw_*`) are at dc5e21a.

**Outcome (31 October).** The committed rule does not beat mover trails, and the frozen-target model on the top-up fails
the give-way gate with its rankings still inverted (STATUS). The step named above for that outcome, P4d's labels on
rollouts, is deferred: a rule at AUC 0.80 gained nothing in missions, so a model that matches it has no clear route to a
gain. Next, the mover-trails baseline's failures are sorted by what sealed the passage, and the fix is chosen from that.
Mover prediction returns if failures remain that need a mover's reaction.

### P4g. The close-up rule (31 October)

**Why.** The sorted failures (STATUS) show the map right and the response wrong: a walker anywhere on the only route
leaves no goal or frontier, and the robot scans where it stopped, with more than 2 m clear for about half of that time.

**Rule** (`close_up`, `routing.close_up`, off by default so lv2 is unchanged). When the route is cut, search again with a
0.10-m footprint, which passes beside the walker, and follow that route to its furthest cell the 0.45-m footprint reaches
and at least 1 m from every track. A walk under 0.25 m keeps the cut, so the robot scans there as before. It uses only the
robot's own map and tracks; the forecasts and contact checks still apply to every step.

**Test** (`configs/experiments/close_up_probe.yaml`, cohorts `cu_obst*`): 80 validation missions with mover trails, paired
with `tp_lat_obst*` (54 of 80), about 3 h on the GPU. Set before it runs: the rule is adopted if it has no collisions and
wins at least twice as many missions as it loses, with the sign test reported. Also read: sealed seconds per mission, and
whether the robot closes up on walkers that then clear, or stands off and waits as long as before.

**Outcome and last round (approved 31 October).** The rule scored 60 of 80 (STATUS), and losses reached half the wins, so
this round goes past P2b's stop, by decision, and is the last one. The stand-off counted every track, so a brief track on
a wall took the target away until it was lost again; only confirmed tracks hold the robot off now.

1. **Fix check** (`configs/experiments/close_up_fix.yaml`, cohorts `cuf_obst*`): the same 80 missions, paired with
   `cu_obst*`. Set before it runs: the fix is kept if it has no collisions, wins at least twice as many missions as it
   loses, and fewer than 15 of the 80 missions spend 60 s or more turning between the cut and scanning within 0.4 m of one
   spot. Otherwise the rule goes back to its first version. Also read: the closest approach to a true walker.
2. **Four walkers** (`configs/experiments/close_up_four_walkers.yaml`): 40 missions paired with `dn_obst4_c1` (5 of 40),
   for information only.
3. **Held-out evaluation** (`configs/experiments/close_up_fix_heldout.yaml`, cohorts `cufh_*`): once, on the rule with the
   fix, paired with lv2's 112 of 140 (cohorts `lb_*`).

**The fix is kept by decision (31 October).** It missed step 1's bar (STATUS), and that rule said the first version
stays. The user overrode it after seeing the result: the mission-by-mission comparison shows the gains coming from the
loop going away, and 13 changed missions are too few to settle a 2-to-1 bar. The held-out mazes have seen neither version,
so the choice does not bias that result, provided one version runs once. Step 2 ran on the first version.

**Outcome (1 November).** Step 3 scored 125 of 140 against lv2's 112 (17 gained, 4 lost, p 0.007) with no collisions (STATUS).
The rule stays on: it is the working planner and the end of hand-rule rounds on this branch.

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

### P6. Handing the walkers to the world model (approved 1 November)

**Why.** The aim is to absorb the hand-built planner into the world model, one part at a time, judging each handover by
parity with the part it replaces. The robot's own motion is already handed over (C5: 91 of 120 against lv2's 87). The walkers
are next: the hand tracker finds them from the depth map and assumes constant velocity. Predicting how a walker reacts stays
parked: three models failed that gate and a rule that scored well on it gained nothing in missions. True walker futures also
changed nothing for lv2 (the module oracles), so this round expects parity, not a gain.

**The order of handover** (each by parity, in missions):
1. The robot's motion forecast. Done.
2. Finding and forecasting walkers. This round.
3. The short-range clearance checks: measured this round, the hand checks keeping authority.
4. The occupancy map.
5. Responses to walkers.
6. Routing and exploration, including the close-up rule.

Steps 4 to 6 need memory the model does not have: it sees 0.3 s. The planner's hard limits, such as the depth stop and a limit
on blind reversing, stay outside the model until a handover shows them unnecessary.

**Finder and tracker** (`go2nav.controllers.walkers`). The model's mover head reads each image column of the four latest
frames. The hand tracker's association and velocity fit stay on top of the detections, so this is a partial handover.

**Shadow run** (`configs/experiments/walker_shadow.yaml`, cohorts `ws_*`): 120 validation missions, C5 with the close-up rule and
mover trails, the hand tracker driving and go2_v2_grid_movers logging its walkers beside it. Scored by `eval.walker_compare`
plan by plan against the true walkers, on walkers within 2 m in the primary camera's view.
- **Bar, set before scoring:** the model finds at least 90% of the share the hand tracker finds, with no more false alarms
  per 100 plans, and a median position error at most 1.25 times the hand tracker's.
- **The hand tracker on the 80 `cuf_*` missions, for scale:** found 68% (95% interval 65-72%), 0.59 false alarms per 100 plans,
  position error 0.148 m, velocity error 0.067 m/s. The bar itself is read from the shadow run's own hand tracks.
- The shadow's walker cohorts are the live test's reference: they change the forecaster from C1 to C5 and nothing else.

**Combined model** (`configs/lewm/go2_v2_walkers.yaml`, `outputs/lewm/go2_v2_walkers`): go2_v2_grid_movers warm-started, 30k steps,
about 7.5 h, corpus v2 plus the give-way units. Its mover labels also apply to the rolled-out states, so the predictor is
taught to carry the walkers (P4f's deferred step), and a free-space head reads 16 rays to 1.5 m (`data.free_labels`) from
encoded and rolled-out states. Labels are never an input. Read offline on validation and transfer: the walker
forecast through rollouts against constant velocity on true positions, the free rays' error, and L1-L4's rollout error against
go2_v2's. C5's decoder needs a refit on this model; until then go2_v2 keeps forecasting the robot.

**Which model drives** (set before the new model is read): go2_v2_walkers, if on the same validation units through the same
tracker it finds at least as many walkers as go2_v2_grid_movers with no more false alarms; otherwise go2_v2_grid_movers.

**Live test** (`configs/experiments/walker_live.yaml`, cohorts `wl_*`): the 80 walker missions with the model's tracks driving
the planner and the mover trails, the hand depth stop left in place.
- **Bar, set before scoring:** at most 3 fewer round trips than the shadow's reference, no collisions. Eighty missions check
  parity loosely, and the sign test is reported beside it.

**Rerun, positions only** (`configs/experiments/walker_held.yaml`, cohorts `wh_*`): the same missions and the same bar, after the
live test failed on false speeds. It passed, so the model gives the planner its walker positions and the planner gets no speeds.

**One model for both jobs** (`configs/experiments/walker_one.yaml`, cohorts `wo_*`): the same 80 missions with go2_v2_walkers also
forecasting the robot, through `go2_v2_walkers_heads`.
- **Bar, set before scoring:** at most 3 fewer round trips than the rerun's 68, no collisions.
- This is a measurement. A switch to one model would be its own piece of work, merged on its own.

**Four walkers** (`configs/experiments/walker_four.yaml`, cohorts `wf_*`): 40 missions paired, the hand tracker against the model's
positions. For information only: no bar and nothing tuned on it.

**If a step fails.** The shadow failing does not stop training, which targets that weakness: the new model repeats the shadow
run instead of driving. Training failing leaves the live test to the existing model, if it passed. The live test failing
leaves the hand tracker in charge, with the reason recorded. No held-out mazes are used in this round.

**One piece at a time.** Each handover is one branch, merged to main before the next starts, so each change has its own
before-and-after in missions. Memory for the model is deferred until this handover is merged.

### P7. One model in the working setup (approved 1 November)

**Why.** The one-model test (`wo_*`) showed go2_v2_walkers can forecast the robot and find the walkers at parity, but missions still
named two models and ran two copies. This piece adopts it and clears two crashes that have ended missions in every round since P2.

**What changes.**
- **One model, one pass.** The forecaster's model also finds the walkers, and both read one encoding of the frames
  (`models.lewm.PacketEncoder`). The hand tracker no longer runs when the model's tracks drive: they replaced its output.
- **The working setup is one file,** `configs/working.yaml`. A cohort takes it with `base:`. Code defaults stay off, so old records reproduce.
- **Crash: a view recovery across a mission phase change.** The port raised where the original drops the recovery
  (`veto_view_round_trip_development.py`). It now drops it.
- **Crash: no floor plane at frame 0** (maze 26 episode 0, one walker). Not a marginal floor as recorded earlier: about 5,770 points
  (53% of the pool) were found, and `dominant_height_cluster` dropped them whole because pruning had not converged in 8 rounds. The
  first frame now retries with `floor.START_REFINEMENTS`, only where the mission would otherwise have ended, so no other
  mission changes. A later frame keeps the 8 rounds: the pose is carried over such a frame.

**Checks, set before anything runs** (`configs/experiments/single_model_check.yaml`, cohorts `smc_*`):
1. **Reproduction.** Six `wo_*` missions, from 104 s to a 480-s timeout, must equal their records exactly: `planning.json`,
   `requests.json`, `walker_shadow.json` and the physics trace. Loading once and encoding once cannot change a number, so this
   replaces the 80-mission rerun proposed earlier. If they differ, the 80 missions run, against 68 with at most 3 fewer and no collisions.
2. **Crash fixes.** The two crashed `wo_*` missions rerun and must pass their crash points. Their results update the record for
   these 80 missions; no other mission can change.

**Held-out evaluation** (`configs/experiments/single_model_heldout.yaml`, cohorts `smh_*`): once, the same 140 missions as `cufh_*`,
the first unexamined evidence for the walker handover. No hand rule changes here, so spent mazes 30-49 are still fair for a model version.
- **Bar, set before scoring:** at most 5 fewer round trips than `cufh_*`'s 125 (the 3-of-80 allowance scaled to 140), no collisions.
  The sign test is reported beside it, against `cufh_*` and against lv2's 112.

**One reference from here on.** The bar's reference moved each round (67, then 68), and an 80-mission bar passes a change truly 3
missions worse 61% of the time, so small losses can add up unseen. From this piece, each handover is measured against the same
record, this piece's 80 missions, with a running total, and the 3-mission allowance applies to that total.

### P8. What the working planner owes to each part (approved 1 November)

**Why.** An audit of the records and code found the evidence for what comes next is thin:
- In `held` mode the model's positions reach two places only, the close-up stand-off and mover-trail marking. The clearance check never
  sees them, and walkers on the map and the depth stop still come from depth. No run shows what these are worth.
- The module oracles that set the handover order were measured on lv2, before close-up and mover trails.
- Every mission so far had polite walkers, which stop for the robot. "No collisions" and "prediction changes nothing" are shown only for those.

This piece adds no capability and changes nothing in the working setup, so it can run before P7 merges. Its results choose the next handover.

**Runs** (`configs/experiments/working_measures.yaml`, cohorts `wm_*`, 80 missions a group unless noted, on the validation mazes):
- **No tracks** (`wm_none_*`): the working setup with the planner given no tracks and no trails, one and two walkers, paired with the
  fixed record (`wo_*` with P7's crash fixes). The same missions are used so that it pairs with that record directly.
- **Module oracles on the current planner**, two and four walkers (40 each), on C1 with the hand tracker so that they run at 14 missions at
  once; C1 and C5 are level on this planner and these modules do not depend on the forecaster. C1 references: `cuf_obst2_c1` and
  `wm_ref_obst4_c1` (40 new missions, the fix included, which `cu_obst4_c1` lacks).
  - `wm_pose_*`: the true pose.
  - `wm_map_*`: the true map (`map_oracle: full`).
  - `wm_true_*`: every walker's true position in place of the tracks, no speeds (`walker_mode: truth`): the ceiling for what P6 handed over.
- **Walkers that do not yield** (`wm_rude_*`, `obstacle_polite: off`), one and two walkers: the hand tracker driving with its speeds and
  straight-line forecast (`walker_mode: shadow`) against the model's positions (`held`), paired.

**Reading, set before anything is scored.**
1. **No tracks.** At most 3 fewer than the fixed record means the walker handover is not shown to change missions at this resolution, and the
   record says so. No bar; nothing is switched off on this result alone.
2. **Next handover.** The oracles are read pooled over the 80 missions, against the C1 references. The pose goes next if the pose oracle has
   the largest net gain, at least 4 missions, with wins at least twice the losses. Otherwise memory goes next, as planned. A map gain
   points to memory too, since the map needs it.
3. **Speed.** It stays parked unless, without yielding, the hand tracker beats the model by more than 3 of 80 or has at least 3 fewer contact stops.

**Compute.** About 7.5 hours on the GPU (the three 80-mission C5 groups) and about 3 hours of C1 missions, one after the other behind P7's
held-out run. C1 is not run beside C5: it is untested, and the machine runs unattended overnight.

### P9. Memory: the walkers the robot no longer sees (approved 6 November)

**Why.** Nearly every walker contact stop on record has been the robot's blind rear: reversing or turning in place into a walker out
of the camera's view. The map does not hold such a walker (`mover_trails`: a track's cells revert to floor out of view), the
hand tracker forgets it after 1.2 s, and the deployed model sees 0.3 s. Of the 8 contact stops out of view with the model's
positions and walkers that do not yield (P8), 6 were with a walker last seen 7.6-11.7 s before, 2 never seen; P6's and P7's
four were seen 3.8-60 s before. A memory of about ten seconds could reach most of them; one that has never seen the walker cannot.

**What is handed over.** What is behind the robot: out of the camera's view, the planner's clearance checks read where the
model remembers walkers, in place of the rule that out of view there is nothing. The geometry (where a remembered place
moves when the robot moves, from the visual odometry's pose) is given; what is there, how long it stays and where a walker
goes are learned (`models.memory`).

**Steps.**
1. **Labels and baselines** (`data.memory_labels`, `eval.memory_baselines`): every corpus v2 unit with walkers, a
   body-frame grid (0.2-m cells, ±1.6 m) of where walkers are, in view or not; scored on the cells out of view, by
   average precision, against what the robot could know without learning (forget after 1.2 s, hold where last seen,
   walk on at the last velocity, a per-cell prior), from true sightings and the true pose, so upper bounds.
2. **Probe H-JEPA v2** (`eval.memory_probe`): its upper levels' snapshots (level 2: 1.6 s apart over 3.2 s, level 3:
   12.8 s), read into the same grid. Stop here if they already remember.
3. **Train the memory.** A body-frame grid (±3.2 m) of learned features, moved by the pose change each step and updated
   at 5 Hz by a convolutional GRU from the world model's state (`training.memory`). Stage A on the deployed model's fixed
   states, to show it learns; stage B jointly with the encoder and the world model's own objectives
   (no frozen level 1 in what is deployed).
4. **Missions.** The planner's clearance checks (forecast paths, turns, the stopping projection and the dispatch depth
   stop: the 1-cm cells) also see each remembered walker out of the camera's field of view, as a disc of the walker's
   radius at the probability-weighted centre of each connected group of cells above `τ`. Routing does not see them. `τ`
   is fixed before any mission: the lowest threshold whose out-of-view precision on 200 training units is at least 0.5.

**Bars, set before anything is scored.**
- **The memory (step 3):** average precision out of view above the best baseline's on the validation units, and in view no
  worse than the deployed state read one frame at a time (the probe's `now`) on the same frames.
- **Missions (step 4),** walkers that do not yield (80, `wm_rude_model_*`'s missions): contact stops with the walker in contact out
  of the forward view at the stop at most 3 (8 in P8 by this count), round trips at least 11. Four walkers (40, `wf_held_c5`'s): contact stops at most 2, round trips at least 21.
  The fixed record (80): within the running allowance of 3 below 70, no contacts. Walkers that do not yield are in every
  acceptance test from here on (decided 6 November).

### P9b. The planner reads the memory as a veto (approved 6 November, after P9's missions failed)

**Why.** P9's memory passed its bar, but its walkers in the 1-cm cells turned a walker held beside the robot into a robot
inside its own footprint, and the turn exits then measured against the walker instead of the walls: 20 contact stops with
walls, none with the memory off (STATUS, "P9: memory, steps 3 and 4").

**The change** (`--walker-memory veto`; P9's use stays as `on`): remembered walkers no longer enter the map's cells. A last
selection stage (`selection.remembered_veto`) rules out any action whose forecast brings a point of the body nearer to a
remembered walker than now, once within 0.45 m, and takes the best clear action left, else hold. The memory, its threshold
and the jointly trained world model are P9's, unchanged.

**Test and stopping** (PLAN "Phases and gates"). First the 80 missions with walkers that do not yield (`memv_rude_*`),
against P9's bars there: out-of-view walker contacts at most 3, round trips at least 11. Stop and report if wall contact
stops reach 5 (P9: 20, memory off: 0), or as soon as either bar is decided. Only if both pass, the four-walker (40) and
fixed-record (80) cohorts against P9's bars.

**Result and what is deferred (6 November).** The veto removed the wall contacts but failed the out-of-view walker bar after
8 missions (STATUS, "P9b"). Piece 3 closes as a measured negative result: the memory exists, passes its offline bar, and is
not yet good enough at the body to keep walkers off. **To revisit after the pose piece** (decided 6 November): the memory
near the body (finer cells within about 0.8 m of the robot, more loss weight there, and the walker's motion rather than a
still position, so the veto can be read against where the walker will be), then P9b's missions again with the same bars and
stopping rule. Better poses from the next piece also sharpen the memory's warp.

### P10. The world model carries the pose through frames visual odometry cannot measure (approved 6 November)

**Why.** With walkers that do not yield, pose loss ends more missions than anything else: 35 of 80 with the model's
positions, 31 with the hand tracker (P8), and about half of P9's and P9b's. In all 66 of P8's, visual odometry raised "neither
camera measures the current frame" with a walker parked about 0.55 m ahead, and **the robot was standing still**: zero commands
and no movement over the last 2 s in every one. The missions ended at a median 166 s of 480, so time was left. Today a frame that
no pass measures ends the mission (`perception.visual_odometry`, `PoseLost`), however briefly the view is blocked and however
little the robot has moved. With polite walkers the true pose bought only 2 of 80 (P8), so whether the pose is worth handing over
turns on walkers that do not yield.

**What is handed over.** Where the robot is while the cameras cannot say: during an outage the pose is carried by the world
model's prediction of its own motion under the commands actually applied, and visual odometry restarts from the carried pose
once frames measure again. Rotation stays with the gyro, as in visual odometry today; what is learned is the translation.

**Steps.**
1. **The upper limit, and how long outages last** (GPU, about 2.5 h). The 80 missions of `wm_rude_model_*` (C5, model positions,
   walkers that do not yield) with the true pose (`wm_pose_rude_*`), visual odometry running beside it in shadow. When the shadow
   loses the pose, it logs the outage and restarts at the true pose on the next frame it can measure, so each mission records
   every outage: its length, how far the robot moved and turned during it, and where the walker was.
2. **The carrier, offline** (GPU, about 1-2 h). On corpus v2's validation units, windows shaped like step 1's outages (length
   and motion; a walker within 0.8 m in the forward view), each frame's position carried by one 100-ms forecast of the applied
   commands from the frames before it, yaw from the gyro. Three carriers on the same windows: the world model (C5's rollout
   and decoder), C1 (the ridge command model, as the bridge) and the commands integrated as if executed exactly (the floor).
3. **The code** (`perception.carried` or similar, unit tests). On `PoseLost` the controller enters a carried outage instead
   of ending: each frame's pose is the carrier's; mapping and planning go on from it. Every frame a fresh visual odometry is
   started there; once it measures 4 frames in a row, its poses, composed with the carried pose at its start, replace the
   carrier, and floor registration reads them as before. The outage ends the mission as now if the robot moves farther during
   it than the distance at which step 2's carrier reaches 5 cm of error at the 90th percentile. Off by default
   (`--pose-carry off|model|c1`).
4. **Missions** (GPU, about 6.5 h): walkers that do not yield (80, the same missions), then the fixed record (80) and four
   walkers (40, `wf_held_c5`'s).

**Bars, set before anything is scored.**
- **Gate after step 1.** The pose goes on only if the true pose gains, against `wm_rude_model_*` (11 round trips), at least 4
  net with wins at least twice the losses (P8's rule for the pose). If not, P10 closes there: pose losses would only turn into
  timeouts or contacts, and the record says so.
- **The carrier (step 2).** The world model carries the pose if its error at the end of the windows is at most 5 cm at the 90th
  percentile, and its median no worse than C1's. If only C1 passes, C1 carries it, and the record says the pose is bridged,
  not handed over.
- **Missions (step 4).** Walkers that do not yield: missions ended by pose loss at most 9 (35 now), round trips at least 11 plus
  half the true pose's net gain from step 1 (rounded down), contact stops at most 13 (10 now). Over every carried outage, the
  position error against the truth when visual odometry takes over again at most 10 cm at the 90th percentile. The fixed
  record (80): within the running allowance of 3 below 70, no contacts. Four walkers (40): contact stops at most 2, round
  trips at least 18 (`wf_held_c5` 21, less 3).
- **Stopping** ("Phases and gates"): every 20 missions. Step 1 stops early only if the gate can no longer be met. Step 4 stops
  when a bar is decided; a new failure in at least 5 of the first 20 (a position jump at takeover, a crash in the carrier) stops it
  as broken.

**Compute.** About 11 GPU-hours, at most 6 world-model missions at once, mazes 10-29 only (50-89 stay held). Step 1 needs only the
shadow log, so it can run while steps 2 and 3 are written.

**Result (7 November).** The gate failed: stopped by the stopping rule after 74 missions, the true pose 3 won, 5 lost against
`wm_rude_model_*` (STATUS, "P10"). Pose losses become timeouts and contacts with walkers out of view. Offline, the deployed world
model fails the carrier bar (90th percentile 28.5 cm) and C1 passes (1.7 cm, limit 0.05 m of carried path). P10 closes as a
negative result; steps 3 and 4 do not run, and `--pose-carry` stays off.

### P11. Getting out of the way of a walker that waits (approved 7 November)

**Why.** A walker that does not yield walks up to the robot and waits for it to leave its path (`sim.obstacles`: it stops while
the robot is within 0.625 m ahead of it, and never turns back). The planner treats the walker as an obstacle and waits too. In
P10's 74 missions with the true pose (a walker "waits on the robot" when it has stood still at least 1 s within 0.70 m with the
robot ahead of its last heading):
- all 40 timeouts end in such a stand-off; it covers a median 81% of the mission, its longest stretch a median 222 s, and the
  robot stands still 94% of it and reverses 2%;
- 22 of the 24 contacts with walkers are with a still walker that had been waiting on the robot, 12 of them out of view in the
  last second, so out-of-view contacts are a symptom of the stand-off;
- no mission with a stand-off of 60 s or more made a round trip: 0 of 55 (P8's runs: 0 of 57).
During stand-offs the walker is within 60° of the heading 86% of the time and in the forward view 71%, so the model's tracks
see most of them. P4f's back-off rule (`planner.give_way`, off in the working setup) backs off 0.40 m at most 3 times per track,
and a walker that does not yield just follows it.

**What changes, and why it is a hand rule first.** This piece adds a response, it does not hand one over: the corpus barely
contains the decision (with a mover waiting near and ahead the robot holds or turns in 93% of the next 1.6 s, P4f), and P4f's
model ranked retreats inverted even after a give-way top-up. So the rule comes first, its encounters become the data, and P12
hands the decision to the world model.

**The rule** (`planner.vacate`, `--vacate on`, off by default).
- **Trigger:** a confirmed track (the model's walkers, `held`) within 0.9 m and in the front half that has stayed within 0.15 m
  of one spot for at least 4 s. Polite walkers turn back after 3 s, so they should rarely trigger it.
- **Retreat:** back along the robot's own travelled path (breadcrumbs every 0.10 m, known floor), facing the walker so that it
  stays in view. Each plan takes the reverse lattice candidate (-0.10 m/s, yaw -0.45 to 0.45 rad/s) whose world-model forecast
  ends nearest a point 0.5 m back along the path, among those whose forecast path stays on observed floor with the footprint's
  0.45 m of clearance from everything but that walker, and ends no nearer it.
- **End:** no confirmed track within 1.2 m for 2 s (the walker has turned off; the planner resumes), or the path runs out, or
  5 plans in a row with no admissible step, or 120 s. Then 10 plans before it can trigger again.
- **Records:** every encounter (its trigger, steps, end and the walker track) is logged in `planning.json`, the seed of P12's data.

**Steps.**
1. **Code and unit tests** (CPU): the rule, `--vacate`, per-mission counts in `cohort`.
2. **Walkers that do not yield** (`vacate_rude_obst{1,2}_c5`, 80, GPU about 3.5 h): the working setup with the rule, paired with
   P8's `wm_rude_model_*` (11 round trips, 10 contact stops).
3. **The fixed record** (`vacate_obst{1,2}_c5`, 80, about 3.5 h): polite walkers, paired with the fixed record (`wo_*` with
   P7's two crash fixes, 70 of 80).
4. **Only if step 2 fails:** the same 80 missions with the true walker positions (`walker_mode: truth`), to tell the rule from
   the tracks.

**Bars, set before anything is scored.**
- **Step 2:** round trips at least 21 (net gain at least 10) with wins at least twice the losses, contact stops at most 10.
- **Step 3:** within the running allowance of 3 below 70, no contacts.
- **Stopping** ("Phases and gates"): every 20 missions; stop when a bar is decided, or as broken if a new failure (a crash, a
  wall contact during a retreat) appears in at least 5 of the first 20.

**Compute.** About 7 GPU-hours, at most 6 world-model missions at once, mazes 10-29 only (50-89 stay held). Step 3 runs only
once step 2 passes.

**Result (7 November).** Stopped during step 1: in six smoke rounds (21 missions where the rule could act) the robot never
got out of the walker's way, and no round trip came of it (13 timeouts, 4 pose losses, 4 contacts; STATUS, "P11"). The
waiting walker stands inside the cameras' blind zone (its face about 0.1 m from the lenses, under the 0.2-m depth minimum),
so the map, the model's walker head and visual odometry all lose it once the robot moves. All 35 "walker gone" endings
checked against the truth were wrong. The robot stops wedged just past a corner, and the true pose does not rescue it.
Steps 2-4 do not run; `--vacate` stays off. The rule as finally coded is in STATUS and `planning/planner.py`.

### P12. The world model decides when to give way (planned; after P11 merges, if P11 passes)

*Superseded (7 November)* by the project review's plan (`docs/REVIEW_2026-11-07.md`, section 6): data that contain the
decision, the world model as the decision-maker, and range sensing as an observation. The sketch below is kept for the
record.

The rule's when and how far become the model's. Sketch, to be set out in full for approval when P11 closes:
1. **Data.** A corpus v2 top-up of kind `vacate`: free units with non-polite walkers, driven like `GiveWayDriver` but with P11's
   retreat in its repertoire and its choices varied on purpose (hold for 2-30 s, retreat for 0.3-3 m, step forward), so that
   the same stand-offs end in different ways: a walker that keeps waiting, one that follows, one that turns off. P11's mission
   encounters say which situations to weight. Gate on the data: at least 25% of the stand-off windows followed by the walker
   moving off the robot's path within 10 s.
2. **Model.** The world model learns what each response leads to: whether the robot's way ahead is clear after H seconds under
   hold or under retreat, either from rollouts read by the mover head or from a response head on the latent, trained with the
   encoder as before. Offline bar: it ranks hold against retreat better than the rule's trigger on held-out units.
3. **Missions.** The model's choice replaces the rule's trigger and end (when to back off, when to go again), on P11's missions;
   bar: no worse than the rule, no more contacts.

The memory near the body (P9b's deferred follow-up) waits until after P12, and returns only if contacts with walkers out of
view remain.

### P13. Review phase A: observations that can show the problem (approved 7 November)

The project review (`docs/REVIEW_2026-11-07.md`, approved 7 November with its plan) puts the world model in charge:
phases A-E. This is phase A. It builds what the later phases need to observe, collect and judge, and changes no
behaviour of the working setup by default. Branch `feature/phase-a-observations`.

**1. A range sensor** (`go2nav/sim/range_sensor.py`, `--range-sensor on`).
- **Model:** after the Go2's L1 LiDAR, 360° in azimuth, idealised to one horizontal ring:
  - 720 beams (0.5° apart) from a mount 0.30 m forward and 0.12 m above the base origin;
  - ranges 0.05-20 m, Gaussian noise σ 1 cm (seeded), one scan with each 10-Hz frame;
  - cast analytically against the wall boxes and the walkers' cylinders, never the robot's own body.

  The walls and walkers are vertical prisms, so one ring sees what a 90° fan would. The real L1's floor returns and
  self-occlusion are left out, and the record says so.
- **Recorded** in each frame's packet as `range` (ranges and validity), so phase B's corpus has it.
- **Tests:** casts against a box and a cylinder, the minimum range, noise that replays exactly.

**2. The hand reference with the sensor.** With `--range-sensor on`, the planner gets the sensor's returns:
- returns within 1 m join the dispatch check's obstacle cells;
- every return joins the map's occupied 1-cm cells at the frame's pose.

Floor and free space still come from the cameras, so routing is unchanged. This is the reference phase D is measured
against, with the same sensor.
- **Measure, no pass mark:** P10's 80 missions with walkers that do not yield (`rs_rude_obst{1,2}_c5`), the working setup
  plus the sensor, paired with `wm_rude_model_*` [P8's run, 11 round trips, 10 contact stops]. Read: round trips;
  contact stops by kind (walker in view, walker out of view, wall); timeouts; pose losses; stand-off seconds. About 3
  GPU-hours.
- The other two reference suites phase D needs (the fixed record's 80 missions, four walkers' 40) run before phase D,
  about 4.5 GPU-hours.

**3. Reactive walkers** (`sim/obstacles.py`, `behaviour: reactive`, alongside polite and non-polite):
- They walk the passage graph as now, but steer round the robot and each other.
- They pass where the gap allows their radius plus 5 cm, and otherwise wait.
- Each walker draws a patience (2-20 s), after which it turns back, and a yielding share decides whether it stops for
  the robot as a polite walker does.
- They never move into the robot (5-cm gap, as now). Their draws are recorded in the placement, so a mission replays
  exactly.
- **Tests:** passing in a wide corridor, waiting then turning back in a narrow one, exact replay.

**4. Layout families** (`go2nav/mazes/families.py`). Every family stays on a cell grid with axis-aligned walls, so the
walkers' passage graph, the scene builder and the casting work unchanged. The pitch becomes a per-layout value, read from
the specification's `evaluation_layout.pitch_m` (every registered set already records it) by the walkers, the scene and
the episode builder.
- **Family A:** today's 4×4 tree mazes, 1.3 m pitch.
- **Family B (development):** 4×4 grids, pitch drawn from 1.1, 1.3 and 1.6 m. A spanning tree plus 2-3 extra passages
  (loops), and one or two 2×2 rooms with their inner walls removed. (5×5 was planned, but the layout identity's exact
  canonical form takes at most 24 cells; the LeWM corpus is already on 5×4 trees.) Set `loops_rooms_v1`: dev_tune 0-9,
  validation 10-29, two episodes each.
- **Family C (held out for good):** offices, a different construction, on a 6×4 grid at a pitch of 1.2 or 1.4 m. A
  corridor along one row; below it rooms of 2×1 cells; above it a 2×2 room or an outer 2×1 room with an inner one behind;
  one door each, sometimes two. Set `offices_v1`: 20 layouts, two episodes each. It is never used for collection,
  tuning, training or smoke tests, and is run only for a claim the user approves.
- **Tests:** connectivity, identities and exclusions as today's sets, walls matching the passages, the pitch read
  through.
- The hand planner gets a few smoke missions on family B. Failures there are information, not something to tune.
  Family C gets none, to stay unseen: its code paths are family B's and the generator's, covered by unit tests and by
  its exact regeneration.

**5. Metrics** (`eval/cohort.py`). Per mission:
- time to the beacon and home;
- path length;
- stand-off seconds (from the true walker states, as `outputs/vacate/standoff.py`);
- contacts per kilometre.

Per comparison, the exact sign test as now, under the stop rule. Each suite's fixed anchor (the run every later one
pairs with) is named in the suite's configuration.

**Gate to phase B.** All five items built and unit-tested, the full suite green, the reference measure reported in
STATUS, and family B smoke-tested with the hand reference. Then this branch merges, and phase B (data that
contain the decision) is set out on its own branch.

**Compute.** About 2-3 days of building, mostly CPU, longer than the review's 1-2 days because of the layout families.
GPU: about 3 hours for the reference measure, then about 4.5 hours for the other reference suites before phase D. At
most 6 world-model missions at once, EGL_DEVICE_ID=1, mazes 50-89 of family A still held.

### P14. Review phase B: data that contain the decision (proposed 8 November; preparation approved 8 November)

**Why.** In corpus v2, when a walker waited near the robot, the robot held or turned in place in 93% of the next 1.6 s
(P4f). The stand-offs that end every failed mission with walkers that do not yield (P11) were almost never resolved in
the data. A model that learns only from observation cannot learn from that what a retreat or a step aside leads to. This
phase records those outcomes, and adds the range sensor to everything the model will see. Nothing in it changes how the
robot is controlled. Branch `feature/phase-b-data`.

**1. Range scans for corpus v2, cast offline** (`go2nav.data.range_backfill`; approved and running 8 November as
preparation). Every frame of corpus v2's 5,200 units gets the scan the range sensor would have given, from the recorded
pose, the walkers' recorded positions and the maze's walls. The output is written to `go2nav_lewm_v2_range`; the corpus
is never touched. Check against the units' own truth labels (`free_rays.npy`): on 24 units the forward and backward beams
differ from the labels by a median of 0.7 cm, 2.1 cm at the 95th percentile.

**2. A family B training set** (`configs/mazes/loops_rooms_corpus_v1.yaml`; preparation). 1,300 layouts with loops and
rooms on 5×4 grids, the corpus mazes' size: lewm_train 0-1199, lewm_val 1200-1299. It excludes every family B evaluation
layout and the held-out family C.

**3. The stand-off driver** (`StandoffDriver`, kind `standoff`). It extends `GiveWayDriver`; its encounters are
stand-offs:
- **Approach.** It faces the nearest walker in sight, then either walks to 0.6-1.1 m from it or stops in its path and
  lets it come.
- **Wait.** It holds for 2-30 s, drawn log-uniform, so both short and long stand-offs occur.
- **Resolve,** chosen at random:
  - retreat along its own path, 0.3-3 m (35%);
  - step aside into the side with most room (20%);
  - turn away and walk off, 1-3 m (20%);
  - hold on, 5-20 s (15%);
  - advance (10%).
- **Safety.** Every command passes the free driver's stand-off rule on the true walls and walkers, so the driver never
  touches anything.
- **Role.** The driver only makes sure the outcomes are observed. No model is trained to imitate it.

**Revised in four pilots** (7 November, collection approved; 48 units each, mazes 0-23 of both families; each set
aside on the data drive, `_set_aside_standoff_*`). The first 69 units of the collection as designed met the gate on
nothing: against walkers that do not yield, 43 stand-offs, none resolved, none with the robot 1 m away. Such a walker
follows a robot that only backs off (both moved 0.4 m and stayed 0.62 m apart), so in a corridor the move that lets it
pass is to get out of its lane at a junction. The driver now:
- **gives way** (35%): back along its own path at 0.25 m/s to the first side opening (0.3-4 m), line up with the
  opening's middle, turn to it at 0.8 rad/s and go 1.0-1.5 m into it at 0.30 m/s;
- waits 1-6 s before resolving (corpus v2 already holds 6.7 h of stand-offs in which nothing happens; *hold on* keeps
  long ones), and retreats at 0.10 or 0.20 m/s; resolutions: give way .35, retreat .15, aside .15, walk off .15,
  hold .10, advance .10;
- **unsticks:** a driver commanding motion that has not got 0.20 m from where it was 6 s ago, with a walker within
  0.8 m, gives way (alternately walks off), and an escape step is kept 1.5 s. 86% of pilot 2's stand-off frames were the
  driver stuck under its own stand-off rule, turning in place for up to 60 s or stepping back and forth;
- **ends a pinned unit:** stuck three times on one spot (between two walkers that do not yield, or a walker and a
  wall: walkers that never give up make that permanent), the unit ends ('PINNED', frames kept).
Free starts also use the layout's own pitch now (they used 1.3 m, so on family B the robot could start against a wall),
and the stand-off units run the no-slip pass (P13 item 6).

| Pilot | Resolved within 10 s, stand-offs | Same, frames | Robot moved 1 m away | Contact stops |
|---|---|---|---|---|
| First 69 units (family A) | - | 7.9% | 4.3% | 1 |
| 1: give way | - | 8.6% | 9.2% | 1 |
| 2: faster | - | 3.7% | 15% | 0 |
| 3: unstick | 21% | 12% | 26% | 1 |
| 4: no progress, pinned | 20% | 5.7% | 31% | 2 (4%) |

**End to end, one unit** (8 November, scratch root; family B maze 5, two reactive walkers): 900 frames, no contact,
40% of the frames in stand-off segments. Its live range scans equal the offline cast at the same recorded poses
exactly (largest difference 0.0 m), so item 1's backfill is the live sensor. Its walkers passed or waited beyond 0.70
m, so the audit counted no stand-offs in it.

**4. A stand-off top-up** (`collect --kinds standoff`).
- **Units:** 1,200 training and 100 validation units of 900 frames each.
- **Layouts:** half on corpus v2's family A mazes (lewm_sets_v1, 5×4 trees), half on family B (loops_rooms_corpus_v1).
- **Walkers:** one to three per unit, 40% of units reactive, 40% non-yielding, 20% polite.
- **Recorded:** everything corpus v2 records, plus `range.npz` from the live sensor, plus each walker's internal state
  (waiting, behaviour, yielding, patience) as labels for probes.
- **Compute:** about 8 hours of collection on the CPU, rendering on the R9700. It runs after the phase A reference
  suites, or beside them with fewer workers if the first minutes stay healthy.

**Gate on the data, set before collection, read before any training.** A stand-off window is a walker standing still
for at least 1 s within 0.70 m with the robot ahead of its last heading.
- At least 20% of stand-offs resolve within 10 s: the walker is out of the stand-off and more than 1.0 m from the
  robot. **Amended 7 November after the pilots** (approved; it first read "at least 25% of stand-off frames"). Counted by
  frames, a few long stand-offs decide the share: 12% and 6% in pilots 3 and 4, whose drivers differ only in how they
  get unstuck, against 21% and 20% counted per stand-off. What a world model can learn from is the number of distinct
  resolutions it sees, not how long the unresolved waits last. `standoff_audit.gate` checks the amended conditions.
- At least 20% of stand-offs see the robot move at least 1 m away from where it stood, further from the walker.

Corpus v2 against the gate (`go2nav.data.standoff_audit`, all 5,200 units; `outputs/phase_a/standoff_audit_v2.json`):

| Corpus v2 | Stand-offs | Stand-off frames | Resolved within 10 s | Robot moved 1 m away |
|---|---|---|---|---|
| All | 3,740 | 241,193 (6.7 h) | 2.6% | 4.1% |
| Walkers that do not yield | 3,374 | 98% of them | 0.9% | 2.2% |
| Polite walkers | 366 | 2% | 88% (they turn back) | 22% |

The corpus already holds 6.7 hours of stand-offs with walkers that do not yield, and under 1% of them resolve: the model
has watched stand-offs in which nothing happens. The give-way top-up's retreats (0.1-0.3 m) did not change that.
- Every walker behaviour is drawn for at least 15% of the units, and both layout families supply at least 15% of the
  stand-off frames. Non-yielding and reactive walkers each supply at least 15% of the stand-off frames. Polite walkers
  stop 0.8 m away and turn back after 3 s, and yielding reactive ones wait 0.40 m short of the robot's outline (a
  centre about 0.95 m away), so under the 0.70-m definition they make few stand-off frames by design.
- Contact stops in at most 5% of units, as corpus v2's gate.

If the gate fails, the driver's weights change and a small top-up runs again, measured the same way. No training starts
on data that fails it. The collection runs with pilot 4's driver (2de1826); pilot 4's 48 units are its first.

**Collected 7-8 November.**

The first full collection, at ed0cbb5 with 1,300 units, failed the gate:

- 18.7% of its 1,623 stand-offs were resolved, against the 20% the gate requires.
- Non-yielding walkers' stand-offs resolved 15.6% of the time, reactive walkers' 24%.
- 127 non-yielding units ended pinned. They held 542 of the 1,192 non-yielding stand-offs, and 5% of those resolved.
- 102 of the 171 pinned units ended in a dead end with a non-yielding walker at its mouth. That walker never turns back, so no move can resolve the stand-off.

The fifth revision was approved on 8 November (89ef678). When the driver is stuck and every way out with 1 m of room runs into a walker within 1.3 m, it is now pinned at once rather than after three tries. The 527 non-yielding units were set aside in `_set_aside_standoff_ny_ed0cbb5` and recollected; pilot 5 (mazes 0-59) was their first 51.

On the same 527 units, the revision resolved the same number of stand-offs (186) out of fewer recorded (945 against 1,192). It records fewer hopeless repeats; it adds no resolutions. Contact stops fell from 17 to 3.

The reactive and polite units stay at ed0cbb5. The revision changes nothing there until a driver is stuck with a walker at the robot. A reactive walker turns back after its patience, so a later collection with reactive walkers should limit the rule to walkers that do not yield.

**The gate on the corpus: passed.**

| | Result | Gate |
|---|---|---|
| Stand-offs resolved within 10 s | 22.1% of 1,376 | ≥20% |
| Robot moved ≥1 m away | 27.9% | ≥20% |
| Stand-off frames, non-yielding / reactive | 74% / 25% | ≥15% each |
| Stand-off frames, families A / B | 46% / 54% | ≥15% each |
| Contact stops, `standoff` / `standoff_b` | 25/650 (3.8%) / 14/650 (2.2%) | ≤5% |

198 units ended pinned.

Indexed 8 November: 6,500 units and 4,963,265 windows of 12 frames; the 31 October index is kept as `index_12_pre_p14.*`.

In the corpus audit, two of corpus v2's accepted shortfalls now pass:

- 38,194 rest-start windows;
- an obstacle in view in 23.7% of frames.

The lateral share is 9.5% of blocks, against its gate of 10%; it was accepted at 8.9%.

**Compute.** Preparation (items 1-2): CPU, under an hour. The driver and the collector: about a day of building. The
collection: about 8 CPU-hours. Mazes 50-89 of family A and all of family C stay unused.

### P15. Review phase C: a model that predicts the interaction (proposed 8 November; approved 8 November: proprioception from the IMU and the commands, heads with stop-gradient, family C untouched)

**Why.** P14 gives data in which a stand-off's outcome follows the robot's choice. Phase C asks whether a
self-supervised predictor learns that outcome. Corpus v2's models scored AUC 0.19-0.40 on the give-way contrast,
inverted, against a privileged ceiling of 0.87. This phase decides the thesis (docs/REVIEW_2026-11-07.md). Branch
`feature/phase-b-data` until P14 merges.

**0. Control** (approved and running, 8 November). `go2_3level_v2`, unchanged, on corpus v2 plus the stand-off units,
with their clips at the upper levels (configs/hjepa/go2_3level_p14_control.yaml). It is scored on the gates below. It
separates the effect of the data from that of the new inputs.

**1. Model: `go2_3level_c`.** H-JEPA as `go2_3level_v2`, levels at 100 ms, 1.6 s and 12.8 s, trained jointly. Upper
gradients flow into level 1, and nothing is frozen.
- **Range input.** Each frame's 720-beam ring (`range.npz`: P14's own units, and corpus v2's backfill in
  `go2nav_lewm_v2_range`) goes into a small 1-D convolution with circular padding, pooled into 8 sectors of 45°. The
  sectors become 8 tokens in the ViT's sequence, with their own position embeddings, so [CLS] attends to the camera
  and the ring together. Invalid beams read as the maximum range, with a mask channel.
- **Proprioception.** The corpus has no joint states. One more token carries what an IMU and the policy give the
  real robot:
  - body angular velocity (`twist`);
  - projected gravity (from the pose's orientation);
  - the applied command history (`applied_history`).

  The base's linear velocity is left out, since hardware has no direct sensor for it. Adding joint states would mean
  recollecting the corpus (about 20 CPU-hours).
- **Rollouts at every level.** The upper levels add a 2-6-step autoregressive rollout loss to the teacher-forced one,
  as level 1 already has with 8 steps. The lengths are drawn per batch, and the weight equals the teacher-forced
  loss's.
- **Heads, with stop-gradient.** They read the latent and never shape it, so the representation stays
  self-supervised:
  - contact within 0.8 s;
  - the robot's own motion over one step and over 1.6 s;
  - walker presence (a walker within 2 m, in view).

  Their labels are privileged and used in training only, from the recorded pose, walls and walkers:
  - contact means the body outline's true clearance falls under 0.10 m;
  - the number of positives is counted before training, and reported.

**2. Gates.** Fixed before any model is scored, offline, on the validation units (both families) and the transfer
units. Family C stays untouched until phase D's held-out missions:
1. **Give-way.** P4e's probe (`mover_probe.give_way`), with labels for the new stand-off validation units. The
   retreat-minus-hold contrast must rank the walkers that respond with AUC ≥ 0.80, at 1.6 s and at 3.2 s.
   - At the 12.8-s level: whether the robot's way is clear after a 2-3 m retreat (from `branch`, with a retreat tape
     of that length), AUC ≥ 0.75.
2. **Contact.** The contact head, AUC ≥ 0.9 at 0.8 s.
3. **Own motion at standstill.** The one-step motion head's bias on held frames (zero command) stays under 1 mm a
   frame. P10 found 3-4 mm.
4. **Walkers.** Localisation no worse than LeWM v2's 0.216 m.

L1 health and action sensitivity must not fall below `go2_3level_v2`'s. The control is scored the same way.

If gate 1 fails on these data, the thesis has a clean negative result: the self-supervised predictor does not learn
the reaction even when it is observed. The next choice is then supervision on predicted states (P4d's labels on
rollouts) or route 1. The record will say which, and why.

**3. Order and compute.**
1. Build and test: about a day, beside the control.
2. A 30-minute smoke run measures throughput and memory under the 85% cap. The 9 extra tokens a frame add about 5%
   to the encoder.
3. Training: about 15-16 h for 60k steps. It starts when the control finishes, since the R9700 holds one of them.
4. Probes on the CPU.

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
