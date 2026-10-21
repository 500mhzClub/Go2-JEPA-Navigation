# H-JEPA timescales: decision record (21 October 2026)

## Decision

| Level | Step | Stride from the level below | Meant to capture |
|---|---|---|---|
| L1 | 100 ms | (one camera frame and its five 50-Hz commands) | local physics: how a command moves the body and changes clearance |
| L2 | 1.6 s | 16 | a manoeuvre: finishing a turn, approaching a wall, passing a doorway |
| L3 | 12.8 s | 8 | progress through the maze: reaching the next junction or the end of a corridor, about one cell |

How the levels train (details in [PLAN.md](PLAN.md)):
- **Context.** L3 predicts from 2 states, as level 2 does on DROID in the paper, so a training clip is 257 frames
  (25.7 s). L2 and L1 train on random crops within that clip.
- **Data.** L1 trains on every clip. L2's and L3's losses use mission and explore clips only, not the scripted free
  units.
- **Macro-actions** are the paper's pooled commands. An intended displacement (dx, dy, dyaw) is the fallback.

**The goal of this branch is a functional baseline**: H-JEPA trained end to end, its level 1 run in the current
harness as C7/C8, its upper levels checked offline. In the current harness, any forecaster only ever has to look
800 ms ahead. The value of the upper levels should show as we hand over the harness's hand-engineered parts to the
world model (route and frontier choice, recovery, memory). These timescales are chosen to be useful then, at a cost
the current corpus can carry now.

## The physical timescales

The RL locomotion policy turns velocity commands into gait, so the world model never sees the 50-Hz joint
coordination; its finest interface is the 10-Hz command.

| Event | Typical duration |
|---|---|
| one command tick | 0.1 s |
| one harness decision (committed action) | 0.4 s; the forecaster looks 0.8 s ahead |
| a 90-degree turn in place (0.45 rad/s) | about 3.5 s |
| crossing one 1.3-m maze cell (at most about 0.2 m/s, plus the turns and holds missions make) | about 8-10 s |
| a corridor or junction-to-junction leg (2-3 cells) | about 20-30 s |
| one outbound route (the farthest cell of a 4 x 4 maze) | about 1-2 minutes |

The flat world model (C5) already covers 0.1-0.8 s. A hierarchy whose upper levels step at 0.2-1.6 s, the paper's
strides of 2 applied to our 100-ms level 1, would sit inside the timescale the flat model and the harness already
handle, which makes it largely redundant. The paper's own top level is about a quarter of an AntMaze task, which
is cell-scale. L3 at 12.8 s is our equivalent.

## The case for shorter timescales

The paper's strides of 2, or DROID's 3, would give upper levels at 0.2-0.8 s.
- **Tested regime.** Every result in the paper uses strides of 2 or 3. Our 16 and 8 are untested, and each
  predictor has to leap further.
- **Easier learning.** A short step is more predictable from its pooled commands, so upper levels are less likely to
  plateau or collapse.
- **More data per epoch.** Short clips fit in every unit. The paper's Push-T failed at three levels because clips no
  longer fit its short episodes.
- **Immediate use in this harness.** A 400- or 800-ms upper level could forecast the harness's horizons in one or two
  steps instead of eight, with less compounding error. That benefit is measurable now.
- **Denser gradients into L1's encoder.** Near-timescale objectives overlap the forecaster's own horizon, so joint
  training could sharpen exactly the features C7 uses.

**What it costs:** no level reaches beyond what the flat model already sees. The hierarchy would answer "can H-JEPA
forecast 800 ms better?" rather than "can a world model take over route-scale decisions?", and the second question
is the one the later branches need.

## The case for longer timescales

A top level of 25 s or more would be region-scale: "find a new room", "take the other branch".
- **Where the planning payoff is.** The harness components we intend to hand over (route planning, frontier
  exploration, return home, escaping a pocket) decide on the scale of corridors and branches, tens of seconds to
  minutes.
- **Stronger abstraction.** The paper (Fig. 5) finds that levels discard fast detail and keep slow state in
  proportion to the gap between the data's fast and slow frequencies. A longer top level widens that gap: gait-driven
  view jitter against which part of the maze the robot is in.
- **Goal costs.** Higher levels give smoother goal-distance landscapes along corridors (paper Fig. 7, Table 1). That
  is the property a learned replacement for route planning needs.

**What it costs now:**
- **Clip length.** A 25.6-s top level with 2 context states needs 51-s clips. The training windows left are 46% of
  the mission and explore units' and 15% of the free units'; at 25.7-s clips they are 73% and 58%.
- **Coherent data.** The paper's §G found upper levels learn only from goal-directed, temporally coherent
  trajectories. Our coherent data, mission and explore units capped at 90-100 s, holds fewer than two independent
  51-s clips per unit.
- **Predictability.** Twenty-five seconds of pooled commands leaves a lot of closed-loop outcome unexplained. The
  level may learn little until a planner (rather than recorded commands) supplies its actions.
- **Feedback.** Longer runs, and no use for the top level in the current harness.

So a longer top level probably needs a new corpus of long, goal-directed episodes: full round trips, exploration
runs across whole mazes.

## Why the middle, now

- **Clear of the flat model.** L2 at 1.6 s is already twice the flat model's horizon, and L3 at 12.8 s is about one
  cell, the paper's own top-level scale.
- **The longest top level the current corpus supports comfortably.** 25.7-s clips keep 73% of the mission and
  explore units' training windows (units of 90-100 s), so no new data collection is needed.
- **No extra compute.** Upper levels encode only the frames they sample, about 20 per clip whatever the strides.
- **Untested strides, mitigated.** Each level's training is checked by the per-level gates (H1 health, H2 action
  sensitivity: true macro-actions must beat shuffled ones and persistence). If L2 or L3 fail them, the fallback is
  an extra level to shrink the leaps: 0.1, 0.8, 3.2, 12.8 s (strides 8, 4, 4), at the same clip length.
- **Free units out of the upper losses.** The scripted free units (segments of 2-6 s) are noise at 12.8 s, and §G's
  explore-versus-stitch result says such data does not teach an abstract level. They still teach L1, where their
  steady runs and starts from rest matter.

## What would change this

- **Planning above the forecaster.** Once route or frontier choice moves into the model, the top level must reach
  the scale of those decisions. Then collect long goal-directed episodes and revisit a 25-50-s level.
- **Upper levels fail H2 at these strides.** Add the intermediate level first, before shortening the top.
- **Dynamic environments.** Moving obstacles act on the 1-10-s scale (an obstacle crossing a corridor). L2 and L3
  sit there already, and should be re-checked against the obstacles' timescales when they arrive.
