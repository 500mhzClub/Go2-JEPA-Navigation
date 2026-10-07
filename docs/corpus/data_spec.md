> The design specification the May 2026 corpus (T1's pretraining data) was generated to; `go2nav.corpus` cites its
> sections as "data spec §N". Kept as written; the rest of this repository does not depend on its H-JEPA sections.

# Fresh Retrain Data Specification for LeWM-Style Quadruped Navigation

Date: 2026-05-13

> **Implementation pointers.** Where this spec talks about generators or
> collectors, the live implementation lives in:
>
> - §8 (scene distribution) + §9 (geometry) + §14 (domain randomization)
>   → [lewm_worlds/families.py](../lewm_worlds/lewm_worlds/families.py),
>   [lewm_worlds/randomization.py](../lewm_worlds/lewm_worlds/randomization.py),
>   shares in [lewm_worlds/splits.py](../lewm_worlds/lewm_worlds/splits.py).
> - §13 (collection policy mix) →
>   [lewm_genesis/collectors/](../lewm_genesis/lewm_genesis/collectors/);
>   see [collection_policy.md](collection_policy.md) for the per-collector
>   reference and audit format.
> - §5.1 `command_source` + privileged route labels →
>   [CommandBlock.msg](../lewm_go2_control/msg/CommandBlock.msg) fields
>   `command_source`, `route_target_id`, `next_waypoint_id`.
> - §16 spawn integrity + cell-scoped `episode_id` →
>   `RolloutRunner._reset_robot_to_spawn`; per-episode spawn pose
>   randomization is on by default
>   (`RolloutConfig.randomize_spawn_pose=True`).
> - §5.3 derived labels →
>   [lewm_worlds/labels/derived.py](../lewm_worlds/lewm_worlds/labels/derived.py)
>   plus [scripts/derive_raw_rollout_labels.py](../scripts/derive_raw_rollout_labels.py).
>   This is an offline post-pass: it rebuilds scene metadata from
>   `family + topology_seed` or a manifest, then joins logged base poses,
>   episode info, and command blocks.

## 1. Scope

This document specifies the data required for a completely fresh retrain of the
LeWMQuad navigation stack. It is simulator-agnostic and robot-agnostic within
the quadruped class: it does not assume Genesis, Mini Pupper, a specific URDF,
or a specific rendering backend. It does assume a quadruped with an egocentric
camera and a low-level locomotion controller that can execute body-frame command
blocks or a simulator-specific equivalent.

The target stack is the repository architecture, not a larger model:

- Vision encoder: ViT-Tiny, 224 px RGB, patch size 14, 192-dim latent.
- Predictor: 6-layer causal transformer, 192-dim residual stream, AdaLN-zero
  action conditioning, 16 heads with 64-dim heads, 2048-dim MLP.
- Projectors: two-layer BatchNorm MLPs with 2048 hidden units.
- Main temporal abstraction: `seq_len=4`, `temporal_stride=5`,
  `action_block_size=5`, `window_stride=5`.
- Main command representation: `active_block`, normally a flattened
  `5 x command_dim` executed command block. In this repository the command dim
  is 3, so the predictor sees 15D macro actions.
- Downstream heads: keep the existing 512-wide MLP/head scale unless an H-JEPA
  implementation explicitly adds the BeliefEncoder, GoalAdapter, LoopClosure,
  and Reachability modules described in `docs/v3_topological_nav_plan.md`.

The primary data lesson from `report.md` is that frame count is not the scarce
resource. Distinct scene topology, reset-correct episodes, action support, hard
aliasing examples, and sequence-level escape examples are the scarce resources.

## 2. LeWM Data Philosophy

LeWM is a joint-embedding predictive architecture, not a reconstruction model.
The data should therefore be built around learnable predictive structure:

- The model learns by predicting future latent embeddings under actions, not by
  decoding pixels. Data must expose action-conditioned differences in future
  state, not merely many visually adjacent frames.
- The encoder and predictor are trained end to end with a SIGReg anti-collapse
  term. Data must keep enough diversity in every batch that SIGReg is not asked
  to rescue a collapsed or nearly constant scene distribution.
- The world model should see contacts, near-wall views, stops, recoveries, and
  failed maneuvers. Filtering all collisions creates a model that is clean
  offline and out of distribution online.
- Privileged simulator state may be stored as labels for audits and downstream
  supervision, but it must not be an input to the deployed model.
- The planner must search action blocks that are represented in the training
  distribution. Unsupported high-dimensional command noise should not be used
  as the main inference action space.
- Evaluation must be by held-out scene topology and held-out scene seed, not by
  adjacent windows or different parallel envs from the same scene.

For navigation, LeWM should be treated as a local dynamics and perception
backbone. It should not be expected to solve long-horizon topological routing
from single-frame latents alone. The fresh corpus must therefore support both:

- Base LeWM training for short-horizon action-conditioned prediction.
- Eventual H-JEPA training for history-conditioned place belief, goal-image
  retrieval, loop closure, reachability, and macro-action routing.

## 3. Definitions

Use these units when auditing the corpus:

- `scene_instance`: one unique environment topology plus obstacle placement,
  landmarks, visual randomization seed, terrain seed, and physical-domain seed.
- `topology_seed`: the graph or navigable-structure seed before visual and
  physics randomization.
- `episode`: a reset-separated robot rollout. A `done` at raw step `t` ends one
  episode; raw step `t + 1` begins a new episode.
- `raw_step`: one high-level command tick observed by the world-model data
  pipeline. Low-level physics substeps may be smaller.
- `macro_step`: one model action block. The repository default is 5 raw steps.
- `macro_transition`: one `(observation_t, active_command_block_t,
  observation_t+1)` transition at the macro timescale.
- `wm_sequence`: one LeWM training sample with 4 observations and 4 command
  blocks at stride 5. It covers 20 raw command ticks in the repository default.
- `belief_window`: a history window for H-JEPA, usually 8 to 32 macro steps.
- `pair_example`: a pair of belief windows, goal images, or memory nodes used
  for contrastive, loop-closure, or reachability training.

All counts below are effective counts after removing invalid windows,
cross-reset windows, corrupted frames, and overrepresented duplicates. A single
scene with many parallel envs cannot substitute for many distinct scenes.

## 4. Data Products

Every fresh retrain should produce three immutable datasets:

1. `raw_rollout`: simulator state, command streams, proprioception, resets,
   contacts, and scene metadata.
2. `rendered_vision`: egocentric RGB replayed or captured from `raw_rollout`,
   with camera-validity flags and copied raw labels.
3. `derived_labels`: simulator-privileged labels computed offline for audits
   and downstream heads.

The base LeWM may train on `rendered_vision` plus actions and reset masks. The
planning and H-JEPA stages train from `rendered_vision` plus `derived_labels`.

At production scale, **Genesis is the authoritative dynamics and rendering
backend** for state, commands, resets, contacts, and RGB/depth (see
[decision_pivot_to_genesis.md](decision_pivot_to_genesis.md)). The
`raw_rollout` and `rendered_vision` schema separation is preserved as a
contract for downstream consumers even though the bulk implementation
collapses them into a single Genesis pass. Gazebo is retained only as a
small audit oracle for dynamics and sensor parity spot checks, not on the
bulk path.

Lidar (`/velodyne_points`, `/unitree_lidar/points`) is **not part of the v3
corpus**. The bringup plan marked it optional; the pivot makes the absence
explicit.

### 4.1 Transport and topic integrity

Rosbag transport loss is not automatically equivalent to bad training data, but
it must never be silent. Every `raw_rollout` conversion must emit:

- `contract_audit`: command/reset loss checks.
- `topic_audit`: per-topic count, record-time rate, source-time rate, maximum
  timestamp gap, and timestamp-regression counts.
- `data_quality_audit`: the selected pass/fail policy.

Policy:

- `smoke`: fail only contract-critical command/reset loss. Non-contract topic
  gaps are reported but do not fail the gate.
- `pilot` and `training`: fail missing command, executed-command, or reset
  events; duplicate or unmatched command IDs; reset-count gaps; missing critical
  streams; timestamp regressions; excessive critical-stream gaps; and low
  observed critical-stream rates.
- `raw_pilot` and `raw_training`: same as `pilot`/`training` for state,
  command, reset, contact, and proprioception streams, but do not require
  `/rgb_image`; vision is supplied later by the render-replay stage.

RGB/proprio/state gaps that pass the hard gate may still invalidate local
training windows. Those windows must be flagged or excluded; they must not be
silently interpolated into valid data.

## 5. Required Schema

### 5.1 Per-step observations

Required:

- `vision`: uint8 RGB, shape `(episodes/envs, time, 3, H, W)` or equivalent.
- `cmd_nominal`: the command requested by the collector.
- `cmd_executed`: the command actually consumed by the low-level controller
  after latency, filtering, clipping, or command-hold effects.
- `done`: reset or termination flag.
- `collision/contact`: physical contact with obstacle or terrain failure.
- `base_position_world`: logged for labels and audits only.
- `base_orientation_world`: quaternion, logged for labels and audits only.
- `joint_position`, `joint_velocity`, `last_low_level_action`: for robot-state
  diagnostics and optional proprio labels.
- `proprio_observation`: if available, the exact proprio vector seen by the
  low-level controller.
- `camera_valid`: false when clipping, invalid depth, missing render, severe
  motion blur artifact, or fallback substitution occurs.

Recommended:

- `base_linear_velocity_body`, `base_angular_velocity_body`.
- `foot_contacts` per leg.
- `command_source`: primitive, teacher, OU/noise, recovery, route-following,
  teleop, or scripted.
- `controller_mode`: gait/controller state if the low-level controller exposes
  one.
- `sim_time`, `control_dt`, `physics_dt`, `decimation`.

### 5.2 Scene metadata

Required:

- `scene_id`: globally unique dataset id.
- `topology_seed`, `visual_seed`, `physics_seed`.
- `scene_family`: one of the families in section 8 or a project-specific
  extension.
- `robot_id`: embodiment and controller id.
- `camera_model`: intrinsics, extrinsics, FOV, near/far planes, resolution.
- `world_bounds`.
- `obstacles`: geometry, pose, material, and semantic class.
- `landmarks`: identity, pose, visibility surface, color/texture family.
- `navigable_graph`: node centers, adjacency, edge widths, local graph type,
  graph diameter, dead-end list, loop/cycle count.

For non-grid environments, `navigable_graph` can be a waypoint graph,
visibility graph, Voronoi graph, or any deterministic abstraction that supports
same-place labels and graph-distance buckets. It must be regenerated from the
scene seed without replaying the robot rollout.

### 5.3 Derived per-step labels

These labels are not deployment inputs.

Required:

- `episode_id`: reset-separated.
- `episode_step`: step since most recent reset.
- `scene_id`: repeated per step.
- `cell_id` or `place_node_id`: scene-scoped topological node.
- `yaw_bin`: heading bin relative to the topological graph.
- `local_graph_type`: corridor, left/right turn, T-junction, crossroad,
  dead-end, doorway, open room, obstacle field, ramp/step, boundary, unknown.
- `nearest_cell_distance`: distance to nearest graph node center.
- `graph_distance_to_landmarks`: BFS or weighted graph distance per landmark.
- `clearance`: distance to nearest obstacle or wall.
- `traversability`: forward free-space score over a short horizon.
- `landmark_visible`, `landmark_identity`, `landmark_bearing`,
  `landmark_range`, with line-of-sight occlusion.
- `integrated_body_motion`: body-frame `dx`, `dy`, and `dyaw` over every action
  block and over H-JEPA history windows.

Implementation status:

- `scripts/derive_raw_rollout_labels.py` writes `derived_labels/labels.jsonl`
  and `derived_labels/summary.json` for compact `messages.jsonl` or per-scene
  rosbag2 MCAP raw rollouts.
- Current fields include `cell_id`, `yaw_bin`, `local_graph_type`,
  `nearest_cell_distance_m`, `bfs_distance_to_landmark`, `clearance_m`,
  `traversability_forward_m`, landmark visibility/bearing/range, and
  integrated body motion over action-block and history windows.
- The pass is labels-only. It is not called by `RolloutRunner` and none of its
  privileged fields are deployment inputs.

Recommended:

- `frontier_gain`: new topological nodes or area reached in a future window.
- `escape_gain`: whether a future sequence leaves the recent local basin.
- `stuck_label`: low displacement despite nonzero command.
- `recovery_label`: contact or near-contact followed by successful separation.
- `route_target_id`: teacher's current target, if a privileged teacher is used.
- `route_progress`: graph distance reduction under the teacher or oracle route.

## 6. Split Protocol

Use scene-level splits. No topology seed, generated graph, landmark placement,
or visual seed may appear in more than one split.

Recommended full retrain split:

| Split | Scene instances | Purpose |
| --- | ---: | --- |
| Train | 2400 | Base LeWM and downstream heads |
| Validation | 300 | Hyperparameters, calibration, early stopping |
| Test-ID | 300 | Held-out scenes from the same registered distribution |
| Test-hard | 300 | Larger or more aliased scenes, not used for tuning |
| Test-transfer | 100 per transfer axis | Optional simulator, camera, texture, or robot changes |

Minimum acceptable split when compute is constrained:

| Split | Scene instances |
| --- | ---: |
| Train | 1000 |
| Validation | 150 |
| Test-ID | 150 |
| Test-hard | 150 |

Anything below 500 train scenes is a diagnostic or smoke-test corpus, not a
fresh retrain corpus for navigation generalization.

## 7. Corpus Size

### 7.1 Full target

Use this as the default target for a serious fresh retrain:

- Unique train scenes: 2400.
- Rollout per train scene: 64 reset-separated episodes or parallel env streams.
- Raw duration per episode/env stream: 800 to 1200 raw steps.
- Valid macro transitions per train scene: at least 10,000.
- Valid LeWM sequences per train scene: cap at 10,000 per epoch to prevent
  large scenes dominating.
- Total effective train LeWM sequences: 20M to 30M.
- Total effective validation LeWM sequences: 2M to 4M.
- Total effective test LeWM sequences: 2M to 4M per test split.

At the repository default of `seq_len=4`, `temporal_stride=5`,
`action_block_size=5`, and `window_stride=5`, an env stream of 1000 raw steps
produces roughly 196 candidate LeWM sequences before filtering. With 64 env
streams per scene, that is about 12,500 candidate sequences per scene.

### 7.2 Minimum target

The minimum target for a complete but lower-confidence retrain:

- Unique train scenes: 1000.
- Rollout per train scene: 48 reset-separated episodes or env streams.
- Raw duration per episode/env stream: 800 to 1000 raw steps.
- Valid macro transitions per train scene: at least 6500.
- Total effective train LeWM sequences: 6M to 10M.
- Validation/test sequences: at least 750k per split.

If the corpus cannot meet these numbers, keep the experiment framed as a
diagnostic. Do not use it to reject the architecture.

### 7.3 Scene count beats per-scene env count

If storage or render time is limited, reduce envs per scene before reducing
scene count. For this task, 2400 scenes with 32 env streams each is preferable
to 300 scenes with 256 env streams each.

## 8. Scene Distribution

The scene distribution should match the navigation task while preserving local
dynamics coverage. The full target uses 3000 ID scenes total
(`2400/300/300`). Counts below are train counts; validation and test should
preserve the same proportions unless explicitly marked as hard test.

| Scene family | Train scenes | Share | Purpose |
| --- | ---: | ---: | --- |
| Open obstacle fields | 240 | 10% | Basic free-space locomotion, obstacle avoidance, camera parallax |
| Local composite motifs | 360 | 15% | T-junctions, S-bends, doorways, slaloms, short dead ends |
| Small enclosed mazes | 360 | 15% | 4x4 to 5x5 graphs, short routes, many successful traversals |
| Medium enclosed mazes | 600 | 25% | 6x6 to 8x8 graphs, the main navigation distribution |
| Large enclosed mazes | 360 | 15% | 9x9 to 12x12 graphs, long horizons for H-JEPA and routing |
| Loop and alias stress mazes | 240 | 10% | Repeated corridors/junctions, false loop closures, symmetry |
| Rough/local dynamics variants | 120 | 5% | Slopes, floor friction, small steps, clutter if the robot supports them |
| Visual/sensor stress variants | 120 | 5% | Lighting, texture, exposure, distractors, camera jitter |

Hard test should overweight large and alias stress scenes:

- 30% large enclosed mazes.
- 30% loop and alias stress mazes.
- 20% medium enclosed mazes.
- 10% rough/local dynamics variants.
- 10% visual/sensor stress variants.

## 9. Geometry Requirements

Use robot-normalized ranges so the spec transfers across quadrupeds.

Let:

- `L_body` = body length.
- `W_body` = body width at hip/shoulder envelope.
- `R_turn` = reliable in-place or arc-turn radius under the low-level
  controller.
- `H_cam` = camera height.

Required ranges:

- Corridor width: `1.6 x W_body` to `3.0 x W_body`, with at least 25% of
  corridors in the difficult `1.6 x W_body` to `2.0 x W_body` band.
- Doorway/gap width: `1.4 x W_body` to `2.4 x W_body`.
- Dead-end depth: `1.0 x L_body` to `5.0 x L_body`.
- Straight corridor length: `2 x L_body` to `12 x L_body`.
- Turn/junction spacing: include both short spacing near `R_turn` and long
  spacing that requires commitment before visual feedback changes.
- Wall height: high enough to occlude the camera's line of sight, unless the
  scene family is explicitly testing low obstacles.
- Landmark placement: reachable and occluded by topology, not visible through
  walls or across impossible routes.

Each train split should include:

- At least 25,000 unique topological nodes across all scenes.
- At least 60,000 directed graph edges.
- At least 5000 dead-end nodes.
- At least 5000 junction nodes.
- At least 2000 loop/cycle structures.
- At least 200,000 pairs of graph-distinct locations whose current frame is
  visually similar under a frozen or early LeWM baseline.

## 10. Situation Coverage

Counts are train-split minimums in effective macro-window examples. A
macro-window is centered on a LeWM sequence or H-JEPA belief window and should
not cross a reset.

| Situation | Minimum examples | Target examples | Notes |
| --- | ---: | ---: | --- |
| Straight free-space walking | 500k | 2M | Balanced speeds and surfaces |
| Slow cautious forward near walls | 300k | 1M | Within 1 to 2 body widths of walls |
| Left/right turns in corridors | 250k each | 750k each | Include both arc and pivot turns |
| T-junction approach | 250k | 750k | All three branch choices represented |
| Crossroad approach | 150k | 400k | Needed for alias and routing |
| Doorway or narrow gap traversal | 250k | 750k | Include failed and successful attempts |
| Dead-end entry | 200k | 600k | Approach before turn-around is visible |
| Dead-end recovery/backout | 150k | 500k | Reverse, pivot, and re-entry cases |
| Wall following | 200k | 600k | Left and right wall balanced |
| Near-collision without contact | 250k | 750k | Clearance below safety threshold |
| Contact/collision with recovery | 100k | 300k | Keep but do not dominate |
| Stuck/low-motion under command | 100k | 300k | Critical for planner failure diagnosis |
| Spin or scan in place | 150k | 400k | Goal and memory disambiguation |
| Landmark first sighting | 100k | 300k | Transition from not visible to visible |
| Landmark occlusion/loss | 100k | 300k | Avoid through-wall shortcuts |
| Goal arrival from multiple headings | 50k | 150k | For GoalAdapter and success detection |
| Frontier expansion | 200k | 600k | Sequence-level exploration target |
| Loop closure revisit | 100k | 300k | Same place after long route |
| False loop closure | 100k | 300k | Similar view, graph-distant place |

For any situation with left/right variants, keep the ratio between variants
within 45/55 unless the robot embodiment makes symmetry impossible.

## 11. Action and Macro-Action Coverage

### 11.1 Command block contract

The logged command block must represent what the low-level controller actually
executed. If the controller has latency, smoothing, clipping, gait-state
conditioning, or safety overrides, store both the nominal command and the
executed command. The predictor should train on executed blocks.

For repository-compatible training:

- `K = 5` raw command ticks per macro action.
- `cmd_dim = 3` for body-frame `[vx, vy, yaw_rate]`, or the equivalent
  robot-specific command vector.
- `active_block` flattens the `K x cmd_dim` block.
- Planner macro-action repeat should equal `K`.

If a different quadruped needs a different command period, choose `K` by
physical effect, not by simulator ticks. One macro step should be long enough
to produce measurable displacement and yaw under normal gait, but short enough
that local collisions remain recoverable by replanning.

### 11.2 Primitive support

The train corpus must contain the primitive family that the planner will
search. Minimum train macro transitions per primitive:

| Primitive family | Minimum transitions | Target transitions |
| --- | ---: | ---: |
| Forward slow/medium/fast | 250k each | 750k each |
| Reverse/backout | 200k | 600k |
| Brake/hold | 150k | 400k |
| Spin left/right | 150k each | 400k each |
| Arc left/right | 200k each | 600k each |
| Lateral step left/right if supported | 100k each | 300k each |
| Wall-follow left/right | 150k each | 400k each |
| Recovery sequence | 100k | 300k |
| Teacher route-following | 500k | 2M |
| Frontier/explore target pursuit | 300k | 1M |

Every primitive should appear in every major scene family with at least 10,000
macro transitions. Recovery and collision-adjacent primitives may be exempt in
open obstacle fields if they would require unrealistic forced failures.

### 11.3 Command distribution audits

Before training, publish:

- Histograms of each command dimension before and after latency reconstruction.
- Joint histograms of forward speed vs yaw rate.
- Fraction of near-zero commands.
- Fraction of saturated/clipped commands.
- Per-primitive success, collision, and displacement distributions.
- Predictor action sensitivity probe on a held-out batch after LeWM training.

If different sampled commands collapse to near-identical physical motion under
the low-level controller, fix the primitive bank or controller before blaming
the world model.

## 12. H-JEPA Data Requirements

The fresh corpus should be collected once but support later H-JEPA work.

### 12.1 BeliefEncoder

For every train scene:

- At least 70% of visited topological nodes should have 16 or more belief
  windows.
- At least 50% of visited nodes should have belief windows from 4 or more
  approach headings or entry directions.
- At least 25% of visited nodes should be revisited after leaving the local
  neighborhood for 8 or more graph hops.
- Boundary frames, where `nearest_cell_distance` is high or cell assignment is
  ambiguous, should be flagged and excluded or downweighted.

Global pair targets:

- Same-place strong-positive pairs: at least 2M train, 200k val.
- Same-place different-yaw weak-positive pairs: at least 1M train, 100k val.
- Same-scene graph-distinct hard negatives: at least 3M train, 300k val.
- Cross-scene capped negatives: at most 30% of pair loss contributions.
- History-disambiguable visual aliases: at least 500k train pairs.
- History-aliased ambiguous pairs: log and mask; do not force apart.

### 12.2 GoalAdapter

For goal-image training:

- Every landmark/goal cell in train scenes should have at least 16 single-image
  goal observations.
- Each goal cell should include at least 4 yaw bins when physically visible.
- Each goal identity or visual family should appear in at least 200 train
  scenes if identity-specific landmarks are used.
- The validation and test goal images must come from held-out scenes.

Pair targets:

- Goal-to-belief positives: at least 1M train, 100k val.
- Goal-to-belief hard negatives within same scene: at least 2M train, 200k val.
- Goal false matches caused by color/texture distractors: at least 200k train.

### 12.3 LoopClosureHead

Loop closure data must be precision-first. A false merge is worse than a missed
merge.

Minimum pair counts:

- True loop closures: 500k train, 50k val.
- Near-place but not same-node negatives: 500k train, 50k val.
- Visually similar graph-distant false-loop negatives: 500k train, 50k val.
- Cross-scene easy negatives: capped to 20% of training pairs.

Acceptance data must include calibration splits separate from threshold tuning.

### 12.4 ReachabilityHead

Reachability should be trained on memory-generated node pairs, not only raw
rollout frame pairs. Required buckets:

- `same/adjacent`
- `2-3`
- `4-7`
- `8-15`
- `>15`
- `unknown/unreliable`

Minimum memory-generated train pairs:

| Bucket | Minimum train pairs | Target train pairs | Minimum val pairs |
| --- | ---: | ---: | ---: |
| same/adjacent | 250k | 1M | 25k |
| 2-3 | 250k | 1M | 25k |
| 4-7 | 250k | 1M | 25k |
| 8-15 | 250k | 1M | 25k |
| >15 | 250k | 1M | 25k |
| unknown/unreliable | 250k | 1M | 25k |

If a bucket cannot meet the minimum, merge adjacent finite buckets before
training. Do not let class weighting hide a missing bucket.

## 13. Collection Policy Mix

The corpus should combine task-success trajectories and dynamics-stress
trajectories. Recommended train mix by macro transition:

| Source | Share | Purpose |
| --- | ---: | --- |
| Privileged route teacher to goals/landmarks | 30% | Successful navigation and goal arrival |
| Privileged frontier/exploration teacher | 20% | Coverage and dead-end discovery |
| Primitive/scripted command curriculum | 20% | Planner action support |
| OU/noisy exploration | 10% | Local dynamics diversity |
| Recovery/contact curriculum | 10% | Wall contact, stuck, backout |
| Loop/revisit curriculum | 10% | Loop closure and memory training |

The privileged teacher is a data collector only. Its graph target, oracle pose,
and route state are labels, not deployment inputs.

## 14. Domain Randomization

Randomize enough to prevent visual shortcuts, but not so much that local
dynamics become unlearnable.

Per scene:

- Wall/floor textures and colors, with landmark colors kept separable from
  background distractors only when the task depends on identity.
- Lighting direction, intensity, exposure, and mild shadows.
- Camera extrinsics within calibrated mounting tolerance.
- Camera noise, mild blur, and compression artifacts if expected at deployment.
- Floor friction and restitution within robot-safe ranges.
- Robot mass/inertia and actuator strength within calibrated tolerances.
- Controller command latency and sensor delay within measured ranges.

Hold out at least one visual theme and one physics theme for test-transfer if
the final claim includes simulator or robot transfer.

## 15. Rendering and Sensor Quality

Invalid vision data poisoned the previous stack. The fresh retrain must include
a hard render-quality gate:

- Camera clipping or invalid depth: below 0.1% of frames; all such frames
  flagged.
- Frame substitution/fallback: below 0.2% of frames; all such frames flagged.
- Landmark line-of-sight labels: obstacle-aware, never FOV-only.
- Near-plane and wall thickness: chosen so wall-contact views show wall
  surfaces, not through-wall geometry.
- Camera timestamp aligned with the command and proprio timestamp.
- Render replay deterministic from raw rollout and scene metadata.
- GPU render-replay jobs must consume a checked `raw_rollout` plus the scene and
  camera manifests, write camera-validity flags, and never rewrite command,
  reset, or episode arrays.

If invalid frames cluster near collisions, do not simply drop all collisions.
Fix the camera placement, near plane, wall thickness, or retraction logic so
contact remains represented correctly.

## 16. Reset and Episode Integrity

Hard requirements:

- No LeWM sequence may train a transition across a reset.
- No downstream pair may use positions or labels across a reset unless the pair
  is explicitly cross-episode and same-scene by design.
- `episode_id` must be reset-separated, not `(file, env)`.
- `scene_id` must be scene-scoped and stable.
- `cell_id` must always be interpreted as `(scene_id, cell_id)`.
- Raw and rendered datasets must preserve identical `done` arrays.

Audit:

- Count cross-reset candidate windows before filtering.
- Assert zero cross-reset windows in the final training index.
- For pair datasets, sample 10,000 pairs and verify episode and scene scoping.

## 17. Evaluation Sets

Evaluation should answer separate questions:

- `val-id`: model selection on held-out scenes from the train distribution.
- `test-id`: final in-distribution scene generalization.
- `test-hard`: long horizons, high aliasing, narrow corridors, many dead ends.
- `test-transfer-visual`: unseen textures, lighting, camera perturbations.
- `test-transfer-physics`: unseen friction, latency, body mass, terrain.
- `test-transfer-robot`: optional, same task on a different quadruped class.

For navigation evaluation, each test scene should define:

- At least 8 start-goal pairs.
- At least 2 short, 3 medium, and 3 long graph-distance routes.
- At least 2 routes requiring a dead-end recovery or loop closure.
- Goal images captured from held-out approach headings.
- Oracle shortest path length for metrics only.

## 18. Pretraining and Downstream Training Gates

### 18.1 Data gates before LeWM training

Do not start the main retrain until these pass:

- Train scene count meets the selected tier.
- Validation/test scenes have no topology or seed overlap with train.
- Valid LeWM sequences meet the selected tier.
- Each major scene family contributes at least 5% of train sequences.
- Render invalid-frame rate is below threshold.
- `raw_rollout` conversion passes the `training` data-quality profile.
- If RGB was generated offline, the render-replay plan and rendered output pass
  the camera-validity and timestamp-alignment gates.
- Per-primitive action support meets section 11.
- Collision/contact examples are present but below 20% of all windows.
- Reset-crossing windows are excluded.
- Derived graph labels are populated for at least 99% of steps in graph scenes.

### 18.2 Gates after LeWM training

Run before downstream heads:

- Treat autoregressive planning-readiness as an explicit gate before advancing
  the `seq_len={4,8,16}` sweep. Run
  `scripts/probe_lewm_rollout_horizons.py`, review horizons
  `1,2,3,5,8,10,16,20`, and approve the exact checkpoint before starting the
  next sequence-length phase. See `docs/lewm_planning_readiness_gate.md`.
- Held-out scene prediction loss is within 20% of train loss trend, not
  diverging.
- `z_proj` per-dimension standard deviation remains non-collapsed.
- Predictor action-sensitivity probe shows terminal predicted latents vary
  materially across supported primitive actions.
- Multi-step open-loop error and persistence, zero-action, and shuffled-action
  baselines are reported for horizons 1, 2, 3, 5, 8, 10, 16, and 20 macro
  steps.
- Latent-distance vs graph-distance Spearman rho is reported by scene family.
- Nearest-neighbor cell confusion is reported by local graph type.

### 18.3 Gates before H-JEPA training

- Belief-window same-place coverage passes section 12.1.
- Visual-alias hard-negative inventory is nontrivial.
- Goal-image diversity passes section 12.2.
- Reachability bucket histograms pass section 12.4.
- Memory dry-run node purity can be computed on validation scenes.

## 19. Failure Points and Mitigations

| Failure point | Symptom | Mitigation in data spec |
| --- | --- | --- |
| Too few scene seeds | Good train/eval loss, poor unseen navigation | Minimum scene counts and scene-level split |
| Scene leakage | Inflated offline head metrics | Split by topology/scene seed only |
| Reset contamination | Impossible displacement or place labels | Reset-separated `episode_id`, zero cross-reset windows |
| Camera clipping | Latents learn through-wall views | Render quality gate and contact-preserving fixes |
| Unsupported planner actions | CEM candidates look different numerically but move the same | Primitive coverage and executed-command logging |
| Single-frame aliasing | Local loops, wrong loop closure | Belief-window data and false-loop negatives |
| Teacher bias | Model only learns shortest-path behavior | Mixed collection policy with primitive/noisy/recovery data |
| Rare recovery scarcity | Planner cannot escape dead ends | Explicit dead-end and recovery counts |
| Landmark shortcuts | Goal head rewards color through walls | LOS-aware labels and distractor false matches |
| Class imbalance | Reachability predicts majority bucket | Minimum per-bucket pair counts |
| Cross-scene negative dominance | Retrieval separates scene style, not place | Cap/downweight cross-scene negatives |
| Privileged leakage | Deployment metrics do not reproduce | Clear input/label boundary |
| Robot transfer mismatch | Different gait or camera invalidates data | Robot metadata, normalized geometry, transfer splits |
| Simulator artifacts | Model learns renderer/physics quirks | visual/physics randomization and transfer tests |

## 20. Recommended Fresh Collection Recipe

This is a simulator-agnostic recipe, expressed in target counts rather than a
specific CLI.

1. Generate 3000 ID scene instances and 300 hard-test scene instances from
   registered seeds.
2. For each train/val/test-ID scene, collect 64 reset-separated env streams of
   800 to 1200 raw steps.
3. For each hard-test scene, collect evaluation-only rollouts and start-goal
   definitions; do not train on hard-test rollouts.
4. Use the collection policy mix in section 13.
5. Store nominal commands, executed commands, controller latency, resets, pose,
   contacts, proprioception, and all scene metadata.
6. Render egocentric RGB deterministically with camera-validity flags.
7. Compute derived topological, landmark, safety, motion, frontier, and
   reachability labels offline.
8. Build fixed train/val/test manifests by scene id.
9. Build LeWM sequence indices with reset filtering and per-scene caps.
10. Build downstream pair datasets only after the LeWM checkpoint and H-JEPA
    phase gates are fixed.

If current infrastructure insists on one scene per chunk, increase chunks to
match the scene counts. If it insists on thousands of parallel envs per scene,
subsample per-scene windows during training so those envs do not erase scene
diversity.

## 21. Acceptance Checklist

The fresh retrain dataset is ready when all of the following are true:

- Scene counts meet the selected full or minimum tier.
- Every scene has deterministic metadata sufficient to regenerate the
  navigable graph.
- Train, validation, test-ID, and test-hard have no scene/topology leakage.
- At least 99% of graph-scene steps have valid `cell_id` or `place_node_id`.
- LeWM has at least 6M effective train sequences for the minimum tier or 20M
  for the full tier.
- Every required situation in section 10 meets its minimum count.
- Every planner primitive in section 11 meets its minimum support.
- Goal, loop-closure, belief, and reachability pair inventories meet section
  12.
- Render invalid-frame rates are below section 15 thresholds.
- Reset-crossing train windows are zero.
- Data manifests and histogram audits are committed alongside the dataset.

Only after this checklist passes should a failed retrain be interpreted as an
architecture or planner problem.
