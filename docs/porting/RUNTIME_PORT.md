# Runtime port: notes and method

How the original mission runtime works, how it is being ported, and how each port is checked. The resume point for
the runtime part of the migration (see also `../MIGRATION_STATUS.md`).

## How the original runtime runs one mission

Entry chain: v14 pinned launcher → cohort runner → `run_go2_dev_mission_pinned_v14_development.py` (harness version,
C1A/C1R variants, dynamics) → `run_go2_dev_mission_stage2_development.py` → `run_go2_dev_mission_development.py`
(fixes, margins, degradation, decoders) → the V4 owner's `run`
(`scripts/run_go2_navigation_capability_completed_support_v4_development.py`).

The owner's loop, at 50 Hz policy ticks (24,000 ticks maximum):
- every fifth tick (10 Hz, `frame = tick // 5`) the session renders a sensor packet: primary RGB + depth, auxiliary
  (45° down) RGB + depth, 500 Hz gyro block, proprioception; it is submitted to the controller and the pipeline is
  drained before the tick continues;
- every tick, `controller.request(now_ns)` returns the command, which the session applies through the locomotion
  policy (10 physics substeps).

Controller pipeline per frame (threads and process pools, but deterministic: the clock is the simulation clock, the
loop drains every queue after each submit, and planning joins the mapping queue first):
1. **pose** (worker pool): dual-camera RGB-D visual odometry with gyro rotation → raw visual pose evidence;
2. **registration** (worker pool) + mission state: floor-plane registration corrects the visual pose; the round-trip
   mission advances on the registered position;
3. every fourth frame: **mapping** (worker pool) updates the routing map; then, from frame 4, **planning**: route,
   motion forecast for six candidate actions × eight horizons (C0–C4 differ here), scoring, safety/recovery selection,
   and a scheduled command (dispatch three ticks later, committed for one tick);
4. **obstacles** (worker pool, its own thread): depth obstacles for the 50 Hz dispatch check.

**Verified (6 October):** running the four worker pools in-process gives an identical mission (C1, prelim maze 30,
recovery on: requests, mission, poses and model calls byte-identical; planning differs only in wall-clock fields).
The clean runtime therefore runs every stage in-process, synchronously, per frame.

## Tools (`tools/porting/`)

| Tool | Use |
|---|---|
| `runtime_mro.py`, `worker_mro.py` | Flattened class order and method ownership of the composed controller and of the four worker objects (`docs/porting/runtime_*.json`, `worker_*.json`) |
| `executed_view.py` | One method chain (or function) with only the lines the P1 trace executed |
| `call_graph.py` | Repository caller → callee graph of one mission (in-process workers, `sys.monitoring`); numba kernels are invisible to it |
| `call_tree.py` | The call tree below a function, from recorded graphs |
| `resolve_globals.py` | What each global name in a composed method resolves to, through `bind()`/`use()` rewiring |
| `replay_bundle.py` | Records a mission's component traffic: every worker-pool call (arguments, result) and every controller submit/request |

Recorded bundles (local, `outputs/porting/bundles/`, about 8.5 GB per mission): `C1_on_prelim30`.

## Method

Each component is ported to clean code from the executed views and checked against a bundle:
- worker components (pose tracker, registration, mapping, obstacles) by replaying their recorded call sequence and
  comparing the decision-relevant outputs at every call;
- the controller by replaying recorded packets and worker results and comparing every dispatch request, mission row
  and planning decision;
- finally the whole mission closed loop against the original (physics trace and outcome).

Diagnostics, provenance flags and self-validation of the original (witness checks, digests, schema assertions) are
not ported unless a decision or a shipped result reads them. numba kernels are ported as numba kernels
(`fastmath=False`) so results stay bit-identical.

## Component notes

### Pose tracker (`_motion`, `SparseCornerCompletionMotion` → model `SparseCornerCompletionPose`, MRO 62)

Consumers read only `current_pose` (through `current_joint_pose`): frame, measured_ns, mode, reference_frame, position,
rotation, gyro rotation, promoted keyframe and reason, keyframe count, digests. Everything else in its 111-key result
is diagnostics or constant flags.

Effective feature frame (`CompletedSparseFeatureFrame150`): Shi–Tomasi corners (quality 0.01, min distance 5, block
3) on depth-valid pixels, kept if liftable (raw-depth bilinear lift valid), ordered by min-eigenvalue response then
y, x; upright SIFT descriptors (size 8, angle 0). If more than 150 are liftable: per-cell cap 13 on a 4×3 grid of
160-px cells, then the first 150. Otherwise all of them, completed to 150 with weaker corners (quality 0.001) not
already present.

Matching lifts with the 5×5 local inverse-depth estimate (cached per observation); feature detection lifts with raw
depth.

### Floor registration, obstacles, routing map (ported, exact on C1_on_prelim30)

- Registration: joint floor plane along the visual up → minimal tilt + heading-restoring twist + height correction
  against frame 0's plane; transport of the last registered correction when the frame has < 100 floor candidates.
  Weak-extent pruning (two variants), validation failures and re-acquisition never ran: `UnportedBranch`.
- Obstacles: own gyro integrator **without** re-orthonormalisation; floor plane along the gyro up; 4-px samples
  3–65 cm above it → 1-cm body cells within ±1 m. Also returns per-camera valid-sample counts (dispatch reads the
  primary count).
- Routing map: B from initial gravity; floor height from frame 0's plane (normalised up); coverage with the
  current plane available → valid-quad rectangle + numba polygon kernel; unavailable → local inverse depth (float64)
  + level-quad index (`eligible_floor_cell_index`); coarse 5-cm obstacles (±160) and fine 1-cm (±800).
- Visual support (tracker diagnostics actually consumed): per-frame selected feature counts (primary, auxiliary) =
  the current frame's keypoint counts after completion (`completed_corner_support`).

### Controller (ported; exact on C1_on_prelim30 and C3_off_prelim30)

Per frame, in order: `submit` (native-context packets; the ledger's preceding requests for the frame) → pose →
`_register` (registration + visual support; mission advance on the registered position; floor re-acquisition streak;
goal/phase update; vetoes of every plan when holding) → every 4th frame: map, then `_plan` → obstacles.

`_plan` gates: startup recovery hold; frames < 10 (native context warm-up); floor waiting; mission hold. Then:
route (`_route` chain, see below) → pose in the map → target heading or route target or scan → `_select_action`
(forecast 6 candidates x 8 horizons, correction, scoring, scan utilities, `_select_clear_prediction` safety/recovery
chain) → `ScheduledCommand.prepare` (dispatch at +300 ms, 400 ms; translation pulses 100 ms) → ledger commit.

`request(now)` (50 Hz): latest eligible plan; latched window veto; dispatch check (stale obstacles > 250 ms hold;
0.45 m disk connector over the remaining plan against 1-cm obstacle cells; translation stopping margin with +0.5 s
and the observation age; primary-depth-missing translation veto [unported]); command gates (startup, visual
recovery, floor waiting, mission hold); committed prefix executed check; ledger veto/record; veto-view recovery arm.

Mission (`StopConditionedSettlingMission` ← `SettledBoundaryRoundTripMission` ← `MeasuredSettlingRoundTripMission`
← `ObservedRoundTripMission`): outbound to goal then home; arrival = within radius (0.02 m in the capability runs)
with ≥ N quiet intervals under zero requested command and measured-quiet boundaries.

`_route` chain (outer → inner): stall watchdog (frontier stall; escape unported) · turn memory weak trigger ·
interrupted-view bookkeeping · coverage translation views (requested patch, camera viewpoint) · measured view arc
(unported part) · prefix-aware/terminal flags · visual support (recovery unported) · camera frontier context ·
routing-memory scope (persistent) · initial panorama survey · frontier visits (stand-off view) · veto-view recovery
(rotate 45° toward the veto's yaw sign) · `route_from_current_pose` with the proposer chain: grid BFS over observed
floor minus 0.45 m inflation, 1-cm-cell connector, clearance-preferred A* refinement, fine goal-cell route.

Forecasters: C1 command-history model (`controllers.forecasters`); C3 V-JEPA 2.1 tokens → T4 predictor per horizon
(each distinct command prefix once) → T7 decoder; C4 one batch of 48 direct readouts (`controllers.neural`). The
candidates' commands are rounded through the model scale in float64 and passed through the safety limits
(`locomotion.safety`) from the newest applied command. C3 replay (recovery off, prelim maze 30): all 1,101 frames and
5,501 requests identical.

Mapping: the dense floor-cell index (`dense_level_floor_quads`, 32-row batches) is the one the tiled density-routed
map ran (its copy keeps the density-routed `__module__`); the sparse index falls back to it on ambiguous last bits.
C2_on_prelim31 mapping: all 443 updates identical.

C1A (`controllers.adaptive.AdaptiveTravelForecaster`) wraps the C1 forecaster: travel ratios per movement type
(`planning.actions.movement`) from the *registered* pose (the original's "tracker pose" is the evidence's corrected
pose) against C1's re-forecast of the planned commands. C1R is C1 with the stage-2 refit command model. Replays: C1A
(uniform friction 0.3, prelim maze 30) all 1,446 frames, 7,226 requests and every plan's ratios identical; C1R (patches,
stage-2 eval maze 30) all 2,593 frames and 12,961 requests identical.
