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
