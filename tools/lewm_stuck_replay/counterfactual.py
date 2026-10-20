"""Ground truth at stuck decisions: replay a run to each chosen decision (physics checked against the trace), snapshot
the simulator (`go2nav.sim.snapshot`, as the C0 oracle), and execute every candidate's tape for 800 ms, rendering
its frames. Writes OUT/d<decision>.npz: motion (6, 8, 3) true body-frame x, y, yaw at 100-800 ms, frames
(6, 8, H, W, 3) the primary RGB at t+100..800 ms, and the history frames (4, H, W, 3) at t-300..t.

    python tools/lewm_stuck_replay/counterfactual.py RUN OUT DECISION[,DECISION...]
"""
import json, math, sys
from pathlib import Path
import numpy as np
from go2nav.data import rerender
from go2nav.planning.actions import command_sequences
from go2nav.sim import snapshot as snapshots
from go2nav.sim.sensors import rotation_xyzw


def main(run, out, decisions):
    from go2nav.sim import scene
    from go2nav.sim.session import MissionSession
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run/'planning.json').read_text())
    todo = {plan[i]['frame']: i for i in decisions if not (out/f'd{i}.npz').exists()}
    if not todo:
        return
    spec, requests = rerender.load(run, 'specification.json'), rerender.load(run, 'requests.json')
    with np.load(run/'native/physics_trace.npz') as archive:
        trace = {k: archive[k].copy() for k in rerender.TRACE_FIELDS}
    scene.init_genesis(spec['procedural_seed'])
    session = MissionSession(spec, out/'visual_meshes', policy_checkpoint=rerender.run_policy(run),
                             dynamics=rerender.run_dynamics(run), demo_textures=rerender.run_demo_textures(run))
    session.settle()
    checked = rerender.check_samples(session, trace, 0)
    recent = []
    last = max(todo)
    for tick, row in enumerate(requests):
        if tick % 5 == 0:
            frame = session.frames
            recent = (recent+[session.frame()[0]['image']['rgb']])[-4:]
            if frame in todo:
                i = todo[frame]
                d = plan[i]
                if d['measured_ns'] != session.sim_ns:
                    raise ValueError('decision time mismatch')
                snap = snapshots.capture(session)
                records = {name: list(getattr(session, name)) for name in snapshots.RECORD_LISTS}
                initial = session.samples[-1]['base_pose_world'].copy()
                Q = rotation_xyzw(initial[3:])
                motions, frames = [], []
                for tape in rerender_tapes(d['committed_prefix']):
                    snapshots.restore(session, snap)
                    for name, value in records.items():
                        setattr(session, name, list(value))
                    session.frames = frame+1
                    start, shots = len(session.samples), []
                    for k, command in enumerate(np.repeat(tape, 5, axis=0)):
                        session.step(command)
                        if k % 5 == 4:
                            shots.append(session.frame()[0]['image']['rgb'])
                    poses = np.stack([s['base_pose_world'] for s in session.samples[start:]])
                    horizon = poses[np.arange(49, 400, 50)]
                    xy = (horizon[:, :3]-initial[:3])@Q
                    yaw = [math.atan2(*(lambda R: (R[1, 0], R[0, 0]))(Q.T@rotation_xyzw(p[3:]))) for p in horizon]
                    motions.append(np.c_[xy[:, :2], yaw])
                    frames.append(np.stack(shots))
                snapshots.restore(session, snap)
                for name, value in records.items():
                    setattr(session, name, value)
                session.frames = frame+1
                np.savez_compressed(out/f'd{i}.npz', motion=np.asarray(motions, np.float32), frames=np.asarray(frames),
                                    history=np.stack(recent), decision=i, frame=frame)
                print(json.dumps(dict(decision=i, frame=frame)), flush=True)
                if frame == last:
                    return
        session.phase = 2
        session.step(row['requested_command'])
        checked = rerender.check_samples(session, trace, checked)


def rerender_tapes(prefix):
    """The six candidates' (8, 3) tapes as the oracle executes them (float32 round trip)."""
    from go2nav.controllers.oracle import encoded_tapes
    return encoded_tapes(prefix, False)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(',')])
