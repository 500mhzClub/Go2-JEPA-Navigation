"""Fixed-tape training collections: the robot executes a scripted command tape (no controller) in a registered scene,
and every 10 Hz frame is kept.

    python -m go2nav.data.collect COLLECTION CASE OUT_DIR

A collection is data/collections/<name>/plan.json: its scene specifications (one per case) and its command tape (one
command per 100-ms tick, held for the tick's five policy steps). Each case settles, then runs the tape in the mission
simulator (`go2nav.sim.session`: visible robot, both cameras, live depth noise with the case's layout index mod 4) and
stops early on a physical stop. Collections of this kind: full_heading (T5a), maze_view_training and
maze_view_transfer (T5b), c3v2_rest_turn (the C3 decoder).

A case folder holds rgb_NNNN.png and auxiliary_rgb_NNNN.png per frame, frames.json (frame, time, physics sample
index), physics_trace.npz, contact_events.json and result.json.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from go2nav import artifacts

REPO = Path(__file__).resolve().parents[2]
COLLECTIONS = REPO/'data/collections'


def plan(name):
    return json.loads((COLLECTIONS/name/'plan.json').read_text())


def collect(name, case, out):
    """Run one case into `out` (created). Returns the result record."""
    import cv2
    import torch
    from go2nav.eval.mission import persist, save
    from go2nav.sim import scene
    from go2nav.sim.session import MissionSession, PhysicalStop
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(4)
    collection = plan(name)
    spec, tape = collection['specifications'][case], collection['tape']
    out = Path(out)
    out.mkdir(parents=True)
    save(out/'specification.json', spec)
    started, stop = time.monotonic(), None
    scene.init_genesis(spec['procedural_seed'])
    session = MissionSession(spec, out/'visual_meshes', policy_checkpoint=artifacts.path('go2_locomotion_policy'),
                             noise_layout=case % 4)
    frames = []
    try:
        session.settle()
        for tick in range(len(tape)*5+1):
            if tick % 5 == 0:
                policy, _, _, _, auxiliary_rgb, measured = session.frame()
                index = len(frames)
                Image.fromarray(policy['image']['rgb']).save(out/f'rgb_{index:04d}.png')
                Image.fromarray(auxiliary_rgb['rgb']).save(out/f'auxiliary_rgb_{index:04d}.png')
                frames.append(dict(frame=index, measured_ns=measured, physical_sample_index=len(session.samples)-1))
            if tick == len(tape)*5:
                break
            session.phase = 2
            session.step(tape[tick//5])
    except PhysicalStop as error:
        stop = str(error)
    finally:
        persist(session, out)
        save(out/'frames.json', frames)
    result = dict(collection=name, case=case, status='COMPLETE' if stop is None else 'PHYSICAL_STOP', physical_stop=stop,
                  camera_frames=len(frames), disallowed_contact=any(bool(s['physics_contact']) for s in session.samples),
                  wall_s=time.monotonic()-started)
    save(out/'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('collection', choices=sorted(d.name for d in COLLECTIONS.iterdir() if (d/'plan.json').exists()))
    p.add_argument('case', type=int)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    import os
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('EGL_DEVICE_ID', '0')
    print(json.dumps(collect(a.collection, a.case, a.out)))


if __name__ == '__main__':
    main()
