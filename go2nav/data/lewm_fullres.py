"""Re-collect LeWM corpus units at full resolution, for encoders that need the camera's 640 x 480 frames (C3's V-JEPA,
under the shared readout protocol, `go2nav.training.readout_protocol vjepa-cache`). The corpus keeps 224 x 168 frames.

    python -m go2nav.data.lewm_fullres collect PROTOCOL.json CORPUS OUT_ROOT [--workers 10]
    python -m go2nav.data.lewm_fullres unit CORPUS OUT_ROOT ROLE MAZE KIND     # one unit (collect runs each in a process)

Collection is deterministic (CPU physics, keyed seeds), so each unit is run again exactly as `lewm_corpus collect_unit`
ran it, into OUT_ROOT (the corpus is not touched), with each primary frame also written as
OUT_ROOT/units/<role>/<unit>/full/NNNN.png. The rerun is then checked against the corpus (verify.json) and marked
`.verified`: every step record and pose must equal the stored ones, and every resized frame too, except that at most
`TOLERATED_FRAMES` frame may differ in fewer than `TOLERATED_PIXELS` pixels. Three of the 600 readout_v1 units have
such a frame (6-45 of 37,632 pixels, at most 12 of 255; two re-collections reproduce it exactly), a rendering
difference from the corpus's own collection rather than a different trajectory; verify.json records it. The resized
frames are then deleted (the corpus holds them). `collect` runs the protocol's units, skipping verified ones, and lists
failures for the next run to retry.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from go2nav.data import lewm_corpus

TOLERATED_FRAMES, TOLERATED_PIXELS = 1, 64


class FullRecorder(lewm_corpus.Recorder):
    """The corpus recorder that also writes each primary frame at full resolution."""
    folder = None

    def frame(self, policy, measured_ns, sample):
        FullRecorder.folder.mkdir(exist_ok=True)
        Image.fromarray(np.asarray(policy['image']['rgb'])).save(FullRecorder.folder/f'{len(self.frames):04d}.png',
                                                                 compress_level=1)
        super().frame(policy, measured_ns, sample)


def compare(folder, stored):
    """The verification record of a re-collected unit against the corpus's."""
    a, b = np.load(folder/'frames.npy'), np.load(stored/'frames.npy', mmap_mode='r')
    with np.load(folder/'steps.npz') as new, np.load(stored/'steps.npz') as old:
        steps_equal = new.files == old.files and all(np.array_equal(new[k], old[k]) for k in old.files)
    report = dict(frames=int(len(a)), stored_frames=int(len(b)), steps_equal=bool(steps_equal))
    if a.shape != b.shape:
        return report | dict(equal=False, accepted=False)
    pixels = (a != b).any(axis=-1).sum(axis=(1, 2))
    differing = np.flatnonzero(pixels)
    report |= dict(equal=bool(steps_equal and not len(differing)), differing_frames=int(len(differing)))
    if len(differing):
        report['differences'] = [dict(frame=int(i), pixels=int(pixels[i]),
                                      max_abs_diff=int(np.abs(a[i].astype(int)-b[i].astype(int)).max())) for i in differing]
    report['accepted'] = bool(steps_equal and len(differing) <= TOLERATED_FRAMES
                              and all(pixels[i] < TOLERATED_PIXELS for i in differing))
    return report


def collect_unit(corpus, out_root, role, maze, kind):
    """Re-collect, compare and mark one unit (see the module docstring)."""
    corpus, out_root, maze = Path(corpus), Path(out_root), int(maze)
    name = lewm_corpus.unit_name(maze, kind)
    folder = out_root/'units'/role/name
    if (folder/'.verified').exists():
        return
    (out_root/'mazes').mkdir(parents=True, exist_ok=True)
    link = out_root/'mazes'/lewm_corpus.SET
    if not link.exists():
        link.symlink_to((corpus/'mazes'/lewm_corpus.SET).resolve())
    FullRecorder.folder = folder.with_name(name+'.tmp')/'full'
    original = lewm_corpus.Recorder
    lewm_corpus.Recorder = FullRecorder
    try:
        meta = lewm_corpus.collect_unit(out_root, role, maze, kind)
    finally:
        lewm_corpus.Recorder = original
    report = dict(unit=f'{role}/{name}', **compare(folder, corpus/'units'/role/name), wall_s=meta['wall_s'])
    (folder/'verify.json').write_text(json.dumps(report)+'\n')
    (folder/'frames.npy').unlink()
    if not report['accepted']:
        raise SystemExit(f'unit differs from the corpus: {report}')
    (folder/'.verified').touch()
    print(json.dumps(report), flush=True)


def collect(protocol, corpus, out_root, *, workers=10):
    """Every unit of a readout protocol not yet verified, `workers` processes at a time; returns the failures."""
    units = json.loads(Path(protocol).read_text())['corpus_units']
    out_root = Path(out_root)
    todo = [u for u in units if not (out_root/u['path']/'.verified').exists()]
    print(json.dumps(dict(units=len(units), todo=len(todo))), flush=True)
    logs = out_root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    started, failed = time.time(), []

    def one(unit):
        _, role, name = unit['path'].split('/')
        maze, kind = name.split('_')
        with open(logs/f'{role}_{name}.log', 'w') as log:
            code = subprocess.run([sys.executable, '-W', 'ignore', '-m', __spec__.name, 'unit', str(corpus), str(out_root),
                                   role, str(int(maze)), kind], stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            failed.append(unit['path'])
        return unit['path'], code
    with ThreadPoolExecutor(workers) as pool:
        for k, (path, code) in enumerate(pool.map(one, todo), 1):
            if k % 20 == 0 or code:
                print(json.dumps(dict(done=k, of=len(todo), last=path, rc=code, wall_s=round(time.time()-started))),
                      flush=True)
    print(json.dumps(dict(finished=len(todo)-len(failed), failed=failed, wall_s=round(time.time()-started))), flush=True)
    return failed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('protocol', type=Path)
    c.add_argument('corpus', type=Path)
    c.add_argument('out_root', type=Path)
    c.add_argument('--workers', type=int, default=10)
    u = sub.add_parser('unit')
    for arg in ('corpus', 'out_root', 'role', 'maze', 'kind'):
        u.add_argument(arg)
    a = p.parse_args()
    if a.command == 'collect':
        sys.exit(1 if collect(a.protocol, a.corpus, a.out_root, workers=a.workers) else 0)
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('EGL_DEVICE_ID', '0')
    collect_unit(a.corpus, a.out_root, a.role, a.maze, a.kind)


if __name__ == '__main__':
    main()
