"""Re-collect one LeWM corpus unit at full resolution, for encoders that need the camera's 640 x 480 frames (C3's
V-JEPA). Collection is deterministic (CPU physics, keyed seeds), so the unit is run again exactly as
`go2nav.data.lewm_corpus collect_unit` ran it, into OUT_ROOT (the corpus is not touched), with each primary frame also
written as OUT_ROOT/units/<role>/<unit>/full/NNNN.png. The rerun is then checked against the corpus: every resized
frame and every pose must equal the stored ones, or the unit is rejected.

    python tools/readout_protocol/fullres_unit.py CORPUS OUT_ROOT ROLE MAZE KIND
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from go2nav.data import lewm_corpus


class FullRecorder(lewm_corpus.Recorder):
    folder = None

    def frame(self, policy, measured_ns, sample):
        rgb = np.asarray(policy['image']['rgb'])
        FullRecorder.folder.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(FullRecorder.folder/f'{len(self.frames):04d}.png', compress_level=1)
        super().frame(policy, measured_ns, sample)


def main(corpus, out_root, role, maze, kind):
    corpus, out_root, maze = Path(corpus), Path(out_root), int(maze)
    name = lewm_corpus.unit_name(maze, kind)
    folder = out_root/'units'/role/name
    if (folder/'.verified').exists():
        return
    (out_root/'mazes').mkdir(parents=True, exist_ok=True)
    link = out_root/'mazes'/'lewm_sets_v1'
    if not link.exists():
        link.symlink_to((corpus/'mazes'/'lewm_sets_v1').resolve())
    partial = folder.with_name(name+'.tmp')
    FullRecorder.folder = partial/'full'
    original = lewm_corpus.Recorder
    lewm_corpus.Recorder = FullRecorder
    try:
        meta = lewm_corpus.collect_unit(out_root, role, maze, kind)
    finally:
        lewm_corpus.Recorder = original
    stored = corpus/'units'/role/name
    a, b = np.load(folder/'frames.npy'), np.load(stored/'frames.npy', mmap_mode='r')
    pa, pb = np.load(folder/'steps.npz')['pose'], np.load(stored/'steps.npz')['pose']
    ok = a.shape == b.shape and np.array_equal(a, b) and np.array_equal(pa, pb)
    report = dict(unit=f'{role}/{name}', frames=int(len(a)), stored_frames=int(len(b)), equal=bool(ok),
                  differing_frames=int((a != b).any(axis=(1, 2, 3)).sum()) if a.shape == b.shape else None,
                  wall_s=meta['wall_s'])
    (folder/'verify.json').write_text(json.dumps(report)+'\n')
    (folder/'frames.npy').unlink()
    if not ok:
        raise SystemExit(f'unit differs from the corpus: {report}')
    (folder/'.verified').touch()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:6])
