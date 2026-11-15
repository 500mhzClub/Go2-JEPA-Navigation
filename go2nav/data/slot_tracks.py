"""D001: matched Genesis units and sequential frozen-s0 observations for P26.

prepare is CPU data conversion. export requires the idle dedicated GPU. Source
walker labels are copied for later scoring; the track extractor never reads them.
"""
import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from go2nav.data import genesis_units, object_tracks, ring_corpus
from go2nav.eval import slot_measures
from go2nav.perception import ring_walkers as rw
from go2nav.perception.range import AZIMUTHS, MOUNT_BODY_M
from go2nav.research import atomic_json, budget_remaining, digest, gpu_run, provenance, require_record, require_training_slot
from go2nav.sim.obstacles import RADIUS_M


def positions(columns, ranges, valid, pose, wall_clearance, image_columns):
    """Mask-column bearings plus nearest valid return; no dynamic labels or track association."""
    result = np.full((len(columns), 2), np.nan, np.float32)
    for k, cols in enumerate(columns):
        if not cols:
            continue
        angles = [slot_measures.bearing_of_column(c) for c in cols]
        half = .5*np.deg2rad(78.323/image_columns)
        beams = valid & (AZIMUTHS >= min(angles)-half) & (AZIMUTHS <= max(angles)+half)
        if not beams.any():
            continue
        j = int(np.flatnonzero(beams)[np.argmin(ranges[beams])])
        radius = float(ranges[j])+RADIUS_M
        body = np.array([MOUNT_BODY_M[0]+radius*np.cos(AZIMUTHS[j]),
                         MOUNT_BODY_M[1]+radius*np.sin(AZIMUTHS[j])])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        point = pose[:2]+np.array([c*body[0]-s*body[1], s*body[0]+c*body[1]])
        if wall_clearance(point[None])[0] > rw.MARGIN_M:
            result[k] = point
    return result


class TrackSeries:
    """Slot identity persists until absence or a >0.5m jump; velocity never crosses a break."""
    def __init__(self, slots=6):
        self.ids = np.full(slots, -1, np.int32)
        self.previous = np.full((slots, 2), np.nan)
        self.histories = [deque() for _ in range(slots)]
        self.next_id = 0
        self.memory = rw.Memory(slots)

    def update(self, frame, xy, robot_xy):
        xy = np.asarray(xy, np.float32)
        seen = np.isfinite(xy).all(-1)
        velocity = np.zeros_like(xy)
        t = frame*.1
        for k, history in enumerate(self.histories):
            if not seen[k]:
                self.ids[k] = -1
                history.clear()
                continue
            if self.ids[k] < 0 or np.linalg.norm(xy[k]-self.previous[k]) > .5:
                self.ids[k], self.next_id = self.next_id, self.next_id+1
                history.clear()
            history.append((t, xy[k].copy()))
            while t-history[0][0] > rw.WINDOW_S:
                history.popleft()
            times = np.array([v[0] for v in history])
            if times[-1]-times[0] >= rw.MIN_SPAN_S:
                velocity[k] = np.polyfit(times-times[-1], np.array([v[1] for v in history]), 1)[0]
        memory = self.memory.update(xy, seen, self.ids, velocity, robot_xy)
        self.previous = xy.copy()
        return dict(xy=xy.copy(), seen=seen, id=self.ids.copy(), velocity=velocity, memory=memory.copy())


def prepare(corpus, ranges, features, out, research_id, workers=8):
    require_record(research_id)
    index = json.loads((Path(features)/'index.json').read_text())
    for role, count in (('lewm_train', 600), ('lewm_val', 100)):
        if len(index['units'][role]) != count:
            raise ValueError('D001 requires the original 600/100 feature-cache unit selection')
    out = Path(out)
    manifest = provenance(research_id, dict(stage='prepare'), [Path(features)/'index.json'])
    for role in ('lewm_train', 'lewm_val'):
        genesis_units.build(corpus, ranges, out, role, workers, units=index['units'][role])
    ring_corpus.pack(out, workers)
    atomic_json(out/'prepare_manifest.json', manifest)
    atomic_json(out/'index.json', dict(research_id=research_id, features_index_sha256=digest(Path(features)/'index.json'),
                units={s: [json.loads(p.read_text())['unit'] for p in sorted((out/s).glob('*.json'))]
                       for s in ('train', 'val')}))


def export(corpus, features, checkpoint, mazes, research_id):
    require_record(research_id)
    require_training_slot()
    corpus, features = Path(corpus), Path(features)
    index = json.loads((features/'index.json').read_text())
    source = json.loads((corpus/'index.json').read_text())
    if source['features_index_sha256'] != digest(features/'index.json'):
        raise ValueError('Prepared units do not match feature selection')
    config = dict(research_id=research_id, checkpoint_sha256=digest(checkpoint),
                  index_sha256=digest(corpus/'index.json'), features_sha256=digest(features/'index.json'))
    old = corpus/'tracks_config.json'
    if old.exists() and json.loads(old.read_text()) != config:
        raise ValueError('Existing track artifacts have different parents')
    atomic_json(old, config)
    from go2nav.models.slots import load_model
    from go2nav.planning.fields import Fields
    from go2nav.planning.wall_raster import WallRaster
    from go2nav.sim import kinematic as kin
    with gpu_run(corpus/'track_run', research_id, config, [Path(checkpoint), features/'index.json']) as manifest:
        torch.cuda.set_per_process_memory_fraction(.80)
        model = load_model(checkpoint, 'cuda:0')
        if model.config.slots != 6 or model.config.patches != np.prod(index['grid']):
            raise ValueError('s0 dimensions do not match the six-slot feature cache')
        slot_measures.set_grid(*index['grid'])
        (corpus/'objects'/'rasters').mkdir(parents=True, exist_ok=True)
        files = []
        for split in ('train', 'val'):
            for key in ('objects', 'engineered'):
                (corpus/key/split).mkdir(parents=True, exist_ok=True)
            for meta_path in sorted((corpus/split).glob('*.json')):
                if budget_remaining(manifest) <= 0:
                    atomic_json(corpus/'track_run'/'result.json', dict(research_id=research_id, status='budget_incomplete'))
                    return
                meta = json.loads(meta_path.read_text())
                target, engineered = corpus/'objects'/split/meta_path.with_suffix('.npz').name, \
                    corpus/'engineered'/split/meta_path.with_suffix('.npz').name
                files.extend((target, engineered))
                if target.exists() and engineered.exists():
                    continue
                # Engineered observations use scan/pose/layout only; labels are never consulted.
                tracks, info = object_tracks.track_episode(meta_path.with_suffix('.npz'), meta, mazes)
                geometry = kin.Geometry(info['spec'])
                fields = Fields(geometry)
                WallRaster.of(geometry).save(corpus/'objects'/'rasters'/f"{meta['maze']:04d}.npz")
                with np.load(meta_path.with_suffix('.npz')) as arrays:
                    ranges, poses = arrays['ranges'].astype(np.float32), arrays['pose']
                features_file = features/('lewm_'+split)/(Path(meta['unit']).name+'.npy')
                x = np.load(features_file, mmap_mode='r')
                if len(x) != len(poses):
                    raise ValueError('Feature and observation frame indices disagree')
                torch.manual_seed(2026111508+meta['index']+(100000 if split == 'val' else 0))
                columns = slot_measures.slot_columns(model, x, 'cuda:0')
                series, rows = TrackSeries(), []
                for t, cols in enumerate(columns):
                    xy = positions(cols, ranges[t], ranges[t] > 0, poses[t], fields.wall_clearance, index['grid'][1])
                    rows.append(series.update(t, xy, poses[t, :2]))
                arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
                for path, values in ((target, arrays), (engineered, tracks)):
                    partial = path.with_suffix('.partial.npz')
                    np.savez_compressed(partial, **values)
                    partial.replace(path)
                print(split, meta['index'], 'tracks exported', flush=True)
        atomic_json(corpus/'tracks_index.json', dict(config=config,
                    files={str(p.relative_to(corpus)): digest(p) for p in files}))
        atomic_json(corpus/'track_run'/'result.json', dict(research_id=research_id,
                    tracks_sha256=digest(corpus/'tracks_index.json'), status='awaiting_literature_review'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'export'])
    for name in ('corpus', 'features', 'research'):
        p.add_argument('--'+name, required=True)
    for name in ('ranges', 'out', 'checkpoint', 'mazes'):
        p.add_argument('--'+name)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare(a.corpus, a.ranges, a.features, a.out, a.research, a.workers)
    else:
        export(a.corpus, a.features, a.checkpoint, a.mazes, a.research)


if __name__ == '__main__':
    main()
