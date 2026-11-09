"""Range scans cast offline for recorded corpus units (data.range_backfill; docs/dynamic/PLAN.md P14 item 1)."""
import json

import numpy as np

from go2nav.data import range_backfill
from go2nav.sim.range_sensor import RangeSensor

SPEC = dict(layout_index=5, geometry=dict(wall_boxes=[dict(centre_xyz=[3., 0., .7], size_xyz=[.08, 4., 1.4], yaw_rad=0.)]))


def corpus(tmp_path, monkeypatch, frames=6):
    """A one-unit corpus: the robot walks +x with a walker 1 m to its left."""
    unit = tmp_path/'corpus/units/lewm_train/0005_standoff'
    unit.mkdir(parents=True)
    pose = np.zeros((frames, 7))
    pose[:, 0], pose[:, 2], pose[:, 6] = np.linspace(0., .5, frames), .30, 1.     # x y z qx qy qz qw
    np.savez(unit/'steps.npz', pose=pose)
    walkers = np.zeros((frames, 1, 2))
    walkers[:, 0] = pose[:, None, :2][:, 0]+[0., 1.]
    np.savez(unit/'dynamic.npz', obstacles_xy=walkers)
    (unit/'meta.json').write_text(json.dumps(dict(set='test_set', maze=5, obstacles=dict(count=1, placement=[dict(height_m=.9)]))))
    (tmp_path/'corpus/index_12.json').write_text(json.dumps(dict(units=[dict(path='units/lewm_train/0005_standoff')])))

    class Set:
        def __init__(self, name, root):
            assert name == 'test_set'

        def maze(self, index):
            assert index == 5
            return SPEC
    monkeypatch.setattr(range_backfill, 'MazeSet', Set)
    return tmp_path/'corpus', pose, walkers


def test_scans_are_the_sensor_at_each_recorded_pose(tmp_path, monkeypatch):
    root, pose, walkers = corpus(tmp_path, monkeypatch)
    ranges, valid = range_backfill.unit_scans(root, dict(path='units/lewm_train/0005_standoff'))
    sensor = RangeSensor(SPEC)
    for k in range(len(pose)):
        expected, ok = sensor.scan(pose[k, :3], (1., 0., 0., 0.), walkers[k], [.9], frame=k)
        assert np.array_equal(valid[k], ok) and np.allclose(ranges[k], expected, atol=1e-2)


def test_backfill_writes_once_beside_the_corpus(tmp_path, monkeypatch):
    root, _, _ = corpus(tmp_path, monkeypatch)
    job = (str(root), str(tmp_path/'out'), dict(path='units/lewm_train/0005_standoff'))
    assert range_backfill.backfill_unit(job)[1] == 'written'
    assert range_backfill.backfill_unit(job)[1] == 'kept'
    with np.load(tmp_path/'out/units/lewm_train/0005_standoff/range.npz') as scans:
        assert scans['ranges_m'].shape == (6, 720) and scans['valid'].any()
    assert not (root/'units/lewm_train/0005_standoff/range.npz').exists()
