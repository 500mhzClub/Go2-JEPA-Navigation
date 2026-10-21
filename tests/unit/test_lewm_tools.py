"""The LeWM tooling around the models: the corpus's motion labels, the readout protocol's resumable training, the
full-resolution re-collection check, the C5 baseline package, the deadlock replay's bookkeeping and the arm registry."""
import json
import math

import numpy as np
import pytest
import torch

from go2nav.controllers import registry
from go2nav.data import lewm_fullres
from go2nav.data.lewm_corpus import body_motion, planar_yaw
from go2nav.eval import stuck_replay
from go2nav.models.lewm import LatentDirectPredictor, LatentMotionDecoder, load_heads
from go2nav.training import lewm_readout, readout_protocol


def quaternion(yaw):
    return [0., 0., math.sin(yaw/2), math.cos(yaw/2)]  # x, y, z, w


def test_body_motion_is_the_motion_in_the_start_frame():
    pose = np.array([[1., 2., .3, *quaternion(math.pi/2)],      # facing +y
                     [1., 2.5, .3, *quaternion(math.pi/2+.2)],  # 0.5 m ahead, 0.2 rad left
                     [0., 2., .3, *quaternion(-math.pi+.1)]])   # 1 m to its right, turned across the wrap
    assert np.allclose(planar_yaw(pose[:, 3:7]), [math.pi/2, math.pi/2+.2, -math.pi+.1])
    motion = body_motion(pose, np.array([0, 0]), np.array([1, 2]))
    assert np.allclose(motion, [[.5, 0., .2], [0., 1., math.pi/2+.1]])  # the yaw change wrapped to (-pi, pi]
    # index arrays of any shape: per window, several horizons
    assert body_motion(pose, np.array([[0], [1]]), np.array([[1, 2], [2, 2]])).shape == (2, 2, 3)


def test_readout_training_resumes_exactly(tmp_path, monkeypatch):
    """A run stopped after a checkpoint and rerun ends with the same weights as one run straight through."""
    recipe = dict(readout_protocol.RECIPE, updates=6, batch=8)
    monkeypatch.setattr(readout_protocol, 'CHECKPOINT_EVERY', 3)
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(40, 4, 8, generator=g), torch.randn(40, 3, generator=g)

    def fresh():
        torch.manual_seed(1)
        return LatentMotionDecoder(8, torch.zeros(3), torch.ones(3))
    batch = lambda m, idx: (m.normalized(x[idx], x[idx, -1]), y[idx])
    straight, _ = readout_protocol.train(fresh(), batch, 40, 5, 'cpu', recipe, checkpoint=tmp_path/'a.ckpt')
    assert not (tmp_path/'a.ckpt').exists()  # removed when the head is done
    save = readout_protocol.atomic_save

    def stop_after_save(value, path):
        save(value, path)
        raise KeyboardInterrupt
    monkeypatch.setattr(readout_protocol, 'atomic_save', stop_after_save)
    with pytest.raises(KeyboardInterrupt):
        readout_protocol.train(fresh(), batch, 40, 5, 'cpu', recipe, checkpoint=tmp_path/'b.ckpt')
    monkeypatch.setattr(readout_protocol, 'atomic_save', save)
    resumed, _ = readout_protocol.train(fresh(), batch, 40, 5, 'cpu', recipe, checkpoint=tmp_path/'b.ckpt')
    assert all(torch.equal(a, b) for a, b in zip(straight.state_dict().values(), resumed.state_dict().values()))


def write_unit(folder, frames, pose):
    folder.mkdir(parents=True)
    np.save(folder/'frames.npy', frames)
    np.savez(folder/'steps.npz', pose=pose)


@pytest.mark.parametrize('pixels, accepted', [(0, True), (lewm_fullres.TOLERATED_PIXELS-1, True),
                                              (lewm_fullres.TOLERATED_PIXELS, False)])
def test_fullres_check_tolerates_one_slightly_different_frame(tmp_path, pixels, accepted):
    frames, pose = np.zeros((3, 8, 16, 3), np.uint8), np.zeros((3, 7))
    write_unit(tmp_path/'stored', frames, pose)
    rerun = frames.copy()
    rerun[1].reshape(-1, 3)[:pixels] = 9
    write_unit(tmp_path/'rerun', rerun, pose)
    report = lewm_fullres.compare(tmp_path/'rerun', tmp_path/'stored')
    assert report['accepted'] is accepted and report['equal'] is (pixels == 0)


def test_fullres_check_rejects_a_different_trajectory(tmp_path):
    frames, pose = np.zeros((3, 8, 16, 3), np.uint8), np.zeros((3, 7))
    write_unit(tmp_path/'stored', frames, pose)
    write_unit(tmp_path/'rerun', frames, pose+1e-9)
    assert not lewm_fullres.compare(tmp_path/'rerun', tmp_path/'stored')['accepted']


def test_package_takes_c5_from_the_base_readout_and_c6_from_the_fit(tmp_path):
    torch.manual_seed(0)
    base = LatentMotionDecoder(8, torch.ones(3), 2*torch.ones(3))
    c5, c6 = LatentMotionDecoder(8, torch.zeros(3), torch.ones(3)), LatentDirectPredictor(8, torch.zeros(3), torch.ones(3))
    torch.save(dict(rollout=base.state_dict(), true=base.state_dict(), latent=8, world_model_sha256='w'), tmp_path/'readout.pt')
    torch.save(dict(c5=c5.state_dict(), c6=c6.state_dict(), latent=8, target_mean=np.zeros(3), target_scale=np.ones(3),
                    world_model_sha256='w'), tmp_path/'fit.pt')
    packaged_c5, packaged_c6 = load_heads(lewm_readout.package(tmp_path/'readout.pt', tmp_path/'fit.pt', tmp_path/'out'))
    assert all(torch.equal(a, b) for a, b in zip(packaged_c5.state_dict().values(), base.state_dict().values()))
    assert all(torch.equal(a, b) for a, b in zip(packaged_c6.state_dict().values(), c6.state_dict().values()))
    assert (tmp_path/'out'/'lineage.json').exists()
    torch.save(dict(c5=c5.state_dict(), c6=c6.state_dict(), latent=8, world_model_sha256='other'), tmp_path/'other.pt')
    with pytest.raises(ValueError):
        lewm_readout.package(tmp_path/'readout.pt', tmp_path/'other.pt', tmp_path/'out2')


def decision(action, modes):
    return dict(action=action, selection=dict(memory_forecast_candidates=[
        dict(action=a, clearance_check_mode=m) for a, m in zip(stuck_replay.ACTIONS, modes)]))


def test_stuck_decisions_and_sampling():
    blocked = ['CLEAR']+['BLOCKED']*5
    planning = [decision('hold', blocked), decision('forward', blocked), decision('hold', ['CLEAR']*6), dict(action='hold')]
    assert stuck_replay.stuck_decisions(planning) == [0]
    stuck = list(range(100, 160))+[200, 260, 500]
    assert stuck_replay.sampled(stuck) == list(range(100, 110))+[125, 150, 200, 500]  # 25, 50, 100, 200+ later
    assert stuck_replay.parse_models(['a=w.pt,f.pt']) == dict(a=('w.pt', 'f.pt'))
    with pytest.raises(ValueError):
        stuck_replay.parse_models(['a=w.pt'])


def test_stuck_list_keeps_failures_the_reference_passed(tmp_path):
    def cohort(name, outcomes):
        rows = []
        for maze, ok in outcomes.items():
            assignment = f'{name}_C5_set_{maze}_ep0'
            (tmp_path/name/'runs'/assignment).mkdir(parents=True)
            (tmp_path/name/'runs'/assignment/'planning.json').write_text(json.dumps([decision('hold', ['CLEAR']+['BLOCKED']*5)]))
            rows.append(dict(assignment=assignment, round_trip=ok, contacts=0))
        (tmp_path/name/'result.json').write_text(json.dumps(dict(rows=rows)))
    cohort('new', {1: False, 2: False, 3: True})
    cohort('old', {1: True, 2: False, 3: True})
    entries = stuck_replay.list_stuck([f'{tmp_path/"new"}:{tmp_path/"old"}'], tmp_path/'list.json')
    assert [e['name'] for e in entries] == ['new_1_ep0'] and entries[0]['stuck_idx'] == [0]
    assert len(stuck_replay.list_stuck([str(tmp_path/'new')], tmp_path/'all.json')) == 2


def test_arm_registry_covers_every_arm():
    assert set(registry.MEMORY_GIB) == set(registry.ARMS)
    assert set(registry.GPU_LANES) <= set(registry.ARMS)
