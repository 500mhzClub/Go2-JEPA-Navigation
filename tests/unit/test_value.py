"""P22b's value of the state: the window, the planner's hook, the data's cost to go (docs/dynamic/PLAN.md)."""
import math

import numpy as np
import pytest
import torch

from go2nav.eval.mission import episode_spec
from go2nav.models import value as vm
from go2nav.planning.fields import Fields
from go2nav.sim import kinematic as kin

FROZEN = dict(GO2NAV_EXACT='1', GO2NAV_RECOVER='1', GO2NAV_WALKER_PRICE='300', GO2NAV_W_STILL='0')


def world(walkers=0, maze=12):
    from go2nav.eval import kinematic_missions as km
    spec, record = episode_spec('validation', maze, 0)
    placement = km.placement_for('validation', maze, 0, spec, record, walkers, 'rude')
    return kin.World(spec, record['home_se2_world'], placement), record


def test_the_window_puts_walkers_walls_and_the_distance_where_they_are_turned_with_the_robot():
    w, record = world()
    fields = Fields(w.geometry)
    walls = torch.as_tensor(vm.wall_grid(fields))[None]
    distance = torch.as_tensor(vm.distance_grid(fields, record['beacon_xy_world']))[None]
    low = torch.as_tensor(fields.low, dtype=torch.float32)
    robot = np.asarray(w.xy, float)
    own = fields.value(fields.static(record['beacon_xy_world']), robot[None])
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))
    for yaw in (0., math.pi/2):
        ahead = robot+np.array([math.cos(yaw), math.sin(yaw)])
        window = vm.draw(walls, distance, low, t(robot[None]), t([yaw]), t(own), t(ahead[None, None]),
                         t([[[.3, 0.]]]), t(np.full((1, 1, vm.MEMORY), .5)), torch.tensor([[True]]))
        assert window.shape == (1, vm.CHANNELS, vm.SIZE, vm.SIZE)
        presence = window[0, 2]
        assert presence[39:41, 31:33].min() == 1 and presence.sum() < 30         # 1 m ahead, on the centre line
        forward, left = window[0, 3, 39, 31], window[0, 4, 39, 31]
        assert torch.allclose(torch.stack((forward, left)),
                              torch.tensor([.3*math.cos(yaw), -.3*math.sin(yaw)])/vm.SPEED_M_S, atol=1e-5)
        assert torch.allclose(window[0, 5:, 39, 31], torch.full((vm.MEMORY,), .5))
        assert abs(float(window[0, 1, 31:33, 31:33].mean())) < .02              # the robot's own distance: zero
        centre = float(w.geometry.wall_distance(robot[None])[0])
        assert abs(float(window[0, 0, 31:33, 31:33].mean())-min(centre, vm.WALL_CLIP_M)) < .1


def test_an_untrained_value_leaves_the_frozen_rule_as_it_is(tmp_path, monkeypatch):
    for k, v in FROZEN.items():
        monkeypatch.setenv(k, v)
    from go2nav.planning.sampling import Planner
    vm.save(tmp_path/'value.pt', vm.ValueNet())
    w, record = world(walkers=4)
    for _ in range(40):
        w.step((.2, 0., 0.))
    snapshot, target = w.snapshot(), record['beacon_xy_world']
    for model in ('true', 'ring_velocity'):
        plain = Planner(w.geometry, horizon_s=3.2, terminal='walkers', model=model)
        valued = Planner(w.geometry, horizon_s=3.2, terminal='walkers', model=model, value=tmp_path/'value.pt')
        for p in (plain, valued):
            if hasattr(p.model, 'observe'):
                p.model.observe(0., w.scan(), w.xy, w.yaw, w.geometry)
        valued.value.observe(0., w.scan(), w.xy, w.yaw, w.geometry, tracker=getattr(valued.model, 'tracker', None))
        requested, _ = plain.proposals(snapshot, target)
        a, terms_a = plain.costs(snapshot, target, requested)
        b, terms_b = valued.costs(snapshot, target, requested)
        assert np.allclose(a, b) and (terms_a['breached'] == terms_b['breached']).all()


def test_the_value_reads_each_slot_where_the_model_leaves_its_walker(tmp_path, monkeypatch):
    from go2nav.planning.value import Value
    vm.save(tmp_path/'value.pt', vm.ValueNet())
    w, record = world()
    value = Value(tmp_path/'value.pt', device='cpu')
    value.bind(Fields(w.geometry))
    value.walkers.now = dict(xy=np.array([[1., 1.], [0., 0.]]), velocity=np.zeros((2, 2)), memory=np.zeros((2, 4)),
                             present=np.array([True, False]))
    seen = {}
    monkeypatch.setattr(vm, 'draw', lambda *args: seen.update(xy=args[6].numpy(), v=args[7].numpy()) or
                        torch.zeros((len(args[3]), vm.CHANNELS, vm.SIZE, vm.SIZE)))
    walkers = np.zeros((2, 5, 1, 2))
    walkers[:, 0] = [1.05, 1.]                                                  # the model's walker at the slot's
    walkers[0, -2:, 0], walkers[1, -2:, 0] = [[2., 1.], [2.03, 1.]], [[1., 2.], [1., 2.04]]
    value.correction(record['beacon_xy_world'], np.zeros((2, 2)), np.zeros(2), walkers, np.zeros(2))
    assert np.allclose(seen['xy'][:, 0], [[2.03, 1.], [1., 2.04]]) and np.allclose(seen['v'][:, 0], [[.3, 0.], [0., .4]])


def test_the_cost_to_go_counts_to_the_leg_s_arrival_or_to_the_failure_and_its_price():
    from go2nav.data.value_missions import cost_to_go
    out = cost_to_go([0, 4, 8, 12], [0, 0, 1, 1], [dict(phase='OUTBOUND', frame=6)], 20, 300.)
    assert np.allclose(out, [.6, .2, 301.2, 300.8])
    out = cost_to_go([0, 8], [0, 1], [dict(phase='OUTBOUND', frame=6), dict(phase='RETURN', frame=30)], 31, 300.)
    assert np.allclose(out, [.6, 2.2])


def test_exploring_changes_only_the_tape_taken(monkeypatch):
    for k, v in FROZEN.items():
        monkeypatch.setenv(k, v)
    from go2nav.data import value_missions as vmi
    from go2nav.planning.sampling import Planner
    w, record = world(walkers=2)
    snapshot, target = w.snapshot(), record['beacon_xy_world']
    plain = Planner(w.geometry, horizon_s=6.4, terminal='walkers', model='true', seed=3)
    monkeypatch.setattr(vmi, 'EXPLORE', 0.)
    never = vmi.exploring(w.geometry, horizon_s=6.4, terminal='walkers', model='true', seed=3, explore_seed=1)
    assert np.array_equal(plain.plan(snapshot, target)[0], never.plan(snapshot, target)[0]) and not never.explored
    monkeypatch.setattr(vmi, 'EXPLORE', 1.)
    always = vmi.exploring(w.geometry, horizon_s=6.4, terminal='walkers', model='true', seed=3, explore_seed=1)
    requested, _ = always.proposals(snapshot, target)
    total, terms = always.costs(snapshot, target, requested)
    k = always.choose(total, terms)
    assert always.explored and not terms['breached'][k]


@pytest.mark.skipif(not __import__('go2nav.data.value_missions', fromlist=['x']).mazes_root().exists(),
                    reason='needs the corpus layouts (GO2NAV_MAZES_ROOT)')
def test_a_branched_mission_repeats_round_0_and_its_branches_add_up(monkeypatch):
    for k, v in FROZEN.items():
        monkeypatch.setenv(k, v)
    from go2nav.data import value_branches as vb
    from go2nav.data import value_missions as vmi
    job = vmi.job(2, 5000, 500)
    meta, arrays = vmi.run(job)
    frames = [int(arrays['frame'][3]), int(arrays['frame'][40])]
    ended, last, moments = vb.run_moments(job, frames, vmi.mazes_root(), 300.)
    assert (ended, last) == (meta['exit'], meta['frames']) and [m[0] for m in moments] == frames
    for frame, leg, names, a in moments:
        assert names[1] == 'hold' and names[2].startswith('target@') and len(names) == vb.TAPES
        reached = a['cut_reached']
        assert np.allclose(a['cost_moment'][reached], a['cost_cut'][reached]+vb.CUT_S)
        assert (a['cost_moment'] > 0).all() and np.isfinite(a['cut_base'][reached]).all()
        assert np.isclose(a['rule'][0], a['rule'].min())                   # the rule's choice first


@pytest.mark.skipif(not __import__('go2nav.data.value_missions', fromlist=['x']).mazes_root().exists(),
                    reason='needs the corpus layouts (GO2NAV_MAZES_ROOT)')
def test_a_data_mission_keeps_every_plan_s_state_and_its_cost_to_go(monkeypatch):
    for k, v in FROZEN.items():
        monkeypatch.setenv(k, v)
    from go2nav.data import value_missions as vmi
    meta, arrays = vmi.run(vmi.job(1, 5000, 500), frames=400)
    count = meta['plans']
    assert count > 50 and all(len(v) == count for v in arrays.values())
    assert (np.diff(arrays['frame']) > 0).all() and (arrays['cost'] >= 0).all()
    assert arrays['slots_xy'].shape == (count, 6, 2) and arrays['memory'].shape == (count, 6, 4)
    assert np.isfinite(arrays['base']).all() and np.isfinite(arrays['own']).all()
