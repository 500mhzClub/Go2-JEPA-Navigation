"""Per-mission measures (eval.metrics; docs/dynamic/PLAN.md P13 item 5)."""
import json

import numpy as np

from go2nav.eval import metrics


def write_run(folder, base_xy, walkers_xy, contact=None):
    """A run folder at 500 Hz: the base path (n, 2) and the walkers (n, w, 2)."""
    native = folder/'native'
    native.mkdir(parents=True)
    n = len(base_xy)
    t = np.arange(n)*.002
    pose = np.zeros((n, 7))
    pose[:, :2], pose[:, 6] = base_xy, 1.
    np.savez(native/'physics_trace.npz', timestamp_s=t, base_pose_world=pose)
    np.savez(native/'obstacle_trace.npz', timestamp_s=t, xy=np.asarray(walkers_xy, float))
    events = [dict(timestamp_s=float(t[-1]), disallowed_contacts=[] if contact is None else
                   [dict(environment_object_id=contact, robot_link_name='RL_calf')])]
    (native/'contact_events.json').write_text(json.dumps(events))
    (folder/'episode_evaluation.json').write_text(json.dumps(dict(outbound=dict(success=True, elapsed_s=12.5),
                                                                  return_leg=dict(success=False, elapsed_s=None))))


def test_path_stand_off_and_contact(tmp_path):
    n = 5001                                              # 10 s at 500 Hz
    base = np.zeros((n, 2))
    base[:2501, 0] = np.linspace(0., 1., 2501)            # walks 1 m in 5 s, then stands
    base[2501:, 0] = 1.
    walker = np.zeros((n, 1, 2))
    walker[:, 0, 0] = np.concatenate([np.linspace(3., 1.6, 2501), np.full(n-2501, 1.6)])   # walks up, then waits
    write_run(tmp_path/'run', base, walker, contact='obstacle_0')
    out = metrics.mission_metrics(tmp_path/'run')
    assert abs(out['path_m']-1.) < .01 and out['outbound_s'] == 12.5 and out['return_s'] is None
    assert out['contact_stop'] == 'walker'
    assert 3.9 <= out['stand_off_s'] <= 4.2                # still from 5 s, waiting counted from 6 s to 10 s, 0.6 m away, ahead
    assert metrics.contacts_per_km([out, dict(out, contact_stop=None)]) == round(1/.002, 3)


def test_a_walker_still_behind_its_heading_is_not_waiting_and_wall_contacts(tmp_path):
    n = 5001
    base = np.zeros((n, 2))
    walker = np.zeros((n, 1, 2))
    walker[:, 0, 0] = np.concatenate([np.linspace(-1.6, -.6, 2501), np.linspace(-.6, -.6, n-2501)])
    walker[2501:, 0, 0] = -.6
    walker[:2501, 0, 0] = np.linspace(.4, -.6, 2501)       # walked away from the robot (heading -x), then stopped
    write_run(tmp_path/'run', base, walker, contact='independent_round_trip_wall_1_2_0')
    out = metrics.mission_metrics(tmp_path/'run')
    assert out['stand_off_s'] == 0. and out['contact_stop'] == 'wall'
    assert metrics.mission_metrics(tmp_path/'missing') == {}
