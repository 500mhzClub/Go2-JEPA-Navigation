"""The rarely reached recovery branches: view recovery, narrow floor patches, a frame without a registered pose,
and the scripted backup and frontier exclusions of the recovery fixes."""
import math
from types import SimpleNamespace

import numpy as np

from go2nav.perception import floor
from go2nav.perception.visual_support import SupportedView
from go2nav.planning import routing
from go2nav.planning.actions import ACTIONS
from go2nav.planning.mission import RoundTripMission
from go2nav.planning.planner import Planner

FRAME_NS = 100_000_000


def yaw(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def test_view_recovery_starts_near_a_strong_view_and_releases_when_support_returns():
    view = SupportedView()
    assert view.advance((120, 30), np.zeros(3), yaw(0.), 0) is None
    assert view.advance((60, 30), np.array([.1, 0., 0.]), yaw(.5), FRAME_NS) is None  # weak but not low
    active = view.advance((40, 30), np.array([.1, 0., 0.]), yaw(.5), 2*FRAME_NS)
    assert active['trigger_ns'] == 2*FRAME_NS and active['measured_ns'] == 0
    assert view.advance((60, 30), np.array([.1, 0., 0.]), yaw(.5), 3*FRAME_NS) is active  # not yet turned back
    assert view.advance((60, 30), np.array([.1, 0., 0.]), yaw(.05), 4*FRAME_NS) is None


def test_view_recovery_needs_the_strong_view_within_20_cm():
    view = SupportedView()
    view.advance((120, 30), np.zeros(3), yaw(0.), 0)
    assert view.advance((10, 10), np.array([.3, 0., 0.]), yaw(0.), FRAME_NS) is None


def test_an_attained_view_without_support_is_retired_after_a_second():
    view = SupportedView()
    view.advance((120, 30), np.zeros(3), yaw(0.), 0)
    states = [view.advance((10, 10), np.zeros(3), yaw(0.), k*FRAME_NS) for k in range(1, 13)]
    assert all(s is not None for s in states[:10]) and states[10] is None
    assert len(view.retirements) == 1 and view.good is None
    assert states[11] is None  # nothing to return to until a new strong frame


def strip(width, n=400, height=-.3):
    """A floor patch 1 m long and `width` wide, as the points of one camera."""
    rng = np.random.default_rng(0)
    return np.column_stack((rng.uniform(.2, 1.2, n), rng.uniform(-width/2, width/2, n), np.full(n, height)))


def test_a_narrow_patch_gives_no_plane_but_a_height_along_up():
    up = np.array([0., 0., 1.])
    points = strip(.01)
    plane = floor.joint_plane(points, np.empty((0, 3)), up)
    assert not plane['available'] and plane['reason'] == floor.TOO_NARROW
    level = floor.gyro_height_plane(points, np.empty((0, 3)), up)
    assert level['available'] and np.isclose(level['offset_body_m'], .3)
    assert floor.gyro_height_plane(strip(.5), np.empty((0, 3)), up)['reason'] is None  # a full plane is kept


def test_narrow_patch_points_off_the_level_are_pruned():
    up = np.array([0., 0., 1.])
    clouds = [np.vstack((strip(.01), [[.5, 0., -.29]])), np.empty((0, 3))]
    masks = [np.ones(len(clouds[0]), bool), np.zeros(0, bool)]
    pruned = floor.condition_narrow_patch(clouds, masks, up, up, per_camera_mean=False)
    assert pruned[0].sum() == len(clouds[0])-1 and not pruned[0][-1]
    wide = [strip(.5), np.empty((0, 3))]
    same = [np.ones(len(wide[0]), bool), np.zeros(0, bool)]
    assert floor.condition_narrow_patch(wide, same, up, up, per_camera_mean=False) is same


def test_a_frame_without_a_pose_holds_and_restarts_the_arrival_dwell():
    mission = RoundTripMission((1., 0.), arrival_radius_m=.02, navigation_frames=100)
    for frame in range(8):
        state = mission.advance(np.array([1., 0., 0.]), frame=frame, requested=(0., 0., 0.))
    assert state['quiet_intervals'] > 0
    state = mission.unavailable(frame=8)
    assert state['hold_required'] and state['quiet_intervals'] == 0 and state['observed_goal_distance_m'] is None
    assert mission.advance(np.array([1., 0., 0.]), frame=9, requested=(0., 0., 0.))['quiet_intervals'] == 0


def test_sticky_exclusions_survive_new_map_evidence_until_they_expire():
    excluded = routing.StickyExclusions({(0, 0)})
    excluded.sticky[(1, 1)] = 100
    excluded.add((1, 1))
    excluded.clear()
    assert excluded == {(1, 1)}
    excluded.expire(99)
    assert excluded == {(1, 1)}
    excluded.expire(100)
    assert not excluded


def open_floor(half_cells=20):
    cells = {(i, j) for i in range(-half_cells, half_cells) for j in range(-half_cells, half_cells)}
    return SimpleNamespace(floor=cells, occupied=set(), fine_occupied=frozenset())


def hold():
    return dict(action='hold', action_index=ACTIONS.index('hold'), requested_command=[0., 0., 0.])


def test_a_scripted_backup_is_five_reverse_steps_recorded_as_holds():
    planner = Planner(arrival_radius_m=.02, recovery=True)
    planner._request_backup('stall')
    steps = [planner._scripted_backup(hold(), open_floor(), np.zeros(3), np.eye(3)) for _ in range(6)]
    assert all(s['action'] == 'hold' and s['requested_command'] == [-.10, 0., 0.] for s in steps[:5])
    assert [s['backup']['step'] for s in steps[:5]] == [1, 2, 3, 4, 5]
    assert steps[5]['requested_command'] == [0., 0., 0.] and not planner.backup_plan


def test_a_backup_step_off_observed_floor_ends_the_script():
    planner = Planner(arrival_radius_m=.02, recovery=True)
    planner._request_backup('deadlock_no_clear_turn')
    step = planner._scripted_backup(hold(), open_floor(), np.array([-.99, 0., 0.]), np.eye(3))
    assert 'backup_aborted' in step and step['requested_command'] == [0., 0., 0.] and planner.backup is None


def still_track(xy, track_id=0, confirmed=True):
    from go2nav.perception.tracks import Track
    track = Track(track_id, frozenset(), 0, history=[(0, np.asarray(xy, float))], motion_updates=2 if confirmed else 0)
    return track


def test_give_way_commits_to_the_distance_whatever_the_mover_then_does():
    planner = Planner(arrival_radius_m=.02, give_way=True)
    planner.tracks = [still_track((1., .2))]
    steps = []
    for k in range(3):
        planner.backup_plan = False
        steps.append(planner._give_way(hold(), open_floor(), np.array([-.1*k, 0., 0.]), np.eye(3)))
        planner.tracks = []                                           # the mover walks off or is lost: still backing off
    assert all(s['requested_command'] == [-.10, 0., 0.] and s['give_way']['step'] == k+1 for k, s in enumerate(steps))
    planner.backup_plan = False
    done = planner._give_way(hold(), open_floor(), np.array([-.4, 0., 0.]), np.eye(3))
    assert done['give_way_ended']['reason'] == 'distance' and done['requested_command'] == [0., 0., 0.]
    assert done['give_way_ended']['moved_m'] == .4 and planner.give_way_cooldown == 10


def test_give_way_ends_when_it_cannot_cover_the_distance():
    planner = Planner(arrival_radius_m=.02, give_way=True)
    planner.tracks = [still_track((1., 0.))]
    for _ in range(26):
        planner.backup_plan = False
        step = planner._give_way(hold(), open_floor(), np.zeros(3), np.eye(3))                    # no reverse under it
    assert step['give_way_ended']['reason'] == 'steps' and step['give_way_ended']['step'] == 26
    planner = Planner(arrival_radius_m=.02, give_way=True)
    planner.tracks = [still_track((0., 0.))]
    step = planner._give_way(hold(), open_floor(), np.array([-.99, 0., 0.]), np.eye(3))           # off observed floor
    assert step['give_way_ended']['reason'] == 'reverse_blocked' and step['requested_command'] == [0., 0., 0.]


def test_only_give_way_may_reverse_closer_to_a_wall_than_the_robot_stands():
    wall = open_floor()
    wall.fine_occupied = frozenset((i, -50) for i in range(-200, 200))            # along y = -0.50 m, in 1-cm cells
    yaw = math.pi/4                                                               # reversing towards it, 2.8 cm a step
    Q = np.array([[math.cos(yaw), -math.sin(yaw), 0.], [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
    kept, detail = Planner._reverse_step(wall, np.zeros(3), Q)
    relaxed, _ = Planner._reverse_step(wall, np.zeros(3), Q, keep_clearance=False)
    assert detail['path_clearance_m'] >= .40 and detail['path_clearance_m'] < detail['current_clearance_m']-.01
    assert relaxed and not kept


def test_give_way_ignores_unconfirmed_moving_or_sideways_tracks_and_advancing_selections():
    planner = Planner(arrival_radius_m=.02, give_way=True)
    moving = still_track((1., 0.), 1)
    moving.velocity = np.array([-.2, 0.])
    for tracks in ([still_track((1., 0.), confirmed=False)], [moving], [still_track((1., 1.))], [still_track((-.5, 0.))]):
        planner.tracks = tracks
        assert 'give_way' not in planner._give_way(hold(), open_floor(), np.zeros(3), np.eye(3))
    planner.tracks = [still_track((1., 0.))]
    advancing = dict(hold(), requested_command=[.15, 0., 0.])
    assert 'give_way' not in planner._give_way(advancing, open_floor(), np.zeros(3), np.eye(3))
