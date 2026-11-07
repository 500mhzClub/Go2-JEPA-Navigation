"""The stand-off data gate (data.standoff_audit; docs/dynamic/PLAN.md P14, amended 7 November)."""
import numpy as np

from go2nav.data import standoff_audit as a


def test_a_walker_that_stops_ahead_of_the_robot_is_a_stand_off():
    n = 40
    base = np.zeros((n, 2))
    walker = np.zeros((n, 2))
    walker[:21, 0], walker[21:, 0] = np.linspace(2., .6, 21), .6     # walks in towards the robot, then waits
    waiting = a.waiting_frames(base, walker)
    assert not waiting[:30].any() and waiting[30:].all()             # still for 1 s first
    assert a.stretches(waiting) == [(30, 40)]


def row(frames, resolved, moved, behaviour='non-yielding', family='lewm_sets_v1'):
    return dict(behaviour=behaviour, family=family, frames=frames, resolved=resolved, moved_away=moved)


def test_the_gate_counts_resolution_per_stand_off():
    rows = ([row(10, True, True)]*2+[row(10, False, False)]*7+[row(400, False, True)]
            + [row(10, True, False, 'reactive', 'loops_rooms_corpus_v1')]*2)
    s = a.summary(rows)
    assert s['all']['resolved_share'] == round(4/12, 4)             # a third of the stand-offs
    assert s['all']['resolved_frame_share'] < .1                    # a tenth of the frames: the long one decides it
    g = a.gate(s)
    assert g['resolved'] and g['moved_away'] and not g['reactive_frames'] and not g['passed']
