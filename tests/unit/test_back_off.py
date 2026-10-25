from go2nav.planning import selection as sel


def selection(paths):
    rows = [dict(action=a, segment_clearances_m=p, nominal_predicted_path_clear=False, reserve_recovery_path_clear=False,
                 required_path_clearance_m=.48 if a != 'hold' else .45) for a, p in paths.items()]
    return dict(action='hold', action_index=0, candidates=[dict(action=a) for a in paths], memory_forecast_candidates=rows)


def test_a_robot_inside_the_footprint_backs_off_only_on_a_lattice():
    s = selection({'hold': [.40]*8, 'cmd-0.10+0.00+0.00': [.40+.004*k for k in range(1, 9)],
                   'cmd+0.15+0.00+0.00': [.40-.01*k for k in range(1, 9)], 'cmd+0.00+0.00+0.45': [.40]*8})
    assert sel.recover_against_hold(s) is s
    backed = sel.recover_against_hold(s, inside=True)
    assert backed['action'] == 'cmd-0.10+0.00+0.00' and backed['memory_forecast_status'] == 'INSIDE_FOOTPRINT_BACK_OFF'


def test_no_back_off_without_enough_gain():
    s = selection({'hold': [.40]*8, 'cmd-0.10+0.00+0.00': [.40+.0005*k for k in range(1, 9)]})
    assert sel.recover_against_hold(s, inside=True) is s
