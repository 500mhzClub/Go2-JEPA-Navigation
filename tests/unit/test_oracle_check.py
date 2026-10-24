import numpy as np

from go2nav.controllers.oracle import OracleForecaster
from go2nav.planning.dispatch import PASSED


def test_a_plan_overridden_at_its_start_is_skipped():
    oracle = OracleForecaster(None)
    branch = dict(applied=np.full((40, 3), .1), poses=np.zeros((400, 7)))
    oracle.branches = [dict(observed_ns=0, branches=[branch])]
    samples = [dict(timestamp_s=t*.002, base_pose_world=np.zeros(7)) for t in range(401)]
    stale = [dict(simulator_ns=0, applied_command=[0., 0., 0.], reason='CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE')]
    check = oracle.verify_executed(stale, samples)
    assert check['passed'] and check['comparisons'][0]['reason'] == 'OVERRIDDEN_AT_PLAN_START'
    passed = [dict(simulator_ns=0, applied_command=[0., 0., 0.], reason=PASSED)]
    assert not oracle.verify_executed(passed, samples)['passed']
