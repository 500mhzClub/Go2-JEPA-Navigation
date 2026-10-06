"""The ported visual odometry reproduces the original pose exactly at every frame of each recorded mission."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/porting'))

pytestmark = [pytest.mark.equivalence, pytest.mark.slow]


def test_pose_matches_every_frame(bundle):
    import replay_check
    from go2nav.perception.visual_odometry import VisualOdometry
    odometry = VisualOdometry()
    for args, _, result in replay_check.calls(bundle, 'pose'):
        packet = args[0]
        pose = odometry.observe(**replay_check.pose_inputs(packet))
        assert not replay_check.compare_pose(pose, result['current_pose']), f'frame {packet.frame}'
