"""Zero means zero (docs/dynamic/PLAN.md P13 item 6): with the rigid solver's no-slip pass, a robot commanded zero stays put."""
from pathlib import Path
import subprocess
import sys

import pytest

from go2nav.eval import cohort

REPO = Path(__file__).resolve().parents[2]
POLICY = REPO/'outputs/locomotion/continuous_seed11/model_1499.pt'

HOLD = """
import sys, math
import numpy as np
from go2nav.sim import scene
from go2nav.sim.session import MissionSession
from go2nav.mazes.sets import MazeSet
scene.init_genesis(1)
s = MissionSession(MazeSet('validation').maze(12), sys.argv[2], policy_checkpoint=sys.argv[3], safety='continuous',
                   noslip=sys.argv[1] == 'on')
s.settle(warm_renderer=False)
for command in ([.2, 0., 0.], [0., 0., 0.]):
    for _ in range(100):
        s.step(np.array(command, np.float32))
p0 = s.samples[-1]['base_pose_world'].copy()
for _ in range(1000):
    s.step(np.zeros(3, np.float32))
p1 = s.samples[-1]['base_pose_world']
print(np.linalg.norm(p1[:2]-p0[:2]))
"""


def test_the_flag_reaches_the_mission():
    command = cohort.mission_command(dict(noslip='on'), 'C5', 'validation', 10, 0, Path('out'))
    assert command[command.index('--noslip')+1] == 'on'
    assert '--noslip' not in cohort.mission_command({}, 'C5', 'validation', 10, 0, Path('out'))


@pytest.mark.skipif(not POLICY.exists(), reason='continuous locomotion policy not available')
def test_a_held_robot_stays_put_only_with_the_no_slip_pass(tmp_path):
    drift = {}
    for flag in ('on', 'off'):
        out = subprocess.run([sys.executable, '-c', HOLD, flag, str(tmp_path/flag), str(POLICY)], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
        drift[flag] = float(out.strip().splitlines()[-1])
    assert drift['on'] < .001                  # 20 s: under 1 mm (measured 0.3 mm)
    assert drift['off'] > .01                  # and the slide it removes (about 1 mm/s)
