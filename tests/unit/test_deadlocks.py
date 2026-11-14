"""The trap test of the deadlock diagnostic (go2nav.eval.deadlocks)."""
import numpy as np

from go2nav.eval.deadlocks import trapped
from go2nav.eval.mission import episode_spec
from go2nav.sim import kinematic as kin
from go2nav.sim import obstacles as moving


def test_a_walker_that_does_not_yield_traps_the_robot_in_a_dead_end_unless_its_target_is_there():
    spec, record = episode_spec('validation', 12, 0)
    w = kin.World(spec, record['home_se2_world'], [])
    g, pitch = w.geometry.neighbours, w.geometry.pitch
    end = next(c for c in sorted(g) if len(g[c]) == 1)                         # a dead end
    way = g[end][0]                                                            # its one passage
    far = next(c for c in sorted(g) if c not in (end, way))
    walker = moving.Walker(g, way, .3, np.random.default_rng(0), polite=False, pitch=pitch)
    w.xy, w.walkers = np.asarray(end, float)*pitch, [walker]
    walker.cell, walker.target = way, end                                      # walking in, 0.52 m from the robot
    walker.xy = w.xy+(np.asarray(way, float)-np.asarray(end, float))*.4*pitch
    assert trapped(w, np.asarray(far, float)*pitch) == (1, True)
    assert trapped(w, np.asarray(end, float)*pitch) == (1, False)
    walker.cell, walker.target = end, way                                      # walking out: not waiting on it
    assert trapped(w, np.asarray(far, float)*pitch) == (0, False)
