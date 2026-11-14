"""P22b's value in the planner (docs/dynamic/PLAN.md): the walkers' slots and memory kept from the ring every frame, and
the network's correction to the frozen rule's distance term at each plan's cut point.

- `WalkerState`: the ring tracker's tracks in `SLOTS` slots with their memory (`perception.ring_walkers.Slots`,
  `Memory`), from the model's own tracker when it has one, else from its own on the frame's scan. The missions of
  P22b's data keep one to record each plan's state (`data.value_missions`).
- `distance_term`: the frozen rule's distance term for the robot at a point with walkers standing at others (the
  walking distance through them over `V_REF`, a sealed way `SEALED_S` more), as `Planner.expected_costs` reads it.
- `Value`: the network (`models.value`) with a `WalkerState`. `correction` draws the window at each tape's end, every
  slot's walker where the model leaves it: the slot is matched to the nearest of the model's walkers at the plan's
  start (within `MATCH_M`; for a model of several futures the first such walker, the true future or the first draw),
  its position and velocity at the end are that walker's, its memory the slot's now. An unmatched slot stays where it
  is.
"""
import numpy as np
import torch

from go2nav.models import value as vm
from go2nav.perception import ring_walkers

SLOTS, MATCH_M = 6, .3


class WalkerState:
    def __init__(self, slots=SLOTS):
        self.own, self.slots, self.memory, self.frame, self.now = None, ring_walkers.Slots(slots), \
            ring_walkers.Memory(slots), 0, None

    def observe(self, t, scan, xy, yaw, geometry, wall_clearance, tracker=None):
        """A frame: the model's `tracker` after its own update, or else the scan (`World.scan()`) for one of ours."""
        if tracker is None:
            if self.own is None:
                self.own = ring_walkers.Tracker()
            found = ring_walkers.detect(*scan, xy, yaw, wall_clearance)
            self.own.update(t, found, ring_walkers.mount_of(xy, yaw), geometry)
            tracker = self.own
        position, seen, ids, _, velocity = self.slots.update(tracker, self.frame)
        memory = self.memory.update(position, seen, ids, velocity, np.asarray(xy, float))
        present = ids >= 0
        self.now = dict(xy=np.where(present[:, None], position, 0.), velocity=np.where(present[:, None], velocity, 0.),
                        memory=np.where(present[:, None], memory, 0.), present=present)
        self.frame += 1


def distance_term(fields, target, xy, walkers_xy):
    """The frozen rule's distance term (seconds) at points xy (K, 2), each with its walkers (K, S, 2) as obstacles
    (NaN or far away for none)."""
    from go2nav.planning.sampling import SEALED_S, V_REF
    static = fields.value(fields.static(target), xy)
    out = np.empty(len(xy))
    for k in range(len(xy)):
        w = walkers_xy[k][np.isfinite(walkers_xy[k]).all(1)]
        value = float(fields.value(fields.with_walkers(target, w), xy[k:k+1])[0])
        sealed = not np.isfinite(value)
        left = static[k] if sealed else value
        out[k] = (left if np.isfinite(left) else 1e3)/V_REF+SEALED_S*sealed
    return out


class Value:
    def __init__(self, path, device=None):
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.net, self.walkers, self.fields, self.targets = vm.load(path, self.device), WalkerState(), None, {}

    def bind(self, fields):
        self.fields = fields
        self.walls = torch.as_tensor(vm.wall_grid(fields), device=self.device)[None]
        self.low = torch.as_tensor(fields.low, dtype=torch.float32, device=self.device)

    def observe(self, t, scan, xy, yaw, geometry, tracker=None):
        self.walkers.observe(t, scan, xy, yaw, geometry, self.fields.wall_clearance, tracker)

    def _distance(self, target):
        key = tuple(np.round(np.asarray(target, float), 3))
        if key not in self.targets:
            self.targets[key] = torch.as_tensor(vm.distance_grid(self.fields, target), device=self.device)[None]
        return self.targets[key]

    @torch.no_grad()
    def correction(self, target, xy, yaw, walkers, base):
        """Seconds (K,) to add to the distance term `base` (K,) of tapes ending at xy (K, 2), yaw (K,), the model's
        walkers along them (K, T, J, 2)."""
        now, count = self.walkers.now, len(xy)
        slots_xy, slots_v, own = cut_inputs(now, walkers, xy, self.fields, target)
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=self.device)
        window = vm.draw(self.walls, self._distance(target), self.low, t(xy), t(yaw), t(own), t(slots_xy), t(slots_v),
                         t(np.tile(now['memory'], (count, 1, 1))),
                         torch.as_tensor(np.tile(now['present'], (count, 1)), device=self.device))
        return self.net(window, t(base), t(own)).cpu().numpy().astype(float)


def cut_inputs(now, walkers, xy, fields, target):
    """The value's walkers at each tape's end (K, S, 2 positions and velocities; each slot where the model leaves its
    walker, see the module docstring) and the robot's walking distance there (K,), from the slots `now`
    (`WalkerState.now`), the model's walkers (K, T, J, 2) and the tapes' ends xy (K, 2)."""
    from go2nav.sim.kinematic import FRAME_S
    count = len(xy)
    slots_xy = np.tile(now['xy'], (count, 1, 1))
    slots_v = np.tile(now['velocity'], (count, 1, 1))
    if walkers.shape[2] and walkers.shape[1] > 1:
        start = walkers[0, 0]                                                        # (J, 2): the same for every tape
        for k in np.flatnonzero(now['present']):
            gap = np.linalg.norm(start-now['xy'][k], axis=1)
            j = int(np.argmin(gap))
            if gap[j] < MATCH_M:
                slots_xy[:, k] = walkers[:, -1, j]
                slots_v[:, k] = (walkers[:, -1, j]-walkers[:, -2, j])/FRAME_S
    own = fields.value(fields.static(target), xy)
    return slots_xy, slots_v, np.where(np.isfinite(own), own, vm.FAR_M)
