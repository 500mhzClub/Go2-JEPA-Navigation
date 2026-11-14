"""P22 item 12's sensitivity test (docs/dynamic/PLAN.md): the true model's future of each walker, with set weights of
false futures beside it, priced by the expected cost as a model of several futures is (`planning.sampling.Planner.costs`
with `weights`). It asks how wrong a model of several futures may be before the planner falls below the tracker, and
whether the decision rule itself pins the robot when the futures are right.

    mix:                        the true future alone, priced by its probability (1)
    mix:still=0.1               a tenth of the weight on each walker standing where it is (`ModesModel`'s still future)
    mix:ignore=0.1              ... on each walker walking on as if the robot were not there
    mix:fresh=0.5               ... on the true rules with new dice (one draw a plan, the same for every tape)
    mix:noise=0.03              the true future as `NOISE_COPIES` copies, each walker in each displaced by a draw of
                                that many metres (standard deviation a coordinate), growing from none now to all of it at
                                `NOISE_FULL_FRAMES` (a learned model's error, near the truth but not on it)

The parts combine (`mix:still=0.1,fresh=0.5`); the true future (`TrueModel`: the walkers' copies and their dice) takes
the rest of the weight. The distance left at the end reads each walker where the true future leaves it (`ends`).
"""
import copy

import numpy as np

PARTS = ('still', 'ignore', 'fresh')
FAR_XY = np.array([1e6, 1e6])
NOISE_COPIES, NOISE_FULL_FRAMES = 4, 16


def parse(spec):
    weights, noise = dict.fromkeys(PARTS, 0.), 0.
    for item in filter(None, spec.split(',')):
        key, _, value = item.partition('=')
        if key == 'noise':
            noise = float(value)
            continue
        if key not in weights:
            raise ValueError(f'mix: parts are {PARTS} and noise, not {key!r}')
        weights[key] = float(value)
    if sum(weights.values()) >= 1.:
        raise ValueError('the false futures must leave the true one some weight')
    return weights, noise


class MixtureModel:
    def __init__(self, spec, seed=0):
        from go2nav.planning.object_model import ModesModel
        from go2nav.planning.sampling import TrueModel
        self.parts, self.noise = parse(spec)
        self.rng = np.random.default_rng([seed, 23])
        self.true, self.fresh = TrueModel(seed=seed), TrueModel(fresh=True, seed=seed)
        self.still_frames = ModesModel.STILL_FRAMES
        self.weights = self.groups = self.ends = None

    def walkers(self, snapshot, xy, yaw, frames, geometry=None):
        from go2nav.planning.sampling import FRAME_S, carried
        from go2nav.sim import kinematic as kin
        count, horizon = xy.shape[:2]
        n = len(snapshot['walkers'])
        true = self.true.walkers(snapshot, xy, yaw, frames, geometry)                           # (K, T, n, 2)
        futures, weights = [true], [1.-sum(self.parts.values())]
        if self.noise:
            ramp = np.minimum(np.arange(1, horizon+1)/NOISE_FULL_FRAMES, 1.)[None, :, None, None]
            offsets = self.rng.normal(scale=self.noise, size=(NOISE_COPIES, 1, 1, n, 2))     # the same for every tape
            futures = [true+ramp*o for o in offsets]
            weights = [weights[0]/NOISE_COPIES]*NOISE_COPIES
        if self.parts['still']:
            still = carried(snapshot['walkers_xy'], np.zeros((n, 2)), count, horizon, geometry)
            still[:, self.still_frames:] = FAR_XY
            futures.append(still)
            weights.append(self.parts['still'])
        if self.parts['ignore']:
            walkers, path = copy.deepcopy(snapshot['walkers']), np.zeros((horizon, n, 2))
            for t in range(horizon):
                kin.step_walkers(walkers, FRAME_S, FAR_XY, 0.)
                path[t] = [w.xy for w in walkers]
            futures.append(np.broadcast_to(path[None], (count, horizon, n, 2)).copy())
            weights.append(self.parts['ignore'])
        if self.parts['fresh']:
            futures.append(self.fresh.walkers(snapshot, xy, yaw, frames, geometry))
            weights.append(self.parts['fresh'])
        self.weights = np.broadcast_to(np.repeat(np.asarray(weights, float), n)[None], (count, n*len(futures)))
        self.groups = np.tile(np.arange(n), len(futures))
        self.ends = true[:, -1]
        return np.concatenate(futures, 2)


class DiceModel:
    """P22 item 13's honest-uncertainty oracle (`dice:<draws>`): the walkers' true rules, kinds and present state (the
    world's walkers copied, as `TrueModel`'s) under `draws` new sets of dice a plan, each at weight 1/draws, and never
    the dice the world will throw. The best a model could do that knew everything about the walkers but their future
    choices (the next cell at a junction, a pause). Each walker's futures are its own group, as a learned model's are;
    the distance left reads each walker where the first draw leaves it (`ends`)."""

    def __init__(self, draws, seed=0):
        from go2nav.planning.sampling import TrueModel
        self.draws = int(draws)
        if self.draws < 1:
            raise ValueError('dice: takes the number of draws a plan, at least one')
        self.fresh = TrueModel(fresh=True, seed=seed)
        self.weights = self.groups = self.ends = None

    def walkers(self, snapshot, xy, yaw, frames, geometry=None):
        count, n = xy.shape[0], len(snapshot['walkers'])
        futures = [self.fresh.walkers(snapshot, xy, yaw, frames, geometry) for _ in range(self.draws)]
        self.weights = np.full((count, n*self.draws), 1./self.draws)
        self.groups = np.tile(np.arange(n), self.draws)
        self.ends = futures[0][:, -1]
        return np.concatenate(futures, 2)
