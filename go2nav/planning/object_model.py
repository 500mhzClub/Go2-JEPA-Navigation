"""The object-level predictor as the walkers' part of the sampling planner's model (docs/dynamic/PLAN.md P22 item 4).

`ObjectModel` runs the ring tracker on every frame's scan (`observe`, as `RingModel`), holds its tracks in fixed slots
(`perception.ring_walkers.Slots`), and keeps the last `history` steps of slots, poses and requests (`record`). At a plan
(`walkers`) every tape is rolled out by `models.object_jepa.ObjectJEPA.rollout` in one batch: the history's tokens
are the same for every tape, the robot's tokens after it are the tape's own path, and the actions are the recorded
requests and then the tape's. The predicted positions (one a 0.4-s step) are put back in the world frame and
interpolated to the frames; empty slots are placed `FAR` away. No simulator state is read.
"""
import numpy as np
import torch

from go2nav.models.object_jepa import (OBJECT_GRID, OBJECT_SPACING, ROBOT_GRID, ROBOT_SPACING, STEP_FRAMES,
                                       actions_input, object_features, robot_features)
from go2nav.perception import ring_walkers
from go2nav.planning.wall_raster import WallRaster

FAR = 1e3


def to_body(points, origin, heading):
    d = np.asarray(points, float)-origin
    c, s = np.cos(heading), np.sin(heading)
    return np.stack((c*d[..., 0]+s*d[..., 1], -s*d[..., 0]+c*d[..., 1]), -1)


def to_world(points, origin, heading):
    c, s = np.cos(heading), np.sin(heading)
    p = np.asarray(points, float)
    return np.stack((c*p[..., 0]-s*p[..., 1], s*p[..., 0]+c*p[..., 1]), -1)+origin


class ObjectModel:
    takes_tapes = True

    def __init__(self, path, device=None):
        from go2nav.training.object_jepa import load_model
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))   # 0.6 s a plan on one core
        self.model = load_model(path, self.device)
        self.history = self.model.config.history
        self.tracker, self.slots = ring_walkers.Tracker(), ring_walkers.Slots(self.model.config.slots)
        self.frames, self.raster, self.fields, self.frame = [], None, None, 0

    def bind(self, fields):
        self.fields, self.raster = fields, WallRaster.of(fields.geometry)

    def observe(self, t, scan, xy, yaw, geometry):
        found = ring_walkers.detect(*scan, xy, yaw, self.fields.wall_clearance)
        self.tracker.update(t, found, ring_walkers.mount_of(xy, yaw), geometry)
        position, seen, ids, _, velocity = self.slots.update(self.tracker, self.frame)
        self.frames.append(dict(xy=position, seen=seen, id=ids, velocity=velocity, pose=np.array([*xy, yaw], float),
                                requested=None))
        self.frames = self.frames[-(STEP_FRAMES*(self.history-1)+1):]
        self.frame += 1

    def record(self, requested):
        self.frames[-1]['requested'] = np.asarray(requested, np.float32)

    def _tokens(self, grid=(OBJECT_GRID, OBJECT_SPACING), velocity=None):
        """The history's object and robot tokens, and the recorded actions of its steps before the last. `grid`: the
        walls' grid round an object; `velocity`: whether the tokens hold it (default: a residual model's do)."""
        h = self.history
        velocity = self.model.config.residual if velocity is None else velocity
        picks = [self.frames[max(len(self.frames)-1-STEP_FRAMES*(h-1-k), 0)] for k in range(h)]
        now = picks[-1]
        origin, heading = now['pose'][:2], float(now['pose'][2])
        objects, robot = [], []
        for f in picks:
            present = (f['id'] == now['id']) & (now['id'] >= 0)
            world = np.where(present[:, None], f['xy'], 0.)
            walls = self.raster.context(world, np.full(len(world), heading), *grid)
            speed = None
            if velocity:                                         # the tracker's velocity, turned into the start frame
                turned = to_body(f['velocity'], np.zeros(2), heading)*(f['seen'] & present)[:, None]
                speed = torch.as_tensor(turned, dtype=torch.float32)
            objects.append(object_features(torch.as_tensor(to_body(world, origin, heading), dtype=torch.float32),
                                           torch.as_tensor(f['seen'] & present), torch.as_tensor(present),
                                           torch.as_tensor(walls, dtype=torch.float32), speed))
            yaw = float(f['pose'][2])
            rw = self.raster.context(f['pose'][None, :2], np.array([yaw]), ROBOT_GRID, ROBOT_SPACING)[0]
            robot.append(robot_features(torch.as_tensor(to_body(f['pose'][:2], origin, heading), dtype=torch.float32),
                                        torch.tensor(yaw-heading), torch.as_tensor(rw, dtype=torch.float32)))
        index = len(self.frames)-1
        past = []
        for k in range(h-1):                                   # the requests after each earlier history step
            start = max(index-STEP_FRAMES*(h-1-k), 0)
            block = [self.frames[min(start+j, index)]['requested'] for j in range(STEP_FRAMES)]
            past.append(np.stack([b if b is not None else np.zeros(3, np.float32) for b in block]))
        present_now = (now['id'] >= 0)
        return torch.stack(objects), torch.stack(robot), np.stack(past) if past else np.zeros((0, STEP_FRAMES, 3)), \
            origin, heading, present_now, now['xy']

    @torch.no_grad()
    def walkers(self, snapshot, xy, yaw, frames, geometry=None, requested=None):
        count, horizon = xy.shape[:2]
        steps = -(-horizon//STEP_FRAMES)
        objects, robot, past, origin, heading, present, now = self._tokens()
        requested = np.asarray(requested, np.float32)
        padded = np.concatenate((requested, np.repeat(requested[:, -1:], steps*STEP_FRAMES-horizon, 1)), 1)
        blocks = np.concatenate((np.broadcast_to(past, (count, *past.shape)),
                                 padded.reshape(count, steps, STEP_FRAMES, 3)), 1)          # (K, H-1+steps, 4, 3)
        actions = actions_input(torch.as_tensor(blocks)).to(self.device)
        ends = np.minimum(np.arange(1, steps+1)*STEP_FRAMES-1, horizon-1)
        future_world, future_yaw = xy[:, ends], yaw[:, ends]
        rw = self.raster.context(future_world, future_yaw, ROBOT_GRID, ROBOT_SPACING)
        future_robot = robot_features(torch.as_tensor(to_body(future_world, origin, heading), dtype=torch.float32),
                                      torch.as_tensor(future_yaw-heading, dtype=torch.float32),
                                      torch.as_tensor(rw, dtype=torch.float32)).to(self.device)

        def walls(position):
            world = to_world(position.cpu().numpy(), origin, heading)
            ctx = self.raster.context(world, np.full(world.shape[:-1], heading), OBJECT_GRID, OBJECT_SPACING)
            return torch.as_tensor(ctx, dtype=torch.float32, device=self.device)

        predicted = self.model.rollout(objects[None].expand(count, -1, -1, -1).to(self.device),
                                       robot[None].expand(count, -1, -1).to(self.device), actions, future_robot,
                                       walls, steps).cpu().numpy()                          # (K, steps, S, 2)
        world = to_world(predicted, origin, heading)
        start = np.where(present[:, None], now, 0.)[None, None].repeat(count, 0)           # step 0: where they are now
        knots = np.concatenate((start, world), 1)                                          # (K, steps+1, S, 2)
        f = np.arange(horizon)
        lower = (f+1)//STEP_FRAMES
        frac = ((f+1) % STEP_FRAMES)/STEP_FRAMES
        a, b = knots[:, np.minimum(lower, steps)], knots[:, np.minimum(lower+1, steps)]
        out = a+(b-a)*frac[None, :, None, None]
        out[:, :, ~present] = FAR
        return out


def to_frames(knots, horizon):
    """Positions at 0.4-s knots (..., steps+1, n, 2), the first where the objects are now, to the frames (..., horizon,
    n, 2), linearly."""
    steps = knots.shape[-3]-1
    f = np.arange(horizon)
    lower, frac = (f+1)//STEP_FRAMES, ((f+1) % STEP_FRAMES)/STEP_FRAMES
    a, b = knots[..., np.minimum(lower, steps), :, :], knots[..., np.minimum(lower+1, steps), :, :]
    return a+(b-a)*frac[:, None, None]


class ModesModel(ObjectModel):
    """The several-futures predictor (`models.object_modes`; PLAN P22 item 8) in the planner: one pass a plan gives
    each object `modes` futures under every tape. The planner keeps clear of every future of probability `P_MIN` or
    more (the `TOP` likeliest at most, each a walker of its own in what `walkers` returns), and reads the distance left
    at the end with each object at its likeliest future (`ends`, which `planning.sampling.Planner.costs` takes when a
    model has it). With `KEEP`, clearance is also kept from each walker standing still and carried on at the tracker's
velocity: without it the planner chose the tapes under which the model had a blocking walker move aside, and drove
into it (30 contact stops in 40 missions where the tracker has 11-16; STATUS P22). `walkers_top` gives the likeliest future alone, for the measures. The model is small enough to run
    on a mission's own core (a pass is tens of milliseconds), so suites run a mission to a core."""
    P_MIN, TOP = .1, 3
    KEEP = True     # also keep clear of each walker standing where it is and carried on at the tracker's velocity
    EXPECT, W_STILL = False, .1   # price a contact by its probability over every future, a walker standing still among them
    weights = groups = None

    def __init__(self, path, device='cpu'):
        from go2nav.training.object_modes import load_model
        self.device = torch.device(device)
        self.model = load_model(path, self.device)
        self.history = self.model.config.history
        self.tracker, self.slots = ring_walkers.Tracker(), ring_walkers.Slots(self.model.config.slots)
        self.frames, self.raster, self.fields, self.frame, self.ends = [], None, None, 0, None

    @torch.no_grad()
    def _futures(self, xy, yaw, requested):
        """World positions (tapes, S, modes, steps+1, 2) from now on, probabilities (tapes, S, modes), present (S,)."""
        from go2nav.models.object_modes import WIDE_GRID, WIDE_SPACING
        count, horizon = xy.shape[:2]
        steps = self.model.config.steps
        objects, robot, past, origin, heading, present, now = self._tokens((WIDE_GRID, WIDE_SPACING), True)
        requested = np.asarray(requested, np.float32)
        frames = steps*STEP_FRAMES
        padded = np.concatenate((requested[:, :frames], np.repeat(requested[:, -1:], max(frames-horizon, 0), 1)), 1)
        blocks = actions_input(torch.as_tensor(padded.reshape(count, steps, STEP_FRAMES, 3)))        # (K, steps, 12)
        before = actions_input(torch.as_tensor(np.asarray(past, np.float32)))                         # (H-1, 12)
        history_actions = torch.cat((before[None].expand(count, -1, -1), blocks[:, :1]), 1)           # (K, H, 12)
        future_actions = torch.cat((blocks[:, 1:], torch.zeros(count, 1, blocks.shape[-1])), 1)       # (K, steps, 12)
        ends = np.minimum(np.arange(1, steps+1)*STEP_FRAMES-1, horizon-1)
        future_world, future_yaw = xy[:, ends], yaw[:, ends]
        rw = self.raster.context(future_world, future_yaw, ROBOT_GRID, ROBOT_SPACING)
        future = robot_features(torch.as_tensor(to_body(future_world, origin, heading), dtype=torch.float32),
                                torch.as_tensor(future_yaw-heading, dtype=torch.float32),
                                torch.as_tensor(rw, dtype=torch.float32))
        positions, p = self.model.futures(objects[None].expand(count, -1, -1, -1).to(self.device),
                                          torch.cat((robot[None].expand(count, -1, -1), history_actions), -1).to(self.device),
                                          torch.cat((future, future_actions), -1).to(self.device))
        world = to_world(positions.cpu().numpy(), origin, heading)                                    # (K, S, M, steps, 2)
        start = np.broadcast_to(np.where(present[:, None], now, 0.)[None, :, None, None], (*world.shape[:3], 1, 2))
        return np.concatenate((start, world), 3), p.cpu().numpy(), present

    def walkers(self, snapshot, xy, yaw, frames, geometry=None, requested=None):
        count, horizon = xy.shape[:2]
        knots, p, present = self._futures(xy, yaw, requested)
        if self.EXPECT:
            from go2nav.planning.sampling import carried
            s, m = p.shape[1:]
            paths = to_frames(knots.reshape(count, s*m, knots.shape[3], 2).transpose(0, 2, 1, 3), horizon)
            at = np.where(present[:, None], self.frames[-1]['xy'], FAR)
            paths = np.concatenate((paths, carried(at, np.zeros_like(at), count, horizon, geometry)), 2)
            live = present[None, :, None]
            self.weights = np.concatenate(((p*(1-self.W_STILL)*live).reshape(count, s*m),
                                           np.broadcast_to(self.W_STILL*present, (count, s))), 1)
            self.groups = np.concatenate((np.repeat(np.arange(s), m), np.arange(s)))
            best = np.take_along_axis(knots, p.argmax(-1)[..., None, None, None], 2)[:, :, 0, -1]
            self.ends = np.where(present[None, :, None], best, FAR)
            keep = np.concatenate((np.repeat(present, m), present))
            return np.where(keep[None, None, :, None], paths, FAR)
        order = np.argsort(-p, -1)[..., :self.TOP]                                                    # (K, S, TOP)
        kept = np.take_along_axis(knots, order[..., None, None], 2)                                   # (K, S, TOP, steps+1, 2)
        likely = np.take_along_axis(p, order, -1) >= self.P_MIN
        likely[..., 0] = True                                                                         # the likeliest always
        likely &= present[None, :, None]
        paths = to_frames(kept.reshape(count, -1, kept.shape[3], 2).transpose(0, 2, 1, 3), horizon)   # (K, horizon, S*TOP, 2)
        flat = likely.reshape(count, -1)
        paths = np.where(flat[:, None, :, None], paths, FAR)
        self.ends = np.where(present[None, :, None], kept[:, :, 0, -1], FAR)                          # (K, S, 2)
        if self.KEEP:
            from go2nav.planning.sampling import carried
            frame = self.frames[-1]
            at = np.where(present[:, None], frame['xy'], FAR)
            still = carried(at, np.zeros_like(at), count, horizon, geometry)
            on = carried(at, np.where(present[:, None], frame['velocity'], 0.), count, horizon, geometry)
            paths = np.concatenate((paths, still, on), 2)
        return paths

    def walkers_top(self, snapshot, xy, yaw, frames, geometry=None, requested=None):
        """The likeliest future of each slot at the frames (tapes, horizon, S, 2)."""
        count, horizon = xy.shape[:2]
        knots, p, present = self._futures(xy, yaw, requested)
        top = np.take_along_axis(knots, p.argmax(-1)[..., None, None, None], 2)[:, :, 0]              # (K, S, steps+1, 2)
        return to_frames(top.transpose(0, 2, 1, 3), horizon)

