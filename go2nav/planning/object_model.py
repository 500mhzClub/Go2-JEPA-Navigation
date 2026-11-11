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
        position, seen, ids, _ = self.slots.update(self.tracker, self.frame)
        self.frames.append(dict(xy=position, seen=seen, id=ids, pose=np.array([*xy, yaw], float), requested=None))
        self.frames = self.frames[-(STEP_FRAMES*(self.history-1)+1):]
        self.frame += 1

    def record(self, requested):
        self.frames[-1]['requested'] = np.asarray(requested, np.float32)

    def _tokens(self):
        """The history's object and robot tokens, and the recorded actions of its steps before the last."""
        h = self.history
        picks = [self.frames[max(len(self.frames)-1-STEP_FRAMES*(h-1-k), 0)] for k in range(h)]
        now = picks[-1]
        origin, heading = now['pose'][:2], float(now['pose'][2])
        objects, robot = [], []
        for f in picks:
            present = (f['id'] == now['id']) & (now['id'] >= 0)
            world = np.where(present[:, None], f['xy'], 0.)
            walls = self.raster.context(world, np.full(len(world), heading), OBJECT_GRID, OBJECT_SPACING)
            objects.append(object_features(torch.as_tensor(to_body(world, origin, heading), dtype=torch.float32),
                                           torch.as_tensor(f['seen'] & present), torch.as_tensor(present),
                                           torch.as_tensor(walls, dtype=torch.float32)))
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
