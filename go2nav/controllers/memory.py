"""The walker memory in missions (docs/dynamic/PLAN.md, P9, step 4): where the model remembers walkers behind the robot.

**Stepping** (`MemoryRunner.step`, every `models.memory.STEP_FRAMES`-th frame with a registered pose): the frame is
encoded on its own by the world model (its state, as the memory was trained on), and the memory is moved by the pose
change since its last step, from the registered (visual odometry) poses, then updated. A frame without a pose is
skipped; the next step's motion spans the gap.

**Remembered walkers** (`remembered`): on the labelled grid round the robot, each connected group of cells (4-connected)
whose occupancy is above `threshold` is one walker, at the group's probability-weighted centre, if that centre is out
of the primary camera's field of view (`eval.mover_probe`'s cone and range: what the depth map sees, it maps) and not
within `BODY_M` of the body origin (no walker stands inside the robot). Each is a disc of the walker's radius in the
1-cm cells (`walkers.disk`), which the planner's clearance checks add to the map's (`Planner.remembered`).
"""
import math

import numpy as np
import torch
from scipy import ndimage

from go2nav.controllers.walkers import disk
from go2nav.data import memory_labels as ml
from go2nav.eval import mover_probe as mp
from go2nav.models.lewm import normalise_frames, resize_frame
from go2nav.models.memory import STEP_FRAMES
from go2nav.sim.obstacles import RADIUS_M

BODY_M = .3


def planar(position, rotation):
    """(x, y, yaw) of a pose in the initial-body frame."""
    return float(position[0]), float(position[1]), math.atan2(rotation[1][0], rotation[0][0])


def relative(previous, current):
    """`current` (x, y, yaw) in the body frame of `previous`."""
    c, s = math.cos(previous[2]), math.sin(previous[2])
    dx, dy = current[0]-previous[0], current[1]-previous[1]
    return c*dx+s*dy, -s*dx+c*dy, math.atan2(math.sin(current[2]-previous[2]), math.cos(current[2]-previous[2]))


def out_of_view(x, y):
    """A body-frame point outside the primary camera's field of view and range."""
    xc = x-mp.CAMERA_X_M
    distance = math.hypot(xc, y)
    return not (.2 < distance < mp.MAX_VIEW_M and abs(math.atan2(y, xc)) <= mp.HALF_FOV_RAD)


def groups(probability, threshold):
    """(x, y, peak) body-frame centres of each 4-connected group of grid cells above `threshold`."""
    labels, n = ndimage.label(probability > threshold)
    out = []
    for k in range(1, n+1):
        w = np.where(labels == k, probability, 0.).ravel()
        x, y = (w[:, None]*ml.GRID).sum(0)/w.sum()
        out.append((float(x), float(y), float(w.max())))
    return out


class MemoryRunner:
    def __init__(self, memory, world_model, *, threshold, device='cuda:0'):
        self.device = torch.device(device)
        self.memory_model, self.world_model = memory.to(self.device).eval(), world_model.to(self.device).eval()
        self.threshold = threshold
        self.state, self.last_pose, self.probability = None, None, None
        self.log = []

    @torch.inference_mode()
    def step(self, frame, packet, position, rotation):
        """One memory step on `packet` (a policy packet) at a registered pose (initial-body frame)."""
        if frame % STEP_FRAMES:
            return False
        pose = planar(position, rotation)
        motion = (0., 0., 0.) if self.last_pose is None else relative(self.last_pose, pose)
        pixels = torch.from_numpy(resize_frame(packet['image']['rgb'])[None]).to(self.device)
        autocast = dict(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda')
        with torch.autocast(**autocast):
            state = self.world_model.encode(normalise_frames(pixels)[:, None])[:, 0].float()
        if self.state is None:
            self.state = self.memory_model.initial(1, self.device)
        with torch.autocast(**autocast):
            self.state = self.memory_model.step(self.state, state, torch.tensor([motion], device=self.device))
            logits = self.memory_model.occupancy(self.state)
        self.probability = torch.sigmoid(logits[0].float()).cpu().numpy()
        self.last_pose = pose
        return True

    def remembered(self, position, rotation, map_from_initial):
        """Map-frame centres of the remembered walkers out of view (see the module docstring), and their peaks."""
        if self.probability is None:
            return []
        B = np.asarray(map_from_initial, float)
        out = []
        for x, y, peak in groups(self.probability, self.threshold):
            if math.hypot(x, y) < BODY_M or not out_of_view(x, y):
                continue
            out.append(((B@(np.asarray(position, float)+np.asarray(rotation, float)@np.array([x, y, 0.])))[:2], peak))
        return out

    @staticmethod
    def cells(remembered):
        """The 1-cm cells of the remembered walkers' discs."""
        return frozenset().union(*(disk(c, RADIUS_M) for c, _ in remembered)) if remembered else frozenset()
