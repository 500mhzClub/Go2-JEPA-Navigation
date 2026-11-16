"""F004 physical forecasting: shared observable inputs and matched paired losses.

This module supplies the scientific training primitives. Full GPU execution needs
the separately qualified F004 supervisor; importing this module launches nothing.
"""
import copy
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from go2nav.data.paired_forecasts import INPUTS, need
from go2nav.models.object_jepa import (OBJECT_GRID, OBJECT_SPACING, ROBOT_GRID, ROBOT_SPACING,
                                     actions_input, object_features, offsets, robot_features)
from go2nav.training.object_jepa import rotate

ARMS = ('factual', 'branch', 'paired')
REPORT = (3, 7, 15)  # Zero-based endpoint indices: 1.6, 3.2, 6.4 seconds.


class Sampler:
    def __init__(self, n, seed, arm):
        need(arm in ARMS and n > 0, 'Invalid sampler arm/population')
        self.n, self.arm = n, arm
        self.rng = np.random.default_rng(seed)

    def draw(self, pairs):
        point = self.rng.integers(self.n, size=pairs)
        a = self.rng.integers(6, size=pairs)
        b = self.rng.integers(5, size=pairs)
        b += b >= a
        future = np.stack((a, b), -1)
        order = np.argsort(self.rng.random((pairs, 6)), axis=-1)
        if self.arm == 'factual':
            future[:] = 6
        return np.repeat(point, 2), future.reshape(-1), np.repeat(order, 2, axis=0)

    def state_dict(self):
        return dict(n=self.n, arm=self.arm, rng=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, state):
        need(state['n'] == self.n and state['arm'] == self.arm, 'Incompatible sampler state')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])


def shuffle_slots(tensor, order):
    """History/target tensors have slot axis 2; order is one shared permutation per pair."""
    shape = (order.shape[0], 1, order.shape[1], *((1,)*(tensor.ndim-3)))
    index = order.reshape(shape).expand(tensor.shape[0], tensor.shape[1], tensor.shape[2], *tensor.shape[3:])
    return torch.gather(tensor, 2, index)


def features(raw, rasters, device, order=None):
    """Build ONLY observable/candidate features; no target, identity match or censor flags read."""
    x = {k: torch.as_tensor(np.array(raw[k], copy=True), device=device) for k in INPUTS}
    pose, xy = x['history_pose'], x['history_xy']
    origin, heading = pose[:, -1, :2], pose[:, -1, 2]
    layout = torch.as_tensor([rasters.index[int(m)] for m in raw['maze']], device=device)
    same = (x['history_id'] == x['history_id'][:, -1:]) & (x['history_id'][:, -1:] >= 0)
    seen = x['history_seen'] & same
    position = rotate(xy-origin[:, None, None], -heading[:, None, None])
    velocity = rotate(x['history_velocity'], -heading[:, None, None])*seen[..., None]
    object_grid = rotate(offsets(OBJECT_GRID, OBJECT_SPACING, device), heading[:, None])
    wall_values = rasters.lookup(layout, torch.nan_to_num(xy)[..., None, :]+object_grid[:, None, None])
    objects = object_features(position, seen, same, wall_values, velocity)
    if order is not None:
        order = torch.as_tensor(order, dtype=torch.long, device=device)
        objects = shuffle_slots(objects, order)
    poses = torch.cat((pose, x['future_pose']), 1)
    robot_grid = rotate(offsets(ROBOT_GRID, ROBOT_SPACING, device)[None, None], poses[..., 2, None])
    robot_walls = rasters.lookup(layout, poses[..., None, :2]+robot_grid)
    robot = robot_features(rotate(poses[..., :2]-origin[:, None], -heading[:, None]),
                           poses[..., 2]-heading[:, None], robot_walls)
    commands = torch.cat((x['past_commands'], x['future_commands']), 1).reshape(len(pose), 18, 4, 3)
    return dict(objects=objects, robot=robot[:, :3], actions=actions_input(commands), future_robot=robot[:, 3:],
                layout=layout, origin=origin, heading=heading)


def targets(raw, inputs, order=None):
    device = inputs['origin'].device
    target = torch.as_tensor(np.array(raw['target'], copy=True), device=device)
    valid = torch.as_tensor(np.array(raw['valid'], copy=True), device=device)
    target = rotate(target-inputs['origin'][:, None, None], -inputs['heading'][:, None, None])
    target = torch.where(valid[..., None], target, 0)
    if order is not None:
        order = torch.as_tensor(order, dtype=torch.long, device=device)
        target, valid = shuffle_slots(target, order), shuffle_slots(valid, order)
    return target, valid


def forecast(model, inputs, rasters, steps=16):
    def walls(position):
        world = rotate(position, inputs['heading'][:, None])+inputs['origin'][:, None]
        grid = rotate(offsets(OBJECT_GRID, OBJECT_SPACING, position.device), inputs['heading'][:, None])
        return rasters.lookup(inputs['layout'], world[..., None, :]+grid[:, None])
    return model.rollout(inputs['objects'], inputs['robot'], inputs['actions'], inputs['future_robot'], walls, steps)


def counts(valid):
    need(valid.ndim == 3 and len(valid) % 2 == 0, 'Unpaired validity masks')
    pair = valid.reshape(-1, 2, *valid.shape[1:])
    return valid.sum(), (pair[:, 0, REPORT] & pair[:, 1, REPORT]).sum()


def objective(prediction, target, valid, arm, denominators=None):
    need(arm in ARMS, 'Unknown arm')
    trajectory_count, pair_count = counts(valid) if denominators is None else denominators
    trajectory = (torch.linalg.vector_norm(prediction-target, dim=-1)*valid).sum()/trajectory_count.clamp(min=1)
    p, y = [x.reshape(-1, 2, *x.shape[1:]) for x in (prediction, target)]
    mask = valid.reshape(-1, 2, *valid.shape[1:])
    both = mask[:, 0, REPORT] & mask[:, 1, REPORT]
    error = (p[:, 1, REPORT]-p[:, 0, REPORT]) - (y[:, 1, REPORT]-y[:, 0, REPORT])
    response = (torch.linalg.vector_norm(error, dim=-1)*both).sum()/pair_count.clamp(min=1)
    return trajectory+(response if arm == 'paired' else 0), trajectory.detach(), response.detach()


def rate(step, *, total=20000, warmup=500):
    """Multiplier for update index0..total-1, ending exactly at zero."""
    if step < warmup:
        return (step+1)/warmup
    return .5*(1+math.cos(math.pi*(step-warmup)/(total-warmup-1)))


def save_checkpoint(path, model, optimizer, sampler, step, binding):
    """Save a restartable local checkpoint atomically; scheduler is a pure function of step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    payload = dict(model=model.state_dict(), config=model.config.to_dict(), optimizer=optimizer.state_dict(),
                   sampler=sampler.state_dict(), step=step, binding=binding,
                   scheduler=dict(next_update=step, formula='F004.rate'),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [])
    try:
        with temp.open('wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def restore_checkpoint(path, model, optimizer, sampler, binding):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    need(checkpoint['binding'] == binding and checkpoint['config'] == model.config.to_dict(), 'Incompatible checkpoint source/data/config')
    need(checkpoint['scheduler'] == dict(next_update=checkpoint['step'], formula='F004.rate'), 'Bad scheduler state')
    need(checkpoint['sampler']['arm'] == sampler.arm and checkpoint['sampler']['n'] == sampler.n, 'Incompatible sampler state')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    sampler.load_state_dict(checkpoint['sampler'])
    torch.set_rng_state(checkpoint['torch_rng'])
    if checkpoint['cuda_rng']:
        need(next(model.parameters()).is_cuda, 'GPU checkpoint cannot resume CPU training')
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint['step']
