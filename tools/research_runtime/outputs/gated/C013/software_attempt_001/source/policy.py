"""C013 explicit inference-only precision policies; no training-state mutation."""
import copy
import torch
from go2nav.models.balls_slots import SlotWorld
from go2nav.training.balls_slots import load_world

POLICIES = ('encoder32_predictor64', 'encoder64_predictor64')


class PrecisionWorld(SlotWorld):
    def __init__(self, world, policy):
        if policy not in POLICIES:
            raise ValueError('Unknown inference precision policy')
        super().__init__(copy.deepcopy(world.encoder), copy.deepcopy(world.predictor))
        self.policy = policy
        self.encoder.to(dtype=torch.float32 if policy == POLICIES[0] else torch.float64)
        self.predictor.double()
        self.eval()

    def encode(self, pixels):
        parameter = next(self.encoder.parameters())
        return self.encoder(pixels.to(device=parameter.device, dtype=parameter.dtype)).double()

    def rollout(self, history, actions, steps):
        parameter = next(self.predictor.parameters())
        return self.predictor.rollout(history.to(parameter), actions.to(parameter), steps)


def load_policy(checkpoint, policy, device='cpu'):
    # Construct on CPU, preserve original values, then move one fresh policy copy.
    return PrecisionWorld(load_world(checkpoint, 'cpu'), policy).to(device).eval()


def choose(results):
    if set(results) != set(POLICIES) or not all(v.get('complete') for v in results.values()):
        raise ValueError('Both complete policy outcomes are required')
    return next((p for p in POLICIES if results[p]['checks_passed']), None)
