"""Learned forecasters from images: C3 (JEPA predictor + motion decoder) and C4 (direct).

**C3** (`JepaForecaster`). The three context frames (t-1 s, t-0.5 s, t) are encoded by the frozen V-JEPA 2.1 ViT-L and
layer-normalised. For each horizon h = 1..8 (100-800 ms), the predictor (T4) predicts the tokens at t+h from the
context, the applied-command history and each candidate's first h commands (each distinct command prefix once);
the decoder (T7) reads the motion from the current frame's and the predicted frame's pooled tokens (and the two past
frames'). Candidates' commands are the planned tapes after the robot's safety limits, as the locomotion layer would
execute them.
"""
import numpy as np
import torch
from torch.nn import functional as F

from go2nav.controllers.forecasters import output
from go2nav.locomotion.safety import apply_limits
from go2nav.models.vjepa import VJepaEncoder, normalise, pool_tokens, preprocess
from go2nav.planning.actions import COMMAND_SCALE, command_sequences

CONTEXT_OFFSETS_NS = (1_000_000_000, 500_000_000, 0)


def planned_commands(prefix, pulse):
    """(6, 8, 3) planned tapes as the original model interface rounded them: float32 commands scaled and unscaled in
    float64 (`apply_limits` then returns float32)."""
    scale = torch.tensor(COMMAND_SCALE)
    scaled = torch.as_tensor(command_sequences(prefix, pulse=pulse), dtype=torch.float32)/scale
    return (scaled*scale).numpy()


def applied_history(packet):
    """(15, 3) float32 applied commands at the newest context frame."""
    return np.asarray(packet['sensor_state']['control']['applied_command']['values'], np.float32)


class JepaForecaster:
    """C3: see the module docstring."""

    def __init__(self, predictor, decoder, control_mean, control_std, *, encoder=None, device='cuda:0'):
        self.device = torch.device(device)
        self.encoder = encoder or VJepaEncoder(device=self.device)
        self.predictor = predictor.to(self.device).eval()
        self.decoder = decoder.to(self.device).eval()
        self.control_mean = torch.as_tensor(control_mean, dtype=torch.float32, device=self.device)
        self.control_std = torch.as_tensor(control_std, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def forecast(self, context, prefix, *, pulse):
        requested = planned_commands(prefix, pulse)
        history = applied_history(context.native[-1])
        applied = apply_limits(requested, np.broadcast_to(history[-1], (6, 3)))
        pixels = torch.stack([preprocess(p['image']['rgb']) for p in context.native]).to(self.device)
        tokens = normalise(self.encoder.tokens(pixels))
        past10, past05, current = pool_tokens(tokens[0:1]), pool_tokens(tokens[1:2]), pool_tokens(tokens[2:3])
        frames = tokens[None]
        control = torch.from_numpy(history[:, [0, 2]].copy()).reshape(3, 5, 2).to(self.device)
        control = ((control-self.control_mean)/self.control_std)[None]
        actions = torch.from_numpy(applied[:, :, [0, 2]].copy()).to(self.device)
        mask = torch.ones(6, 768, dtype=torch.bool, device=self.device)
        motions = []
        for horizon in range(1, 9):
            unique, inverse = torch.unique(actions[:, :horizon].reshape(6, -1), dim=0, return_inverse=True)
            first = torch.stack([(inverse == j).nonzero()[0, 0] for j in range(len(unique))])
            n = len(unique)
            predicted = self.predictor(frames.expand(n, -1, -1, -1), actions[first],
                                       torch.full((n,), horizon, dtype=torch.long, device=self.device), mask[:n],
                                       control=control.expand(n, -1, -1, -1))
            predicted = F.layer_norm(predicted.float(), (1024,))
            motion = self.decoder(current.expand(n, -1, -1), pool_tokens(predicted), past05.expand(n, -1, -1),
                                  past10.expand(n, -1, -1))
            motions.append(motion[inverse])
        motion = torch.stack(motions, dim=1).cpu()
        return torch.cat((motion[:, :, :2], motion[:, :, 2:3].sin(), motion[:, :, 2:3].cos(),
                          torch.full((6, 8, 1), -1000.)), dim=-1).numpy()


class DirectForecaster:
    """C4: the motion at each horizon read directly from the three context frames' pooled tokens, the applied-command
    history and the candidate's commands (no predicted future frame); all 6 x 8 forecasts in one batch."""

    def __init__(self, predictor, control_mean, control_std, *, encoder=None, device='cuda:0'):
        self.device = torch.device(device)
        self.encoder = encoder or VJepaEncoder(device=self.device)
        self.predictor = predictor.to(self.device).eval()
        self.control_mean = torch.as_tensor(control_mean, dtype=torch.float32, device=self.device)
        self.control_std = torch.as_tensor(control_std, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def forecast(self, context, prefix, *, pulse):
        requested = planned_commands(prefix, pulse)
        history = applied_history(context.native[-1])
        applied = apply_limits(requested, np.broadcast_to(history[-1], (6, 3)))
        pixels = torch.stack([preprocess(p['image']['rgb']) for p in context.native]).to(self.device)
        features = pool_tokens(F.layer_norm(self.encoder.tokens(pixels).float(), (1024,)))
        past = torch.from_numpy(history[:, [0, 2]].copy()).reshape(3, 5, 2).to(self.device)
        past = (past-self.control_mean)/self.control_std
        actions = torch.tensor(applied[:, :, [0, 2]], device=self.device)
        horizons = torch.arange(1, 9, device=self.device).repeat(6)
        motion = self.predictor(features[None].expand(48, -1, -1, -1), past[None].expand(48, -1, -1, -1),
                                actions[:, None].expand(-1, 8, -1, -1).reshape(48, 8, 2), horizons).reshape(6, 8, 3).cpu()
        return torch.cat((motion[:, :, :2], motion[:, :, 2:3].sin(), motion[:, :, 2:3].cos(),
                          torch.full((6, 8, 1), -1000.)), dim=-1).numpy()
