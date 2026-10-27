"""LeWM forecasters: C5 (the world model's rollout read by its decoder) and C6 (its matched direct head).

**C5** (`LeWMForecaster`). The four latest frames (t-300 .. t ms, the policy packets every forecaster receives) are
resized to 224 x 168 (`models.lewm.resize_frame`, as in training) and encoded. The world model then rolls out eight
100-ms steps for each candidate (the six actions, or any candidate set such as the command lattice; all in one
batch). The actions after the history frames are the
three applied commands the packet records for the last three ticks; the candidates' are their planned tapes after
the robot's safety limits, as the locomotion layer would execute them (as for C3). Each tick's command fills its
five-step action block. The decoder (`LatentMotionDecoder`) reads the motion at each horizon from the latents alone.

**C6** (`LeWMDirectForecaster`). The same four frames' z, the normalised 1.5-s applied-command history and each
candidate's commands go straight to the motion head (`LatentDirectPredictor`), all N x 8 forecasts in one batch.

A grid world model (P4b) rolls out its whole state; the heads read [CLS]'s z (`LeWorldModel.cls`). Both return
(N, 8, 5) float32: x, y, sin yaw, cos yaw, and the contact logit -1000 (contact is not predicted).
"""
import numpy as np
import torch

from go2nav.controllers.neural import applied_history, planned_commands
from go2nav.locomotion.safety import apply_limits
from go2nav.models.lewm import COMMAND_AXES, control_stats, normalise_frames, resize_frame
from go2nav.planning.actions import ACTIONS

FRAME_SPACING_NS = 100_000_000


def output(motion):
    """(N, 8, 3) [x, y, yaw] to the (N, 8, 5) forecast."""
    return torch.cat((motion[:, :, :2], motion[:, :, 2:3].sin(), motion[:, :, 2:3].cos(),
                      torch.full((*motion.shape[:2], 1), -1000.)), dim=-1).numpy().astype(np.float32)


class _LeWMInputs:
    def __init__(self, world_model, device):
        self.device = torch.device(device)
        self.model = world_model.to(self.device).eval()
        dim = self.model.config.command_dim
        self.axes = COMMAND_AXES[dim]
        self.mean, self.std = (torch.as_tensor(v, device=self.device) for v in control_stats(dim))

    def inputs(self, context, prefix, pulse, candidates=ACTIONS):
        """The history frames' state (1, 4, D), or (1, 4, 1+G, D) for a grid model, the applied history (15, 3), and
        the candidates' executed tapes (N, 8, 3)."""
        packets = context.history
        times = [p['image']['measured_ns'] for p in packets]
        if len(packets) != self.model.config.history or np.any(np.diff(times) != FRAME_SPACING_NS):
            raise ValueError('four frames 100 ms apart (t-300 .. t ms) required')
        history = applied_history(packets[-1])
        executed = apply_limits(planned_commands(prefix, pulse, candidates),
                                np.broadcast_to(history[-1], (len(candidates), 3)))
        frames = torch.from_numpy(np.stack([resize_frame(p['image']['rgb']) for p in packets])).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            z = self.model.encode(normalise_frames(frames)[None]).float()
        return z, history, executed

    def normalised(self, commands):
        """Commands (..., 3) to the normalised components the model takes: (forward, yaw) (..., 2), or all three."""
        commands = torch.as_tensor(np.asarray(commands)[..., self.axes], dtype=torch.float32, device=self.device)
        return (commands-self.mean)/self.std


class LeWMForecaster(_LeWMInputs):
    """C5: see the module docstring."""

    def __init__(self, world_model, decoder, *, device='cuda:0'):
        super().__init__(world_model, device)
        self.decoder = decoder.to(self.device).eval()

    @torch.inference_mode()
    def forecast(self, context, prefix, *, pulse, candidates=ACTIONS):
        z, history, executed = self.inputs(context, prefix, pulse, candidates)
        n = len(executed)
        past = self.normalised(history[-3:])[None].expand(n, -1, -1)
        commands = torch.cat((past, self.normalised(executed)), dim=1)            # (N, 11, command_dim)
        actions = commands[:, :, None].expand(-1, -1, 5, -1).flatten(2)            # each tick's command, five times
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            rolled = self.model.rollout(z.expand(n, *z.shape[1:]), actions, 8)
        rolled, z = self.model.cls(rolled).float(), self.model.cls(z)             # (N, 8, D), (1, 4, D)
        motion = self.decoder(z.expand(n*8, -1, -1), rolled.reshape(n*8, -1))     # candidate-major, horizons 1-8
        return output(motion.reshape(n, 8, 3).cpu())


class LeWMDirectForecaster(_LeWMInputs):
    """C6: see the module docstring."""

    def __init__(self, world_model, head, *, device='cuda:0'):
        super().__init__(world_model, device)
        self.head = head.to(self.device).eval()

    @torch.inference_mode()
    def forecast(self, context, prefix, *, pulse, candidates=ACTIONS):
        z, history, executed = self.inputs(context, prefix, pulse, candidates)
        n, z = len(executed), self.model.cls(z)
        past = self.normalised(history)[None].expand(n*8, -1, -1)
        tapes = self.normalised(executed)[:, None].expand(-1, 8, -1, -1).reshape(n*8, 8, len(self.axes))
        horizons = torch.arange(1, 9, device=self.device).repeat(n)
        motion = self.head(z.expand(n*8, -1, -1), past, tapes, horizons)
        return output(motion.reshape(n, 8, 3).cpu())
