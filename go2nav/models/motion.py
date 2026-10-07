"""Motion heads: from V-JEPA features to the body-frame motion (x, y, yaw) of a candidate action.

All heads work on pooled tokens (B, 192, 1024) (`vjepa.pool_tokens`) and predict the motion at one horizon, in units of
metres and radians, through a per-channel normalisation (`target_mean`, `target_scale`) fitted on training data.

- `MotionReadout` (T5): reads motion from the current frame and the predictor's predicted future frame:
  [current, future - current] per token, projected to 32 channels, then an MLP. It is what makes the JEPA predictor a
  motion forecaster.
- `MotionDecoder` (C3's decoder, T7): a larger readout that also sees how the scene changed over the last second
  ([current - frame 0.5 s ago, current - frame 1 s ago]). It is built from a trained `MotionReadout` so that it starts
  exactly equal to it: copied weights in the first channels and units, zero weights everywhere else.
- `DirectMotionPredictor` (C4, T7): the supervised alternative to C3. It sees the same three frames' features (no
  predictor), the applied-command history and the candidate command tape, and regresses motion directly.

Parameter names match the original modules, so the shipped checkpoints load directly.
"""
import copy

import torch
from torch import nn

BASE_PROJ, BASE_HIDDEN = 32, 128
POOLED_TOKENS = 192


class MotionReadout(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(2048, BASE_PROJ), nn.GELU())
        self.decode = nn.Sequential(nn.Flatten(1), nn.Linear(POOLED_TOKENS*BASE_PROJ, BASE_HIDDEN), nn.GELU(),
                                    nn.Linear(BASE_HIDDEN, 3))
        self.register_buffer('target_mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('target_scale', torch.as_tensor(scale, dtype=torch.float32).clone())

    def normalized(self, current, future):
        return self.decode(self.project(torch.cat((current, future-current), dim=-1)))

    def forward(self, current, future):
        return self.normalized(current, future)*self.target_scale+self.target_mean


class MotionDecoder(nn.Module):
    """A `MotionReadout` grown in width (`proj` channels per token, `hidden` units) and depth (residual blocks), with
    optional extra inputs. Every added weight starts at zero, so the decoder starts exactly equal to `base`.

    - `past_frames`: per token also [current - past05, current - past10] (pooled features 0.5 s and 1 s ago).
    - `history`: the normalised 1.5-s applied-command history, embedded and added to the hidden layer.
    The deployed C3 decoder uses past_frames, proj 64, hidden 896, depth 1.
    """

    def __init__(self, base, past_frames=False, history=False, proj=BASE_PROJ, hidden=BASE_HIDDEN, depth=0):
        super().__init__()
        self.past_frames, self.history = past_frames, history
        self.config = dict(proj=proj, hidden=hidden, depth=depth)
        base = copy.deepcopy(base)
        self.register_buffer('target_mean', base.target_mean.clone())
        self.register_buffer('target_scale', base.target_scale.clone())
        linear, base_hidden, base_out = base.project[0], base.decode[1], base.decode[3]
        self.project = nn.Sequential(nn.Linear(4096 if past_frames else 2048, proj), nn.GELU())
        self.flatten, self.act = base.decode[0], base.decode[2]
        self.hidden = nn.Linear(POOLED_TOKENS*proj, hidden)
        self.out = nn.Linear(hidden, 3)
        with torch.no_grad():
            self.project[0].weight[:BASE_PROJ].zero_()
            self.project[0].weight[:BASE_PROJ, :2048] = linear.weight
            self.project[0].bias[:BASE_PROJ] = linear.bias
            w = self.hidden.weight.view(hidden, POOLED_TOKENS, proj)  # token-major flatten: input index t*proj + c
            w[:BASE_HIDDEN].zero_()
            w[:BASE_HIDDEN, :, :BASE_PROJ] = base_hidden.weight.view(BASE_HIDDEN, POOLED_TOKENS, BASE_PROJ)
            self.hidden.bias[:BASE_HIDDEN] = base_hidden.bias
            self.out.weight.zero_()
            self.out.weight[:, :BASE_HIDDEN] = base_out.weight
            self.out.bias.copy_(base_out.bias)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
                                    for _ in range(depth))
        with torch.no_grad():
            for block in self.blocks:
                block[2].weight.zero_()
                block[2].bias.zero_()
        if history:
            self.history_embedding = nn.Sequential(nn.Linear(30, 64), nn.GELU(), nn.Linear(64, hidden))
            with torch.no_grad():
                self.history_embedding[2].weight.zero_()
                self.history_embedding[2].bias.zero_()

    def normalized(self, current, future, past05=None, past10=None, history=None):
        parts = [current, future-current]
        if self.past_frames:
            parts += [current-past05, current-past10]
        h = self.hidden(self.flatten(self.project(torch.cat(parts, dim=-1))))
        if self.history:
            h = h+self.history_embedding(history.reshape(len(history), -1))
        a = self.act(h)
        for block in self.blocks:
            a = a+block(a)
        return self.out(a)

    def forward(self, current, future, past05=None, past10=None, history=None):
        return self.normalized(current, future, past05, past10, history)*self.target_scale+self.target_mean


class DirectMotionPredictor(nn.Module):
    """C4: features of the three context frames + command history + the candidate's commands up to the horizon."""

    def __init__(self, target_mean, target_scale):
        super().__init__()
        self.projection = nn.Linear(1024, 32)
        self.network = nn.Sequential(nn.Linear(3*POOLED_TOKENS*32+47, 896), nn.GELU(), nn.Linear(896, 896), nn.GELU(),
                                     nn.Linear(896, 3))
        self.register_buffer('target_mean', torch.as_tensor(target_mean, dtype=torch.float32).clone())
        self.register_buffer('target_scale', torch.as_tensor(target_scale, dtype=torch.float32).clone())

    def normalized(self, features, past_control, future_action, horizon):
        """features (B, 3, 192, 1024); past_control (B, 3, 5, 2); future_action (B, 8, 2); horizon (B,) in 1-8."""
        x = nn.functional.gelu(self.projection(features)).flatten(1)
        steps = torch.arange(8, device=features.device)[None, :]
        future = future_action*(steps < horizon[:, None])[:, :, None]
        condition = torch.cat((past_control.flatten(1), future.flatten(1), horizon[:, None].float()/8), dim=1)
        return self.network(torch.cat((x, condition), dim=1))

    def forward(self, features, past_control, future_action, horizon):
        return self.normalized(features, past_control, future_action, horizon)*self.target_scale+self.target_mean


def load_motion_readout(path):
    """A trained `MotionReadout` (T5) from {'model_state_dict': ...}."""
    state = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    readout = MotionReadout(state['target_mean'], state['target_scale'])
    readout.load_state_dict(state)
    return readout.eval().requires_grad_(False)


class ReadoutDecoder(nn.Module):
    """A plain `MotionReadout` behind the C3 decoder's interface: the v4-era C3 read motion from the current and
    predicted features alone (the C3-v2 fresh checks used one with C3-v2 weights)."""

    def __init__(self, readout):
        super().__init__()
        self.readout = readout

    def forward(self, current, future, past05=None, past10=None):
        return self.readout(current, future)


def load_c3_decoder(path, base_readout):
    """The C3 decoder of a fit file: a decoder-fix/stage-2 fit (`load_decoder_and_c4`) or a plain readout checkpoint
    ({'model_state_dict': ...}, wrapped in `ReadoutDecoder`)."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if 'readout_config' in state:
        return load_decoder_and_c4(path, base_readout)[0]
    return ReadoutDecoder(load_motion_readout(path)).eval()


def load_decoder_and_c4(path, base_readout):
    """The C3 decoder and the matched C4 from one fit file ({'readout', 'readout_config', 'variant', 'c4'})."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    decoder = MotionDecoder(base_readout, past_frames=state['variant'] == 'past_frames', history=state['variant'] == 'history',
                            **state['readout_config'])
    decoder.load_state_dict(state['readout'])
    c4 = DirectMotionPredictor(state['c4']['target_mean'], state['c4']['target_scale'])
    c4.load_state_dict(state['c4'])
    return decoder.eval().requires_grad_(False), c4.eval().requires_grad_(False)
