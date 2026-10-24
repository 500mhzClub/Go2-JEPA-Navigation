"""Motion forecasters: what each controller predicts each candidate action will do.

A forecaster returns, for the six actions (`go2nav.planning.actions`; C1 also for any candidate set) and eight 100-ms
horizons, the body-frame motion
(x, y, sin yaw, cos yaw, contact logit) from the plan's frame, as float32 (as the original models produced it). Contact
is not predicted (the logit is -1000). The controllers differ only here:

- **C1** (`CommandHistoryForecaster`): the command tape alone, through the fitted command-history model (T6).
- **C3** / **C4**: learned from images (`go2nav.controllers.neural`).
- **C0**: the simulator itself (`go2nav.controllers.oracle`).
"""
import numpy as np

from go2nav.models.command import CommandHistoryModel, history_features
from go2nav.planning.actions import ACTIONS, command_sequences


def output(xy_yaw):
    """(N, 8, 5) float32 forecast from (N, 8, 3) [x, y, yaw]."""
    result = np.zeros((len(xy_yaw), 8, 5), np.float32)
    result[:, :, 3] = 1.
    result[:, :, :2] = xy_yaw[:, :, :2]
    result[:, :, 2], result[:, :, 3] = np.sin(xy_yaw[:, :, 2]), np.cos(xy_yaw[:, :, 2])
    result[:, :, 4] = -1000.
    return result


def command_history(packets):
    """The command model's 420 history features from the last four frames' policy packets (t-300 .. t ms)."""
    rows = [p['sensor_state']['control']['applied_command'] for p in packets]
    return history_features([r['values'] for r in rows], [r['valid'] for r in rows], [r['measured_ns'] for r in rows],
                            [p['sensor_state']['decision_ns'] for p in packets])


class CommandHistoryForecaster:
    """C1: nominal integration of each command tape plus the learned command-history correction."""

    def __init__(self, model_path):
        self.model = CommandHistoryModel.load(model_path)

    def forecast(self, context, prefix, *, pulse, candidates=ACTIONS):
        commands = command_sequences(prefix, pulse=pulse, candidates=candidates)
        return output(self.model.predict(commands, command_history(context.history)))
