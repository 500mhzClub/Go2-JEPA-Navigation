"""The grid readout: grid tokens taken from the probe's flattened states in the right order, and the patch head fits."""
import numpy as np
import torch

from go2nav.eval import grid_readout as gr


def test_grid_tokens_take_the_cells_of_the_history_then_the_future_state():
    w, latent, grid = 3, 4, (3, 4)
    state = torch.arange(w*4*13*latent, dtype=torch.float32).reshape(w, 4, 13, latent)  # (windows, frames, tokens, D)
    future = -state[:, :2]                                                           # two horizons
    features = dict(zh=state.flatten(2).half(), true=future.flatten(2).half())
    rows = np.array([2, 0])
    history = gr.grid_tokens(features, 'true', rows, None, grid, latent)
    assert history.shape == (2, 4, 3, 4, latent)
    assert np.array_equal(history[0, 3, 1, 2], state[2, 3, 1+4*1+2].half().numpy())  # row 1, column 2: token 7
    stacked = gr.grid_tokens(features, 'true', rows, 1, grid, latent)
    assert stacked.shape == (2, 5, 3, 4, latent)
    assert np.array_equal(stacked[1, 4], future[0, 1, 1:].reshape(3, 4, latent).half().numpy())


def test_the_patch_head_learns_which_cell_holds_the_mover():
    rng = np.random.default_rng(0)
    n, latent = 600, 8
    x = rng.normal(0, .1, (n, 4, 3, 4, latent)).astype(np.float16)
    cell = rng.integers(0, 12, n)
    x[np.arange(n), :, cell//4, cell % 4, 0] += 2.                      # the mover's cell lights up channel 0
    y = np.stack((cell//4, cell % 4), 1).astype(np.float32)
    units = np.arange(n)//10
    predict = gr.fit(x, y, units, device='cpu', epochs=6)
    assert np.median(np.abs(predict(x)-y)) < .5
