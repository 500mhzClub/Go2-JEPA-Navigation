import numpy as np
import pytest
import torch

from go2nav.eval.balls_interim import action_controls, paired_gain, sample


def test_interim_samples_are_fixed_and_exclude_test_and_future_overflow():
    episodes, starts = sample(100, 100, 'val')
    np.testing.assert_array_equal(episodes, np.repeat(np.arange(100), 5))
    np.testing.assert_array_equal(starts, sample(100, 100, 'val')[1])
    assert starts.min() >= 2 and starts.max()+4 < 100
    assert len(sample(1000, 100, 'train')[0]) == 4096
    with pytest.raises(ValueError, match='never reads'):
        sample(1000, 100, 'test')


def test_action_controls_preserve_history_and_input():
    actions = torch.arange(24).reshape(2, 6, 2).double()
    before = actions.clone()
    controls = action_controls(actions)
    assert torch.equal(actions, before)
    for control in controls.values():
        assert torch.equal(control[:, :2], actions[:, :2])
    assert torch.equal(controls['reversed'][:, 2:], actions[:, 2:].flip(1))
    assert torch.count_nonzero(controls['zero'][:, 2:]) == 0
    assert torch.equal(controls['rollout'], actions)


def test_paired_gain_sign_and_episode_sampling():
    truth = np.zeros((10, 8))
    factual = np.ones_like(truth)
    baseline = np.full_like(truth, 2.)
    episodes = np.repeat([0, 1], 5)
    gain = paired_gain(baseline, factual, truth, episodes)
    assert gain['red'] == dict(mean=6., ci95=[6., 6.])
    assert gain['white'] == dict(mean=6., ci95=[6., 6.])
    reverse = paired_gain(factual, baseline, truth, episodes)
    assert reverse['red']['mean'] == -6.
