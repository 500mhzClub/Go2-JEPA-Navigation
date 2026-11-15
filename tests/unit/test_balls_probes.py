import numpy as np
import torch

from go2nav.eval.balls_probes import apply_linear, correlation, fit_linear, targets


def test_probe_target_order_is_fixed_without_matching_test_predictions():
    red = np.array([[4., 5.]])
    white = np.array([[[3., 8.], [2., 7.], [2., 1.]]])
    expected = np.array([[4, 5, 2, 1, 2, 7, 3, 8]])
    assert np.array_equal(targets(red, white), expected)
    assert np.array_equal(targets(red, white[:, [2, 0, 1]]), expected)


def test_train_fitted_probe_applies_same_transform_to_heldout_data():
    torch.manual_seed(42)
    train, test = torch.randn(64, 4), torch.randn(20, 4)+3
    w = torch.randn(4, 2)
    decoder = fit_linear(train, train@w+2)
    prediction = apply_linear(test, decoder).numpy()
    truth = (test@w+2).numpy()
    assert np.min(correlation(prediction, truth)) > .999
    assert np.max(abs(prediction-truth)) < .03
