import copy

import torch

from go2nav.eval.slots_recovery import mismatches


def test_changed_optimizer_state_and_missing_key_are_not_hidden():
    expected = dict(model={'weight': torch.ones(2)}, optimizer={'state': {0: {'exp_avg': torch.zeros(2)}}})
    actual = copy.deepcopy(expected)
    assert not mismatches(expected, actual)
    actual['optimizer']['state'][0]['exp_avg'][1] = .01
    assert mismatches(expected, actual) == ['/optimizer/state/0/exp_avg']
    del actual['model']['weight']
    assert '/model/keys' in mismatches(expected, actual)


def test_rng_is_exact_and_nonfinite_state_cannot_pass():
    expected = {'draw': torch.tensor([1., 2.])}
    assert not mismatches(expected, copy.deepcopy(expected), rtol=0, atol=0)
    assert mismatches(expected, {'draw': torch.tensor([1., 2.+1e-6])}, rtol=0, atol=0)
    assert mismatches(torch.tensor([float('nan')]), torch.tensor([float('nan')]))
    assert mismatches(torch.zeros(2), torch.zeros(1, 2))
