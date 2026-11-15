import json

import numpy as np
import pytest
import torch

from go2nav.eval.slots_checkpoint import cache_parent, rollout_grouping, windows
from go2nav.eval.slots_recovery import finite_values
from go2nav.research import digest


def test_fixed_windows_are_distinct_split_specific_and_have_four_frames():
    for n in (1, 3, 6):
        train = windows((1000, 100), n, 'train')
        repeat = windows((1000, 100), n, 'train')
        val = windows((1000, 100), n, 'val')
        assert all(np.array_equal(a, b) for a, b in zip(train, repeat))
        assert not np.array_equal(train[0], val[0])
        assert len(set(zip(*train))) == 16
        assert (train[1] >= 0).all() and (train[1]+3 < 100).all()
    with pytest.raises(ValueError):
        windows((1000, 100), 3, 'test')


def test_nonfinite_outputs_or_gradients_are_explicit_failures():
    finite_values({'output': torch.ones(2)}, 'outputs')
    for value in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(FloatingPointError, match='weight'):
            finite_values({'weight': torch.tensor([value])}, 'gradients')


def test_candidate_grouping_detects_cross_batch_dependence():
    class Model:
        def __init__(self, mix=False):
            self.mix = mix

        def rollout(self, history, actions, steps):
            values = actions[:, 2:2+steps, :1, None]
            if self.mix:
                values = values.mean(0, keepdim=True)
            return history[:, -1:]+values

    history, actions = torch.zeros(100, 3, 2, 4), torch.arange(2000).reshape(100, 10, 2).float()/100
    assert rollout_grouping(Model(), history, actions)['passed']
    failed = rollout_grouping(Model(mix=True), history, actions)
    assert not failed['passed'] and not failed['checks']['separate']['passed']


def test_cache_bytes_and_training_identity_are_verified(tmp_path):
    cache, data = tmp_path/'cache', tmp_path/'data'
    cache.mkdir()
    encoder = tmp_path/'encoder.pt'
    encoder.write_bytes(b'frozen encoder')
    index = dict(n=3, seed=0, encoder=str(encoder), encoder_sha256=digest(encoder), splits={}, data_metadata={})
    for split in ('train', 'val'):
        (cache/f'{split}.npy').write_bytes(b'frozen states')
        index['splits'][split] = digest(cache/f'{split}.npy')
        folder = data/'n3'/split
        folder.mkdir(parents=True)
        (folder/'meta.json').write_text('{}')
        index['data_metadata'][f'{split}/meta.json'] = digest(folder/'meta.json')
    (cache/'index.json').write_text(json.dumps(index))
    state = dict(cache=str(cache), cache_sha256=digest(cache/'index.json'), n=3, seed=0)
    assert cache_parent(state, data)[0] == cache
    with pytest.raises(ValueError, match='n/seed'):
        cache_parent(dict(state, seed=1), data)
    (cache/'train.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        cache_parent(state, data)
