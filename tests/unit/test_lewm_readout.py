"""The LeWM base readout: trains both heads and round-trips into the C5 decoder's shape."""
import torch

from go2nav.models.lewm import LatentMotionDecoder
from go2nav.training import lewm_readout


def test_heads_train_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(lewm_readout, 'UPDATES', 50)
    monkeypatch.setattr(lewm_readout, 'BATCH', 16)
    g = torch.Generator().manual_seed(0)
    data = dict(zh=torch.randn(64, 4, 8, generator=g), true=torch.randn(64, 8, 8, generator=g),
                rollout=torch.randn(64, 8, 8, generator=g), y=torch.randn(64, 8, 3, generator=g))
    heads = {kind: lewm_readout.train_head(data, kind, 1, 'cpu')[0] for kind in ('rollout', 'true')}
    torch.save(dict({k: h.state_dict() for k, h in heads.items()}, latent=8), tmp_path/'readout.pt')
    base = lewm_readout.load_base(tmp_path/'readout.pt', 'rollout')
    decoder = LatentMotionDecoder(8, torch.zeros(3), torch.ones(3))
    decoder.load_state_dict(base.state_dict())
    assert torch.equal(decoder.target_mean, heads['rollout'].target_mean)  # the base's normalisation comes along
    z, future = torch.randn(5, 4, 8), torch.randn(5, 8)
    assert torch.allclose(decoder(z, future), heads['rollout'](z, future))
