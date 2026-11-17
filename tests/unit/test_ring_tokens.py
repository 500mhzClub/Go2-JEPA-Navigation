"""P28 (docs/dynamic/PLAN.md P28): the ring world's object-token JEPA and its measures, on made-up scenes."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')


def batch(b=8, steps=16):
    from go2nav.models.object_jepa import ACTION_DIM, OBJECT_DIM, ROBOT_DIM
    g = torch.Generator().manual_seed(0)
    r = lambda *shape: torch.randn(*shape, generator=g)
    seen = (torch.rand(b, 1, 6, generator=g) < .4).expand(-1, 3, -1)
    valid = seen[:, :1].expand(-1, steps, -1) & (torch.rand(b, 1, 6, generator=g) < .8)
    flag = lambda x, m: torch.cat((x[..., :2], m[..., None].float(), m[..., None].float(), x[..., 4:]), -1)*m[..., None]
    return dict(objects=flag(r(b, 3, 6, OBJECT_DIM+2), seen), robot=r(b, 3, ROBOT_DIM), actions=r(b, 2+steps, ACTION_DIM),
                future_robot=r(b, steps, ROBOT_DIM), future_objects=flag(r(b, steps, 6, OBJECT_DIM+2), valid),
                truth=r(b, steps, 6, 2), valid=valid, seen=seen, present=seen)


def test_the_tokens_roll_out_in_latent_space_with_the_robot_s_token_given():
    from go2nav.models.ring_tokens import PositionHead, RingTokenJEPA
    model, b = RingTokenJEPA().eval(), batch()
    z = model.encode(b['objects'], b['robot'])
    robot = model.encode_robot(b['future_robot'])
    assert z.shape == (8, 3, 7, 128) and robot.shape == (8, 16, 128)
    alive = b['present'][:, -1]
    rolled = model.rollout(z, b['actions'], robot, 16, alive)
    assert rolled.shape == (8, 16, 7, 128) and PositionHead(model.config)(rolled[:, :, 1:]).shape == (8, 16, 6, 2)
    assert torch.equal(z[:, :, 1:][~b['present']], model.absent.expand_as(z[:, :, 1:][~b['present']]))   # empty slots
    assert torch.equal(rolled[:, :, 1:][~alive[:, None].expand(-1, 16, -1)][0], model.absent)
    other = model.rollout(z, b['actions'], robot.flip(0), 16, alive)             # another path for the robot from step 2 on
    assert torch.allclose(rolled[:, 0], other[:, 0], atol=1e-5) and not torch.allclose(rolled[:, 1], other[:, 1], atol=1e-4)


def test_the_objective_counts_only_tokens_with_a_target():
    from go2nav.models.lewm import SIGReg
    from go2nav.models.ring_tokens import RingTokenJEPA, token_losses
    model, b, sigreg = RingTokenJEPA(), batch(), SIGReg(num_proj=16)
    g = lambda: torch.Generator().manual_seed(1)
    out = token_losses(model, b, rollout_steps=8, sigreg=sigreg, lam=.09, generator=g())
    out['loss'].backward()
    assert all(torch.isfinite(v) for v in out.values())
    spoilt = b['future_objects'].clone()
    spoilt[..., :2] = torch.where(b['valid'][..., None], spoilt[..., :2], spoilt[..., :2]+5.)   # empty slots' positions
    changed = dict(b, future_objects=spoilt)
    model.eval()                                                          # BatchNorm's batch statistics would differ
    with torch.no_grad():
        a = token_losses(model, b, rollout_steps=8, sigreg=sigreg, lam=.09, generator=g())
        c = token_losses(model, changed, rollout_steps=8, sigreg=sigreg, lam=.09, generator=g())
    assert torch.allclose(a['rollout'], c['rollout'], atol=1e-6)          # tokens without a target do not count


def test_the_summary_scores_the_truth_as_perfect_and_holding_as_no_response():
    from go2nav.eval import ring_tokens
    rng = np.random.default_rng(0)
    n = 60
    history = rng.normal(0, 1, (n, 3, 6, 2)).astype(np.float32)
    drift = np.cumsum(rng.normal(0, .05, (n, 7, 16, 6, 2)), 2)
    valid = np.zeros((n, 7, 16, 6), bool)
    valid[:, :, :, :2] = True
    data = dict(maze=np.arange(n) % 12, target=((history[:, -1][:, None, None]+drift)*valid[..., None]).astype(np.float32),
                valid=valid, walker=np.where(np.arange(6) < 2, 0, -1)[None].repeat(n, 0).astype(np.int16),
                waiting=np.zeros((n, 6), bool), kind=(np.arange(n) % 3+1).astype(np.int8),
                future_commands=rng.normal(0, 1, (n, 7, 64, 3)).astype(np.float32), history_xy=history,
                history_velocity=np.zeros((n, 3, 6, 2), np.float32), history_seen=np.ones((n, 3, 6), bool))
    perfect = ring_tokens.summary(data, data['target'], draws=100)['3.2']
    assert perfect['fde']['median'] < 1e-6 and perfect['response']['error'] < 1e-6
    assert abs(perfect['response']['slope']-1) < 1e-6 and perfect['response']['false_rate'] in (0., None)
    still = ring_tokens.summary(data, np.broadcast_to(history[:, -1][:, None, None], data['target'].shape), draws=100)['3.2']
    assert still['fde']['median'] > .05 and abs(still['response']['error']-still['response']['zero']) < 1e-9
    assert '| truth | 3.2 s |' in ring_tokens.table(dict(truth={'3.2': perfect}))


def test_the_velocity_prior_starts_as_constant_velocity_in_latent_space():
    from go2nav.models.ring_tokens import RingTokenConfig, RingTokenJEPA
    from go2nav.training import recipe_bench as training
    toy = training.build('q4').eval()
    z = torch.randn(2, 3, 4, 128)
    out = toy.predict(z, torch.zeros(2, 3, 2))
    assert torch.allclose(out[:, 1:], 2*z[:, 1:]-z[:, :-1], atol=1e-5) and torch.allclose(out[:, 0], z[:, 0], atol=1e-5)
    ring = RingTokenJEPA(RingTokenConfig(residual=True, velocity=True)).eval()
    z = torch.randn(2, 3, 7, 128)
    z[0, 0, 2] = ring.absent.detach()                                      # slot 1 enters at the second step
    present = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1]]).bool()
    out = ring.predict(z, torch.zeros(2, 3, 12), present)
    assert torch.allclose(out[1, 2, 1], 2*z[1, 2, 1]-z[1, 1, 1], atol=1e-5)
    assert torch.allclose(out[0, 1, 2], z[0, 1, 2], atol=1e-5)          # next to an empty slot: no velocity
    assert torch.equal(out[0, 2, 3], ring.absent)                          # empty at the branch point: stays empty
