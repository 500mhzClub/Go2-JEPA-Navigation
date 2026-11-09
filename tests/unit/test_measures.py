"""Measures that isolate prediction (eval.measures, training.lewm.input_shares; docs/dynamic/PLAN.md P16)."""
from types import SimpleNamespace

import numpy as np
import torch

from go2nav.data import observations as o
from go2nav.eval import measures as m
from go2nav.eval import mover_probe as mp
from go2nav.models.lewm import LeWMConfig, LeWorldModel
from go2nav.training.lewm import MAX_ACTION_SHARE, action_share, input_shares


def test_an_auc_passes_when_it_closes_half_the_gap_from_its_null_to_the_ceiling():
    assert m.half_gap(.8, .5, .9) == dict(skill=.75, passed=True)
    assert not m.half_gap(.69, .5, .9)['passed']
    assert not m.half_gap(.95, .9, .9)['passed']          # a ceiling no better than the null leaves nothing to close
    assert not m.half_gap(None, .5, .9)['passed']


def test_the_slope_separates_a_reaction_from_a_leak():
    rng = np.random.default_rng(0)
    true, groups = rng.normal(size=(400, 2))*.1, np.repeat(np.arange(40), 10)
    reacting = m.slope(.8*true+np.array([.03, 0.]), true, groups, draws=100)
    assert abs(reacting['slope']-.8) < .01 and reacting['slope_95'][0] > .5 and abs(reacting['offset_m']-.03) < .005
    leaking = m.slope(np.tile([-.1, 0.], (400, 1)), true, groups, draws=100)      # every walker moved alike
    assert abs(leaking['slope']) < 1e-6 and abs(leaking['offset_m']-.1) < 1e-6


def labels(windows=120, seed=0):
    """Give-way windows in a val and a transfer unit, the walker ahead; half respond to a retreat by following."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(12), windows//12)
    xy = np.column_stack([rng.uniform(.5, 1.2, windows), rng.uniform(-.3, .3, windows)])
    follows = rng.random(windows) < .5
    contrast = np.where(follows[:, None], [-.3, 0.], [0., 0.])+rng.normal(size=(windows, 2))*.002
    horizons = len(mp.HORIZONS)
    stay = np.repeat(xy[:, None], horizons, 1)
    return dict(units=np.array([f'units/lewm_val/{k:04d}_{"transfer" if k % 2 else "mission"}' for k in range(12)]),
                unit=unit, t=np.arange(windows), xy_t=xy, previous=np.zeros((windows, 3), np.float32),
                target=stay+rng.normal(size=(windows, horizons, 2))*.2, cf_hold=stay, cf_advance=stay.copy(),
                cf_retreat=stay+contrast[:, None], moving=np.zeros(windows, bool), waiting=follows,
                constant_velocity=stay), contrast


def features(data, displacement, tapes):
    """Probe features of a toy model whose rollout adds the walker's displacement to two latent dimensions."""
    windows, horizons = len(data['t']), len(mp.HORIZONS)
    z = torch.randn(windows, 4, 6, generator=torch.Generator().manual_seed(1))
    def rolled(d):
        out = z[:, -1:].repeat(1, horizons, 1)
        out[:, :, :2] += torch.as_tensor(d, dtype=torch.float32)
        return out.half()
    return dict(zh=z.half(), ok=torch.ones(windows, dtype=bool), true=rolled(displacement),
                executed=rolled(displacement), **{k: rolled(v) for k, v in tapes.items()})


def test_matched_windows_need_a_robot_that_stood_still():
    data, _ = labels()
    history = np.zeros((len(data['t']), 15, 3), np.float32)
    history[:10, -1, 0] = .2                                    # it was still moving 100 ms before
    data['previous'][10:20, 2] = .3                             # or turning just before t
    keep = m.matched(data, history, np.arange(len(data['t'])), 2)
    assert not keep[:20].any() and keep[20:].all()


def test_the_reaction_passes_for_a_model_that_predicts_it_and_fails_for_one_that_leaks():
    data, contrast = labels()
    history = np.zeros((len(data['t']), 15, 3), np.float32)
    executed = data['target']-data['xy_t'][:, None]
    zero = np.zeros_like(executed)
    ceiling = {tape: {h: dict(val=.95, transfer=.95) for h in m.HORIZON_S.values()} for tape in mp.GIVE_WAY_CONTRASTS}
    run = lambda tapes: m.reaction(data, features(data, executed, tapes), data, features(data, executed, tapes), history,
                                   ceiling, updates=20, draws=50)['ridge']['retreat']
    follows = np.repeat(contrast[:, None], len(mp.HORIZONS), 1)
    good = run(dict(hold=zero, advance=zero, retreat=follows))
    assert m.reaction_passes(good) and abs(good[1.6]['val']['slope']-1) < .05 and good[3.2]['transfer']['auc'] > .95
    leak = run(dict(hold=zero, advance=zero, retreat=zero+np.array([-.1, 0.])))
    assert not m.reaction_passes(leak) and abs(leak[1.6]['val']['slope']) < .05
    assert abs(leak[1.6]['val']['offset_m']-.1) < .01 and .25 < leak[1.6]['val']['auc'] < .75     # chance, on 60 windows


def test_new_contacts_score_the_rollout_against_the_present():
    rng = np.random.default_rng(0)
    def windows(count):
        later = rng.random(count) < .2
        zh = torch.randn(count, 4, 6)
        rolled = zh[:, -1].clone()
        rolled[:, 0] = torch.as_tensor(later, dtype=torch.float32)*3+torch.randn(count)*.1      # only the rollout knows
        return dict(zh=zh.half(), rolled=rolled.half(), contact_t=np.zeros(count, bool), contact_h=later,
                    unit=np.repeat(np.arange(count//20), 20), split=np.where(np.arange(count) % 2, 'val', 'transfer'),
                    history=rng.normal(size=(count, 15, 3)).astype(np.float32),
                    tape=rng.normal(size=(count, 8, 3)).astype(np.float32))
    torch.manual_seed(0)
    report = m.new_contacts(SimpleNamespace(heads=None), windows(800), windows(400), updates=200)
    assert report['best']['model']['val'] > .95 and report['best']['present_only']['val'] < .7
    assert report['best']['scene_blind']['transfer'] < .7 and report['pass']


def test_the_advantage_is_read_from_the_two_tapes_rollouts():
    rng = np.random.default_rng(0)
    def side(count):
        data, _ = labels(count)
        data.update(clear_side=(rng.random(count) < .5).astype(np.float32),
                    clear_hold_long=(rng.random(count) < .3).astype(np.float32), side_found=np.ones(count, np.float32))
        now = torch.randn(count, 6)
        def rolled(clear):
            out = now.clone()
            out[:, 0] += torch.as_tensor(clear)*3                                   # the rollout holds the outcome
            return out
        return data, dict(now=now, back=torch.randn(count, 6), side=rolled(data['clear_side']),
                          hold=rolled(data['clear_hold_long']), rows=np.arange(count),
                          side_tape=np.ones((count, 24), np.float32), hold_tape=np.zeros((count, 24), np.float32))
    torch.manual_seed(0)
    (dtrain, ftrain), (deval, feval) = side(600), side(360)
    history = np.zeros((600, 15, 3), np.float32)
    report = m.advantage(ftrain, feval, dtrain, deval, history, history[:360], updates=100)
    assert report['best']['rollout']['val'] > .95 and report['best']['critic_on_state']['val'] < .65
    assert report['present_only'] == dict(val=.5, transfer=.5) and report['rollout_over_critic']['transfer'] > .3


def tiny(**sensors):
    return LeWorldModel(LeWMConfig(image_hw=(28, 28), width=32, depth=1, heads=2, mlp=64, latent=16, projector_hidden=32,
                                   predictor_depth=1, predictor_heads=2, predictor_head_dim=8, predictor_mlp=32,
                                   command_dim=3, **sensors))


def test_input_shares_find_the_input_a_latent_is_made_of():
    torch.manual_seed(0)
    model = tiny(range_sectors=8, proprio_dim=o.PROPRIO_DIM)
    pixels, obs, g = torch.randn(64, 3, 28, 28), torch.rand(64, o.OBS_DIM), torch.Generator().manual_seed(0)
    assert set(input_shares(model, pixels, obs, g)) == {'image', 'ring', 'proprio'}
    with torch.no_grad():                         # the other sensors' tokens made constant: the latent is the image
        for layer in (model.observations.ring[-1], model.observations.body[-1]):
            layer.weight.zero_()
    shares = input_shares(model, pixels, obs, g)
    assert shares['ring'] == 0 and shares['proprio'] == 0 and abs(shares['image']-1) < .2
    model = tiny(range_sectors=8, proprio_dim=o.PROPRIO_DIM)
    with torch.no_grad():                         # the camera blinded: nothing of the latent is the image
        model.encoder.patch_embed.weight.zero_()
    shares = input_shares(model, pixels, obs, g)
    assert shares['image'] == 0 and shares['ring']+shares['proprio'] > .5
    assert input_shares(tiny(), pixels, None, g).keys() == {'image'}


def test_the_action_share_tells_a_camera_model_from_a_command_code():
    assert action_share(.040, .105, .93) < .05 < MAX_ACTION_SHARE          # the control's validation, one step
    assert action_share(.0076, .82, 1.) > MAX_ACTION_SHARE                 # go2_3level_c's
