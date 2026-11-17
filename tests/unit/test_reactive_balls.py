"""P27's world (docs/dynamic/PLAN.md P27): reactive balls, their kinds, contact, and exact branches."""
import numpy as np

from go2nav.sim import reactive_balls as world


def state(agent, obstacle, cruise, kind):
    s = world.reset(np.random.default_rng(0), 1)
    s.agent[:] = agent
    s.obstacle[0, 0], s.cruise[0, 0], s.velocity[0, 0], s.kind[0, 0] = obstacle, cruise, cruise, kind
    s.obstacle[0, 1:], s.cruise[0, 1:], s.kind[0, 1:] = (60., 60.), 0., world.INDIFFERENT    # the others out of the way
    return s


def test_an_indifferent_obstacle_cruises_past_the_agent_and_bounces_off_a_wall():
    s = world.step(state((30., 30.), (38., 30.), (1., 0.), world.INDIFFERENT), np.zeros((1, 2)))
    assert np.allclose(s.obstacle[0, 0], (39., 30.)) and s.near[0, 0] and not s.contact[0, 0]
    s = world.step(state((10., 10.), (60.5, 30.), (1.5, .5), world.INDIFFERENT), np.zeros((1, 2)))
    assert np.allclose(s.obstacle[0, 0], (60., 30.5)) and np.allclose(s.cruise[0, 0], (-1.5, .5)) and s.bounce[0, 0]


def test_a_yielding_obstacle_is_pushed_away_within_the_radius_and_cruises_beyond_it():
    s = world.step(state((30., 30.), (38., 30.), (0., 1.), world.YIELDING), np.zeros((1, 2)))
    assert np.allclose(s.obstacle[0, 0], (38.+world.FLEE, 31.))
    s = world.step(state((30., 30.), (30.+world.REACT+1, 30.), (0., 1.), world.YIELDING), np.zeros((1, 2)))
    assert np.allclose(s.obstacle[0, 0], (30.+world.REACT+1, 31.))


def test_a_pausing_obstacle_stops_near_the_agent_and_resumes_after_its_hidden_delay():
    s = state((30., 30.), (38., 30.), (0., 1.), world.PAUSING)
    s = world.step(s, np.zeros((1, 2)))
    delay = s.delays[0, 0, 0]
    assert np.allclose(s.obstacle[0, 0], (38., 30.)) and s.timer[0, 0] == delay and 5 <= delay <= 15
    s.agent[:] = (5., 5.)                                                 # the agent leaves
    for _ in range(delay):
        s = world.step(s, np.zeros((1, 2)))
        assert np.allclose(s.obstacle[0, 0], (38., 30.))
    s = world.step(s, np.zeros((1, 2)))
    assert np.allclose(s.obstacle[0, 0], (38., 31.))


def test_the_agent_pushes_an_obstacle_it_touches_along_the_normal():
    s = world.step(state((30., 30.), (37., 30.), (0., 0.), world.INDIFFERENT), np.array([[3., 0.]]))
    assert s.contact[0, 0] and np.allclose(s.agent[0], (33., 30.)) and np.allclose(s.obstacle[0, 0], (39., 30.))
    s = world.step(state((54., 30.), (60., 30.), (0., 0.), world.INDIFFERENT), np.array([[3., 0.]]))   # against a wall
    assert np.allclose(s.obstacle[0, 0], (61., 30.)) and np.allclose(s.agent[0], (55., 30.))


def test_branches_are_exact_copies_under_four_tapes_and_leave_the_state_alone():
    rng = np.random.default_rng(3)
    s = world.reset(rng, 40)
    for _ in range(10):
        s = world.step(s, rng.normal(0, 2, (40, 2)))
    before = s.take(slice(None))
    actions, target, out = world.branches(np.random.default_rng(1), s)
    assert all(np.array_equal(getattr(s, k), getattr(before, k)) for k in vars(s))
    assert actions.shape == (40, 4, 16, 2) and out['obstacle'].shape == (40, 4, 16, 3, 2)
    assert not actions[:, 0].any()                                        # hold
    replay = before.take(slice(None))
    for k in range(16):                                                   # replaying a tape reproduces it exactly
        replay = world.step(replay, actions[:, 1, k])
    assert np.array_equal(replay.obstacle, out['obstacle'][:, 1, -1]) and np.array_equal(replay.agent, out['agent'][:, 1, -1])
    d0 = np.linalg.norm(before.obstacle[np.arange(40), target]-before.agent, axis=-1)
    d = lambda tape: np.linalg.norm(out['obstacle'][np.arange(40), tape, 3, target]-out['agent'][:, tape, 3], axis=-1)
    assert (d(1) <= d0+1e-6).mean() > .8 and (d(2) >= d(1)).all()         # approach closes, retreat does not
    response = np.linalg.norm(out['obstacle'][:, 1, -1]-out['obstacle'][:, 0, -1], axis=-1)
    assert (response > 1.5).mean() > .1 and (response < 1e-9).mean() > .2  # some obstacles respond, some do not


def test_episodes_follow_their_actions_and_the_two_variants_differ_only_by_rings():
    ep, branch = world.episodes(np.random.default_rng(5), 6, 40, moments=2, branch_frames=True)
    assert ep['frames'].shape == (6, 40, 64, 64, 3) and ep['actions'].shape == (6, 39, 2)
    assert branch['t0'].shape == (6, 2) and branch['frames'].shape == (6, 2, 4, 16, 64, 64, 3)
    free = ~ep['contact'][:, 1:].any(-1)
    moved = np.clip(ep['agent'][:, :-1]+ep['actions'], 3., 61.)
    assert np.allclose(ep['agent'][:, 1:][free], moved[free], atol=1e-4)
    assert np.allclose(ep['velocity'][:, 1:], ep['obstacle'][:, 1:]-ep['obstacle'][:, :-1], atol=1e-4)
    ringed = (ep['kind'] != world.INDIFFERENT).any(-1)
    same = (ep['frames'] == ep['frames_t1']).all((1, 2, 3, 4))
    assert (same == ~ringed).all()
    gap = np.linalg.norm(ep['obstacle']-ep['agent'][:, :, None], axis=-1)[:, 1:]          # a reset may overlap them
    assert (gap < 2*world.RADIUS-1e-3).mean() < .005                      # contact keeps them apart (but for corners)
    xs = np.arange(64)+.5                                                  # each obstacle is drawn where it is
    frame, k = ep['frames'][0, 0].astype(float), 1
    w = frame[..., 2]*(frame[..., 2] > 200)
    assert abs((w.sum(0)*xs).sum()/w.sum()-ep['obstacle'][0, 0, k, 0]) < .6


def test_a_data_chunk_is_a_pure_function_of_its_seed():
    from go2nav.data import reactive_balls as data
    a, b = data.chunk(('interactive', 'val', 1, 2)), data.chunk(('interactive', 'val', 1, 2))
    assert np.array_equal(a[1][0]['frames'], b[1][0]['frames']) and np.array_equal(a[1][1]['obstacle'], b[1][1]['obstacle'])
    assert not np.array_equal(a[1][0]['agent'], data.chunk(('confounded', 'val', 1, 2))[1][0]['agent'])


def bench_of(episodes=30, frames=60, moments=3):
    from go2nav.eval import recipe_bench as rb
    ep, branch = world.episodes(np.random.default_rng(11), episodes, frames, moments=moments)
    bench = rb.Bench.__new__(rb.Bench)
    bench.labels, bench.branches, bench.device = ep, branch, None
    bench.positions = np.concatenate((ep['agent'][:, :, None], ep['obstacle']), 2)
    return rb, bench


def test_the_measures_score_the_truth_as_perfect_and_a_blind_forecaster_as_blind():
    rb, bench = bench_of()

    def truth(bench, e, t0, future):
        """The true futures, looked up: the episode's own where the tape is its own, else the matching branch tape's
        (a tape that matches neither, as the shuffled ones, is answered by holding)."""
        out = rb.hold(bench, e, t0, future)
        steps = t0[:, None]+np.arange(rb.STEPS)
        own = (np.abs(bench.labels['actions'][e[:, None], steps]-future) < 1e-6).all((1, 2))
        out[own] = bench.positions[e[:, None], steps+1][own]
        m = (bench.branches['t0'][e] == t0[:, None]).argmax(1)
        for tape in range(4):
            hit = ~own & (bench.branches['t0'][e, m] == t0) & (np.abs(bench.branches['actions'][e, m, tape]-future) < 1e-6).all((1, 2))
            out[hit] = np.concatenate((bench.branches['agent'][e, m, tape][:, :, None], bench.branches['obstacle'][e, m, tape]), 2)[hit]
        return out

    row = rb.measure(truth, bench, draws=200)
    assert row['E'][8]['error'] < 1e-5 and row['E'][8]['ratio'] > 100
    assert row['X'][8]['bounce']['error'] < 1e-5 and row['X'][8]['bounce']['cv'] > 1.
    assert row['X'][8]['straight']['n'] > row['X'][8]['bounce']['n'] > 0
    A = row['A'][8]['all']
    assert A['ratio'] < 1e-5 and A['false_rate'] == 0. and abs(A['slope']-1.) < 1e-6 and A['ci'][0] > .99
    assert row['A'][8]['contact']['responsive'] > 0 and row['A'][8]['distance']['responsive'] > 0
    assert rb.bars(row | dict(R=dict(agent=.2)))['all']
    blind = rb.measure(rb.constant_velocity, bench, draws=200)
    A = blind['A'][8]['all']
    assert abs(A['ratio']-1.) < 1e-9 and A['false_rate'] == 0. and abs(A['slope']) < 1e-9     # zero-response
    assert abs(blind['E'][8]['ratio']-1.) < 1e-9 and not rb.bars(blind)['A'] and not rb.bars(blind)['X']
    assert '| truth |' in rb.table(dict(truth=row), row)


def test_the_heads_read_a_state_into_four_positions_and_the_oracle_rolls_out():
    import pytest
    torch = pytest.importorskip('torch')
    from go2nav.eval import recipe_bench as rb
    from go2nav.training import recipe_bench as training
    assert rb.build_head(training.model_config())(torch.zeros(5, 16, 128)).shape == (5, 16, 4, 2)
    grid = training.model_config(grid=(8, 8))
    assert grid.tokens == 65 and rb.build_head(grid)(torch.zeros(5, 16, 65, 128)).shape == (5, 16, 4, 2)
    out = rb.OracleS(kinds=True).rollout(torch.rand(6, 3, 4, 2)*64, torch.zeros(6, 3), torch.zeros(6, 16, 2))
    assert out.shape == (6, 16, 4, 2)
    model = training.LeWorldModel(grid)
    z = model.encode(torch.zeros(2, 3, 3, 64, 64))
    assert model.rollout(z, torch.zeros(2, 18, 2), 16).shape == (2, 16, 65, 128)


def test_the_motion_weights_follow_the_discs_and_leave_the_unweighted_loss_unchanged():
    import pytest
    torch = pytest.importorskip('torch')
    from go2nav.models.lewm import SIGReg, losses
    from go2nav.training import recipe_bench as training
    from go2nav.training.balls import to_pixels
    ep, _ = world.episodes(np.random.default_rng(2), 4, 11)
    pixels, actions = to_pixels(torch.from_numpy(ep['frames'])), torch.from_numpy(ep['actions'])/3
    change = training.patch_motion(pixels, 8)
    assert change.shape == (4, 10, 64)
    row, col = (ep['agent'][0, 1, 1]//8).astype(int), (ep['agent'][0, 1, 0]//8).astype(int)
    moved = np.linalg.norm(ep['agent'][0, 1]-ep['agent'][0, 0]) > .5
    assert not moved or float(change[0, 0, row*8+col]) > float(change[0, 0].median())
    assert float((change == 0).float().mean()) > .5                      # most patches are still
    model = training.LeWorldModel(training.model_config(**training.RECIPES['r2']['model']))
    sigreg = SIGReg(num_proj=32)
    g = lambda: torch.Generator().manual_seed(0)
    flat = training.motion_losses(model, pixels, actions, rollout_steps=8, sigreg=sigreg, lam=.09, floor=1., generator=g())
    plain = losses(model, pixels, actions, rollout_steps=8, sigreg=sigreg, lam=.09, generator=g())
    assert torch.allclose(flat['loss'], plain['loss'], atol=1e-5)        # floor 1: every weight is 1
    weighted = training.motion_losses(model, pixels, actions, rollout_steps=8, sigreg=sigreg, lam=.09, floor=.1, generator=g())
    assert torch.isfinite(weighted['loss']) and not torch.allclose(weighted['grid_pred'], plain['grid_pred'])


def test_the_detector_places_every_object_and_the_object_tokens_train():
    import pytest
    torch = pytest.importorskip('torch')
    from go2nav.eval import recipe_bench as rb
    from go2nav.models import object_tokens as ot
    from go2nav.models.lewm import SIGReg
    from go2nav.training import recipe_bench as training
    from go2nav.training.balls import to_pixels
    ep, _ = world.episodes(np.random.default_rng(4), 8, 30)
    truth = np.concatenate((ep['agent'][:, :, None], ep['obstacle']), 2).reshape(-1, 4, 2)
    for key in ('frames', 'frames_t1'):                                   # with and without the kinds' rings
        xy, seen = ot.detect(to_pixels(torch.from_numpy(ep[key])).flatten(0, 1))
        error = np.linalg.norm(xy.numpy()-truth, axis=-1)
        assert np.median(error) < .1 and (error < 1.).mean() > .93 and seen.float().mean() > .98
    pixels, actions = to_pixels(torch.from_numpy(ep['frames'][:, :11])), torch.from_numpy(ep['actions'][:, :10])/3
    window = ot.crops(pixels[:, 0], torch.from_numpy(ep['agent'][:, 0])[:, None], 16)
    assert window.shape == (8, 1, 3, 16, 16) and float(window[:, 0, 0, 7:9, 7:9].min()) > .99    # centred on the agent
    model = training.build('r4')
    z = model.encode(pixels)
    assert z.shape == (8, 11, 4, 128) and model.rollout(z[:, :3], actions, 8).shape == (8, 8, 4, 128)
    out = ot.token_losses(model, pixels, actions, rollout_steps=8, sigreg=SIGReg(num_proj=32), lam=.09)
    out['loss'].backward()
    assert torch.isfinite(out['loss']) and rb.build_head(model.config)(z).shape == (8, 11, 4, 2)


def test_the_branch_terms_vanish_for_a_model_that_predicts_both_tapes():
    import pytest
    torch = pytest.importorskip('torch')
    from go2nav.training import recipe_bench as training

    class Exact:
        def encode(self, pixels):
            return pixels.flatten(2)[..., :5]

        def rollout(self, zh, tape, steps):
            return tape[:, 2:, :1]+torch.zeros(1, steps, 5)

    history = torch.zeros(3, 3, 3, 8, 8)
    tapes = [torch.randn(3, 10, 2) for _ in range(2)]
    futures = [t[:, 2:, :1, None, None].expand(-1, -1, 3, 8, 8) for t in tapes]
    out = training.branch_losses(Exact(), history, futures, tapes)
    assert float(out['branch']) < 1e-12 and float(out['paired']) < 1e-12
    out = training.branch_losses(Exact(), history, futures, tapes[::-1])       # the tapes swapped: both terms see it
    assert float(out['branch']) > .1 and float(out['paired']) > .1


def test_the_weighted_branch_terms_run_on_a_patch_grid_model():
    import pytest
    torch = pytest.importorskip('torch')
    from go2nav.training import recipe_bench as training
    from go2nav.training.balls import to_pixels
    ep, branch = world.episodes(np.random.default_rng(6), 2, 30, moments=1, branch_frames=True)
    t0 = branch['t0'][:, 0].astype(int)
    history = to_pixels(torch.from_numpy(np.stack([ep['frames'][i, t-2:t+1] for i, t in enumerate(t0)])))
    futures = [to_pixels(torch.from_numpy(branch['frames'][:, 0, tape, :8])) for tape in (0, 1)]
    tapes = [torch.cat((torch.zeros(2, 2, 2), torch.from_numpy(branch['actions'][:, 0, tape, :8])/3), 1) for tape in (0, 1)]
    model = training.build('r3b')
    out = training.branch_losses(model, history, futures, tapes, floor=.1)
    plain = training.branch_losses(model, history, futures, tapes)
    assert torch.isfinite(out['branch']) and torch.isfinite(out['paired']) and not torch.allclose(out['branch'], plain['branch'])
    flat = training.branch_losses(model, history, futures, tapes, floor=1.)   # floor 1: every weight is 1
    cls_and_grid = lambda d: d[:, :, 0].pow(2).mean()+d[:, :, 1:].pow(2).mean()
    zh, za = model.encode(history), model.encode(futures[0])
    assert torch.isfinite(flat['branch']) and float(cls_and_grid(model.rollout(zh, tapes[0], 8)-za)) > 0


def test_the_memory_reads_only_the_past_and_the_bench_splits_by_what_was_shown():
    import pytest
    torch = pytest.importorskip('torch')
    from go2nav.training import recipe_bench as training
    model = training.build('q3m').eval()
    z, actions = torch.randn(2, 20, 4, 128), torch.randn(2, 19, 2)
    m = model.memory_states(z, actions)
    z2 = z.clone(); z2[:, 12:] += 5.                                        # change the future of frame 11
    m2 = model.memory_states(z2, actions)
    assert m.shape == (2, 20, 4, 128) and torch.allclose(m[:, :12], m2[:, :12], atol=1e-6) and not torch.allclose(m[:, 12], m2[:, 12])
    out = model.rollout(z[:, :3], torch.zeros(2, 10, 2), 8, m[:, :3])
    assert out.shape == (2, 8, 4, 128)
    rb, bench = bench_of(episodes=12, frames=60, moments=2)
    row = rb.measure(rb.constant_velocity, bench, draws=50)
    parts = row['A'][8]
    assert parts['shown']['pairs']+parts['hidden']['pairs'] == parts['all']['pairs']
    assert parts['shown_near']['pairs'] <= parts['shown']['pairs']
