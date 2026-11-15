import numpy as np
import torch

from go2nav.data import balls as data
from go2nav.eval import balls_planning as planning
from go2nav.sim import balls


def test_test_stream_is_disjoint_and_repeatable():
    assert len(set(data.SPLIT_INDEX.values())) == 3
    parts = [data.chunk((1, split, 0, 1))[1] for split in ('train', 'val', 'test')]
    assert not np.array_equal(parts[0]['frames'], parts[2]['frames'])
    assert not np.array_equal(parts[1]['frames'], parts[2]['frames'])
    assert np.array_equal(parts[2]['frames'], data.chunk((1, 'test', 0, 1))[1]['frames'])


def test_task_inputs_do_not_include_intervening_frames_or_future_actions():
    frames = np.arange(100)[None, :, None, None, None]*np.ones((1, 1, 2, 2, 3), np.uint8)
    actions = np.arange(99)[None, :, None]*np.ones((1, 1, 2))
    task = planning.task_inputs(frames, actions, 0, 20, 8)
    assert task.history[:, 0, 0, 0].tolist() == [18, 19, 20]
    assert task.history_actions[:, 0].tolist() == [18, 19]
    assert task.goal[0, 0, 0] == 28
    assert set(task.__dataclass_fields__) == {'history', 'history_actions', 'goal'}


def test_recorded_and_direct_commands_reach_goals_with_correct_time_index():
    ep = balls.episodes(np.random.default_rng(3), 20, 3, 100)
    for episode in range(20):
        for h in (4, 6, 8):
            t = planning.task_start(3, episode, h)
            start, goal = ep['red'][episode, t], ep['red'][episode, t+h]
            for actions in (ep['actions'][episode, t:t+h], np.broadcast_to((goal-start)/h, (h, 2))):
                assert np.linalg.norm(actions, axis=-1).max() <= 3+1e-5
                assert np.linalg.norm(planning.red_path(start, actions)[-1]-goal) < 1e-4


def test_cem_is_paired_and_satisfies_action_constraint():
    start, goal = np.array([30, 30]), np.array([35, 36])
    cost = lambda a: ((planning.red_path(start, a)[:, -1]-goal)**2).sum(-1)
    first = planning.cem(cost, 4, [1, 2, 3])
    assert np.array_equal(first, planning.cem(cost, 4, [1, 2, 3]))
    assert np.linalg.norm(first, axis=-1).max() <= 3+1e-6
    assert np.linalg.norm(planning.red_path(start, first)[-1]-goal) < 1


def test_slot_goal_cost_is_permutation_invariant():
    x = torch.randn(3, 5, 8)
    goal = x[:, [3, 1, 0, 4, 2]].clone()
    assert torch.equal(planning.slot_distance(x, goal), torch.zeros(3))
    assert torch.allclose(planning.slot_distance(x+1, goal), planning.slot_distance(x[:, [4, 3, 2, 1, 0]]+1, goal))
