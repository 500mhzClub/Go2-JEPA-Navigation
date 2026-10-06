"""Benchmark a locomotion policy: how well does it track each velocity primitive?

    python -m go2nav.locomotion.benchmark CHECKPOINT --out result.json [--friction 0.3] [--robots 32]
    python -m go2nav.locomotion.benchmark --compare original.json retrained.json

Two runs in the training environment (flat floor; optional uniform friction), with the policy acting deterministically:

**Steady tracking.** Each of the nine primitives is commanded to `robots` robots for `horizon_s` seconds. After a
`warmup_s` settling period, per primitive:
- the achieved body velocity (vx, vy, yaw rate), averaged over time per robot, then mean and spread over robots;
- the RMS tracking error against the command;
- resets (falls: roll or pitch above 10 degrees).

**Switching.** Every robot cycles through all primitives in a shuffled order, switching every `switch_s` seconds. Per
switch: the mean absolute tracking error over the first second after the switch, and over the rest of the interval.

The command enters only through the policy's observation, exactly as in missions. `--compare` prints two results side by
side with differences.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from go2nav import rocm
from go2nav.locomotion.policy import build_actor, load_config

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO/'configs/locomotion/contract_ppo.yaml'


def _setup(checkpoint, n_envs, friction, seed):
    import genesis as gs
    from go2nav.locomotion.env import Go2LocomotionEnv
    cfg = load_config(checkpoint)
    reference = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg = dict(reference, **cfg)  # an original checkpoint's cfgs.pkl has no primitive bank: use the reference bank
    cfg['env'] = dict(cfg['env'], resampling_time_s=1e4)  # commands are set here, never resampled
    rocm.configure()
    gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=seed, performance_mode=True)
    env = Go2LocomotionEnv(n_envs, cfg, friction=friction)
    return env, build_actor(checkpoint, cfg, gs.device), cfg['primitives']


def _rollout(env, actor, commands_per_step):
    """Run with the given command schedule (steps x envs x 3); return achieved (steps x envs x 3) and reset flags."""
    achieved, resets = [], []
    env.reset()
    with torch.no_grad():
        for commands in commands_per_step:
            env.commands.copy_(commands)
            env._update_observation()
            _, _, done, _ = env.step(actor(env.get_observations(), stochastic_output=False))
            achieved.append(torch.cat([env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]], dim=1).cpu().numpy())
            resets.append(done.cpu().numpy())
    return np.stack(achieved), np.stack(resets)


def steady(env, actor, primitives, robots, horizon_s, warmup_s):
    names = list(primitives)
    bank = torch.tensor([primitives[n] for n in names], dtype=torch.float32, device=env.device)
    commands = bank.repeat_interleave(robots, dim=0)
    steps = int(round(horizon_s/env.dt))
    achieved, resets = _rollout(env, actor, [commands]*steps)
    window = achieved[int(round(warmup_s/env.dt)):]
    result = {}
    for i, name in enumerate(names):
        rows = slice(i*robots, (i+1)*robots)
        per_robot = window[:, rows].mean(axis=0)
        error = window[:, rows]-np.asarray(primitives[name])
        result[name] = dict(command=primitives[name], achieved_mean=per_robot.mean(axis=0).tolist(),
                            achieved_std=per_robot.std(axis=0).tolist(),
                            rms_error=np.sqrt((error**2).mean(axis=(0, 1))).tolist(),
                            resets=int(resets[:, rows].sum()))
    return result


def switching(env, actor, primitives, switch_s, seed):
    names = list(primitives)
    rng = np.random.default_rng(seed)
    orders = np.stack([rng.permutation(len(names)) for _ in range(env.num_envs)], axis=1)  # switches x envs
    bank = np.asarray([primitives[n] for n in names], dtype=np.float32)
    per_switch = int(round(switch_s/env.dt))
    schedule = np.repeat(bank[orders], per_switch, axis=0)  # steps x envs x 3
    achieved, resets = _rollout(env, actor, [torch.as_tensor(c, device=env.device) for c in schedule])
    error = np.abs(achieved-schedule).reshape(len(names), per_switch, env.num_envs, 3)[1:]  # skip the start
    first = int(round(1.0/env.dt))
    return dict(switch_s=switch_s, switches=int(error.shape[0]*env.num_envs),
                first_second_mean_abs_error=error[:, :first].mean(axis=(0, 1, 2)).tolist(),
                settled_mean_abs_error=error[:, first:].mean(axis=(0, 1, 2)).tolist(),
                resets=int(resets.sum()))


def run(checkpoint, *, robots=32, horizon_s=8.0, warmup_s=2.0, switch_s=2.0, friction=None, seed=20260516):
    env, actor, primitives = _setup(checkpoint, robots*len(load_primitives()), friction, seed)
    return dict(checkpoint=str(checkpoint), friction=friction, robots_per_primitive=robots, horizon_s=horizon_s,
                warmup_s=warmup_s, seed=seed, axes=['vx_mps', 'vy_mps', 'yaw_rate_radps'],
                steady=steady(env, actor, primitives, robots, horizon_s, warmup_s),
                switching=switching(env, actor, primitives, switch_s, seed))


def load_primitives():
    return yaml.safe_load(DEFAULT_CONFIG.read_text())['primitives']


def compare(a, b):
    """Print two benchmark results side by side."""
    lines = [f"{'primitive':16s} {'axis':6s} {'command':>8s} {'A achieved':>11s} {'B achieved':>11s} {'A rms':>7s} {'B rms':>7s} {'resets A/B':>10s}"]
    for name, ra in a['steady'].items():
        rb = b['steady'][name]
        for k, axis in enumerate(('vx', 'vy', 'yaw')):
            lines.append(f"{name:16s} {axis:6s} {ra['command'][k]:8.3f} {ra['achieved_mean'][k]:11.3f} {rb['achieved_mean'][k]:11.3f} "
                         f"{ra['rms_error'][k]:7.3f} {rb['rms_error'][k]:7.3f} {ra['resets']:>5d}/{rb['resets']:<4d}")
    sa, sb = a['switching'], b['switching']
    lines.append(f"switching: first-second error A {np.round(sa['first_second_mean_abs_error'], 3).tolist()} "
                 f"B {np.round(sb['first_second_mean_abs_error'], 3).tolist()}; settled A {np.round(sa['settled_mean_abs_error'], 3).tolist()} "
                 f"B {np.round(sb['settled_mean_abs_error'], 3).tolist()}; resets {sa['resets']}/{sb['resets']}")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', nargs='?')
    p.add_argument('--out')
    p.add_argument('--friction', type=float)
    p.add_argument('--robots', type=int, default=32)
    p.add_argument('--compare', nargs=2, metavar=('A_JSON', 'B_JSON'))
    a = p.parse_args()
    if a.compare:
        print(compare(*(json.loads(Path(x).read_text()) for x in a.compare)))
        return
    result = run(a.checkpoint, robots=a.robots, friction=a.friction)
    Path(a.out).write_text(json.dumps(result, indent=1)+'\n')
    print(compare(result, result).split('\n')[0])
    for name, r in result['steady'].items():
        print(f"{name:16s} command {r['command']} achieved {np.round(r['achieved_mean'], 3).tolist()} "
              f"rms {np.round(r['rms_error'], 3).tolist()} resets {r['resets']}")


if __name__ == '__main__':
    main()
