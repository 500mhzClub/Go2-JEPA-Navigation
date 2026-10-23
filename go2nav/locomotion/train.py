"""Train the Go2 locomotion policy from scratch with PPO (rsl-rl).

    python -m go2nav.locomotion.train --out outputs/locomotion/seed11 [--config configs/locomotion/contract_ppo.yaml]
        [--seed 11] [--num-envs 4096] [--max-iterations 501]

Writes model_<iteration>.pt every `save_interval` iterations (and the final one) to --out, with:
- config.yaml: the exact training config (what `policy.LocomotionPolicy` reads);
- lineage.json: seed, config SHA-256, code commit, package versions and checkpoint SHA-256s;
- tensorboard logs.
The defaults reproduce the shipped policy's training run (model_500.pt; seed 11, 4096 environments, 501 iterations).
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

import yaml

from go2nav import rocm

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO/'configs/locomotion/contract_ppo.yaml'


def code_commit():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain', 'go2nav', 'configs'], cwd=REPO, capture_output=True, text=True)
    return dict(commit=result.stdout.strip() or None, uncommitted_changes=bool(dirty.stdout.strip()))


def train(config_path, out, *, seed, num_envs, max_iterations):
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    from go2nav.locomotion.env import Go2LocomotionEnv

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config_text = Path(config_path).read_text()
    cfg = yaml.safe_load(config_text)
    cfg['run'] = dict(num_envs=num_envs, max_iterations=max_iterations, seed=seed)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    rocm.configure()
    gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=seed, performance_mode=True)
    env = Go2LocomotionEnv(num_envs, cfg)
    runner = OnPolicyRunner(env, dict(cfg['train'], run_name=out.name), str(out), device=gs.device)
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    checkpoints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('model_*.pt'))}
    lineage = dict(component='go2_locomotion_policy', seed=seed, num_envs=num_envs, max_iterations=max_iterations,
                   config=str(Path(config_path).resolve().relative_to(REPO)) if Path(config_path).resolve().is_relative_to(REPO) else Path(config_path).name, config_sha256=hashlib.sha256(config_text.encode()).hexdigest(),
                   code=code_commit(), packages={p: metadata.version(p) for p in ('genesis-world', 'torch', 'rsl-rl-lib')},
                   parents=[], checkpoints=checkpoints)
    (out/'lineage.json').write_text(json.dumps(lineage, indent=1)+'\n')
    return lineage


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--seed', type=int, help="default: the config's run.seed")
    p.add_argument('--num-envs', type=int, help="default: the config's run.num_envs")
    p.add_argument('--max-iterations', type=int, help="default: the config's run.max_iterations")
    a = p.parse_args()
    defaults = yaml.safe_load(Path(a.config).read_text())['run']
    a.seed, a.num_envs, a.max_iterations = (defaults[k] if v is None else v for k, v in
                                            (('seed', a.seed), ('num_envs', a.num_envs), ('max_iterations', a.max_iterations)))
    lineage = train(a.config, a.out, seed=a.seed, num_envs=a.num_envs, max_iterations=a.max_iterations)
    print(json.dumps(dict(out=a.out, checkpoints=list(lineage['checkpoints'])), indent=1))


if __name__ == '__main__':
    main()
