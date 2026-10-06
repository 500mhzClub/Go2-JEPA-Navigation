"""Locomotion: the training config matches the shipped policy's, and the ported runtime policy acts identically."""
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO/'configs/locomotion/contract_ppo.yaml').read_text())
SHIPPED = Path(os.environ.get('GO2NAV_LOCOMOTION_CHECKPOINT',
                              '/home/andrewknowles/Workspace/LeWMQuad-v3/models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt'))


@pytest.mark.skipif(not SHIPPED.exists(), reason='shipped locomotion checkpoint not available')
def test_config_matches_shipped_training_config():
    env, obs, reward, command, train = pickle.loads((SHIPPED.parent/'cfgs.pkl').read_bytes())
    assert CONFIG['env'] == env
    assert CONFIG['obs'] == obs
    assert CONFIG['reward'] == reward
    assert {k: v for k, v in train.items() if k not in ('run_name', 'save_interval')} == \
        {k: v for k, v in CONFIG['train'].items() if k != 'save_interval'}
    assert list(CONFIG['primitives']) == command['lewm_command_names']
    assert [list(map(float, v)) for v in CONFIG['primitives'].values()] == command['lewm_command_bank']


@pytest.mark.equivalence
@pytest.mark.skipif('GO2NAV_SNAPSHOT' not in os.environ or not SHIPPED.exists(), reason='needs the frozen snapshot')
def test_runtime_policy_matches_original_wrapper():
    snapshot = Path(os.environ['GO2NAV_SNAPSHOT'])
    sys.path[:0] = [str(snapshot/'lewm_genesis'), str(snapshot/'lewm_worlds'), str(snapshot)]
    from lewm_genesis.rollout import GenesisGo2PPOPolicy
    from go2nav.locomotion.policy import LocomotionPolicy
    original = GenesisGo2PPOPolicy(checkpoint_path=SHIPPED, device='cpu')
    ported = LocomotionPolicy(SHIPPED, device='cpu')
    rng = np.random.default_rng(0)
    for _ in range(20):  # a short sequence, so the action latency state is exercised
        quat = rng.normal(size=(3, 4)).astype(np.float32)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        inputs = dict(base_quat_xyzw=quat, base_ang_vel_world=rng.normal(size=(3, 3)).astype(np.float32),
                      joint_pos=rng.normal(size=(3, 12)).astype(np.float32), joint_vel=rng.normal(size=(3, 12)).astype(np.float32),
                      command=rng.choice(list(CONFIG['primitives'].values()), size=3).astype(np.float32))
        np.testing.assert_array_equal(ported.act(**inputs), original.act(inputs))
