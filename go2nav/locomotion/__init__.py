"""The Go2 locomotion policy: a PPO controller that tracks body-velocity commands.

The navigation stack issues one of nine velocity primitives (hold, forward slow/medium/fast, backward, yaw left/right,
arc left/right; configs/locomotion/contract_ppo.yaml) every 100 ms. This policy turns the current command into 12
joint-position targets at 50 Hz.

- `env`: the vectorised Genesis training environment (a port of the Genesis Go2 example, with the primitive bank).
- `train`: training from scratch with rsl-rl PPO; writes checkpoints, config.yaml and lineage.json.
- `policy`: the run-time policy used in missions (`LocomotionPolicy`); bit-identical to the original wrapper.
- `benchmark`: per-primitive tracking and switching benchmark, with side-by-side comparison of two policies.
"""
