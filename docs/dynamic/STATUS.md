# Dynamic environments and continuous commands: status

The place to resume from. Plan: [PLAN.md](PLAN.md). Branch `feature/dynamic-continuous`.

## P1: locomotion policy on command ranges (23 October)

`outputs/locomotion/continuous_seed11`:
- `go2nav.locomotion.train --config configs/locomotion/continuous_ppo.yaml`, seed 11, 4,096 robots, 1,500 iterations,
  about 22 min on the R9700.
- Commands are drawn from a mixture: 50% uniform over forward -0.25 to 0.45 m/s, lateral ±0.20 m/s and yaw ±0.8
  rad/s; 30% the nine primitives; 20% hold.

Benchmark (`go2nav.locomotion.benchmark`; `outputs/locomotion/benchmark_{original_lattice,continuous_seed11}{,_mu0.3}.json`).
The lattice is 90 commands (6 forward x 3 lateral x 5 yaw), 8 robots each. Errors are RMS in (forward m/s, lateral
m/s, yaw rad/s). The shipped policy is measured on the same grid.

| | Shipped policy | Continuous policy |
|---|---|---|
| Lattice within M1 (0.05 m/s, 0.1 rad/s, no resets), nominal floor | 17/90 | **87/90** |
| Lattice within M1, friction 0.3 | 13/90 | 80/90 |
| Lattice median RMS, nominal | 0.034, 0.188, 0.066 | **0.019, 0.022, 0.027** |
| Lattice worst RMS, nominal | 0.144, 0.288, 0.173 | 0.043, 0.062, 0.074 |
| Continuous switching, settled error, nominal | 0.032, 0.105, 0.045 | 0.016, 0.016, 0.019 |
| Continuous switching resets, friction 0.3 | 15 | 5 |

- **M1 (the range): 87/90.**
  - The three misses are the extreme corners: full reverse, full lateral and yaw 0.4-0.8 rad/s.
  - The worst lateral RMS there is 0.062 against the 0.05 threshold. Noted, not retrained.
  - The shipped policy does not track lateral motion at all.
- **M2 (the nine primitives): pass.** They track as well or better on nominal floor, with two small regressions:
  - hold drifts in yaw (mean -0.008 rad/s against -0.002; RMS 0.017 against 0.003);
  - backward undershoots (-0.175 m/s against -0.189).
- **Friction 0.3: far better.** The shipped policy reaches only 0.166 m/s on the 0.30 m/s command and drifts sideways
  at 0.095 m/s; the new policy reaches 0.322 with no drift. R5 at friction 0.30 may improve for every arm, so the
  static baselines may not be "level" on the new policy. M3 measures this.
- **M3:** C1 on the continuous policy, on the 100 benchmark episodes (`loco_continuous_c1.yaml`, cohorts
  `loco_cont_*_c1`). Paired with C1 on the shipped policy, 72/100. Running.

## P2 spike: kinematic obstacles in Genesis (23 October)

A cylinder (0.35 m across, 0.9 m tall) is added as a fixed entity and moved with `set_pos(..., zero_velocity=True)`
every physics step, on the CPU backend. It walks at 0.3 m/s into a Go2 holding position under the policy.

- **Deterministic:** two runs give the same robot and obstacle trace hash and pixel-identical frames.
- **Contacts:** reported in `robot.get_contacts` as robot link against obstacle link: first contact at 3.29 s, 6.8 N.
  The kinematic body then shoves the robot aside, so the mission's contact stop must fire first.
- **Rendering and depth follow the moved pose:** depth along the centre row tracks the obstacle from 1.0 m down to
  0.24 m.
- A free (non-fixed) body moved the same way also works, but with larger contact forces (47 N). Fixed is the choice.
- **Needed in `sim/scene.py`:** `fix_raster_order` checks that the visible population is exactly floor, walls and
  robot, so it must admit the obstacles. The contact topology must name obstacle links for attribution.
