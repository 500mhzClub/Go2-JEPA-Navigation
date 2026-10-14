# Third-party components

This repository is licensed under the Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). It vendors no
third-party code; the components below are dependencies, downloads or data it uses.

| Component | Use here | Licence | Distribution |
|---|---|---|---|
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 0.4.6 (simulation), 0.3.14 (corpus render) | simulator; `go2nav/locomotion/env.py` is a modified port of its Go2 locomotion example (`examples/locomotion/go2_env.py`), with the origin and the one change stated in the module docstring | Apache-2.0 | dependency, not vendored |
| Unitree Go2 robot model (URDF and meshes) | the simulated robot | shipped inside the Genesis package (`genesis/assets/urdf/go2`), which carries no separate licence file for it | loaded from the installed Genesis package; not redistributed here |
| [rsl-rl](https://github.com/leggedrobotics/rsl_rl) 5.4.1 | PPO for the locomotion policy | BSD-3-Clause | dependency, not vendored |
| [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) ViT-L/16 (distilled from ViT-G, 384 px) | the frozen image encoder of C3 and C4 | the vjepa2 repository's licence: MIT, except three data-augmentation utilities under Apache-2.0 (none used here). The weights are the official checkpoint linked from that repository's README (`vjepa2_1_vitl_dist_vitG_384.pt`); the README states no separate terms for them (checked at the pinned commit `204698b`) | not re-hosted: `python artifacts/fetch.py vjepa` downloads the weights from Meta's server (SHA-256 checked) and the source at the pinned commit |
| [ambientCG](https://ambientcg.com) textures | `assets/textures` (corpus render, demonstration surfaces and videos) | CC0 1.0 | vendored (1K colour maps only); see `assets/textures/README.md` |

The trained checkpoints listed in `artifacts/manifest.yaml` (locomotion policy, T1-T8, C1) are this programme's own
and are covered by this repository's licence; the encoder weights are loaded separately and are not part of them (the
largest checkpoint has 17.2 M parameters).

`configs/corpus/go2_platform_manifest.yaml` is the May 2026 platform plan. It names a ROS 2 Go2 package
(`khaledgabr77/unitree_go2_ros2`) with its licence marked pending; that package was never vendored or used. The robot
is Genesis's bundled Go2 (`genesis_urdf: genesis_builtin_go2` in the same file). The file is kept unchanged because the
corpus scene packs are built from it.
