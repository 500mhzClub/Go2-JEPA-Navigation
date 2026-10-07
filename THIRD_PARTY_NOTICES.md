# Third-party components

Draft for the pre-publication review. The repository's own licence is still to be chosen.

| Component | Use here | Licence | Distribution |
|---|---|---|---|
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 0.4.6 (simulation), 0.3.14 (corpus render) | simulator; `go2nav/locomotion/env.py` is a port of its Go2 locomotion example (`examples/locomotion/go2_env.py`), with the origin and the one change stated in the module docstring | Apache-2.0 | dependency, not vendored; the Go2 URDF and meshes are loaded from the installed Genesis package and not redistributed |
| [rsl-rl](https://github.com/leggedrobotics/rsl_rl) 5.4.1 | PPO for the locomotion policy | BSD-3-Clause | dependency, not vendored |
| [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) ViT-L (distilled from ViT-G, 384 px) | the frozen image encoder of C3 and C4 | source: MIT with Apache-2.0 portions (as recorded in `artifacts/manifest.yaml`); weights: terms to be confirmed in review | not re-hosted: `python artifacts/fetch.py vjepa` downloads the weights from Meta's server and the source at a pinned commit |
| [ambientCG](https://ambientcg.com) textures | `assets/textures` (corpus render, demonstration surfaces and videos) | CC0 1.0 | vendored (1K colour maps only); see `assets/textures/README.md` |

The trained checkpoints listed in `artifacts/manifest.yaml` (locomotion policy, T1-T8, C1) are this programme's own; the
encoder weights are loaded separately and are not part of them.
