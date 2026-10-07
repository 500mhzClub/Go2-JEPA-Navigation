# Scene textures (CC0)

Diffuse (albedo) color maps applied to floor / wall / obstacle surfaces at
render time (`go2nav/corpus/textures.py` and `go2nav/corpus/render.py`). Bricks097 and
WoodFloor043 also colour the demonstration surfaces (`go2nav/sim/demo_appearance.py`), and
WoodFloor043 and PaintedPlaster017 the hero video's presentation scene (`go2nav/eval/hero_video.py`). Landmarks and
visual-stress distractors are intentionally **not** textured — their solid
colors are task identity / decoys.

## License

All textures here are from **ambientCG** (https://ambientcg.com) and are
released under the **Creative Commons CC0 1.0 Universal** license (public
domain): *"You can copy, modify, distribute and perform the assets, even for
commercial purposes, all without asking permission."* No attribution is
required; redistribution and bundling into this repository is permitted.

Only the `*_Color.jpg` (1K) map of each material is vendored; other PBR maps
(normal/roughness/AO/displacement) were dropped — the pipeline is diffuse-only.

## Provenance

| category | asset | source |
| --- | --- | --- |
| floor | Concrete034 | https://ambientcg.com/a/Concrete034 |
| floor | Tiles093 | https://ambientcg.com/a/Tiles093 |
| floor | WoodFloor043 | https://ambientcg.com/a/WoodFloor043 |
| floor | PavingStones131 | https://ambientcg.com/a/PavingStones131 |
| wall | Bricks097 | https://ambientcg.com/a/Bricks097 |
| wall | Plaster001 | https://ambientcg.com/a/Plaster001 |
| wall | PaintedPlaster017 | https://ambientcg.com/a/PaintedPlaster017 |
| wall | Concrete045 | https://ambientcg.com/a/Concrete045 |
| obstacle | Metal055A | https://ambientcg.com/a/Metal055A |
| obstacle | Cardboard004 | https://ambientcg.com/a/Cardboard004 |
| obstacle | Wood067 | https://ambientcg.com/a/Wood067 |
| obstacle | Concrete036 | https://ambientcg.com/a/Concrete036 |

Re-fetch: `https://ambientcg.com/get?file=<AssetId>_1K-JPG.zip`, keep `*_Color.jpg`.
