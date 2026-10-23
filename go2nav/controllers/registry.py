"""The arms (controllers) by name, with what the cohort runner needs to schedule their missions. The one list that the
mission runner, the cohort runner and `go2nav.controllers.arms` share; it imports nothing heavy.

- `MEMORY_GIB`: host memory a mission of the arm needs (the cohort runner starts one only when it is available).
- `GPU_LANES`: arms whose concurrent missions are capped, by lane: `c3` (the V-JEPA ViT-L encoder and the dense
  predictor; `--c3-lanes`) and `lewm` (the small LeWorldModel, C5-C8; `--lewm-lanes`). Other arms are not capped.
"""
ARMS = ('C0', 'C1', 'C1A', 'C1R', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8')
MEMORY_GIB = dict(C0=5, C1=4, C1A=4, C1R=4, C2=4, C3=7, C4=6, C5=5, C6=5, C7=5, C8=5)
GPU_LANES = dict(C3='c3', C5='lewm', C6='lewm', C7='lewm', C8='lewm')
