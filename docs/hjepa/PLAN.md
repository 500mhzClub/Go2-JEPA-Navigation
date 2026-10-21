# H-JEPA on the Go2 (draft for review, 21 October 2026)

H-JEPA (Zhang, Terver, Rabbat, LeCun and Balestriero, arXiv:2610.06805) stacks action-conditioned JEPAs. Each level
predicts farther ahead in its own latent space, and the hierarchy trains **jointly, end to end**:
- one objective, the sum of every level's loss;
- each upper level's prediction gradients flow through the encoders below it, which is what makes the levels
  abstract (paper §3);
- no stop-gradient, no EMA targets, no pretrained weights.

The paper's level 1 is LeWM. In its headline runs every level trains from scratch in one run. Stagewise training
(freeze level 1, train level 2 on top) appears only in its ablations, and loses on matched data (AntMaze 54% vs 40%).

> Note: `~/Downloads/h-jepa.pdf` is a different document (an internal "H-JEPA Quadruped Navigator" design, v4.0). It
> puts a stop-gradient between the high-level loss and the shared trunk. This plan follows the arXiv paper.

Goal of this branch: a working H-JEPA in the **current harness and environment**, benchmarked against the frozen
baselines (C1 72, C3 50/60, C4 55/80, C5 71, C6 71; docs/lewm/STATUS.md). Moving obstacles and failure analysis
come after, on later branches.

## Architecture

**Level 1: LeWM, as span8w** (`go2_idm_span8w.yaml`):
- ViT-Tiny encoder on 224 x 168 frames, z of 192 dimensions, pixels only;
- AdaLN-zero causal predictor over 4 frames; each 100-ms step's action is its block of 5 (forward, yaw) commands;
- teacher-forced next-z loss plus an 8-step rollout loss;
- SIGReg 0.08, and inverse dynamics (weight 100) over 1- and 8-step spans weighted 1:3.

These are the settings that gave the C5/C6 baselines. Level 1 is **trained from scratch** inside the hierarchy, not
initialised from span8w.

**Levels 2 and 3** (paper §2.1, §C):

| Setting | Choice |
|---|---|
| Stride `s` | 4 at both levels, window `w` 1. Level 2 steps 400 ms, level 3 steps 1.6 s (see below) |
| State encoder | two-layer MLP (hidden 384), applied to each lower-level latent on its own, then a BatchNorm projector, as at level 1; latent 192 |
| Macro-action encoder | DROID's form: the raw commands of the `s` lower steps concatenated (4 x 10 values at level 2), an MLP (hidden 64) to 8 dimensions, then LayerNorm. Level 3 pools four level-2 macro-actions the same way |
| Predictor | causal AdaLN-zero transformer like level 1's, 4 blocks; context `c` 3 states, teacher-forced one-step loss as in the paper (level 1 keeps its rollout loss) |
| Anti-collapse | SIGReg 0.16 per upper level (DROID's level 2); inverse dynamics weight 50, regressing the detached macro-action embedding from consecutive states; SIGReg 0.005 on the macro-action embeddings |

Why stride 4 rather than the paper's 2:
- Our level-1 step (100 ms) is finer than the paper's: DROID's level 1 runs at 5 fps with a level-2 stride of 3,
  so its level 2 steps 600 ms.
- With stride 2, three levels would reach only 400 ms per top-level step. That's no further than level 1 already
  sees, and it wouldn't serve the later long-horizon work.
- The frequency gap that drives abstraction (paper Fig. 5) is wide here. Gait-driven camera bob and body roll/pitch
  vary at 2-4 Hz, while place and heading drift over seconds.

**Training** (paper §C, Table 5):
- One objective: `L = sum over levels of (prediction + lambda * SIGReg + gamma * IDM)`, plus the macro-action SIGReg
  terms.
- AdamW, warm-up then cosine, gradient clipping per level, bf16. Learning rate 5e-5 at level 1, as span8w; upper
  levels the same (DROID's choice; the simulator runs used a fifth).

## Data

The existing LeWM corpus: 3,599 training units and 400 validation units, 3.28M frames.
- A sample is one 49-frame clip (4.9 s) within one unit, so 4 top-level states at stride 16. About 94% of the
  12-frame windows' starts allow one; `lewm_corpus index --length 49`.
- Per clip, each level trains on its own random crop (paper §C):
  - level 1: 12 consecutive frames;
  - level 2: 4 consecutive level-2 states (every 4th frame);
  - level 3: the 4 states at stride 16.
- That is about 20 encoded frames per clip, against 12 now. Batches stay at 16 mazes x 8 clips.
- No new data collection.

## Evaluation

**H1 health**, per level:
- prediction loss doesn't fall towards zero (the collapse signature, paper §C);
- SIGReg settles near its Gaussian floor;
- effective rank and per-dimension z spread.

**H2 action sensitivity**, per level: rollout error with the true (macro-)actions at most 0.7 x that with shuffled
ones, and below persistence. This is the current L2 gate, applied at every level.

**H3 level-1 quality**, against span8w on the same checks:
- the C5-style readout (corpus readout, `lewm_readout package`) through gates L3/L4;
- the turn gate (`eval.lewm_turn_gate`);
- deadlock escapes (`eval.stuck_replay evaluate` on the stored counterfactuals).

Bar: L4 at most 5.0 mm, the C5 baseline.

**H4 abstraction** (paper Fig. 4, 7), on validation units:
- per level, MLP-probe NMSE of a slow entity (maze position and heading) and a fast one (body roll and pitch from
  the pose quaternion, and the body-frame twist);
- per level, latent distance to an anchor against shortest-path distance through the maze (the planning cost
  geometry).

These are reported, not gated. They show whether the levels separate timescales as the paper predicts.

**Benchmarks**:
- New arms **C7** (H-JEPA level 1's rollout read by its decoder) and **C8** (the direct head on its encoder), run
  through the existing `LeWMForecaster`. Level 1 is saved in the LeWM checkpoint format, so no new controller code is
  needed.
- The same 100 episodes as C5/C6, paired comparison against C5 71 and C6 71.

In the current harness only the forecaster slot changes. The upper levels are trained and measured here; they plan
(subgoals, abstract goal costs) once the environment needs long horizons.

## Code

| File | What |
|---|---|
| `go2nav/models/hjepa.py` | `UpperLevel` (state encoder, macro-action encoder, predictor, projector, IDM head), `HJEPA` (a `LeWorldModel` plus upper levels), the joint `losses`, save/load, and level-1 export in LeWM format |
| `go2nav/data/lewm_dataset.py` | clip sampling at a given length (the index already supports any length) |
| `go2nav/training/hjepa.py` | training on the `training.lewm` pattern: resumable checkpoints, keyed batches, per-level validation |
| `go2nav/eval/hjepa_levels.py` | H4 probes and cost geometry |
| `configs/hjepa/go2_3level.yaml` | the configuration above |
| `controllers/registry.py`, `arms.py` | C7 and C8 |
| tests | per-level shapes, gradient flow from upper losses into level 1's encoder, resume, export round trip |

## Budget and order

1. Code and tests: about 1 day.
2. A 15,000-step screen, about 3 h. Check H1, H2 and a first H3 read before committing to the full run.
3. A full run of 60,000 steps (span8w's budget), about 12-13 h at an estimated 0.75 s per step.
4. Readouts and offline evaluation, about 2 h.
5. C7/C8 benchmarks, about 8 h at 6 lanes.

In all, about 1.1 GPU-days, within the approved budget. A second seed would follow only if the comparison with C5
is close.

## Decisions for review

1. **Three levels, strides 4 and 4** (100 ms, 400 ms, 1.6 s). The alternative is the paper's stride 2, or two levels.
2. **Upper-level settings from DROID** (SIGReg 0.16, IDM 50, macro-action SIGReg 0.005, shared learning rate):
   DROID is the paper's setting nearest ours, varied scenes with inverse dynamics needed.
3. **Level 1 trained from scratch** with span8w's settings, rather than warm-started from span8w.
4. **Arm names C7/C8.**
