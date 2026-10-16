# LeWM in the benchmark suite: plan

Approved by Andrew on 16 October 2026 (all five decisions as recommended). Progress and the place to resume from:
[STATUS.md](STATUS.md). Paper: Maes, Le Lidec, Scieur, LeCun and Balestriero, *LeWorldModel: Stable End-to-End
Joint-Embedding Predictive Architecture from Pixels*, arXiv:2603.19312; code github.com/lucas-maes/le-wm (MIT).

## Role

LeWM enters the benchmark the way C3 does: as a **motion forecaster inside the unchanged harness**, so that, as for
every arm, only the forecast differs.

| Arm | Forecast |
|---|---|
| **C5** | LeWM's encoder and predictor, trained jointly from Go2 pixels with SIGReg, roll each candidate's commands forward in latent space; a decoder reads body-frame motion from the latents alone (C3's counterpart) |
| **C6** | a direct head on LeWM's own encoder features, the command history and the candidate tape (C4's counterpart) |

C5 against C6 asks whether the LeWM predictor adds anything beyond its encoder; C5 against C3 compares an 18M-parameter
JEPA trained end to end in the task domain with a frozen 304M foundation encoder plus a 17M predictor.

The paper's own use case, CEM planning to a goal image, cannot drive these missions: they are 100-480 s egocentric
round trips to coordinate goals with a 2-cm arrival radius, which need mapping and routing (LeWMQuad-v2 tried latent
CEM planning with energy heads and failed). Its other analyses carry over: probing (the decoder is a probe; gate L3),
and surprise (offline analysis on the R6 strip missions).

## Why it might fail, and the design answers

1. **Many scenes.** Each paper environment is one fixed scene, so SIGReg's spread must come from controllable state.
   Across 1,000+ mazes, maze identity can satisfy SIGReg while consecutive frames barely change and the predictor
   ignores actions (LeWMQuad-v3 measured exactly this). Answer: batches of 16 mazes x 8 windows; counterfactual actions
   in the data (25% random substitutions in the explore units); gate L2 against shuffled actions.
2. **Small per-step motion** (2 cm or 2.6 degrees per 100 ms) and an 8-step deployment rollout. Answer: an 8-step
   rollout term in the loss (the paper-faithful objective is trained alongside for comparison).
3. **Precision.** R3 measured the tolerance: 10 mm of random forecast error at 700 ms keeps C1 at 20/20, 20 mm 19/20,
   40 mm 7/20; C3's decoder scores 8.5 mm and C4 5.0 mm on T7's metric. Answer: gate L4 at 15 mm before any mission.

## Configuration

| Quantity | Go2 value | Paper | Reason |
|---|---|---|---|
| Model step | 100 ms: one frame, five 50-Hz policy steps | frameskip 5 | the paper's construction exactly; the harness plans in 100-ms ticks (prefix 3, commit 4, pulse 1, scores at 300 and 700 ms) |
| Action | the tick's five executed (forward, yaw) commands, normalised: 10 values | 5 actions | the paper's block; a veto can change the command mid-tick |
| History | 4 frames (t-300 .. t ms) | 3 | velocity and acceleration; the history every forecaster already receives |
| Rollout | 8 steps (100-800 ms), all six candidates in one batch | horizon 5 | the harness's horizons |
| Training window | 12 frames | 4 | for the rollout term |
| Input | 640 x 480 resized to 224 x 168 (full field of view), patch 14: 192 tokens | 224 x 224 | |
| Encoder | ViT-Tiny/14 from scratch, [CLS] then BatchNorm MLP projector to z (192) | same | fallbacks only on a capacity failure: 320 x 240, then ViT-Small |
| Predictor | 6 blocks, 16 x 64 heads, MLP 2048, dropout 0.1, AdaLN-zero, about 10.8M | same | |
| Loss | teacher-forced MSE + 8-step rollout MSE + 0.09 SIGReg (1,024 projections, 17 knots) | first and last | rollout term added |
| Batches | 16 mazes x 8 windows | 128 i.i.d. | |
| Optimiser | AdamW 5e-5, weight decay 1e-3, warm-up 2,000 then cosine, clip 1, bf16; 60,000 steps (about 7.6 h) | same | |

Configs: `configs/lewm/go2.yaml` (the Go2 configuration) and `configs/lewm/paper.yaml` (the paper's objective and
batches, same history). Training is compiled whole-loss (`torch.compile`, autotuned GEMMs): 453 ms per step against
756 eager on the R9700.

## Data

A new corpus from the mission simulator (Go2 camera, the shipped locomotion policy, 10 Hz), on new mazes:
`configs/mazes/lewm_sets_v1.yaml`, 1,200 training and 100 validation layouts.

- **5 x 4 grid (amendment).** The 4 x 4 family is nearly exhausted (about 945 unused topologies after the 357
  registered ones), and the rigorous-phase sealed set, drawn from the same family and not in any registry, would almost
  surely be repeated by 1,300 new 4 x 4 mazes. A 5 x 4 grid (same pitch, walls and start cell) can never repeat a
  4 x 4 maze's topology or embedding, so no evaluation maze, sealed or not, can enter training. Training and
  validation layouts are topology-disjoint.
- **Units** per maze: `explore` (C1 missions with a random action in place of 25% of plans; 900 frames), `mission`
  (C1 unaltered, recovery on for even mazes; 1,000 frames), `free` (a scripted driver: harness action structure,
  cruises and arcs, Ornstein-Uhlenbeck commands, wall approaches, under a privileged stand-off rule; 600 frames) and,
  on validation mazes, `transfer`. About 3.3 million frames, 4,000 units.
- **Visual variety (amendment, Andrew).** Free units draw a random look (`go2nav.sim.random_appearance`: grey,
  tinted, hued, checker and CC0-texture cell surfaces; light direction, colour and intensity, ambient, shadows).
  Missions keep the benchmark's grey cells and light, which their visual tracker is tuned to. Transfer units use a
  held-out look (Bricks097 walls, Tiles093 floor, dim colour-cast light) that no training unit draws.
- **Rendering** on the R9700 (EGL device 1): the integrated GPU (device 0, the default) saturated; renders are
  bit-identical on both.
- **Gate L0** (amended from shares to counts): at least 20,000 windows of each of the six movement types (missions
  turn a lot, so cruise and start-from-rest stay a few percent but are tens of thousands of windows), contact stops
  under 5% of units, command-to-motion lag at zero.

## Gates (offline, before missions)

| Gate | Pass |
|---|---|
| L0 data | as above |
| L1 health | SIGReg plateaus; z spread and effective rank not collapsed |
| L2 action sensitivity | from rollout step 3, true-action error at most 0.7 x the shuffled-action error and below persistence |
| L3 probe | C5's decoder on the true future z: median XY at most 10 mm and RMS yaw at most 1.5 degrees (eval_onpolicy, 800 ms) |
| L4 forecast | C5's T7 score at most 15 mm |

Fallbacks, in order, if L2 or L4 fails: batch composition and lambda; 320 x 240; ViT-Small; last, a latent-difference
action decoder (Delta-JEPA), reported as a variant.

## Benchmarks

`configs/experiments/lewm_benchmarks.yaml` (310 missions): C5 and C6 on R4's, R5's and R6's exact episodes, compared
with the recorded cohorts; C3 at friction 0.25 and C3/C4 on marked strips (never run before); every learned arm on the
demonstration surfaces (visual transfer). Then three seed chains (`go2nav.training.lewm_chain --seed-offset`).
Phase 2: C5-dyn (a corpus with friction 0.25-1 and strips), the surprise analysis, the May-corpus and data-matched
ablations.

## Rules kept

Long jobs checkpoint and resume (Andrew); they run as systemd user services from a frozen worktree. Everything that
gains from it runs in parallel (process pools, GPU batching, compiled training). Existing arms and experiments must
not change: checked by rerunning missions against their records.
