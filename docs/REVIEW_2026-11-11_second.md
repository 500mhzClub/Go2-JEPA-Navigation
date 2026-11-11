# Second review, 11 November 2026 (evening): what the day missed

Asked for on the evening of 11 November, with P22's first runs training and P23 blocked: review the whole session
against its objectives, re-evaluate the literature, find what was missed, and plan the next steps. Approved the same
evening ("All approved"), with one change: no model is trained on the CPU, as a fallback or otherwise; all training
runs on the dedicated GPU.

## Verdict

The planner objective is met. The world-model objective has no positive result yet: one clean negative (P21's ring
JEPA) and a pivot (docs/REVIEW_2026-11-11.md) whose direction holds. Seven things were missed; they change the plan in
three places: the yardstick, P22's parameterisation, and the order of the Genesis and slot pieces.

## Against the objectives

| Objective | Where it stands |
|---|---|
| The cost planner sorted (10 November) | Merged. Its third round lost 9 round trips with four walkers (27 to 18), undiagnosed by decision |
| Back to the world model as soon as possible | A day went on P21, which P4b to P4f and the August review in LeWMQuad-v3 already argued against; they were not re-read before it was proposed |
| A self-supervised JEPA and planning, few hand rules | No learned model has yet beaten the hand tracker at anything. The count of decision-path constants (9 November) has not been reported since P18; it resumes at the next merge |
| A generic recipe for dynamic, novel worlds | Not tested outside one maze family, and that family flatters the question (below) |

## Measured for this review (CPU, `outputs/review_1111/`)

**1. The walkers are almost deterministic.** 764 val windows of P20's corpus with a walker in sight within 3 m,
replayed exactly and branched 8 times with fresh dice. Median position error (m):

| | 1.6 s | 3.2 s | 6.4 s |
|---|---|---|---|
| Two independent futures of one state | 0.00 | 0.00 | 0.04 |
| The best a deterministic predictor can do | 0.00 | 0.00 | 0.18 |
| Carried on at the true velocity | 0.00 | 0.18 | 0.88 |
| Carried on at the tracker's velocity (training validation) | 0.052 | 0.322 | 0.930 |
| M1 (no object masking), final | 0.070 | 0.179 | 0.411 |

So stochastic predictors are not what is missing here, and the achievable error is far below M0's and M1's.

**2. The robot's choice matters often.** In 338 close encounters (a walker in sight within 1.5 m), holding against
backing away moves the walker by more than 5 cm in 17% of them at 1.6 s, 32% at 3.2 s and 41% at 6.4 s (0.17, 0.27
and 0.47 m when it does). The reaction measure has signal.

**3. Where a walker model can pay** (P17's planner, 6.4 s, the `walkers` terminal distance; round trips, contact
stops in brackets; one seed except where a range is given):

| Suite | The ring tracker (`ring_velocity`) | The true model | Room |
|---|---|---|---|
| Not yielding, 1-2 walkers (80) | 59-62 (7-12) | 69 (0) | about 8 |
| **Not yielding, 4 walkers (40)** | **13 (16)** | **29 (0)** | **16** |
| Reactive, 1-2 walkers (80) | 74 (6) | 80 (0); the rules with fresh dice 80 | 6 |
| Reactive, 4 walkers (40) | 35 (4) | not finished | at most 5 |
| Polite, 1-2 walkers (80) | 78 (1-2) | 80 | 2 |
| Loops and rooms (family B), not yielding, 1-2 (80) | 73 (7); frozen 74 (1) | 76 (0) | 2-3 |
| Loops and rooms, reactive, 1-2 (80) | 78 (2) | not finished | at most 2 |

Nearly all the room is contact stops. Where the layout has loops the walker model barely changes round trips: the
difficulty is mostly a walker sealing the one route of a tree maze.

## What was missed

1. **P22 is trajectory prediction, not a JEPA.** It predicts the metric positions of engineered tracks. That field's
   literature was not consulted: constant velocity is a famously strong baseline (Schoeller et al. 2020), which M0 and
   M1 reproduce (worse than it at 1.6 s), and the standard remedy is to predict a correction to it. P22 stands as the
   ceiling for what object-level dynamics can give the planner.
2. **The yardstick.** P22's planner bar (60 with 1-2 walkers) tests parity on a suite with about 8 missions of room
   and a spread of 2-4. The sensitive measures are four walkers that do not yield, contact stops, and the offline
   errors against the floor above.
3. **This world is easy.** Scripted, near-deterministic walkers on a cell graph, with an effect specific to tree
   mazes, cannot carry a claim of a generic recipe. A second world is needed.
4. **Genesis does not need slots first.** The mission harness already tracks walkers (the hand tracker on the built
   map, and the earlier model's walker finder). The predictor can be tested there on those tracks; slots are the stage
   at which learned perception replaces the tracker.
5. **Official code exists.** C-JEPA's repository holds VideoSAUR training for Push-T and CLEVRER, extracted slots and
   planning scripts; VideoSAUR's own code is MIT. P23's slot model was written from memory. This repository's LeWM is
   a line-by-line port, and the slot model should be.
6. **Frozen features are not uniformly better in this project's records.** For the robot's own motion the from-scratch
   LeWM beat frozen V-JEPA 2.1 (9.4 against 14.2 mm at 0.8 s, docs/lewm/STATUS.md), which the first review left out.
   Frozen features are for the scene and its objects; the robot's own motion stays learned or odometric.
7. **Two faults of the evening.** Replaying a corpus episode through its planner stopped being exact once the
   tracker's detection was sped up (found here; the counterfactual measure now replays the recorded requests; nothing
   had been scored with it). And the slot trainer's GPU crash is not a code error (the same steps run clean on the
   CPU), yet about 40 page faults were caused on the display card while testing it.

## The literature, re-evaluated

- **Holds:** frozen dense features, object tokens and a masked object-level predictor remain the best-supported
  recipe.
- **The nearest work on novel environments:** InJePA (arXiv 2610.23896, September 2026) navigates unseen Habitat MP3D
  scenes with frozen V-JEPA 2.1, DINOv3 or CroCo encoders, a 4-frame history and no map: 53-70% success at about 8 m.
  It lists dynamic environments as future work. Moving agents are the open gap.
- **P22's field:** planning-informed prediction (PiP, ECCV 2020); interventional behaviour prediction (Tang et al.
  2022, arXiv 2204.08665: conditioning on the ego plan from observational data over-trusts the ego's influence; half
  of P20's corpus is planner-driven, so intervention frames should be weighted or separated); learned relational
  dynamics with lookahead for crowd navigation (RGL, arXiv 1909.13165); a camera-only world model among humans
  (NavDreams, arXiv 2203.12299).
- **Slots under camera motion:** SlotContrast (CVPR 2025, arXiv 2412.14295) is more consistent over time than
  VideoSAUR (video FG-ARI 82.9 against 73.9 on MOVi-E) and reports that camera motion remains hard; motion-guided
  discovery (MoTok, CVPR 2023) is the usual remedy.
- **Stochastic JEPAs** (Branch-JEPA, arXiv 2607.05238, and others) are not needed here; they would be for real agents.
- **A second world:** Zhu et al.'s bouncing-ball task (arXiv 2609.04264), where LeWM plans at 0-3%. No code was
  found; it is simple to rebuild, so its figures are a reference, not a like-for-like comparison.

## Plan (approved)

| Step | What | Passes when |
|---|---|---|
| 1 | **P22 revised:** the prediction as a correction to constant velocity, the tracker's velocity in each token (M2); with object masking (M3); with six-step training rollouts (M4); then a variant weighting intervention frames | (a) no worse than the tracker's constant velocity at 1.6 s and at most half its error at 3.2 and 6.4 s; (b) the reaction slope's interval above 0.5; (c) four walkers that do not yield, four seeds: at least 6 round trips above the tracker and half its contact stops |
| 2 | **P23 by port:** the official VideoSAUR code on the DINOv2 features, on the GPU | 90% of walkers in view bound |
| 3 | **The predictor in Genesis on the harness's tracks:** offline on recorded missions, then the long-horizon planner on the built map | offline: beats constant velocity on Genesis tracks; missions: against P18's merged figures. Only after step 1 passes offline |
| 4 | **A second world:** the bouncing-ball task rebuilt; LeWM and the object recipe | LeWM near zero as published; the recipe clearly above it |
| 5 | Slots replace the tracker; encoder adaptation; the H-JEPA decision | later |

All training on the dedicated GPU. The slot trainer's fault is to be solved there (alone on the card first).

## References added by this review

- Schoeller et al. 2020, What the constant velocity model can teach us about pedestrian motion prediction:
  https://arxiv.org/abs/1903.07933
- Song et al. 2020, PiP: planning-informed trajectory prediction: https://arxiv.org/abs/2003.11476
- Tang, Zhan, Tomizuka 2022, Interventional behavior prediction: https://arxiv.org/abs/2204.08665
- Chen et al. 2020, Relational graph learning for crowd navigation: https://arxiv.org/abs/1909.13165
- Dugas et al. 2022, NavDreams: https://arxiv.org/abs/2203.12299
- Zhou et al. 2026, InJePA: https://arxiv.org/abs/2610.23896
- Song et al. 2026, Branch-JEPA: https://arxiv.org/abs/2607.05238
- Manasyan et al. 2025, SlotContrast: https://arxiv.org/abs/2412.14295
- Bao et al. 2023, Object discovery from motion-guided tokens: https://arxiv.org/abs/2303.15555
- Baldassarre et al. 2025, DINO-world: https://arxiv.org/abs/2507.19468
- Code: https://github.com/galilai-group/cjepa, https://github.com/martius-lab/videosaur,
  https://github.com/galilai-group/stable-worldmodel

Figures quoted from papers come from their text as read on 11 November and should be checked before they are relied
on in writing.
