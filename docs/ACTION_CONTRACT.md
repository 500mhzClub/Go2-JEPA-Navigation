# The navigation-locomotion action contract: from primitives to continuous commands (open, 22 October 2026)

Status: noted, not scheduled. In dynamic environments the discrete action set is likely to cap performance. The
contract between the learned navigation component and the RL locomotion policy should become a continuous command
space. This note records what the contract is now, what a change touches, and when to make it.

## The contract today

The contract is discrete at both ends.

- **Navigation side** (`go2nav/planning/actions.py`): six actions, each committed for 400 ms after a 300-ms prefix of
  already-committed commands.

  | Action | Command (forward m/s, lateral m/s, yaw rad/s) |
  |---|---|
  | hold | (0, 0, 0) |
  | forward | (0.20, 0, 0) |
  | left arc, right arc | (0.16, 0, ±0.45) |
  | left turn, right turn | (0, 0, ±0.45) |

  Lateral is always zero. C1-C8 all choose among these six: each controller scores the candidates and dispatches one.
- **Locomotion side** (`configs/locomotion/contract_ppo.yaml`): the PPO policy was trained only on nine fixed
  primitives.
  - Hold, 0.20, 0.25 and 0.30 m/s forward, and 0.20 m/s backward.
  - Turns at ±0.45 rad/s and arcs at 0.20 m/s with ±0.45 rad/s.
  - No lateral motion, one turn rate, and no command jitter.

  Tracking is benchmarked only at those points (`docs/validation/locomotion_retrain_2026-10-06.md`).
- **Corpus** (`go2nav/data/lewm_corpus.py`): mostly primitives, too. The free driver's random-walk segments (20% of
  free segments) do issue continuous commands, 0-0.22 m/s and ±0.5 rad/s, so the world models have seen a little
  off-primitive driving. The policy has not been checked there.
- **World models**: already continuous on the input side. The action is the normalised command block (forward and yaw
  per 100-ms tick; lateral is dropped).

## Why it matters in dynamic environments

With moving obstacles the robot needs to:
- modulate speed: slow down to yield or wait, or speed up to pass through a gap;
- step sideways or back away;
- make finer heading changes;
- react inside the current 700-ms commitment.

None of these exists in the six-action set. In static mazes the missions do not show that the set binds: C1, C5 and C6
fail in other ways, through contacts at low friction, pose loss and deadlocks. With moving agents it probably will
bind, and it would cap any learned planner whatever its world model.

## What a change touches, cheapest first

1. **Locomotion policy.** This is more than re-recording data: the policy must be retrained on command ranges, for
   example forward -0.3 to 0.5 m/s, lateral ±0.2 m/s and yaw ±1 rad/s.
   - The upstream Genesis example samples ranges; this port replaced that with the primitive bank.
   - Training takes about 8 minutes on the R9700.
   - Benchmark tracking across the whole envelope, and confirm the old primitives still track as well. Then C1-C4 can
     rerun unchanged on the new policy, so the baselines stay comparable.
2. **Corpus.** Re-record with commands drawn across the envelope: continuous random walks, plus continuous
   perturbations in place of the six-action explore substitutions. The current corpus took about 10 hours with 28
   workers.
3. **World models.** Little change: add lateral to the action vector (15 values per block instead of 10) and retrain.
   That is about 14 hours for LeWM or H-JEPA. H-JEPA's macro-action encoders take any action vector.
4. **Planning and harness** (the expensive part). Candidate scoring over six actions becomes optimisation over
   continuous commands, for example sampling (CEM or MPPI) through the world model. The hand-engineered harness assumes
   the primitives throughout: dispatch safety checks, recovery, latches and the commit timing. A middle step keeps
   discrete scoring but over a richer sampled set, before full continuous optimisation.

## When

- **Do it with the moving-obstacle corpus, not before or after.** That corpus has to be recorded anyway. Recording it
  on the primitive contract and again later doubles the most expensive data step.
- **Order:**
  1. Retrain and benchmark the locomotion policy over ranges.
  2. Check C1 on the new policy against its 72.
  3. Record the moving-obstacle corpus with wide commands.
  4. Retrain the world models.
  5. Leave the planner on discrete candidates at first, so each change is measured on its own.
- **Measure the cap before paying for the planner rework.** In the first moving-obstacle missions, classify failures
  where no primitive could avoid the obstacle but a slower, sideways or backing command could have. If that count is
  large, it justifies moving to continuous optimisation.
