# Physical obstacle forecasting: clarified project priority

The user clarified on 16 November that the primary goal is accurate prediction of
dynamic obstacles, both independently and in response to the agent's actions,
with a path to Go2. Learned heads at inference are acceptable. Cleaner latent
representations are a possible improvement, not a requirement for success. Toy
experiments remain useful when they answer the relevant questions faster.

This clarification governs subsequent priorities. Historical P25/P26 results,
protocols and incomplete comparison rows remain preserved. They are not marked
complete or converted into evidence for the clarified goal. [F001](records/F001.json)
records the literature and source audit behind this documentation update; it is
not a new experiment or a scientific result review.

The subsequent [F002 evidence audit](FORECAST_EVIDENCE_AUDIT_2026-11-16.md)
is complete: saved paired vectors show useful M5 responses but excessive false
responses. It identifies the next comparison—matched deterministic forecasting
with ordinary versus branched data and a separately tested paired-response loss.
Read that evidence before preregistering the next experiment; do not repeat the
audit or treat its proposed comparison as an already-qualified launch protocol.

## What we already tested, and what P25 adds

Go2's earlier P4/P4b/P4d/P4f work already used ridge, MLP and spatial readout heads,
including evaluation through rollouts and additional mover supervision. P16
tested matched action-response magnitudes with controls for command-history
leakage. Those tests already established that low latent prediction loss and
readable current state need not yield useful physical forecasts. Repeating that
general conclusion in balls is not the project's main contribution.

P25 asks the narrower question of whether the selected representation and
training changes remedy a failure in a controlled setting. Its three different
uses of heads must be kept separate:

| Component | Actual role | What it does not establish |
|---|---|---|
| L1 Fourier head | Supervised training loss on encoded and predicted states | It is not used to interpret trajectories during planning |
| C001 ridge probe | Read positions from current and four-step predicted latents | A failed linear readout does not exclude a useful nonlinear head |
| B001 planner | Optimize terminal latent distance, with slot matching for object models | It does not test planning with decoded physical trajectories |

The original P25 rebuild condition is also unresolved: L0's expected failure was
not reproduced. The plan explicitly withholds the paper-comparison claim until
that mismatch is explained. Keep the local observations; do not tune L0 into
failure or treat additional planning replications as an explanation.

## Questions that must govern subsequent experiments

| Question | Required evidence |
|---|---|
| Is useful obstacle state recoverable? | Train-only learned readouts with held-out selection and evaluation, including appropriate nonlinear or spatial heads where the linear test is insufficient |
| Are independent futures accurate? | Physical trajectory errors by horizon against both persistence and constant velocity, including motion changes and difficult subsets |
| Are action-induced changes accurate? | Matched branches from the same state/history under different candidate actions; score the direction and magnitude of the obstacle trajectory difference, including non-reacting controls |
| Does the model contribute beyond its head? | Shared-head encoded-future versus predicted-future diagnostics and a capacity-matched history/action predictor without latent rollout; either may be a useful forecaster, but attribution must remain distinct |
| Can the result transfer to Go2? | The intended observation and action contract, detection/identity/occlusion coverage, physical errors and tails, unseen episodes/layouts, and measured inference cost |

Use physical units for errors. Report missing objects alongside errors on detected
objects. A good aggregate cannot hide failures on waiting, yielding, non-yielding
or occluded obstacles. An action-sensitive latent is not sufficient: predicted
physical changes must agree with the paired physical outcomes. Keep stochastic
future limits explicit; a single recorded outcome is not proof of all possible
responses.

For a learned head, distinguish training on encoded current states from training
on predicted states and future labels. A powerful head may itself learn dynamics;
that can meet the project goal, but must be compared with a direct history/action
forecaster rather than attributed automatically to the latent rollout. Diagnostic
future-frame encodings are unavailable at deployment and must remain diagnostic.

## Use the cheapest testbed that contains the phenomenon

- The existing balls test independent motion and representation/readout choices.
  Their white balls do not react to the controlled ball. They can supply a
  no-response control, not evidence of learned reactive behavior.
- The existing ring-world reaction evaluator already replays and branches the
  same world under hold and retreat, preserves the starting state, and measures
  predicted versus true obstacle displacement differences. Audit and reuse its
  existing results before constructing another reactive toy world or rerunning it.
- Offline Genesis P26 remains relevant to the observation-to-forecast connection.
  Its current code supplies recorded future robot poses and commands. This is a
  conditional forecast test, not proof of accurate responses to alternative
  actions or of forecasting from commands alone. P24's branched Genesis reaction
  test was not run. New Genesis collection remains outside the current scope.

The strongest existing relevant reference is M5's physical trajectory predictor,
which already uses a learned displacement head. Historical Genesis results show
useful aggregate forecasts but weaker performance for walkers waiting on the
robot. Its class name or lack of latent-only inference is not grounds to discount
that contribution. The historical reports do not establish the missing Genesis
counterfactual reaction result.

## Immediate execution order

1. Preserve and finish the in-flight O0 evaluation; independently inspect and
   review the actual terminal artifacts. It remains a local diagnostic.
2. Audit the existing Go2/ring learned-head and reaction evidence against the
   questions above. Identify a genuinely missing comparison before another run.
3. Preregister the smallest useful forecasting experiment, using the existing toy
   or ring setting where it gives an iteration advantage. Freeze data splits,
   head capacity/training, horizon, controls, accuracy bars and stop rules before
   implementation. Then implement, verify, run and review it through the usual
   literature gates.
4. Carry a useful forecast pipeline into the offline Genesis comparison. Preserve
   P26's observation coverage checks and explicitly test the remaining action-
   response gap before claiming the project objective is met.

Do not automatically launch O1 planning or additional toy planning replications
just to fill the former matrix. They remain pending and may be justified by a
specific forecasting question. Existing launchers' predecessor checks remain
intact; any revised execution route needs its own record and verification, not
fabricated reviews or a silent bypass. No new model, experiment or metric was
executed by this clarification. GPU-hour limits remain removed.

## Literature and source basis

[Zhu et al., Appendix A.3 and A.10](https://arxiv.org/html/2609.04264v3)
separate training auxiliary losses, inference-time physical heads and linear/MLP
readouts. Their direct planning example uses a position head; our current Fourier
head must not be described as an already-qualified position decoder.
[Trajectron++, Sections 3–5](https://arxiv.org/pdf/2001.03093) conditions forecasts
on candidate robot futures and evaluates physical trajectory errors. That
motivates the input contract, not adoption of its architecture or proof of local
counterfactual accuracy. [Causal-JEPA, Section 4](https://arxiv.org/html/2602.11389v2)
treats masking as a prediction method; it does not make latent purity our goal.

Local basis: [historical status](../dynamic/STATUS.md),
[original plan](../dynamic/PLAN.md),
[repository review](../REVIEW_2026-11-15_repository.md), and the source files
identified at commit `feb35f4` in F001. Existing numerical result-review bindings
were checked; no historical experiment was rerun for this account.
