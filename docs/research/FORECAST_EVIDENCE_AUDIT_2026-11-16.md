# Existing forecast evidence and the missing reaction comparison

M5 contains useful physical forecasting and action-response information. Its
response predictions also change too much when the action has little or no true
effect. This is a specific failure to address next. A repeated demonstration that
latent loss does not guarantee planning success would not resolve it.

This audit follows the [user's clarified priority](FORECASTING_PRIORITY_2026-11-16.md).
[F002](records/F002.json) fixed the inputs and analysis before implementation.
It used the CPU to analyse saved predictions; no model inference, training,
simulator replay or new collection was performed.

## What the existing files establish

Twenty exact input files are bound in F002, including model configurations,
reaction rows, historical forecast summaries, Go2 P16 measures, branch-data
summaries and evaluator/trainer source.

- The three saved P16 Go2 evaluations contain both ridge and MLP readouts at
  levels1 and2, and all have `reaction.pass=false`. For example, v2's level1
  1.6s validation reaction slope is0.032 with ridge and0.136 with MLP. These
  artifacts support the earlier report that merely making the state readable
  did not solve accurate reactions. This audit did not refit those heads or
  reconstruct their original predictions.
- M5's saved ring forecast medians are0.0329/0.1007/0.3254m at1.6/3.2/6.4s,
  against constant velocity0.0539/0.2716/0.7761m. Its learned physical displacement
  head is part of a useful forecaster, not a disqualifying workaround.
- Saved Genesis M5 medians are0.0279/0.0736/0.2509m overall. Waiting walkers
  remain weaker:0.0232/0.0575/0.1198m against persistence0.0053/0.0089/0.0361m.
  These are historical summary values, not newly rerun forecasts. The unchanged
  offline evaluator conditions on recorded future robot poses and only scores
  walkers meeting its tracking, visibility and distance conditions.
- Existing ring branch summaries report739,290 training and75,918 validation
  branches; the directories contain4,500/500 episode files. The source groups
  alternative tapes at the same starting moment with copied world state and
  walker random state. File counts and summary contents were inspected; the
  entire payload and historical generation provenance were not audited here.
- N4's saved config uses50% branch windows and the first2,250 training episodes.
  M5's saved config and deterministic training route use ordinary windows,
  six-step rollouts and distance loss. The presence of a `Branches` loader in
  that module does not mean M5's training loop used it. N4 is a different model
  and training recipe, so its result is not a controlled test of branched M5.

## New analysis of the saved paired reactions

All three files—M5, M6 and N4—contain the same511 accepted windows from338
episodes. Each row compares a6.4s hold against retreat at0.2m/s for2s followed by
hold, with the same starting history/world. All keys, labels and truth vectors
match exactly. The original nine model/horizon summaries, including their
rounded episode-bootstrap intervals, reconstruct exactly.

The measured vector is **obstacle position under retreat minus obstacle position
under hold**. A zero-response baseline predicts that this difference is zero;
it does **not** predict that the obstacle itself remains stationary.

| Horizon | M5 mean response-vector error | Zero-response error | M5 minus zero,95% interval |
|---|---:|---:|---:|
|1.6s|0.0408m|0.0323m|+0.0086m [0.0038,0.0135]|
|3.2s|0.1077m|0.0928m|+0.0150m [0.0031,0.0267]|
|6.4s|0.2508m|0.2104m|+0.0404m [0.0169,0.0647]|

There is useful signal on windows with true action effects above the existing
5cm threshold, but unwanted sensitivity elsewhere:

| Horizon | Responsive windows | M5 error / zero error on responsive windows | False response above5cm on nonresponsive windows |
|---|---:|---:|---:|
|1.6s|91|0.1189 /0.1731m|13.6% of420|
|3.2s|167|0.2015 /0.2799m|30.5% of344|
|6.4s|215|0.4229 /0.4985m|47.0% of296|

M5 improves responsive-window mean error by0.0542m [0.0391,0.0707],
0.0784m [0.0559,0.1025] and0.0755m [0.0343,0.1176]. Its mean direction errors
on those windows are37.1/36.1/45.0degrees. Even among exactly unchanged truths
(norm at most1e-6m), predicted effects exceed5cm in12.0/28.4/46.6% of windows.
Thus the missing capability includes recognizing when actions have no effect,
as well as getting genuine responses right.

M6 and N4 have larger pooled response-vector errors than M5 at every horizon.
For example, at3.2s their errors are0.1543m and0.2209m, against M5's0.1077m;
their nonresponse false-positive rates are59.0% and75.6%. These are descriptive
comparisons across different recipes, not proof that longer rollouts or branched
data caused the degradation. N4's several-futures distribution is not assessed
by its saved most-likely prediction alone.

## Interpretation and limits

F002 is `measured_negative` against its joint favorable criterion: M5's paired
vector error must improve on zero response both overall and on responsive
windows. It improves only the latter. The original P22 slope bar remains failed.
Neither result erases the earlier aggregate trajectory gains.

[Trajectron++, Sections3–5](https://arxiv.org/pdf/2001.03093) motivates conditioning
on candidate robot futures and evaluating physical errors. Our additional paired
difference test checks whether the predicted change agrees with the simulated
change. It does not establish real-world human causality or adopt that paper's
stochastic architecture.

Intervals use2,000 paired resamples of whole episodes, following the resampling
principle in [Efron, Section2](https://blogs.helsinki.fi/bk-club/files/2012/05/Efron_Bootstrap_AS1979.pdf).
They describe these historical validation episodes, already inspected in earlier
work. They are not fresh confirmatory evidence, do not account for repeated
model selection and are not adjusted for multiple comparisons.

The old evaluator requested600 windows and retained511. Its source excludes
contact-containing branches and unmatched observations; individual rejection
reasons were not saved. These files omit absolute branch trajectories, so this
audit cannot recover full trajectory ADE, collision timing or missing-object
coverage. The original files also lack exact checkpoint/source digest bindings;
F002 preserves their present bytes without retroactively certifying execution.

## Next discriminating experiment

Use the existing ring data to compare the same deterministic physical forecaster
under ordinary-window training, branch-window training, and branch-window training
with an explicit paired action-effect objective. Keep its learned displacement
head, model capacity, initial weights and update/sample allocation matched.
This separates exposure to alternative actions from supervision of the difference
between them; N4's earlier result does not isolate that question.

Before implementation, a new record must audit the branch payload, shared-history
alignment, persistent object identities, contact/visibility masks and available
targets. Freeze episode/layout splits, selection, losses and evaluation bars.
Keep independent/recorded-motion forecasts, persistence and constant velocity,
paired true responses and nonresponses, and coverage reporting. Do not improve
the pooled score merely by suppressing all response predictions. Short histories,
hidden behavior/timers and stochastic futures remain competing explanations.

This is the selected next question, not a preregistered training protocol or a
proven remedy. No collection or stochastic model expansion is authorized here.
P26's observation-to-forecast test and the unfinished historical P25/P26 rows
remain visible. Finish and review the in-flight O0 run; do not automatically launch
O1 planning.

## Reproduction and evidence

Tracked source: [analysis](../../tools/research/f002_reaction_audit.py) and
[independent verification](../../tools/research/f002_verify.py).
Five software check groups pass. The independent verifier uses scalar math and
explicit repeated-row episode sampling; it checks1,701 values,81
model/horizon/stratum groups and54 intervals, with all20 input hashes intact.

Local artifacts under `outputs/gated/F002`:

- `result.json`: `d9b0a8e9182e20bbb2aa964f7bd7b7ee1b2a27ae2c5b1bb0933010314402030a`
- `verification.json` and `review.json`: exact result/source-bound verification
  and substantive post-result literature review.
- `software/result.json` and `software/review.json`: implementation checks and
  the reviewed prerequisite for the actual analysis.

The commands in the analysis source refuse to overwrite existing results and
require the unchanged input digests. Use the repository research Python with
GPU visibility disabled. The full scientific result remains distinct from the
successful numerical verification.
