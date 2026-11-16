# Development narrative and current assessment — 16 November 2026

**Stopped at the user’s request, 16 November 2026, 16:03 UTC.** No further experiments are authorized by the historical instructions below. F004 factual training ended at update2,361; its latest saved checkpoint is2,000. Branch/paired training and F004 forecast evaluation never started. See the [full work account](FULL_WORK_ACCOUNT_2026-11-16.md) for the current status, results and explanation of the drift from JEPA-plus-head research to a direct supervised baseline.

This is a dated synthesis of reviewed evidence while full O0 planning is still
running. It is preparation for the final report, not completion of P25/P26. It
introduces no new experiment, statistic or scientific interpretation. The
historical basis is the [15 November repository review](../REVIEW_2026-11-15_repository.md);
later statements refer to the individual reports below. Forty-two underlying
result-review digests were checked before writing this account.

The project has built a useful experimental platform and demonstrated a useful
learned object-dynamics component. It has not yet connected self-supervised visual
representation, prediction and planning into a demonstrated improvement for Go2.
The current programme is valuable because it tests those missing connections
separately and preserves unsuccessful results.

## How development reached the present experiment

The October migration made the inherited simulator, locomotion, data and model
code reproducible. Replay and equivalence checks gave later experiments a sound
engineering base. The static-maze experiments then showed that better offline
representation measurements did not reliably improve missions over command-based
motion prediction. That task offered limited evidence for the extra information
a learned world model should contribute.

Moving obstacles made the question more demanding, but initially the gains came
largely from sensing, memory, action search and navigation rules. Subsequent
diagnostics exposed camera blind zones, interface and motion confounds, and
measurements that could improve without useful future prediction. The important
development was learning to separate these failures rather than attributing every
mission outcome to the model.

The ring-world work then isolated object dynamics. M5 was a substantive positive
result on engineered tracks: the historical review documents fewer contact stops
at similar completion under a frozen rule, with more timeouts. Its Genesis
trajectory result was also useful but conditional on simulator pose, full layout
geometry and the evaluated visibility cohort. M5 predicts explicit positions;
it is an object-dynamics reference, not proof of the latent JEPA claim. The value
and planner branches did not establish the broader navigation result. These are
historical findings, not new reruns of those suites.

P25 therefore moved to a controlled world with known physical state. L0 tests the
ported global latent model; L1 adds physical auxiliary supervision; O0 and O1
share a frozen slot encoder and differ in the predictor's masking/recovery
objective. P26 is the separate bridge back to the existing Go2 simulation data:
can camera-derived objects feed the already useful dynamics pipeline?

## What the current execution added

The first priority was valid inference. The original six trainers encountered
operational stalls, but their checkpoints were preserved and sequential recovery
completed all six at 303,000 steps. Full-state audits and the final numerical
checks passed. Training-derived normalization calibration and the qualified
float64/math inference policy produced derived inference artifacts without
retraining the learned weights. Recovery does not identify the original hang's
root cause. Numerical admission does not establish physical usefulness.
[Final baseline inference](FINAL_INFERENCE_2026-11-16.md) records the scope and
limitations of those checks.

All six seed0 baselines then completed physical probes and 18,000 planning cases,
with independent saved-case reconstruction. L0's H4 endpoint success was
97.6%, 97.8% and 97.1% at n1/n3/n6. L1 was slightly better at n1, worse at n3,
and much worse at n6. The expected weak-L0 reference pattern was not reproduced.
Consequently the original O1-minus-L0 bar of thirty percentage points cannot be
met against these particular baseline point estimates. The bar and the mismatch
remain in the report; the task is not being retuned to manufacture the expected
failure. These results cover one training seed per count.
[Baseline planning report](FINAL_BASELINE_PLANNING_2026-11-16.md)

The first RGB slot encoder and both predictors completed their training, state
audits and deployed numerical admission. Their downstream evidence is weak:

| Measurement | Reviewed observation | What remains unresolved |
|---|---|---|
| Encoder mask binding | 0 of 20,000 ball appearances reached IoU 0.5, including the mostly visible subset | Why this recipe failed to produce object-aligned masks; how information is distributed in its latents |
| Four-step physical readout | O1 reduced red/white RMSE by 23.01%/8.93% relative to O0; both were worse than holding the current estimate | Whether the predictor, representation or training/inference history mismatch explains the loss of useful prediction |
| O0 planning, frozen interim prefix | 2/495 successes, against L0 486/495 and L1 469/495 on the same tasks | Full-run result, O1 planning, seed variation and the contribution of encoder, dynamics and latent goal cost |

The evidence is in the [binding](OBJECT_BINDING_N3_2026-11-16.md),
[paired probe](OBJECT_PROBES_N3_2026-11-16.md) and
[interim planning](PRELIMINARY_OBJECT_PLANNING_2026-11-16.md) reports. The interim
prefix is the preserved U008 snapshot from 10:33:50 UTC, comprising the first
165 complete episodes. It is not a readout of the latest growing output and does
not replace the full run or admit its successor.

## How the literature constrains the explanation

The physical-probing paper separates linear access to state from more entangled
information. Our probe uses a different fit size and ridge penalty; a poor linear
readout cannot establish that all physical information is absent. Nor can low
latent loss establish planning quality. Those distinctions matter when the strong
L0 planning result and weak white-ball readout coexist.
[Section 3 and Appendix A.10](https://arxiv.org/html/2609.04264v3)

Causal-JEPA's masking objective changes the information available to the predictor
while retaining an identity anchor. Its discussion distinguishes predictive
influence from recovery of true causal mechanisms. Here, failed mask binding
prevents treating the slot indices as demonstrated individual objects; a lower
masked-model error is not evidence that causal interaction has been recovered.
[Sections 4.1–4.2 and Remark 2](https://arxiv.org/html/2602.11389v2)

VideoSAUR distinguishes frame-level mask fit from consistency across video. Its
reported non-object groupings under reconstruction concern a different decoder
and feature setting. That supports investigating grouping, but does not identify
the cause of this RGB encoder's failure. Our per-frame binding measure also does
not establish stable tracking.
[Sections 3.3–4.1](https://arxiv.org/html/2306.04829v2)

These sections were revisited before this synthesis. No additional model change
or diagnostic is adopted from them here.

## What this can establish for Go2

The transferable aim is a tested way to retain small moving objects and predict
their futures from observations. The toy weights themselves are not the proposed
robot controller. P25 asks whether the representation and latent planner work on
a controlled problem; P26 asks whether a separately frozen camera encoder can
supply useful tracks to the existing Genesis dynamics pipeline.

Several limits are decisive. The balls pass through one another, their autonomous
motion does not react to the controlled ball, and endpoint planning has no
collision cost. Successful P25 planning therefore cannot demonstrate interactive
collision avoidance. P26 is offline and conditional on its recorded data and
observation assumptions. It cannot by itself demonstrate closed-loop mission
improvement, onboard state estimation, robustness to stale observations or a
real-time deployment deadline. The camera encoder used in P26 is separate from
the RGB balls encoder, so the latter's negative result does not settle P26.

The strongest parts of the development process are preserved controls, matched
cohorts, checkpoint custody, independent inspections and candid negative results.
Its recurring weakness has been allowing improvements to the surrounding
navigation system or offline metrics to stand in for the full predictive
contribution. The present approved sequence keeps that contribution testable.

## Work still required

Finish and independently inspect full O0 planning, review it against the
literature, then run the matched O1 experiment. Continue with the frozen-s0 P26
export, the original 20,000-step S1 recipe and the matched five-arm offline
comparison. Then complete the additional n3 seeds and n1/n6 object runs in the
approved order. P26 must retain detection, identity, occlusion and trajectory
coverage alongside conditional prediction errors.

The final P25 comparison requires all eighteen model rows and five encoder-binding
results. At the reviewed input-preparation snapshot, only six model rows have
complete comparison inputs; fifteen individual planning/probe/binding results
are admissible. That is input availability, not a new aggregate result.
The [completion inventory](COMPLETION_INVENTORY_2026-11-16.md) and
[runbook](RUNBOOK.md) define the remaining evidence and commands. The final report
must include the source/dependency audit and the separately verified E004 archive.

The user removed GPU-hour caps; elapsed accounting, serial full GPU execution
and literature/result gates remain. No planner optimization, new simulator
collection, dense-feature baseline, value model or hierarchy experiment is being
added to this stage. Full scientific completion and robot transfer remain unproven.

The local synthesis manifest is
`outputs/gated/report_preparation/development_narrative_20261116/inputs.json`.
It binds the dated historical review, source reports, primary-source consultation
and exact results/reviews used here. Checking those digests does not substitute
for the individual scientific inspections or for the unfinished experiments.
