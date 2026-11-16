# P25/P26 execution

Runtime update on 16 November: all six original experiments completed 303,000 steps
serially after the authorized reboot. All six actual A015 artifact audits and
A014 completion reviews pass. All six final A003 numerical evaluations and their
A016 operational reviews also pass, under `outputs/gated/A003_recovered/RUN`
and `outputs/gated/A016/runs/RUN`. No original trainer or final evaluator remains
active. Their historical source pins and output directories must be preserved;
do not relaunch completed cases. See the [final inference report](FINAL_INFERENCE_2026-11-16.md).

The n3 seed0 RGB encoder is now training under C005 after primary-method
consultation and unchanged data/source checks. A CPU-only read of its atomic
checkpoint confirms 4,269 updates across all 57 optimizer states and the scheduler,
in epoch1 after its first validation. Current exact process identities and
heartbeat are under `outputs/gated/C005/runs/slots_n3_s0`, with the supervisor
launch identity in `outputs/gated/C005/launch_slots_n3_s0.json`. Do not relaunch it.
These counters establish progress; learned object binding remains unmeasured.
Keep the active gated checkout pinned to81171a2; documentation updates belong in
the isolated validation checkout until the child finishes.

After terminal completion, inspect and literature-review actual C001 and C005
artifacts before selected-checkpoint C004, cache and O0/O1. Object-model probes and
paired learned planning follow, then P26 before extra seeds/counts. The six final
baseline probes are already complete through C006, with a reviewed partial C002
audit under `outputs/gated/C002/final_baseline_probes`; do not rerun them. At admission,
47.699 of48 additional GPU-hours and27.997 of28 B/C hours remained; current use
is in the shared ledger. No further approval is needed for this approved scope.

A017 completed the requested CPU comparison of frozen L1 n1 snapshots at 105,000
and 220,000 steps while this trainer continued. Calibration and physical probes
both have exact-result literature reviews under `outputs/gated/A017`. See the
[checkpoint progress report](CHECKPOINT_PROGRESS_2026-11-15.md) for the measurements,
controls and figures. This diagnostic does not replace any final evaluation gate.

The following A015 commands document the completed audits. Before approving a run,
the artifact audit executes from the
unchanged original checkout (replace the run name for each successor):

```sh
cd ../Go2-JEPA-Navigation-p25
env -u PYTHONPATH HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" outputs/gated/A015/verify_final.py --run l1_n6_s0
cd ../Go2-JEPA-Navigation-p25-validation
```

The audit refuses live or unsuccessful children. It checks source/data lineage,
both final artifacts at303000, exact CPU state restoration, optimizer mapping and
update counts, the original schedule, RNG structure and complete metric coverage.
Results go to fresh `outputs/gated/A015/finals/RUN` directories. Consult primary
sources and review each actual audit, then bind the A014 run's completion review
to that audit before launching the next index. The A015 software review covers
synthetic fixtures only; it cannot pass a real run or replace A003.
Each A014 completion review must also include `final_audit_review` (the path from
this checkout to its A015 review) and `final_audit_review_sha256` (that file's digest).

The failed A010 attempt, A011 stack and both A012 timeouts remain preserved.
See [the runtime record](RUNTIME_RECOVERY_2026-11-15.md).
Launch no concurrent GPU work or silent retry. The A014 plan and all-six A003
checks are complete and reviewed. A007's preliminary GPU diagnostic exhausted its declared
cap; do not retry or infer probe scores from it.

This branch implements the approved offline scope. It does not claim that pending scientific performance gates pass.
Run from this checkout with the repository's Python 3.12 / ROCm environment. `PY` below names its interpreter;
`BALLS_DATA`, `CORPUS_V2`, `RANGES`, `DINO_FEATURES` and `MAZES` name existing local data roots. Outputs are local,
ignored artifacts. Original runs remain on their pinned checkout and are never resumed by these commands.

Use `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `PYTORCH_TUNABLEOP_ENABLED=0` and a clean `PYTHONPATH`.
Tests additionally use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; otherwise a system ROS plugin requires unrelated packages.
New training refuses CPU fallback and refuses to start while another repository trainer is active. Small numerical
checks are the only work allowed alongside the six existing jobs. Device intervals are recorded in
`outputs/gated/gpu_intervals.json`; the default allowance is 48 hours. Set `GO2NAV_GPU_LEDGER` only to relocate that
same shared ledger, never to reset the allowance.

## Literature gate

Before implementation/experimentation: create a complete preregistration under `docs/research/records` from the
existing examples, after reading relevant primary methods/appendices and official technical sources. Record the
actual design consequences, contrary evidence and immutable protocol. Reusing a record is allowed for its declared
verification suite and seeds. Changed scientific protocols require a new record.

After each result: revisit the relevant literature and write `review.json` beside `result.json`. Required fields:
`research_id`, `result` (relative filename), `result_sha256`, `reviewed_at`, `sources` (URL, version, sections,
consequence), `measurements`, `uncertainty`, `explanations`, `next_step`, `status`. Use `passed`, `measured_negative`,
`technical_blocked`, or `budget_incomplete` accurately. The tools check completeness and result hashes; they cannot
perform or attest to the literature reading. Never generate a passing review merely to unblock a command.

Each next stage takes the review of its exact parent artifact. Keep labels out of representation/predictor training;
executed commands are permitted. The post-result explanation must distinguish implementation, representation,
prediction, planning and generalization. Existing A001/A002/B001/C001 numerical reviews are in local run directories.

## A: final inference artifacts

All six cases below completed and passed their individual reviews on 16 November.
These commands document the frozen protocol; the existing result directories must
not be reused. A003 ran one case at a time through the reviewed A016 supervisor:

```sh
python3 outputs/gated/A016/run_a003.py --run l0_n1_s0
```

Use the frozen order `l0_n1_s0`, `l0_n3_s0`, `l0_n6_s0`, `l1_n1_s0`,
`l1_n3_s0`, `l1_n6_s0`. Raw numerical outputs are under
`outputs/gated/A003_recovered/RUN`; operational outputs are under
`outputs/gated/A016/runs/RUN`. Review both exact results before the next case.
The wrapper checks all six final lineages, requires an unused output directory,
and runs the unchanged evaluator from pinned commit `161fe8b`. It enforces the
remaining4-hour A allocation and48-hour shared allowance, reserving45 seconds
for bounded cleanup. Overlapping outer/inner intervals count once; unresolved
child termination leaves accounting open. It never approves a scientific result.

The wrapper executes the equivalent of this diagnostic command from the pinned
gated checkout:

```sh
"$PY" -m go2nav.eval.balls_inference --checkpoint outputs/balls/recovery_A014/l0_n3_s0/model.pt \
  --data "$BALLS_DATA" --out outputs/gated/A003_recovered/l0_n3_s0 --research A003
```

A003 uses train-only calibration and float64/math inference, selected by A002 at unchanged tolerances. The source
checkpoint is copied through an open inode, learned weights remain unchanged, and `calibrated.pt` is explicitly
inference-only. Review the final numerical result before planning. No final-checkpoint validity is inferred from
the successful step-40,000 checks.

The retired original workspace follow-up used `go2nav.eval.await_balls --runs outputs/balls/runs --data "$BALLS_DATA"`
with `--out outputs/gated/A003` and `--pids` listing all six original trainer PIDs. It records their process identities,
waits up to 24 hours without signalling them, verifies final step counts and unchanged source/config hashes, and
runs only the A003 suite. `outputs/gated/A003/state.json` records progress. It exits awaiting literature review;
it cannot approve its own results or launch a dependent experiment. Its old state is
historical; do not restart it with the retired PIDs. A016 now handles recovered final paths.

Operational exception A006: the user authorized immediate retirement of two hung original L1 processes.
The original waiting A003 queue was stopped before them and archived in `outputs/gated/A006/queue_before`.
It is no longer active. Four originals continue; two full-state copies at105000 have passed the A005 CPU
preflight and await serial continuation under the original7dfc837 trainer. Do not relaunch the old six-PID
queue or assume that all final files will appear under `outputs/balls/runs`. Resolve original/recovery paths
with their checkpoint lineage, verify final303000/configuration, then run this same A003 suite sequentially.
Literature review is required after each recovery result and each numerical result. The user-approved
exception changes process handling only; it does not authorize device resets or changes to the experiments.

## B: fixed test tasks and planning

These test splits and controls have already been generated and evaluated in this workspace:

```sh
"$PY" -m go2nav.data.balls build --out "$BALLS_DATA" --split test --research B001
"$PY" -m go2nav.eval.balls_planning --data "$BALLS_DATA" --n 3 \
  --out outputs/gated/B001/n3 --research B001
```

Existing complete data splits are preserved; evaluators require a new output directory. Each n uses 1,000 new
episodes and horizons 4/6/8. Recorded/direct/true-CEM endpoint success was 100% at every n/horizon; hold and random
controls are retained. Five groups are evaluation groups, not five training seeds. No collision dynamics are added.

```sh
"$PY" -m go2nav.eval.balls_planning --data "$BALLS_DATA" --n 3 --research B001 \
  --checkpoint outputs/gated/A003_recovered/l0_n3_s0/calibrated.pt \
  --inference-review outputs/gated/A003_recovered/l0_n3_s0/review.json \
  --controls outputs/gated/B001/n3/result.json --out outputs/gated/planning/l0_n3_s0
```

The same CLI loads reviewed O0/O1 checkpoints. It never supplies intermediate future frames or truth positions to
the learned planner. CEM uses final elite mean; goal slots are independently encoded and Hungarian-matched.

## C: three-ball core, then extra seeds/counts

C006 completed all six baseline probes on spare CPUs while the GPU slot trainer
continued. Its eight helper groups and all six actual numerical/operational cases
were inspected and literature-reviewed. L1n6 is a measured scientific negative;
all operational cases passed. See the [full results](FINAL_BASELINE_PROBES_2026-11-16.md).
The following command documents the completed execution, not a relaunch. The
registered order was L0 n3, L1 n3, L0 n1, L1 n1, L0 n6, L1 n6, all seed0.

```sh
python3 outputs/gated/C006/run_one.py --run l0_n3_s0
```

Each case needs its own `C006/pre_RUN.json` with substantive primary-source
consultation and exact existing train/test data hashes. The wrapper requires
reviewed final A003/A015/A016 parents, the C006 software review and prior case
reviews. It imports unchanged C001 probe functions with exactly the existing
samples, controls, ridge coefficient and episode bootstrap. C006 explicitly
records CPU execution as an adaptation; it performs no neural training or new
calibration. CPU/GPU arithmetic is not assumed identical. Actual finite, grouping,
solver residual, frozen-weight and data/source checks remain mandatory.

Use one eight-thread CPU child, at most 16 GiB RSS and 900 seconds including
cleanup. Scientific artifacts go to `outputs/gated/probes/RUN`; separate
operational evidence goes to `outputs/gated/C006/supervision/RUN`. Inspect and
literature-review both exact results before the next case. No review or successor
is automatic. The concurrent GPU trainer and its pinned checkout remain unchanged.

C005 adds an external deadline around unchanged C001 commands: the smaller of
the remaining 28-hour B/C and 48-hour shared allowances, reserving 45 seconds for
identity-checked cleanup. Its 18 CPU process/accounting checks passed and received
literature review. It preserves the last atomic checkpoint on a deadline stop;
updates after that checkpoint may be lost. No successor or review is automatic.
Launch the first stage with its recorded scientific consultation:

```sh
python3 outputs/gated/C005/run_stage.py --stage slots --n 3 --seed 0 --python "$PY" \
  --preconsultation outputs/gated/C005/pre_slots_n3_s0.json
```

The wrapper accepts `slots`, `extract`, `o0` and `o1`, plus the corresponding
checkpoint/cache/parent-review arguments below. Each new stage needs its own
substantive consultation and unused output directory. Scientific outputs remain
under `outputs/gated/C001/STAGE_nN_sSEED`; operational outputs are under
`outputs/gated/C005/runs/STAGE_nN_sSEED`. Inspect and review both actual results
before dependent work. Direct commands below document the unchanged child
protocol; use the supervisor for execution so the B/C cap is also enforced.

C005 revision2 corrects a prospective accounting error: C004 belongs to the
validation reserve, as its preregistration already states. V1 counted it against
B/C. No C004 interval exists yet, so the live run's allowance and deadline are
unchanged. The audit and original evidence are preserved under
`outputs/gated/C005/allocation_audit` and `outputs/gated/C005/record_v1.json`.
The isolated `outputs/gated/C005/v2` implementation passed eight focused CPU
checks and literature review, inheriting unchanged v1 process-supervision checks.

After the live v1 run finishes and its actual results receive review, use
`outputs/gated/C005/v2/run_stage.py` for subsequent C001 stages. First move to the
updated checkout and verify that `v2/software/result.json`'s `research_sha256`
equals the current C005 record digest. V2 operational outputs and software review
are under `outputs/gated/C005/v2`; scientific outputs remain under C001.
Do not change the live v1 source, its deadline or the pinned gated checkout.

C003 has also passed a bounded three-update GPU diagnostic of the actual checkpoint helpers, optimizer state,
schedule, cursor and RNG restoration, followed by CPU/GPU output/gradient and batch-grouping checks. Its synthetic
inputs and short run do not establish full-loop crash recovery, convergence or final trained-checkpoint validity.
To reproduce in a new output directory:

```sh
"$PY" -m go2nav.eval.slots_recovery --out outputs/gated/C003/recovery --research C003
```

Review that result before dependent work. The completed review is `outputs/gated/C003/recovery/review.json`.
Use it as the numerical-verification review for the following training stages; model-source hashes must match.

```sh
"$PY" -m go2nav.training.balls_slots slots --data "$BALLS_DATA" --n 3 --seed 0 --research C001 \
  --verification-review outputs/gated/C003/recovery/review.json --out outputs/gated/C001/slots_n3_s0
```

Review the selected validation reconstruction result, then validate that selected encoder on the fixed real
train/validation windows. The numerical check uses no test examples or optimizer updates:

```sh
"$PY" -m go2nav.eval.slots_checkpoint --data "$BALLS_DATA" --research C004 \
  --checkpoint outputs/gated/C001/slots_n3_s0/model.pt \
  --parent-review outputs/gated/C001/slots_n3_s0/review.json --out outputs/gated/C004/slots_n3_s0
```

Review C004's actual result before extracting the frozen cache:

```sh
"$PY" -m go2nav.training.balls_slots extract --data "$BALLS_DATA" --n 3 --research C001 \
  --checkpoint outputs/gated/C001/slots_n3_s0/model.pt \
  --parent-review outputs/gated/C004/slots_n3_s0/review.json --out outputs/gated/C001/cache_n3_s0
```

Review cache integrity, then run `o0` and `o1` sequentially with identical seed/cache:

```sh
"$PY" -m go2nav.training.balls_slots o0 --data "$BALLS_DATA" --n 3 --seed 0 --research C001 \
  --cache outputs/gated/C001/cache_n3_s0 --parent-review outputs/gated/C001/cache_n3_s0/review.json \
  --verification-review outputs/gated/C003/recovery/review.json --out outputs/gated/C001/o0_n3_s0
```

The source-corrected O1 mask retains the earliest identity anchor. The RGB encoder is a deterministic SAVi-style
adaptation, not stochastic SAVi or a claimed exact reproduction. After reviewing each selected predictor's training
result, validate its real-input derivatives and 100-candidate/eight-step inference grouping (shown for O1; also run O0):

```sh
"$PY" -m go2nav.eval.slots_checkpoint --data "$BALLS_DATA" --research C004 \
  --checkpoint outputs/gated/C001/o1_n3_s0/model.pt --parent-review outputs/gated/C001/o1_n3_s0/review.json \
  --encoder-review outputs/gated/C004/slots_n3_s0/review.json --out outputs/gated/C004/o1_n3_s0
```

Use its passing C004 review as `--inference-review` for B001 learned planning and `--parent-review` for probes.
Never relax numerical tolerance after a failure without a new preregistration and literature review. Probe
extraction/fitting uses current train labels only; a fixed decoder scores current and four-step future positions,
with action-reversal/no-rollout controls:

```sh
"$PY" -m go2nav.eval.balls_probes --data "$BALLS_DATA" --n 3 --research C001 \
  --checkpoint outputs/gated/C001/o1_n3_s0/model.pt --parent-review outputs/gated/C004/o1_n3_s0/review.json \
  --out outputs/gated/probes/o1_n3_s0
```

Probe convention: lexicographically sorted white positions, no test-label slot assignment; fixed ridge 0.001 on
train-standardized features, 4,096 train windows, five test windows per episode, 2,000 episode bootstrap samples.
Report per-axis correlations and uncertainty; retain training seeds separately. Run P26 before spending remaining
allocation on n3 seeds1/2 and n1/n6. Extra L0/L1 seeds use `go2nav.training.balls ... --research C001` with the original
training recipe. No test-driven tuning to force L0 failure or O1's claimed margin.

## C: binding and final comparison

Before the final P25 report, measure binding with the selected, reviewed RGB encoder:

```sh
"$PY" -m go2nav.eval.balls_binding --data "$BALLS_DATA" --n 3 --research C002 \
  --checkpoint outputs/gated/C001/slots_n3_s0/model.pt \
  --parent-review outputs/gated/C004/slots_n3_s0/review.json --out outputs/gated/binding/slots_n3_s0
```

The primary measure keeps every geometric disc, including occluded balls; visible-area diagnostics are separate.
The decoder mask's best IoU must be at least 0.5. These are per-image measures, not proof of temporal identity.
Consult the literature after this result before interpreting it or changing any successor experiment.

For final comparisons, provide an index of `{n, seed, kind, planning, probes, binding}` entries with paths to
reviewed result files. A local index of the planned 18 model runs is `outputs/gated/p25_index.json`.
Legacy result reviews that lack a machine-readable training seed must include `run_identity: {n, seed, kind}`
verified against their training artifact and parent manifest. Do not infer seed identity from a filename alone.

```sh
"$PY" -m go2nav.eval.balls_comparison --index outputs/gated/p25_index.json \
  --out outputs/gated/comparison --research C002
```

The report retains negative reviewed measurements and lists missing/budget/technical evidence explicitly.
It reports paired episode intervals and the original numerical bars, separately for each training seed;
it is itself a result requiring literature review before scientific interpretation. The local probe's 4,096
training windows and ridge 0.001 differ from Zhu Appendix A.10's 50,000 and 1e-6; report that adaptation.

## D: frozen s0, offline Genesis

Preparation has completed for the original 600/100 feature-cache unit selection:

```sh
"$PY" -m go2nav.data.slot_tracks prepare --corpus "$CORPUS_V2" --ranges "$RANGES" \
  --features "$DINO_FEATURES" --out outputs/gated/D001/corpus --research D001
"$PY" -m go2nav.data.slot_tracks export --corpus outputs/gated/D001/corpus --features "$DINO_FEATURES" \
  --checkpoint outputs/camera_slots/s0/model.pt --mazes "$MAZES" --research D001
```

The exporter uses live scans when present and the existing backfilled scans otherwise. It emits matched
`objects/{train,val}` slot tracks and `engineered/{train,val}` tracker tracks. No new simulator collection or s0
training. Review track coverage/identity and numerical validity before training S1:

```sh
"$PY" -m go2nav.training.object_jepa train --corpus outputs/gated/D001/corpus \
  --out outputs/gated/D001/s1 --config configs/object/m5.yaml --set seed=0 --research D001 \
  --parent-review outputs/gated/D001/corpus/track_run/review.json
"$PY" -m go2nav.eval.slot_dynamics --corpus outputs/gated/D001/corpus \
  --m5 outputs/object_dynamics/m5/model.pt --s1 outputs/gated/D001/s1/model.pt --mazes "$MAZES" \
  --tracks-review outputs/gated/D001/corpus/track_run/review.json --s1-review outputs/gated/D001/s1/review.json \
  --out outputs/gated/D001/comparison --research D001
```

S1 selects minimum mean label-free validation error across 1.6/3.2/6.4s. The paired report retains misses and
occluded targets, one-to-one matching, detection precision/recall, switches/fragmentation, waiting, quantiles,
coverage and the common-case comparison. Historical `object_measures offline` remains available unchanged.
All new Genesis claims remain conditional on simulator pose/full layout and a previously used validation corpus.

The full historical default-path regression is independent of new model training. Run its documented CPU
reference path with GPUs hidden, two workers and one thread per worker:

```sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -m go2nav.eval.legacy_regression --corpus outputs/genesis/units_val --mazes "$MAZES" \
  --model outputs/object_dynamics/m5/model.pt --reference outputs/genesis/m5_genesis_offline.json \
  --out outputs/gated/D002/legacy --research D002
```

D002 checks all 400 historical units, every cohort count and every reported median. It must receive its own
post-result literature/source review. This reference regression does not establish the new paired P26 bars.
The first replay completed in 296.74 seconds: all 144 numeric fields exactly matched the published summary,
including counts and all waiting/kind breakdowns. Its positive review is `outputs/gated/D002/legacy/review.json`.

## Evidence bundle

```sh
"$PY" -m go2nav.eval.research_report --root outputs/gated --out outputs/gated/report_final --bundle \
  --source-commit 7dfc837 --source-commit 161fe8b --source-commit 98e6fcf
```

Use a new output directory for every snapshot; existing reports are preserved. E002 separates measured result
status from reviewed status, and leaves missing reviews explicitly awaiting literature review. The archive
contains current source/configs/docs, exact historical source blobs, small evidence artifacts, raw diagnostic
text and recovery scripts. Hashes and archive members use the same captured bytes; every member is verified
before the final archive filename is published. `verification.json` records its digest and payload count.
Large datasets and checkpoints remain local at their recorded paths and digests; this is not a standalone
dataset release. An interim bundle is not evidence that pending P25/P26 experiments are complete.

The latest reviewed interim capture through A017 is
`outputs/gated/report_checkpoint_progress_20261115/reproducibility.tar.gz` (2,103
payload files, 58 reviewed result rows). Its independent verification and substantive
review are under `outputs/gated/E002/progress_archive`. Keep it unchanged; final
training, planning and P26 results must enter a later bundle after their own reviews.
