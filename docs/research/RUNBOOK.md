# P25/P26 execution

Runtime update on 16 November: all six original experiments completed 303,000 steps
serially after the authorized reboot. All six actual A015 artifact audits and
A014 completion reviews pass. All six final A003 numerical evaluations and their
A016 operational reviews also pass, under `outputs/gated/A003_recovered/RUN`
and `outputs/gated/A016/runs/RUN`. No original trainer or final evaluator remains
active. Their historical source pins and output directories must be preserved;
do not relaunch completed cases. See the [final inference report](FINAL_INFERENCE_2026-11-16.md).

The n3 seed0 RGB encoder completed23752updates across eight epochs. Actual C008
terminal audit, C001 training and C005 operation reviews pass; the original81171a2
source binding is preserved. Selected-checkpoint C004 then passed all1104 sampled
comparisons, with separate C004 and C007 reviews. Do not relaunch these completed
cases. See the [encoder report](RGB_ENCODER_TRAINING_2026-11-16.md).

Shared frozen-feature extraction completed normally at06:20UTC through C005v2.
Full cache inspection and separate C001/C005 reviews pass; all70.4million
float32 values are finite, expected shapes/payloads and source/data/encoder
identities match. Its historical source remains6925f47. Evidence is under
`outputs/gated/C001/extract_n3_s0` and `outputs/gated/C005/v2/runs/extract_n3_s0`.
Do not rerun extraction. O0n3seed0 completed30epochs/11370updates through C005v2,
with historical runtime commit a75523b. Its actual C008/C001/C005 reviews pass.
Selected-weight C004 found four long-rollout grouping failures, despite all
1392 small-batch comparisons passing. The numerical review is technical_blocked;
normal C007 completion does not override it. C012 subsequently completed its
bounded precision diagnostic; all180 saved-array summaries and separate actual
inspection/science/operation reviews pass. Float32 regrouping failures appear
at steps7/8; common-history float32 and recursive float64 comparisons pass.
Preserve C004 and all checkpoints. C013 subsequently tested both complete
precision paths: encoder64/predictor64 passes2772comparisons; mixed precision
fails one long-rollout CPU/GPU comparison. Actual inspection/science/operation
reviews pass under `outputs/gated/C013/retry_001`. The initial validator shape
failure and two earlier software versions remain preserved. See the
[precision report](OBJECT_INFERENCE_PRECISION_2026-11-16.md).
C014 preregisters explicit full64 evaluator integration and future selected-model
checks. Its loader and three object evaluation boundaries are implemented in
the validation checkout; four CPU software groups pass with an exact-result
review under `outputs/gated/C014/software_loader`. This covers only the loader,
not the still-pending numerical validator/supervisor or C009v2 route. Original
C009 requires its original C004/source bindings and must not be bypassed.
O1n3seed0 completed30epochs/11370updates normally through C005v2 from a75523b.
Actual C008 audit and separate C001/C005 reviews pass. Selected epoch28 latent
validation MSE is0.0040267887; final resume state remains separately preserved.
See the [paired O1 report](OBJECT_PREDICTOR_O1_2026-11-16.md). No trainer or GPU
interval remains active;43.361additionalGPUhours remain. Preserve the gated
checkout for unchanged training/legacy helpers; new inference work continues
in the isolated validation checkout. O1 numerical and physical results remain
pending. Do not relaunch completed training or extraction.
C008 is a training-state audit and does not cover extraction. Keep each future
active trainer's runtime pinned through its actual C008 and completion reviews.

Object-model probes and paired learned planning follow, then P26 before extra
seeds/counts. All six original baseline planning cases and18000 saved-case
reconstructions are complete and reviewed. L1n6 is a measured negative with
successful operation. See the [combined baseline report](FINAL_BASELINE_PLANNING_2026-11-16.md). The six final
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

B002 has completed a bounded CPU feasibility check of the unchanged final L0n3
planner. All three horizons, repeated H8 actions and the100-candidate/eight-step
grouping checks pass, with maximum sampled cost discrepancy5.116e-13. The rough
full-model runtime estimate is27.39minutes; one episode does not establish an ETA
or planning success. Its six software groups and actual scientific/operational
results have separate literature reviews under `outputs/gated/B002`.

B003 now provides complete CPU execution of the six already-planned baseline
evaluations while the GPU encoder trains. Seven adapter checks match every
synthetic B001 case field except latency, independently verify scoring, retain
the original summaries and verify all six actual final parents. Use this path
for those six baseline cases; do not also launch their GPU command above:

```sh
python3 outputs/gated/B003/run_one.py --run l0_n3_s0
```

The fixed order is L0n3, L1n3, L0n1, L1n1, L0n6, L1n6, all seed0. Each needs a
fresh `B003/pre_RUN.json` consultation binding checkpoint, parent, protocol and
the existing test data. Run from the clean validation checkout, with unchanged
source and reviewed helpers. One eight-thread child has a two-hour wall cap
including15seconds cleanup and16GiB RSS cap; it creates no GPU interval. Each
model still evaluates all1000 episodes at horizons4/6/8 with unchanged CEM,
numerical tolerances, scoring and bootstrap. No training or calibration occurs.

Scientific artifacts go to `outputs/gated/planning/RUN`; operational artifacts
go to `outputs/gated/B003/operations/RUN`. Inspect all3000 raw cases and their
task pairing/scoring, then consult literature and review both actual results
before the next model. Partial cases and deadline stops remain incomplete;
they cannot fill comparison cells. CPU/GPU arithmetic and measured latencies
are not assumed equivalent. Keep the concurrent GPU trainer and pinned source
unchanged, and keep the CPU helper/source files fixed while its child is live.

After confirmed normal termination, use the independently reviewed saved-case
inspector before writing the scientific and operational reviews:

```sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" outputs/gated/B003/inspection/v2/verify_cases.py --run l0_n3_s0
```

It reconstructs every stored action sequence and all summaries, checks pairing
with the reviewed controls, and verifies saved numerical arrays. Its five
software groups include the actual legacy control schema and deliberate corruptions.
The initial inspector's missing-field failure is preserved under
`inspection/attempt_001_l0_n3_s0`. V2 and both actual n3 inspections are reviewed;
later actual outputs still need separate review. It never reruns a model or
automatically approves a result. L0n3 and L1n3 planning are complete; do not relaunch
them. Their endpoint rates are97.8/98.2/97.7% and94.7/94.2/94.7% respectively.
See the [completed pair](BASELINE_PLANNING_N3_2026-11-16.md). Next is L0n1.

## C: three-ball core, then extra seeds/counts

After each C001 slot/predictor training run exits normally, keep its runtime
checkout pinned until its terminal artifacts have been audited and reviewed.
C008 supplies a CPU-only audit from the validation checkout:

```sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" outputs/gated/C008/audit.py --run slots_n3_s0
```

Revisit the declared sources and record the per-case consultation before use.
The audit requires actual successful C005 terminal evidence and closed owned
intervals. It checks full update counts, model/optimizer/scheduler restoration,
finite epoch logs, validation selection and source/data identity without forward
passes or updates. Twelve focused software groups are reviewed; initial source
inventory routing evidence is preserved under `C008/software_attempt_001`.
Actual C008, C001 scientific and C005 operational results need separate substantive
reviews before C007/C004 or any checkout update. An earlier selected model's epoch
tag remains source-bound metadata, not independently retained historical weights.
Actual C004 numerical checks and later binding/planning are still required.

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
python3 outputs/gated/C007/run_check.py --run slots_n3_s0 --python "$PY" \
  --preconsultation outputs/gated/C007/pre_slots_n3_s0.json
```

C007 runs unchanged C004 under external CPU supervision. Its thirteen focused
CPU check groups passed and received exact-result literature review. This does
not establish actual GPU timeout recovery or selected-model numerical validity.
Each invocation has the smaller of the existing half-hour cap, remaining 8-hour
validation reserve and remaining 48-hour global allowance, including 45 seconds
for identity-checked cleanup. Nested wrapper/child intervals count once. If
termination cannot be confirmed, accounting stays open.

Use a clean updated checkout only after the live trainer's actual C001 and C005
reviews pass. The wrapper requires its reviewed exact sources, C004 protocol,
selected checkpoint, actual training reviews and a fresh substantive consultation
binding all six existing train/validation data files. It refuses active training,
GPU processes, open ledger intervals and reused output directories. Scientific
results remain under `outputs/gated/C004/RUN`; operational evidence is under
`outputs/gated/C007/runs/RUN`. Neither reviews nor successors are automatic.

Inspect and literature-review both actual C004 and C007 results before extracting
the frozen cache:

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
python3 outputs/gated/C007/run_check.py --run o1_n3_s0 --python "$PY" \
  --preconsultation outputs/gated/C007/pre_o1_n3_s0.json
```

The predictor invocation additionally requires its actual reviewed C004 encoder.
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
allocation on n3 seeds1/2 and n1/n6. Extra L0/L1 seeds retain
`go2nav.training.balls ... --research C001` and the original training recipe,
executed through C010 below. No test-driven tuning to force L0 failure or O1's claimed margin.

### Additional baseline seeds after core and P26

C010 supervises exactly L0/L1 n3 seeds1/2, with fresh outputs under `C001/RUN`:

```sh
python3 outputs/gated/C010/run_train.py --run l0_n3_s1 --python "$PY" \
  --preconsultation outputs/gated/C010/pre_l0_n3_s1.json
```

Its fifteen focused CPU groups are reviewed. All ten inspected computation
methods match the original baseline source, and defaults match the original
recipe. Six train/validation data files must match reviewed A014 provenance.
The full prerequisite fixture covers P26, core results, earlier training audits,
seed ordering and incomplete/tampered inputs. Actual current admission refuses
the missing P26 export review before GPU access; no extra training has started.

Reviewed n3seed0 planning/probes/binding and normal completed P26 export, S1 and
comparison are required. A measured-negative comparison remains valid evidence;
technical or budget-incomplete results do not establish completion. Seed2 also
requires the full seed1 four-model planning/probe/binding comparison. Each earlier
baseline needs separate C001 training and C010 operation reviews plus an actual
completion audit bound by `final_audit_review` and its digest. That audit must bind
both the303000-step full checkpoint and final model.

C010 uses unchanged global48/B/C28 allowances, with45seconds reserved for cleanup;
outer rows charge C001 with a C010 supervisor marker. All generated files and
input hashes are retained. It refuses existing output directories and provides
no automatic resume, retry or successor. Actual operations are under
`outputs/gated/C010/runs/RUN` and require their own review.

The original A003, B003 and C006 records cover only the six seed0 baselines.
A019 now covers calibrated inference for the four additional n3 baselines, with
reviewed software. C011 now covers the unchanged additional-seed CPU probes.
B004 covers the unchanged additional-seed CPU planning with reviewed software.
Do not silently apply the original
records to seeds1/2. C010 software checks alone establish neither an actual
completed training state nor numerical/scientific validity.

After a normal C010 completion, use A018 before changing its runtime checkout:

```sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --kill-after=5s 175s "$PY" outputs/gated/A018/audit.py --run l0_n3_s1 \
  --preconsultation outputs/gated/A018/pre_l0_n3_s1.json
```

Create the consultation only after inspecting the actual terminal evidence and
revisiting the relevant sources. It binds the name, A018 record, software review,
C001/C010 raw results, full checkpoint and final model digests. The audit verifies
the original runtime source inventory there and executable source in the audit
checkout, all admitted input bytes, exact process termination and closed owned
ledger rows. Snapshot files and audit results go to `A018/runs/RUN`.

Nine focused software groups pass: four default L0/L1 seed fixtures,29 deliberate
state/config/optimizer/RNG/log corruptions,14 invalid terminal/accounting cases,
source routing and the expected missing-actual-run refusal. The complete-state
check requires127/131 populated optimizer states, all303000 counters, exact
model/head and optimizer/scheduler restoration, isolated CPU RNG restoration,
606training and16validation rows. No forward, backward or update occurs.
GPU RNG is checked only as bytes; no actual extra-seed checkpoint has been audited.

Inspect and separately literature-review actual A018, C001 training and C010
operations. Bind `final_audit_review` and its SHA256 in the C001 review; retain
the same reference in the operational review. A successful artifact audit does
not replace the still-required numerical inference gate or scientific evaluation.

After those actual reviews, run the unchanged numerical evaluator through A019:

```sh
python3 outputs/gated/A019/run_check.py --run l0_n3_s1 --python "$PY" \
  --preconsultation outputs/gated/A019/pre_l0_n3_s1.json
```

The fresh consultation binds `name`, `research_sha256`,
`scientific_research_sha256` (both A019), `software_review_sha256` and the complete
`inputs_sha256` mapping returned by A019's `stages.inputs`, plus sources,
consultation time and consequences. It requires the original A014-bound n3
train/validation bytes and actual C001/C010/A018 evidence. Science goes to
`A019/inference/RUN`, operations to `A019/runs/RUN`. All paths are under
`outputs/gated`. The four names are L0/L1 n3 seeds1/2; seed0 reruns are refused.

A019 reuses the original numerical module without changing its computation or
sample policy:4096 train calibration windows,64 train/validation diagnostics,
float64/math and the original tolerances. Its non-A003 branch uses the original
0.10 allocator fraction; actual memory fit remains unmeasured. The external
supervisor enforces1800seconds including45cleanup within the existing shared
A4/global48 allowances. Both nested intervals count once. No extra allowance,
automatic retry, successor or scientific review is provided.

After actual normal completion, inspect the outputs and revisit the sources,
then use the CPU artifact verifier:

```sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --kill-after=5s 175s "$PY" outputs/gated/A019/verify_result.py --run l0_n3_s1 \
  --preconsultation outputs/gated/A019/pre_artifact_l0_n3_s1.json
```

That consultation binds `name`, `research_sha256`, `scientific_result_sha256`,
`operational_result_sha256`, `software_review_sha256`, sources, time and
consequences. Verification goes to `A019/verification/RUN`. It checks all270
reported comparisons with127 gradient tensors per phase, source/seed identity,
finite inference-only float64 export, exact learned model/head weights and only
permitted BN-buffer changes, including reported calibration deltas. It does not
rerun GPU arithmetic or re-estimate the calibration moments. Auxiliary-head
gradients are outside the latent-only parity loss.

Thirteen CPU software groups, including four default artifact fixtures and18
corruptions, are reviewed. No actual additional-seed numerical result exists.
Review actual numerical, verification and operational results separately before
probes/planning; bind the verification digest and verified run identity to the
numerical review because the unchanged calibrated export omits a top-level seed.

### Additional-seed CPU probes

C011 extends the original C006 computation to L0/L1 n3 seeds1/2 in that order
within each seed. Eighteen focused software groups are reviewed; the initial
test-variable naming failure and its negative review remain under
`C011/software_attempt_001`. No actual additional-seed probe has run.

After actual C001/C010/A018 and all three A019 reviews, use:

```sh
python3 outputs/gated/C011/run_one.py --run l0_n3_s1 --python "$PY"
```

Before launch, create `C011/pre_RUN.json` with name, C011 record digest,
software-review digest, complete input hashes, original train/test data hashes,
consultation time, sources and consequences. The complete files are returned by
`parents.numerical_parent`, `parents.original_probe_data` and
`parents.previous_reviews`; all three sets are bound. Original data bytes remain
those used by C006. A fresh consultation cannot substitute a new dataset.

The A019 numerical review must contain exact `run_identity` with kind/n/seed,
`verification_review` pointing to `A019/verification/RUN/review.json`, and
`verification_review_sha256`. Its actual verification and operation result hashes
must agree, including all270 reported comparisons and preserved model/head weights.
Synthetic software fixtures cannot satisfy these actual-run requirements.

Use one eight-thread CPU child,16GiB maximum sampled RSS,900seconds including15
for cleanup and3600seconds across four cases. The existing B003 lock serializes
these probes with CPU planning. There is no GPU interval or automatic review,
retry or successor. Science goes to `probes/RUN`; separate operations go to
`C011/supervision/RUN`. Independently inspect actual saved sample indices, truth,
predictions, point statistics and ridge residual before substantive literature
review of science and operations. Preserve measured negatives and incomplete cases.

### Additional-seed CPU planning

B004 covers the same four additional baseline names after actual C011 probe
reviews. Its thirteen software groups and independent-inspector source are
reviewed. The initial missing inspector output-parent finding, passing fixture
result and negative review are retained under `B004/software_attempt_001`.

```sh
python3 outputs/gated/B004/run_one.py --run l0_n3_s1 --python "$PY"
```

Create `B004/pre_RUN.json` after consulting on the actual parents. Bind `name`,
`research_sha256`, `b001_record_sha256`, `checkpoint_sha256`,
`software_review_sha256`, complete `inputs_sha256` from `parents.inputs`,
its original `data_sha256`, consultation time, sources and consequences.
The actual same-seed C011 probe can be passed or a valid measured negative,
but its operation must have completed normally. Seed0 paths are excluded.

The fixed3000-case B003 computation,100/10/20 CEM, original task seeds,
float64/math and candidate checks are retained. One eight-thread CPU child
has7200seconds including15cleanup,16GiB sampled RSS and zero GPU intervals;
the four cases total at most28800seconds. The B003 lock serializes CPU work.
Science goes to `planning/RUN`; operations go to `B004/operations/RUN`.

After actual normal completion, use the independent inspector:

```sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --kill-after=5s 115s "$PY" outputs/gated/B004/verify_cases.py --run l0_n3_s1
```

It reconstructs all saved actions against original truth/control tasks and
reproduces scores and bootstrap summaries without model inference. Review
inspection, science and operations separately after revisiting the literature.
Bind `inspection_review` and `inspection_review_sha256` in the science review;
successor admission verifies these plus actual operation and artifact digests.
No additional-seed planning result currently exists.

### External deadlines for selected-object evaluations

Use the reviewed C009 supervisor for actual object planning, probes and binding
instead of launching the illustrative raw commands above/below directly:

```sh
python3 outputs/gated/C009/run_eval.py --stage planning --run o1_n3_s0 --python "$PY" \
  --preconsultation outputs/gated/C009/pre_planning_o1_n3_s0.json
```

Stages are `planning`, `probes`, and `binding`; binding names identify the encoder
(for example `slots_n3_s0`). The 21 routes retain the approved model/count/seed
scope and exact existing scientific modules. Fresh consultations bind complete
inputs, actual C004/C007 selected-model reviews, cache/encoder parents and
original C006 dataset bytes; planning additionally requires reviewed B001 controls.

Ten focused CPU groups pass, including normal/deadline children and rejection of
changed data despite a fresh consultation. The initial data-binding gap and its
review remain under `C009/software_attempt_001`. This software evidence does not
demonstrate an actual GPU evaluation or recovery from an uninterruptible stall.

C009 enforces the smaller remaining global48, B/C28 and scientific allowance,
with45seconds reserved for cleanup. Existing B0012-hour and C0021-hour cumulative
caps remain. Outer ledger rows carry the underlying scientific ID and an explicit
C009 supervisor marker, so existing supervisors count the time correctly and
nested intervals count once. Close only owned intervals after confirmed exit.
Operations live under `C009/runs/STAGE/RUN`; scientific outputs retain their
original paths. Inspect raw cases and separately literature-review science and
operations before any successor. There is no automatic review or retry.

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

The scientific commands below define the unchanged stages. Execute future GPU
stages through the reviewed D003 external supervisor, after reviewed n3 seed0
planning/probes for L0/L1/O0/O1, slot binding and the existing adapter/legacy checks.
For each stage, consult the literature and bind its actual inputs and prior reviews:

```sh
python3 outputs/gated/D003/run_stage.py --stage export --python "$PY" \
  --preconsultation outputs/gated/D003/pre_export.json
```

Use `--stage s1` and then `--stage compare` only after their separate scientific
and operational predecessor reviews. D003 enforces the existing cumulative eight
GPU hours and global allowance, with 45 seconds reserved for owned-child cleanup.
Its eleven CPU check groups pass; the initial null-input fixture failure and
source snapshot remain under `outputs/gated/D003/software_attempt_001`. The
software review is not an actual P26 result or proof of GPU recovery. Fresh
stage consultations and all actual outcomes still require substantive review.

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

## Latest execution state — 16 November 2026

All six original baseline planning cases and18,000 saved-case reconstructions are
complete and reviewed. L1n6 is a measured scientific negative, with successful
operation. O0n3seed0 completed30epochs/11370updates and passed C008/C001/C005
reviews. Its actual C004 check is technical_blocked: four100candidate8step
regrouping comparisons exceed fixed tolerance; all1392 short-window comparisons
and two exact rollout repeats pass. Preserve the failure and a75523b source.
No training or evaluation is currently active. Before dependent object work,
preregister a bounded precision/backend/accumulation diagnostic; do not overwrite
C004 evidence, silently relax tolerances or infer a scientific O0 negative.
The full C002 comparison, O1/object science, offline P26, further seeds/counts and
actual final archive remain pending. See the all-six and O0 reports.

## Evidence bundle

```sh
"$PY" outputs/gated/E003/report.py --root outputs/gated --out outputs/gated/report_final --bundle \
  --source-commit 7dfc837 --source-commit 161fe8b --source-commit 98e6fcf
```

Use E003 for the final capture after fresh literature consultation and manual verification of its record,
helper sources and exact software review. Its reviewed capacity extension admits 512 MiB total, retaining
the 64 MiB individual-file limit; six software groups include all nine inherited cases, a verified 280 MiB
synthetic archive and both oversize refusals. This does not establish actual final evidence coverage.
Include additional executed source revisions as required. Preserve the original E002 helper and all
historical archives. Any actual overflow must stop capture without silent exclusions or limit changes.

Use a new output directory for every snapshot; existing reports are preserved. The inherited E002 logic separates measured result
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
