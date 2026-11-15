# P25/P26 execution

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

After each original run finishes, repeat for all six L0/L1 runs under the same policy:

```sh
"$PY" -m go2nav.eval.balls_inference --checkpoint outputs/balls/runs/l0_n3_s0/model.pt \
  --data "$BALLS_DATA" --out outputs/gated/A003/l0_n3_s0 --research A003
```

A003 uses train-only calibration and float64/math inference, selected by A002 at unchanged tolerances. The source
checkpoint is copied through an open inode, learned weights remain unchanged, and `calibrated.pt` is explicitly
inference-only. Review the final numerical result before planning. No final-checkpoint validity is inferred from
the successful step-40,000 checks.

The workspace follow-up uses `go2nav.eval.await_balls --runs outputs/balls/runs --data "$BALLS_DATA"`
with `--out outputs/gated/A003` and `--pids` listing all six original trainer PIDs. It records their process identities,
waits up to 24 hours without signalling them, verifies final step counts and unchanged source/config hashes, and
runs only the A003 suite. `outputs/gated/A003/state.json` records progress. It exits awaiting literature review;
it cannot approve its own results or launch a dependent experiment. Keep this checkout unchanged while it waits.

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
  --checkpoint outputs/gated/A003/l0_n3_s0/calibrated.pt \
  --inference-review outputs/gated/A003/l0_n3_s0/review.json \
  --controls outputs/gated/B001/n3/result.json --out outputs/gated/planning/l0_n3_s0
```

The same CLI loads reviewed O0/O1 checkpoints. It never supplies intermediate future frames or truth positions to
the learned planner. CEM uses final elite mean; goal slots are independently encoded and Hungarian-matched.

## C: three-ball core, then extra seeds/counts

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
"$PY" -m go2nav.eval.research_report --root outputs/gated --out outputs/gated/report --bundle
```

The archive contains source/configs/research records and small evidence artifacts with a digest manifest. Large
datasets and checkpoints remain local at the paths and digests recorded in run manifests; they are not published.
