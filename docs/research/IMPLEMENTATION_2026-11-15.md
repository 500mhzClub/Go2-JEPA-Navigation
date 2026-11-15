# Implementation evidence, 15 November 2026

Implementation branch: `feature/p25-p26-gated`, based on `7dfc837`. The six existing L0/L1 trainers continue on
their original pinned checkout. Camera-slot components were imported from `98e6fcf`; s0 weights remain frozen.
This document reports completed implementation checks and outstanding experiments, not completion of P25/P26.

## Completed measurements

| Area | Measurement | Interpretation and limit |
|---|---|---|
| A001 | Six checkpoint snapshots at step 40,000; train-only BN calibration changes no learned weights | Calibration lowers validation prediction error in all six. This supports a moment-mismatch explanation, not physical representation sufficiency. |
| A001/A002 | L0 passes float32 checks; L1 exceeds the declared batching/device tolerance. Math attention alone does not fix L1; all six pass float64/math forward/backward and grouping checks | Use one float64 inference policy for both arms. Final checkpoints still require A003. No nonfinite state was found; original jobs continue. |
| B001 | 1,000 new episodes per n, five groups, horizons 4/6/8. Recorded/direct/true-CEM success 100% in every cell | Action timing, reachable goals and optimizer controls pass. Learned latent-cost planning remains unmeasured. |
| C001 | CPU/GPU forward and parameter-gradient parity passes for the RGB slots and masked predictor at batch 2/three frames | Small-batch numerical validity only; full training convergence and object binding remain unmeasured. |
| D001 | Original feature-cache selection converted:600 train / 100 val; all 80 overlapping old validation units retain identical arrays | No new simulation collected. Live-scan and backfilled-scan sources are both supported. |
| D001 | Recorded engineered observations versus the live tracker produce identical M5 predictions at three checked windows | Adapter check on one unit; not a rerun of the full historical aggregate. |
| Verification | Broader unit suite:321 passed, 9 skipped, 6 deselected. Subsequent targeted suite:62 passed; final gate/cache/queue suite:27 passed | Skips and deselections are retained; GPU numerical checks run separately on the dedicated device. Counts overlap and must not be added. |

Validation prediction MSE, 64 fixed windows per snapshot:

| Model | Saved statistics | Train-only recalibration |
|---|---:|---:|
| L0,n1 |0.051068|0.003506|
| L0,n3 |0.010201|0.003700|
| L0,n6 |0.040614|0.012197|
| L1,n1 |0.289109|0.011024|
| L1,n3 |0.064576|0.017946|
| L1,n6 |0.174290|0.046077|

These are intermediate diagnostic samples, not final validation or generalization estimates. Calibration uses
4,096 training windows, encoder moments first and predictor moments second. Original checkpoints are unchanged.
Float32 L1 discrepancies remain reported; changing precision does not retroactively validate every training update.

The fixed random control succeeds in 1.2–2.7% of tasks, depending on n/horizon; hold succeeds in 2.6–8.3%. Collision-free
success is recorded separately. Balls pass through one another, so this test cannot establish interactive navigation.

## Implemented paths

- Research preregistration and result-review validation; source/config/input digests; append-only run-manifest history;
  dedicated-device checks and shared elapsed-device budget accounting. Reviewed cache/track indexes are checked
  against actual artifact bytes; selected predictors bind their frozen encoder/cache lineage.
- Safe L0/L1 resume validation before log/config changes, atomic checkpoints, restored RNG, finite-gradient rejection,
  inference-only calibration exports and checkpoint lineage.
- Fixed held-out task generation, paired CEM, optimizer controls, per-case output and episode uncertainty; independent
  goal encoding, Hungarian slot distance and train-only linear probes with fixed sorting and controls.
- RGB slot encoder, sequential carry, reconstruction trainer and frozen embedding cache; identical O0/O1 initialization
  and order, separate mask RNG and no position labels in either trainer. Paper inspection corrected the mask to retain
  the first-frame identity anchor before training.
- Matched Genesis adapter, range-assisted slot tracks with identity/velocity resets, recorded-observation model input,
  S1 training/checkpoint selection and paired offline coverage/occlusion/identity/error metrics.
- Local report/archive builder and correction of the unsupported value-impossibility claim (E001). V2 remains a measured negative.
- Final-checkpoint follow-up waits for the preserved jobs, checks completion and unchanged code, runs only A003,
  and stops for literature consultation before any dependent experiment.

## Remaining evidence

1. Finish the six preserved jobs; calibrate/review final checkpoints under A003.
2. Train/review n3 seed 0 slots, frozen cache, O0/O1; run learned planning and probes.
3. Export/review frozen-s0 tracks, train/review S1, run the paired P26 comparison and full legacy-summary regression.
4. Subject to the remaining 48-hour allowance, run the additional n3 seeds and n1/n6 object models; report all seeds.
5. Consult literature after each result before its successor. Report any negative/technical/budget outcome without
   claiming that an unfinished experiment failed scientifically.

Only about 0.0111 additional GPU device-hours had been consumed by the completed diagnostics. The shared ledger is
authoritative as later work runs. No new full training has run concurrently with the original jobs.

No claim is yet supported about O1's planning margins, position-probe thresholds, P26's detection/error bars, superiority
over dense DINO-WM, or deployment beyond conditional offline simulation. See the [runbook](RUNBOOK.md) for commands.
