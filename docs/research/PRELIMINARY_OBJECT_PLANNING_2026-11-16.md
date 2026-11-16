# Preliminary O0 planning assessment — 16 November 2026

The current object-slot baseline O0 is performing substantially worse than the
existing global-latent baselines on the checked prefix. This is an interim result,
not completion of the full experiment or a verdict on O1.

The frozen snapshot at10:33:50UTC contains497saved cases. The comparison uses
all495cases from the first165fully completed episodes, covering each episode at
horizons4,6,8. Two trailing rows from the next episode are excluded. Every arm is
matched on the original task, start/goal and random seed.

| Model/control | H4 successes | H6 successes | H8 successes | Total |
|---|---:|---:|---:|---:|
| O0: object-slot predictor | 1/165 | 0/165 | 1/165 | 2/495 |
| L0: plain global-latent baseline | 162/165 | 161/165 | 163/165 | 486/495 |
| L1: auxiliary-supervised baseline | 154/165 | 157/165 | 158/165 | 469/495 |
| True-dynamics CEM | 165/165 | 165/165 | 165/165 | 495/495 |
| Hold position | 12/165 | 9/165 | 3/165 | 24/495 |
| Random actions | 2/165 | 4/165 | 3/165 | 9/495 |

O0 median endpoint error increases from7.265pixels atH4 to9.869atH6 and11.761
atH8; success requires at most1pixel. The current O0 pipeline has poor utility
on these tasks, even though the selected checkpoint passed numerical admission.
The strong controls demonstrate solvability. They do not identify whether the
encoder, learned dynamics, latent goal cost or their combination causes failure.

The already-reviewed encoder binding result is0/20,000 appearances aboveIoU0.5.
O1's masked-history training reduces decoded four-step red/white RMSE23.01%/8.93%
relative to O0, but neither beats holding the current estimate. These are reasons
for concern about representation and prediction. O1 planning is still unmeasured.

The literature was consulted before this readout and revisited after it.
[Causal-JEPA Appendix G](https://arxiv.org/html/2602.11389v2) aligns slots for a
latent planning cost; alignment alone does not establish physical object binding.
[VideoSAUR Appendix B.4](https://arxiv.org/html/2306.04829v2) documents non-object
groupings under reconstruction in a different decoder/feature setting. It makes
poor grouping a plausible hypothesis, not an established explanation here.
The [physical-probing appendix](https://arxiv.org/html/2609.04264v3) distinguishes
linear accessibility from more entangled information; failed linear readout does
not prove the latter absent.

Only16.5% of the intended cases and one training seed are included, all within
the first evaluation group. No sequential significance or population interval is
claimed. This readout cannot replace the full run's terminal/provenance inspection
or admit O1. It establishes no closed-loop Go2 improvement or collision avoidance.
P26 uses a separately frozen camera encoder and remains a distinct test.

The unchanged qualified inspectors independently reconstruct all1485learned-arm
case scores, actions and task identities. Original control reviews remain valid;
2475control scores replay exactly in original float32 arithmetic, with all flags
also matching float64. Two interim-reader failures are preserved: a historical
control-schema assumption and an extra float64 near-zero comparison. The latter
has maximum discrepancy0.00002158pixels and no success/collision changes. No
learned-case tolerance, checkpoint, planner or scientific output was altered.

Evidence: `outputs/gated/U008/interim_003/result.json` and its hash-bound review;
immutable snapshots, source hashes, failed attempts and control diagnostic remain
under `outputs/gated/U008`. The U008 protocol is in `records/U008.json`.

Next: finish full O0 unchanged, inspect and review it, then run matched O1.
Continue P26 and the approved replications. If the paired result remains poor,
representation quality and training/inference history consistency are stronger
candidates for a future discriminating experiment than assuming extra predictor
training will solve it. Any changed experiment needs its own literature gate.
