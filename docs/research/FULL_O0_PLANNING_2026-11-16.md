# Full O0 planning result

O0 completed all 3,000 preregistered planning cases: 1,000 three-ball episodes at
each of horizons 4, 6 and 8. Its endpoint success is 1.5%, 1.1% and 1.2% respectively.
The full result is a reviewed scientific negative for this slot model and the
fixed latent-distance planner. Independent scoring and execution reviews pass.

| Horizon | Success | Episode-bootstrap 95% interval | Median endpoint error | Collision-free success |
|---|---:|---:|---:|---:|
| 4 | 15/1,000 (1.5%) | 0.8–2.3% | 7.8291 pixels | 1.3% |
| 6 | 11/1,000 (1.1%) | 0.5–1.8% | 9.5546 pixels | 1.1% |
| 8 | 12/1,000 (1.2%) | 0.6–1.9% | 11.1945 pixels | 0.9% |

The separately verified recorded, direct and true-model CEM controls each reach
100% at every horizon. The learned-only result file's `controls_passed: false`
comes from its summary lacking control-arm rows; it is not a failed control run.
The actual control evidence is separately bound and verified by U002.

U002 independently reconstructed all saved task/action records, endpoint scoring,
success classifications and summaries. Its largest endpoint discrepancy was
0.00001036 pixels; the nearest success boundary was 0.00272238 pixels away.
This verifies the stored decisions and scores, not the unsaved neural candidates
inside CEM. The existing full64 C014 numerical evidence remains bound to the run.

The job completed normally in 19,738.97 seconds (5.483 hours), with no stop signals
or runtime errors. Both elapsed-accounting intervals are closed. Its source remains
pinned at `4a68a72c6d47d60dfffff92ac7278bd15126a530` in the performance checkout.
Earlier partial/interim results remain preserved.

The interpretation is limited to one n3 seed and the local task. Poor slot binding
and physical probes are compatible with the planning failure, but these results do
not isolate encoder, predictor and cost contributions. The three horizons share
episodes; their rows are not independent samples. O1 planning remains pending,
and this O0 result does not establish whether masking helps.

This run did not use a physical forecasting head at inference. It supplies no new
evidence against learned physical heads, nor does the independent-motion balls
environment test obstacle reactions to robot actions. The distinction is explicit
in [Zhu et al., Appendix A.3](https://arxiv.org/html/2609.04264v3), which evaluates
decoded physical predictions as an additional planning signal.
[Causal-JEPA, Appendices B and G](https://arxiv.org/html/2602.11389v2) distinguishes
observability masking from interventions on physical dynamics and describes
matched-slot latent planning; our local adaptation is not a full reproduction.

The next work follows [F004's physical forecasting comparison](MATCHED_FORECASTING_2026-11-16.md).
O1 planning and the historical matrix remain incomplete and do not advance
automatically. The original weak-L0 paper condition also remains unreproduced.

| Artifact | SHA-256 |
|---|---|
| Scientific result | `f5cac869f9654de60bcad113549a1f52de7b480dd9e261683caa614e3fe96680` |
| Cases | `82e561b5788115bacb88bb5e2f4d3b1075c28e4c01de390e40043a76ccd20c3e` |
| Science review | `43f33c79ce27da30b97d59d4247e0682e2467d174f11e7c11ce8ee3892f5203c` |
| Independent inspection review | `dc95e6bf0ee07b1e914ffd759417f3746351cb2be0e0b5aab42b7f540b1f2e68` |
| Operations review | `698a97a2357c0397c3f720497ae3bcaa49c68450c8e23231b222bae6c5213cd2` |

Science/inspection artifacts are in `outputs/gated/planning_full/o0_n3_s0`;
operational artifacts are in `outputs/gated/U001/operations/o0_n3_s0`.
