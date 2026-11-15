# Research workflow

The user requires literature consultation as a gate before every new implementation,
experiment, or interpretation of a positive, negative, or inconclusive result.
This applies to future work in this repository, not only the current P25/P26 stage.

Before implementation or experimentation, read relevant primary methods/appendices
or official technical documentation/source. Abstracts alone are insufficient. Record
the question, hypothesis, sources and sections read, differences from this project,
contrary evidence, frozen protocol, success/stop criteria, resource limit and checks
under `docs/research/records`. Reuse a record only within its declared protocol,
seeds and verification suite. Changed protocols need a new record/version before use.

After each result, revisit the literature before interpreting it or starting dependent
work. Write a review bound to the exact result digest, including measurements,
uncertainty, explanations and the next discriminating step. Follow
`docs/research/README.md` and `docs/research/RUNBOOK.md` for the schema and commands.
Do not create a passing review merely to unblock execution. Hash/schema checks do
not themselves constitute literature consultation or scientific review.

Keep implementation validity, representation, prediction, planning and generalization
claims separate. Report missing coverage, unsuccessful runs and exhausted budgets;
unfinished experiments are not scientific negatives. Preserve previous evidence.

For the approved P25/P26 scope, preserve the six existing trainers and their pinned
checkout. New full GPU work is sequential after those jobs finish; declared small
diagnostics may coexist below the memory ceiling. Use the shared 48-hour additional
device-time ledger and no training CPU fallback. No new Genesis missions, collection,
planner tuning, value models, hierarchy or stochastic variants are authorized by this
stage. Dense-feature superiority is untested.

Use an isolated checkout for implementation while jobs run. A queued follow-up binds
to source/config hashes: inspect its state before editing that checkout. The A003
queue ends awaiting literature review and must not automatically advance itself.
