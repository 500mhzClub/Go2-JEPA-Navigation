# Remaining approved work after the GPU-time override

This is an execution inventory as of16 November2026, not a new scientific
analysis. All global, stage and per-run GPU-hour limits are removed under
[U001](records/U001.json). Progressing jobs continue; elapsed accounting,
serial GPU use and substantive literature/result reviews remain required.

## Scientific sequence

| Stage | Completed evidence | Remaining work |
|---|---|---|
| Original baselines | Six seed0 L0/L1 trainings, final-state audits, numerical admission, probes and full planning;18,000 saved planning cases independently inspected | Reuse these exact results |
| First object core, n3seed0 | Encoder/cache, O0/O1 training and numerical admission; binding and both probes reviewed | Full O0 planning is active; inspect with U002 and review, then run O1 through U001 and inspect/review |
| Offline Genesis P26 | Legacy400-unit regression and600/100-unit prepared inputs; U003 launcher qualified | Frozen-s0 export,20,000-step S1 and matched five-arm comparison, each with its own actual result review |
| Three-ball additional seeds | C010/A018/A019/C011/B004 software prepared; no actual extra-seed runs | Seed1 L0/L1 plus encoder/cache/O0/O1 and all measures; complete seed1 before seed2 |
| Other object counts | Seed0 L0/L1 results complete at n1/n6 | Own encoder/cache and O1 at n1 and n6, with numerical admission, binding, probes and planning |
| Final comparison | C002 comparator implemented; historical partial baseline-probe audit preserved | All18 rows:10 baselines and8 object models, with five distinct encoder-binding results |
| Final report/archive | E003 integrity/capacity software and U004 accounting metadata correction reviewed | Requirement-by-requirement audit, high-level development narrative, final reviewed comparisons and verified archive |

The18 comparison rows are n3 seeds0/1/2 × L0/L1/O0/O1, plus n1/n6
seed0 × L0/L1/O1. `outputs/gated/U004/planned_comparison_index.json` names
the intended inputs. Only first-pair object planning uses `planning_full`;
later paths must be reconciled to their actual launch manifests before use.
File presence does not establish review or completion. The current metadata
snapshot is `outputs/gated/U004/requirements_observation.json`.

## Remaining execution integration

U001 already runs the first O0/O1 pair without GPU caps, and U002 checks its
complete outputs. U003 preserves original P26 commands and paths while applying
the same accounting policy. Their reviewed helpers and the active runtime
checkout must remain unchanged.

Before later GPU stages, adapt their operational admission in the isolated
followthrough checkout. Keep scientific computation and existing reviews intact:

- C010's U005 adapter is now qualified: old48/28-hour caps are removed and the
  first-pair core uses full planning plus U002/U001 evidence. Seven software
  groups pass; actual P26 prerequisites remain absent. U006 now qualifies A018
  metadata/source admission with unchanged original state and terminal checks;
  six integration groups pass. All actual extra-seed artifacts remain absent.
- C005v2's U009 adapter now removes the old48/28-hour caps for all14 remaining
  encoder/cache/predictor stages. Seven CPU integration groups pass with unchanged
  C001 recipes and approved count/seed order. C008 still needs explicit U009
  operational metadata and historical source reconciliation before actual use.
- A019's U007 adapter now removes old48/4-hour and per-call GPU deadlines;
  eight integration groups pass. Numerical computation and artifact checks
  remain unchanged. Actual additional-seed numerical results remain absent.
- C009v2 still enforces old scientific allocations for later object measures.
  Extend the accounting-only execution route to those already-approved cases,
  retaining C014 admission and independent inspections.
- C014's later selected-checkpoint numerical checks also retain old outer,
  reserve/global and internal per-call deadlines. Remove those under U001
  while preserving every numerical comparison and tolerance.
- CPU-only C011/B004 have separate wall/RSS bounds. They are not GPU-hour caps;
  do not silently change their scientific or resource protocols. Their source
  and record inventories match the preserved gated checkout at a75523b; use
  that checkout after revalidating complete actual parent/data admission.

These are pending operational corrections, not reasons to interrupt current O0
or to tune the planner. Consult the relevant official documentation before each
implementation and review its actual checks before use. Missing scientific
predecessors remain missing; valid measured negatives satisfy completion gates
where the existing protocol permits them.

## Final report and reproducibility requirements

The final narrative must connect the original Go2 motivation, baseline recovery,
completed baseline/object evidence, P26 and replications. Distinguish measured
results from hypotheses and unfinished claims; retain the original declared
bars, local deviations from papers and separate training-seed uncertainty.
No closed-loop robot transfer is established by this offline work.

Run C002 only with exact reviewed inputs and verified model/count/seed identities.
Independently inspect paired-case coverage and the requirement table before
interpreting it. Do not overwrite historical partial or failed artifacts.

Use the reviewed U004 reporter with actual executed source commits, ignored
runtime helpers, consultations, inspections, reviews and failures. Reconcile
the selected paths against every executed operation, including retired runs.
The archive remains a local reproducibility bundle: large datasets/checkpoints
need explicit external manifests and are not included as a standalone release.

The metadata-only observation at10:03UTC selected1,339 run files totaling
471,179,895bytes (about449.35MiB), with no selected file exceeding64MiB.
This excludes current and historical source snapshots and future artifacts.
The existing512MiB total limit may therefore be insufficient. Recheck actual
complete capacity before capture; any necessary versioned capacity change must
preserve all selected evidence and receive its own documented review. The user
removed GPU time limits, not archive integrity or evidence coverage rules.

This reporting approach follows the claims/limitations/reproducibility items in
the [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist).
U004 preserves the same-byte archive operations described by
[Python tarfile](https://docs.python.org/3.12/library/tarfile.html) and historical
blob capture described by [Git cat-file](https://git-scm.com/docs/git-cat-file).
These sources were consulted before the correction and revisited for its
software review; they do not establish scientific success.
