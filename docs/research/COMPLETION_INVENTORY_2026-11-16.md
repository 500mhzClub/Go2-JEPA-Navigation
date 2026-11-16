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
| Final report/archive | E004 capacity, P26 evidence selection and inherited integrity checks reviewed | Requirement-by-requirement audit, high-level development narrative, final reviewed comparisons and verified archive |

The18 comparison rows are n3 seeds0/1/2 × L0/L1/O0/O1, plus n1/n6
seed0 × L0/L1/O1. `outputs/gated/p25_final_index.json` is the prepared runnable index; its
byte-identical U004 inventory remains preserved. Only first-pair object planning
uses `planning_full`; later paths must be reconciled to actual launch manifests.
The reviewed C002 input preparation accepts15 available components and six
complete baseline rows;12planning,10probe and4binding components remain missing.
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
  C001 recipes and approved count/seed order. U010 now qualifies C008's U009
  metadata and exact historical source reconciliation: seven groups pass with
  unchanged full-state checks. No actual later checkpoint audit exists yet.
- A019's U007 adapter now removes old48/4-hour and per-call GPU deadlines;
  eight integration groups pass. Numerical computation and artifact checks
  remain unchanged. Actual additional-seed numerical results remain absent.
- C009v2's U012 adapter now removes old outer/inner allocations for the16 later
  object measures. Eight CPU groups and44 metadata corruption refusals pass.
  Original C014 admission, commands and all three inspection calculations remain;
  actual later outcomes and per-case admission checks are still pending.
- C014's U011 adapter now removes outer, reserve/global and internal numerical
  deadlines. Eight CPU groups pass, including 46 metadata corruption refusals;
  original numerical loop, tolerances and inspection calculations remain unchanged.
  No actual later selected checkpoint has completed this suite yet.
- CPU-only C011/B004 have separate wall/RSS bounds. They are not GPU-hour caps;
  do not silently change their scientific or resource protocols. Their source
  and record inventories match the preserved gated source at a75523b. U013
  fixes a B004 launcher path that otherwise refuses its required probe output.
  Six software groups pass, including all four real-child launcher fixtures.
  Use the isolated `Go2-JEPA-Navigation-p25-cpu` checkout with unchanged C011
  and U013 planning supervision; revalidate actual parent/data admission.

Current O0 continues unchanged; planner tuning remains outside the approved scope.
Consult the relevant official documentation before each
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

Use the reviewed E004 reporter with actual executed source commits, ignored
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

A subsequent metadata-only check selected1,489 run files totaling
475,038,721bytes (about453.03MiB); none exceeded64MiB. Source snapshots and
future evidence remain outside this count. U013 corrects execution routing,
not the separate pending archive-capacity requirement.

E004 now provides a reviewed2GiB aggregate limit while retaining64MiB per file.
It also fixes the old exclusion of `D001/corpus/track_run` evidence and includes
corpus JSON lineage. Six software groups pass, including a verified576MiB fixture,
P26 metadata selection and both oversize refusals. Earlier capacity observations
above remain historical; actual final capacity/coverage must still be verified.
