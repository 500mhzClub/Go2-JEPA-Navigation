# Later baseline CPU execution

This checkout preserves the C011/B004 scientific source from
`a75523bde0d1bb57c46ffb2fdcbbd036e87a36a8`. The approved experiment order,
CPU limits, numerical checks and scientific reviews are unchanged.

U013 corrects a B004 supervisor path: the original launcher treated its required
existing probe directory as a new planning output. The evaluator and independent
inspector already used `outputs/gated/planning/RUN`. The new supervisor checks
and collects that directory and records explicit U013 source/review qualification.

After actual reviewed training, full-state audit and numerical admission, run
C011 probes with `python3 outputs/gated/C011/run_one.py --run RUN --python "$PY"`.
After the corresponding probe science/operation reviews and fresh B004/U013
consultation, run `python3 outputs/gated/U013/run_one.py --run RUN --python "$PY"`.
Use the unchanged `outputs/gated/B004/verify_cases.py` afterward, with GPUs hidden.
Science, independent inspection and operation outcomes each need substantive
literature review. No successor is automatic.

The fresh B004 consultation additionally binds `u013_record_sha256` and
`u013_software_review_sha256`. The complete instructions and current scientific
state are maintained in the followthrough branch RUNBOOK. U013 has six reviewed
software groups; no actual extra-seed planning result exists yet. All fixtures
and their failures remain local under `outputs/gated/U013`.

The source snapshots in `tools/research_runtime/outputs/gated/U013` preserve the
qualified helper and failed test variants. Restore selected source only to its
original relative path with the separately retained exact software reviews,
consultations, datasets and checkpoints. Original C011/B004 helpers and broader
runtime history are published on `feature/p25-followthrough`.
