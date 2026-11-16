# Research runtime source archive

This directory preserves the experiment scripts that previously lived beneath
ignored `outputs/` directories, including historical and failed variants.
The package implementation, configurations and tests remain in their normal
repository locations. `manifest.json` records each original relative path,
SHA256, size and mode; every copied source file is byte-identical to that path
at capture time.

The `outputs/gated/` subtree contains P25/P26 launchers, audits, numerical checks,
inspectors and their software tests through U013. Other subtrees preserve earlier
Go2 experiment scripts. The original repository review is also tracked under
`docs/REVIEW_2026-11-15_repository.md`.

These are source snapshots, not automatically runnable experiments. Scripts retain
their original working-directory assumptions and machine paths. For restoration,
use the manifest to copy the selected qualified version to its original relative
path in an unused checkout, then supply the separately retained datasets,
checkpoints and exact consultation/result-review evidence required by the runbook.
Historical failed variants remain evidence, not recommended entry points.

The full source history is published on the corresponding feature branches.
Large generated artifacts and third-party tool installations are not part of this
source archive. Creating this snapshot does not modify any live runtime file.

The U010–U013 controlled-change JSON files are included alongside their source
because their software checks verify the exact inherited transformations.
