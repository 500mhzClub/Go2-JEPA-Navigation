#!/bin/bash
# P14 item 5: index corpus v2 with the stand-off units (the 31 Oct index kept as index_12_pre_p14.*), then the stand-off
# gate (standoff_audit) and the corpus audit (contact stops by kind). 24 workers beside the four-walker missions.
set -euo pipefail
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
R=/mnt/workspace_drive/go2nav_lewm_v2; O=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/phase_b
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-phase-b
$PY -W ignore -c "import json; from go2nav.data import lewm_corpus as v1; print(json.dumps(v1.build_index('$R', length=12, workers=24), indent=1))" > $O/index_12_summary.json
$PY -W ignore -m go2nav.data.standoff_audit $R --kinds standoff standoff_b --workers 24 --out $O/standoff_gate.json > /dev/null
$PY -W ignore -c "import json; from go2nav.data import lewm_corpus_v2 as v2; print(json.dumps(v2.audit('$R', workers=24), indent=1))" > $O/corpus_audit_p14.json
echo done
