#!/bin/bash
# End-to-end, stage A: the May corpus from scratch for T1 (build, the 80 T1 scenes' rollouts, plans and labels), the
# T1 rows (compared with the shipped rows) and T1's 18,101 rendered frames. From the frozen e2e worktree.
set -eu
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-e2e
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
RV=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_render_vulkan/bin/python
export PYTHONPATH=.
E=outputs/e2e
mkdir -p $E
echo "stage A start $(git rev-parse --short HEAD) $(date)"
[ -f $E/corpus/corpus.json ] || $PY -m go2nav.corpus.build $E/corpus
$PY -m go2nav.corpus.datagen rollout $E/corpus $E/datagen --scenes data/training/t1_temporal/frames.txt --workers 4 --post-workers 4
$PY -m go2nav.data.t1_rows $E/datagen --out $E/t1_rows.jsonl --workers 8
cmp $E/t1_rows.jsonl data/training/t1_temporal/rows.jsonl && echo "T1 rows identical to the shipped rows"
$PY -m go2nav.corpus.datagen render $E/corpus $E/datagen --render-python $RV --workers 6 --textures --frames-list data/training/t1_temporal/frames.txt
echo "stage A done $(date)"
