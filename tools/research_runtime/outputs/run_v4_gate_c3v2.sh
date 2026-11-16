#!/bin/bash
# v4 gate: the C3-v2 fresh check 0 (C3 with the C3-v2 readout) vs the original run, from the gate2 worktree.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-gate2
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
O=outputs/missions/v4_gates
mkdir -p $O
[ -f $O/C3v2_check00/result.json ] || { rm -rf $O/C3v2_check00; $PY -m go2nav.eval.mission --arm C3 --set c3v2_sets_v1 --maze 0 --recovery off --harness v4 --decoder-fit c3v2_readout --out $O/C3v2_check00 > $O/C3v2_check00.log 2>&1; }
echo "C3v2_check00 $(cat $O/C3v2_check00/result.json 2>/dev/null) $(date)"
