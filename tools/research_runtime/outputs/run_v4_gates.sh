#!/bin/bash
# v4-harness gate missions (fresh port runs vs original v4-era runs), from the frozen gate worktree.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-gate
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
O=outputs/missions/v4_gates
mkdir -p $O
run() { name=$1; shift; [ -f $O/$name/result.json ] || { rm -rf $O/$name; $PY -m go2nav.eval.mission "$@" --harness v4 --out $O/$name > $O/$name.log 2>&1; }; echo "$name $(cat $O/$name/result.json 2>/dev/null) $(date +%H:%M)"; }
run C1_off_prelim30 --arm C1 --set prelim_test_v1 --maze 30 --recovery off --coverage-fix off &
run C2_off_prelim30 --arm C2 --set prelim_test_v1 --maze 30 --recovery off --coverage-fix off &
run C4_off_prelim30 --arm C4 --set prelim_test_v1 --maze 30 --recovery off --coverage-fix off &
run C1_on_prelim33 --arm C1 --set prelim_test_v1 --maze 33 --recovery on --coverage-fix off &
run C1_c3v3_m00 --arm C1 --set c3v3_sets_v1 --maze 0 --recovery off --coverage-fix off &
run C1_sens_scale0p5_prelim30 --arm C1 --set prelim_test_v1 --maze 30 --recovery off --degrade scale:0.5 &
wait
run C3_off_prelim30 --arm C3 --set prelim_test_v1 --maze 30 --recovery off --coverage-fix off &
run C3v2_check00 --arm C3 --set c3v2_sets_v1 --maze 0 --recovery off --coverage-fix off --decoder-fit c3v2_readout &
wait
echo "gates done $(date)"
