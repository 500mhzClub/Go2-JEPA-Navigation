#!/usr/bin/env bash
# P25: L0 and L1 for 1, 3 and 6 balls, seed 0, six at once on the R9700, from the pinned checkout. Resumable: rerun.
set -u
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation-p25
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTORCH_TUNABLEOP_ENABLED=0 PYTHONUNBUFFERED=1
LOG=outputs/balls/chain.log
echo "[$(date '+%d %b %H:%M')] chain31 start $(git rev-parse --short HEAD)" >> $LOG
pids=()
for model in l0 l1; do for n in 1 3 6; do
  out=outputs/balls/runs/${model}_n${n}_s0
  mkdir -p $out
  $PY -m go2nav.training.balls train --data outputs/data/balls --n $n --model $model --out $out --seed 0 >> $out/train.log 2>&1 &
  pids+=($!)
done; done
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail+1)); done
echo "[$(date '+%d %b %H:%M')] chain31 done, failed runs: $fail" >> $LOG
