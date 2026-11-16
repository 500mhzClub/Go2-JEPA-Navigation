#!/bin/bash
# End-to-end, stages B-D (from the e2e worktree, at the commit printed below):
#   B: the decoder-source missions checked against their originals, then arranged with their verified replays;
#      decoder contexts rebuilt and compared;
#   C: the T1-T8 chain retrained from the regenerated frames (new encoder features);
#   D: C3 on the e2e models (prelim 30-49).
# Stage B stops if a decoder-source mission differs from its original (a different training set would follow).
set -eu
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation-e2e
cd $W
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
export PYTHONPATH=.
E=outputs/e2e
wait_for() { until [ -e "$1" ]; do sleep 120; done; echo "ready: $1 $(date)"; }
wait_for outputs/cohorts/c3v3_onpolicy/result.json
wait_for outputs/cohorts/c3v2_check/result.json
wait_for outputs/cohorts/s2rec2/result.json
until grep -q "stage A done" outputs/e2e_stage_a.log; do sleep 120; done
echo "stage B start $(git rev-parse --short HEAD) $(date)"
bash outputs/check_decoder_sources.sh > outputs/decoder_sources.check.txt 2>&1
same=$(grep -c "requests: identical;trace: identical;frames: identical;mission: identical;outcome: identical" outputs/decoder_sources.check.txt || true)
echo "decoder-source missions identical to their originals: $same of 32"
if [ "$same" != 32 ]; then echo "STOP: decoder-source missions differ (outputs/decoder_sources.check.txt)"; exit 1; fi
$PY -m go2nav.data.mission_sources outputs/recordings --cohorts outputs/cohorts --workers 6
$PY -m go2nav.data.samples decoder --recordings-root outputs/recordings --out $E/training/decoder/contexts.jsonl
cmp $E/training/decoder/contexts.jsonl data/training/decoder/contexts.jsonl && echo "decoder contexts identical to the shipped ones" || echo "decoder contexts DIFFER"
echo "stage C start $(date)"
$PY -m go2nav.training.chain $E/chain --render-root $E/datagen/render --recordings-root outputs/recordings --features $E/features
echo "stage D start $(date)"
$PY -m go2nav.eval.cohort configs/experiments/e2e_chain.yaml --workers 2 --c3-lanes 2 --out outputs/cohorts
echo "end-to-end done $(date)"
