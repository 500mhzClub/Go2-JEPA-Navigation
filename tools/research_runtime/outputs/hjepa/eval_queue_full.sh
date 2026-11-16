#!/bin/bash
# H-JEPA full-run evaluation, queued behind the full run (hjepa-full). Every step skips work already done, so a rerun resumes.
set -euo pipefail
W=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=$W-lewm-bench
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
C=/mnt/workspace_drive/go2nav_lewm
S=$W/outputs/hjepa/full_3level
E=$S/eval
log() { echo "[$(date +%H:%M:%S)] $*"; }
while systemctl --user is-active -q hjepa-full; do sleep 60; done
[ -f $S/hjepa.pt ] && [ -f $S/lewm.pt ] || { log "full run did not finish (no hjepa.pt/lewm.pt)"; exit 1; }
git -C $B checkout -q --detach a3ac49d
cd $B
export PYTHONPATH=$B
mkdir -p $E
log "decoder-context cache"
[ -f $E/cache/rollout.done ] || $PY -W ignore -m go2nav.training.lewm_decoder cache --world-model $S/lewm.pt --recordings-root $W/outputs/recordings --out $E/cache
log "readout protocol heads (C7p, C8)"
$PY -W ignore -m go2nav.training.readout_protocol lewm --protocol $W/outputs/lewm/protocol/readout_v1.json --corpus $C --world-model $S/lewm.pt --out $E/protocol --cache $E/cache
log "corpus readout (C7)"
$PY -W ignore -m go2nav.training.lewm_readout corpus --world-model $S/lewm.pt --corpus $C --out $E/readout
[ -f $E/c7/fit_base_only.pt ] || $PY -W ignore -m go2nav.training.lewm_readout package --readout $E/readout/readout.pt --c6-from $E/protocol/fit.pt --out $E/c7
printf 'hjepa_world_model: %s\nhjepa_c7_decoder_and_c8: %s\n' $S/lewm.pt $E/c7/fit_base_only.pt > $E/models.yaml
log "L4 (T7 score) of C7 and C8"
[ -s $E/l4.json ] || $PY -W ignore -m go2nav.training.lewm_decoder evaluate --cache $E/cache --checkpoint $E/c7/fit_base_only.pt > $E/l4.json
log "turn gate"
[ -f $E/turn_gate.json ] || $PY -W ignore -m go2nav.eval.lewm_turn_gate --world-model $S/lewm.pt --corpus $C --decoder c7=$E/c7/fit_base_only.pt --out $E/turn_gate.json
log "latent measures"
[ -f $E/latent_measures.json ] || $PY -W ignore -m go2nav.eval.latent_measures lewm --world-model $S/lewm.pt --cache $E/cache --out $E/latent_measures.json
log "deadlock states against the simulator's ground truth"
[ -f $E/stuck_eval.json ] || $PY -W ignore -m go2nav.eval.stuck_replay evaluate --list $W/outputs/lewm/stuck_counterfactual/list.json \
  --dir $W/outputs/lewm/stuck_counterfactual --out $E/stuck_eval.json \
  --model hjepa=$S/lewm.pt,$E/c7/fit_base_only.pt \
  --model span8w=$W/outputs/lewm/chain_span8w/world/lewm.pt,$W/outputs/lewm/readout/chain_span8w_corpus/base_only/fit_base_only.pt
log "what each level captured"
[ -f $E/levels.json ] || $PY -W ignore -m go2nav.eval.hjepa_levels $S/hjepa.pt --corpus $C --out $E/levels.json
log "C7/C8 benchmarks (200 missions, 6 LeWM lanes)"
PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=1 $PY -W ignore -m go2nav.eval.cohort configs/experiments/lewm_benchmarks_hjepa_full.yaml --workers 8 --lewm-lanes 6 --resume
log "done"
