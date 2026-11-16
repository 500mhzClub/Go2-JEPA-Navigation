#!/bin/bash
# Trace the original bulk rollout of one corpus scene: trace_rollout.sh SPLIT FAMILY OFFSET SEED N_ENVS N_BLOCKS
set -eo pipefail
S=/home/andrewknowles/Workspace/LeWMQuad-v3-snapshot-2026-10-06
G=/home/andrewknowles/Workspace/Go2-JEPA-Navigation
B=/tmp/claude-1000/-home-andrewknowles-Workspace-LeWMQuad-v3/1ff72248-fe5c-4f88-9830-977630ad1e82/scratchpad/ros_msgs
source /opt/ros/jazzy/setup.bash
source $B/install/setup.bash 2>/dev/null || true
cd $S
export PYTHONPATH="$S/lewm_genesis:$S/lewm_worlds:$S:${PYTHONPATH:-}"
/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python $G/tools/porting/line_trace.py \
  --out $G/outputs/porting/corpus/${7:-executed_rollout.json} --root lewm_genesis --root lewm_worlds --root scripts -- \
  scripts/genesis_bulk_rollout.py --scene-corpus /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/scene_corpus/minimum_tex_20260520T211541Z \
  --split $1 --family $2 --scene-offset $3 --scene-limit 1 --seed $4 --n-envs $5 --n-blocks $6 --backend cpu --no-rgb \
  --recovery-interlock-clearance-m 0 --log-progress-every-blocks 0 --no-writer --out $G/outputs/porting/corpus/trace_rollouts/$2_$3
