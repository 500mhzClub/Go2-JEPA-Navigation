#!/bin/bash
# Rerun the original bulk rollout of one May-corpus lineage scene (porting check).
set -eo pipefail
S=/home/andrewknowles/Workspace/LeWMQuad-v3-snapshot-2026-10-06
OUT=/home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/porting/corpus/rollout_orig
source /opt/ros/jazzy/setup.bash
source /tmp/claude-1000/-home-andrewknowles-Workspace-LeWMQuad-v3/1ff72248-fe5c-4f88-9830-977630ad1e82/scratchpad/ros_msgs/install/setup.bash 2>/dev/null || true
cd $S
export PYTHONPATH="$S/lewm_genesis:$S/lewm_worlds:$S:${PYTHONPATH:-}"
/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/genesis_bulk_rollout.py \
  --scene-corpus /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/scene_corpus/minimum_tex_20260520T211541Z \
  --split test_hard --family large_enclosed_maze --scene-offset 10 --scene-limit 1 --seed 17 \
  --n-envs 48 --n-blocks 200 --backend cpu --no-rgb --recovery-interlock-clearance-m 0 \
  --log-progress-every-blocks 0 --out $OUT/rollout
