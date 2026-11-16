#!/bin/bash
# Trace one lineage scene of each remaining family, two at a time.
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation/outputs/porting/corpus
run() { env -i HOME=$HOME PATH=/usr/bin:/bin bash trace_rollout.sh $1 $2 $3 $4 48 200 executed_rollout_$2.json > trace_rollout_$2.log 2>&1; }
run train local_composite_motifs 5 12 & run train loop_alias_stress 1 8 & wait
run val medium_enclosed_maze 6 13 & run train open_obstacle_field 4 11 & wait
run train rough_local_dynamics 1 8 & run val small_enclosed_maze 2 9 & wait
run train visual_sensor_stress 9 16
echo FAMILY_TRACES_DONE
