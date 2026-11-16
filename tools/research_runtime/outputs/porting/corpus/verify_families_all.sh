#!/bin/bash
# Ported rollout of one lineage scene per family (all eight), compared with the May messages. Usage: verify_families.sh [OUT]
cd /home/andrewknowles/Workspace/Go2-JEPA-Navigation
PY=/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
R=/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/datagen_full/rollout
O=${1:-outputs/porting/corpus/port_families}
one() {  # split family offset seed scene chunk
  $PY -m go2nav.corpus.bulk outputs/porting/corpus/port_build $O/$2 --split $1 --family $2 --scene-offset $3 --scene-limit 1 --seed $4 > $O/$2.log 2>&1
  echo "$2 $5 $(python3 tools/porting/compare_messages.py $O/$2/$5/messages.jsonl $R/$1/$2/$6/raw/$5/messages.jsonl | head -1)"
}
mkdir -p $O
python3 - <<'PY' > $O/jobs.txt
import json
w = json.load(open('outputs/porting/corpus/lineage_scenes.json'))
seen = set()
for scene, (split, family, index, seed, _) in sorted(w.items()):
    if family in seen: continue
    seen.add(family)
    chunk = f'chunk_{(index//40)*40:04d}'
    print(split, family, index, seed, scene, chunk)
PY
cat $O/jobs.txt
while read -r a b c d e f; do one $a $b $c $d $e $f & while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 5; done; done < $O/jobs.txt
wait
echo VERIFY_DONE
