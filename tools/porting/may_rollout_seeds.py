"""Write data/corpus/may_2026_rollout_seeds.json: the bulk-rollout seed of every scene of the May corpus (porting tool).

    python tools/porting/may_rollout_seeds.py

The May rollouts ran each (split, family) in chunks of 40 scenes in name order, scene k of a chunk with seed 7 + k.
Two scenes (train rough_local_dynamics slots 6 and 38) made the rigid solver produce NaNs; they were replaced by
re-seeded scenes (salt 1), each rolled out alone (seed 7), after the first pass. The replacements have different
names, so the final corpus's name order differs from the order the first pass used. A scene's seed is therefore its
position in the *original* list (with the two original scenes), not in the final one: 36 of that family's 50 training
scenes sit one position off.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go2nav.corpus.manifest import stable_scene_id  # noqa: E402
from go2nav.corpus.plan import _scene_seed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REPLACED = {('train', 'rough_local_dynamics', 6), ('train', 'rough_local_dynamics', 38)}
FIRST_SEED, CHUNK = 7, 40


def main():
    plan = json.loads((REPO/'data/corpus/may_2026_plan.json').read_text())
    groups, seeds = {}, {}
    for a in plan['assignments']:
        key = (a['split'], a['family'], a['scene_index'])
        if key in REPLACED:
            original = _scene_seed(plan_seed=plan['plan_seed'], split=a['split'], family=a['family'], index=a['scene_index'], salt=0)
            name = stable_scene_id(a['family'], original)
            seeds[a['scene_id']] = FIRST_SEED  # the replacement, rolled out alone
        else:
            name = a['scene_id']
        groups.setdefault((a['split'], a['family']), []).append(name)
    for names in groups.values():
        for position, name in enumerate(sorted(names)):
            seeds.setdefault(name, FIRST_SEED+position % CHUNK)
    current = {a['scene_id'] for a in plan['assignments']}
    out = REPO/'data/corpus/may_2026_rollout_seeds.json'
    out.write_text(json.dumps({name: seeds[name] for name in sorted(current)}, indent=0)+'\n')
    print(f'{out}: {len(current)} scenes')


if __name__ == '__main__':
    main()
