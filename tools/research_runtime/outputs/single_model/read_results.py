"""Paired tables for P7 and P8, from the cohorts' result.json. Written before the runs finished, so the reading cannot follow the numbers.

    python outputs/single_model/read_results.py record     # the fixed record: wo_* with P7's crash fixes
    python outputs/single_model/read_results.py heldout    # smh_* against cufh_* and lb_*
    python outputs/single_model/read_results.py measures   # wm_* against their references
"""
import collections
import json
import re
import sys
from math import comb
from pathlib import Path

ROOT = Path('outputs/cohorts')


def sign_test(gained, lost):
    n = gained+lost
    if not n:
        return 1.
    return min(1., 2*sum(comb(n, k) for k in range(min(gained, lost)+1))/2**n)


def rows(cohort):
    path = ROOT/cohort/'result.json'
    if not path.exists():
        return None
    return {tuple(map(int, re.search(r'_(\d+)_ep(\d)$', r['assignment']).groups())): r
            for r in json.loads(path.read_text())['rows']}


def failure(r):
    if r['round_trip']:
        return 'ok'
    e = r.get('source_error') or ''
    for key, label in (('PhysicalStop', 'contact'), ('PoseLost', 'pose'), ('Unported', 'crash')):
        if key in e:
            return label
    return 'error' if e else 'timeout'


def contacts(table):
    return sum(failure(r) == 'contact' or r['hard'] > 0 or r['contacts'] > 0 for r in table.values())


def pair(a, b):
    k = set(a) & set(b)
    gained = sum(b[x]['round_trip'] and not a[x]['round_trip'] for x in k)
    lost = sum(a[x]['round_trip'] and not b[x]['round_trip'] for x in k)
    return dict(n=len(k), a=sum(a[x]['round_trip'] for x in k), b=sum(b[x]['round_trip'] for x in k), gained=gained, lost=lost,
                p=round(sign_test(gained, lost), 3))


def merged(*tables):
    """Tables of different walker counts as one, the count in the key."""
    return {(i, *k): v for i, t in enumerate(tables) for k, v in t.items()}


def show(label, table):
    print(f"{label}: {sum(r['round_trip'] for r in table.values())} of {len(table)}, contacts {contacts(table)},",
          dict(collections.Counter(failure(r) for r in table.values() if not r['round_trip'])))


def record():
    table = {}
    for n in (1, 2):
        base = rows(f'wo_obst{n}_c5')
        fix = rows(f'smc_fix_obst{n}') or {}
        table.update({(n, *k): fix.get(k, r) for k, r in base.items()})
    return table


def heldout():
    names = (('R4 (static, mazes 30-49, ep 0)', 'r4'), ('R6 reference (static)', 'r6_ref'), ('held-out static', 'heldout_static'),
             ('held-out one walker', 'heldout_obst1'), ('held-out two walkers', 'heldout_obst2'))
    groups = dict(lv2=[], hand=[], model=[])
    for label, tag in names:
        lb = next(rows(c) for c in (f'lb_{tag}_c1', f'lb_{tag.replace("heldout_", "")}_c1') if rows(c))
        hand, model = rows(f'cufh_{tag}_c1'), rows(f'smh_{tag}_c5')
        if model is None:
            print(label, 'not finished')
            continue
        for group, table in zip(groups.values(), (lb, hand, model)):
            group.append(table)
        print(f"{label}: lv2 {sum(r['round_trip'] for r in lb.values())}, hand tracker on C1 {sum(r['round_trip'] for r in hand.values())}, "
              f"one model {sum(r['round_trip'] for r in model.values())} of {len(model)}, against hand {pair(hand, model)}")
    if len(groups['model']) == len(names):
        lv2, hand, model = (merged(*g) for g in groups.values())
        print('\nall 140: against cufh', pair(hand, model), ' against lv2', pair(lv2, model))
        show('one model', model)
        show('hand tracker on C1', hand)
        print('bar: at least 120 round trips and no contacts:', sum(r['round_trip'] for r in model.values()) >= 120 and contacts(model) == 0)


def measures():
    fixed = record()
    none = merged(*[rows(f'wm_none_obst{n}_c5') or {} for n in (1, 2)])
    if none:
        print('no tracks against the fixed record:', pair({k: fixed[(k[0]+1, *k[1:])] for k in none}, none))
        show('no tracks', none)
    ref2, ref4 = rows('cuf_obst2_c1'), rows('wm_ref_obst4_c1')
    if ref4:
        show('C1 reference, 2 walkers', ref2)
        show('C1 reference, 4 walkers', ref4)
    for tag in ('pose', 'map', 'true'):
        two, four = rows(f'wm_{tag}_obst2_c1'), rows(f'wm_{tag}_obst4_c1')
        if two and four and ref4:
            pooled = pair(merged(ref2, ref4), merged(two, four))
            print(f'{tag} oracle: 2 walkers {pair(ref2, two)}; 4 walkers {pair(ref4, four)}; pooled {pooled}; '
                  f"net {pooled['gained']-pooled['lost']}, wins at least twice losses {pooled['gained'] >= 2*pooled['lost']}")
            show(f'  {tag}, 2 and 4 walkers', merged(two, four))
    hand = merged(*[rows(f'wm_rude_hand_obst{n}_c5') or {} for n in (1, 2)])
    model = merged(*[rows(f'wm_rude_model_obst{n}_c5') or {} for n in (1, 2)])
    if hand and model:
        print('walkers that do not yield, hand against model:', pair(hand, model))
        show('  hand tracker', hand)
        show('  model positions', model)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'record':
        table = record()
        show('the fixed record', table)
        print('against wo_*:', pair(merged(*[rows(f'wo_obst{n}_c5') for n in (1, 2)]), {(k[0]-1, *k[1:]): v for k, v in table.items()}))
    else:
        dict(heldout=heldout, measures=measures)[mode]()
