"""P9b's stopping rule (PLAN "Phases and gates", P9b): tallies the veto cohorts and prints PASS, FAIL <why> or RUNNING.
    python outputs/memory/veto_check.py rude|rest"""
import json, sys
from pathlib import Path
import numpy as np
from go2nav.eval.walker_compare import planar_yaw, in_reach
ROOT, CACHE = Path('outputs/cohorts'), Path('outputs/memory/veto_seen.json')
seen = json.loads(CACHE.read_text()) if CACHE.exists() else {}

def stop(run):
    """None, ('wall',), ('walker', out_of_view)."""
    ev = [x for x in json.loads((run/'native/contact_events.json').read_text()) if x['disallowed_contacts']]
    if not ev:
        return None
    oid = ev[0]['disallowed_contacts'][0]['environment_object_id']
    if not oid.startswith('obstacle'):
        return ('wall',)
    if run.name not in seen:
        T, j = ev[0]['timestamp_s'], int(oid.split('_')[-1])
        tr, o = np.load(run/'native/physics_trace.npz'), np.load(run/'native/obstacle_trace.npz')
        walls = json.loads((run/'specification.json').read_text())['geometry']['wall_boxes']
        i = len(tr['timestamp_s'])-1; k = min(np.searchsorted(o['timestamp_s'], T), len(o['timestamp_s'])-1)
        last = False
        for t in (T-.1, T):
            i = min(np.searchsorted(tr['timestamp_s'], t), len(tr['timestamp_s'])-1); k = min(np.searchsorted(o['timestamp_s'], t), len(o['timestamp_s'])-1)
            last |= bool(in_reach(tr['base_pose_world'][i, :2], float(planar_yaw(tr['base_pose_world'][i])), o['xy'][k, j], walls))
        seen[run.name] = not last
        CACHE.write_text(json.dumps(seen))
    return ('walker', seen[run.name])

def tally(cohorts, total):
    n = rt = wall = walker = out = 0
    for c in cohorts:
        for run in sorted((ROOT/c/'runs').glob('*')) if (ROOT/c/'runs').exists() else []:
            if 'attempt' in run.name or not (run/'episode_evaluation.json').exists():
                continue
            e = json.loads((run/'episode_evaluation.json').read_text())
            n += 1; rt += bool(e['round_trip_success'])
            s = stop(run)
            wall += s == ('wall',); walker += bool(s and s[0] == 'walker'); out += s == ('walker', True)
    return dict(done=n, of=total, round_trips=rt, wall=wall, walker=walker, walker_out_of_view=out)

def verdict(t, *, walls, out_of_view=None, contacts=None, round_trips):
    left = t['of']-t['done']
    if t['wall'] >= walls: return f"FAIL wall contact stops {t['wall']} (stop at {walls})"
    if out_of_view is not None and t['walker_out_of_view'] > out_of_view: return f"FAIL out-of-view walker contacts {t['walker_out_of_view']} > {out_of_view}"
    if contacts is not None and t['wall']+t['walker'] > contacts: return f"FAIL contact stops {t['wall']+t['walker']} > {contacts}"
    if t['round_trips']+left < round_trips: return f"FAIL round trips {t['round_trips']} + {left} left < {round_trips}"
    return 'PASS' if not left else 'RUNNING'

if sys.argv[1] == 'rude':
    t = tally(['memv_rude_obst1_c5', 'memv_rude_obst2_c5'], 80)
    print(json.dumps(t), verdict(t, walls=5, out_of_view=3, round_trips=11))
else:
    four, fixed = tally(['memv_four_c5'], 40), tally(['memv_obst1_c5', 'memv_obst2_c5'], 80)
    v4 = verdict(four, walls=5, contacts=2, round_trips=21)
    vf = verdict(fixed, walls=1, contacts=0, round_trips=67)
    print(json.dumps(dict(four=four, fixed=fixed)), 'four:', v4, '| fixed:', vf)
    print('PASS' if v4 == vf == 'PASS' else next((v for v in (v4, vf) if v.startswith('FAIL')), 'RUNNING'))
