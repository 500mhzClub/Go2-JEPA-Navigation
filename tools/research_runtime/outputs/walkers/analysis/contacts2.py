import json, glob, collections, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
def one(path):
    run = Path(path)
    rows = [p for p in json.loads((run/'planning.json').read_text()) if 'selection' in p]
    cmd = np.array([p['selection'].get('requested_command', [0, 0, 0])[:3] for p in rows], float).reshape(-1, 3)
    hard = None
    c = json.loads((run/'native/contact_events.json').read_text())
    hit = next((x for x in c if x['disallowed_contacts']), None)
    last = None
    if hit:
        t = hit['timestamp_s']; recent = [p for p in rows if t-3 <= p['measured_ns']/1e9 <= t]
        rc = np.array([p['selection'].get('requested_command', [0, 0, 0])[:3] for p in recent], float).reshape(-1, 3)
        last = dict(raw=hit['disallowed_contacts'][0], back=float((rc[:, 0] < 0).mean()) if len(rc) else None, phase=hit.get('phase'))
    tr = [len([t for t in p['selection'].get('obstacle_tracks') or [] if t['motion_updates'] >= 3]) for p in rows]
    return dict(plans=len(rows), back=int((cmd[:, 0] < 0).sum()), tracks=float(np.mean(tr)) if tr else 0, last=last, name=run.name)
if __name__ == '__main__':
    for c in sys.argv[1:]:
        runs = sorted(glob.glob(f'outputs/cohorts/{c}/runs/*'))
        with ProcessPoolExecutor(10) as p: rs = list(p.map(one, runs))
        print(c, 'plans', sum(r['plans'] for r in rs), 'reverse share', round(sum(r['back'] for r in rs)/sum(r['plans'] for r in rs), 3), 'mean confirmed tracks per plan', round(np.mean([r['tracks'] for r in rs]), 2))
        for r in rs:
            if r['last']:
                print('  ', r['name'].split('validation_')[1], 'reversing share, last 3 s', None if r['last']['back'] is None else round(r['last']['back'], 2), json.dumps(r['last']['raw'])[:230])
