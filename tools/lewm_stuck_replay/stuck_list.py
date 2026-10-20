"""Episodes span8w C5 failed and idm100 C5 passed: failure kind and the stuck decisions (hold, every moving candidate
BLOCKED after the turn re-checks)."""
import json, re
from pathlib import Path
C = Path('outputs/cohorts')
key = lambda a: re.search(r'_(\d+)_ep(\d+)$', a).groups()
out = []
for tag in ('r4', 'r5_mu030', 'r5_mu025', 'r6_ref', 'r6_strips'):
    new = {key(r['assignment']): r for r in json.loads((C/f'lewm_{tag}_c5_span8w/result.json').read_text())['rows']}
    old = {key(r['assignment']): r for r in json.loads((C/f'lewm_{tag}_c5/result.json').read_text())['rows']}
    for k in new:
        if old[k]['round_trip'] and not new[k]['round_trip']:
            run = C/f'lewm_{tag}_c5_span8w/runs'/new[k]['assignment']
            p = json.loads((run/'planning.json').read_text())
            stuck = [i for i, d in enumerate(p) if d.get('selection') and d['action'] == 'hold'
                     and all(c['clearance_check_mode'] == 'BLOCKED' for c in d['selection'].get('memory_forecast_candidates', []) if c['action'] != 'hold')
                     and d['selection'].get('memory_forecast_candidates')]
            out.append(dict(cond=tag, run=str(run), contacts=new[k]['contacts'], decisions=len(p), stuck=len(stuck),
                            first_stuck=stuck[0] if stuck else None, stuck_idx=stuck))
            print(f"{tag:10s} maze {k[0]} contacts={new[k]['contacts']} decisions={len(p)} stuck={len(stuck)} first={stuck[0] if stuck else None}")
Path('outputs/lewm/stuck_replay').mkdir(parents=True, exist_ok=True)
Path('outputs/lewm/stuck_replay/stuck_list.json').write_text(json.dumps(out))
