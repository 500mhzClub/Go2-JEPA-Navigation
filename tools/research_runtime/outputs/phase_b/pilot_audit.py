"""P14 pilot: the stand-off gate over the stand-off units present (not yet indexed)."""
import json, glob, os, sys, collections
from go2nav.data import standoff_audit as a
R = '/mnt/workspace_drive/go2nav_lewm_v2'
rows, beh, stops, n = [], collections.Counter(), collections.Counter(), 0
for f in sorted(glob.glob(R+'/units/lewm_*/*_standoff*')):
    if not os.path.exists(f+'/.done'):
        continue
    n += 1
    m = json.load(open(f+'/meta.json'))
    stops[m['physical_stop']] += 1
    pl = m['obstacles']['placement']
    beh[a.behaviour(pl[0]) if pl else 'none'] += 1
    rows += a.audit_unit((R, dict(path=f[len(R)+1:])))
s = a.summary(rows)
print(n, 'units; behaviours drawn', dict(beh), '; stops', dict(stops))
print('all', s['all'])
for key in ('behaviour', 'family'):
    for k, v in s[key].items():
        print(' ', k, v)
