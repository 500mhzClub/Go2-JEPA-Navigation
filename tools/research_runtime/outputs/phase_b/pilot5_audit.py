"""P14 pilot 5: the recollected non-yielding units against the same units of ed0cbb5, and the gate on the corpus as
it would stand (kept reactive and polite units + the new non-yielding ones, the old non-yielding scaled out)."""
import json, glob, os, collections, sys
from go2nav.data import standoff_audit as a
R = '/mnt/workspace_drive/go2nav_lewm_v2'; OLD = R+'/_set_aside_standoff_ny_ed0cbb5'
def units(root):
    out = {}
    for f in sorted(glob.glob(root+'/units/lewm_*/*_standoff*')):
        if os.path.exists(f+'/.done'):
            out[f[len(root)+1:]] = (root, f)
    return out
new_all, old_all = units(R), units(OLD)
def rows_of(root, path):
    m = json.load(open(root+'/'+path+'/meta.json'))
    return m, a.audit_unit((root, dict(path=path)))
new_ny, kept, old_same, stops_new, stops_old = [], [], [], collections.Counter(), collections.Counter()
for path, (root, f) in new_all.items():
    m, rs = rows_of(root, path)
    if m['code'].get('commit', '')[:7] == '89ef678' or path in old_all:
        new_ny += rs; stops_new[m['physical_stop']] += 1
        if path in old_all:
            mo, ro = rows_of(OLD, path); old_same += ro; stops_old[mo['physical_stop']] += 1
    else:
        kept += rs
print('recollected', sum(stops_new.values()), 'units; stops', dict(stops_new), '| same units at ed0cbb5', dict(stops_old))
print('new', a.summary(new_ny)['all']); print('old', a.summary(old_same)['all'])
# projection: kept + new non-yielding rows scaled to the 527 units
scale = 527/max(sum(stops_new.values()), 1)
proj = kept+new_ny*int(round(scale)) if scale >= 1 else kept+new_ny
s = a.summary(kept+[r for r in new_ny for _ in range(int(round(scale)))])
print('projected all', s['all']); print('projected gate', a.gate(s))
