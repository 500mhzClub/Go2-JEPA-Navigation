"""Per forecaster: decisions with an escape (a moving candidate not BLOCKED), at the first stuck decision and over all
sampled; turn forecasts' translation and their dip below the start clearance."""
import json, statistics
from pathlib import Path
O = Path('outputs/lewm/stuck_replay')
ARMS = ['logged', 'C5_span8w', 'C5_idm100', 'C6_span8w', 'C6_idm100']
tot = {a: [0, 0, 0] for a in ARMS}
dips = {a: [] for a in ARMS[1:]}; trans = {a: [] for a in ARMS[1:]}
resid = []
for f in sorted(p for p in O.glob('*.json') if p.name != 'stuck_list.json'):
    rows = json.loads(f.read_text())
    line = f'{f.stem:14s}'
    for a in ARMS:
        esc = [any(v['mode'] != 'BLOCKED' for k, v in r[a].items() if k != 'hold') for r in rows]
        tot[a][0] += esc[0]; tot[a][1] += sum(esc); tot[a][2] += len(esc)
        line += f' {a}: {"Y" if esc[0] else "n"} {sum(esc):2d}/{len(esc)}'
    for r in rows:
        resid.append(r['wall_resid_mm'])
        for a in ARMS[1:]:
            for t in ('left_turn', 'right_turn'):
                seg = r[a][t]['seg']
                dips[a].append(1000*(seg[0]-min(seg)))
                xy = r[a+'_xy'][['hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn'].index(t)]
                trans[a].append(1000*max((x*x+y*y)**.5 for x, y in xy))
    print(line)
print('\ntotals (first stuck decision escapes / sampled decisions with an escape):')
for a in ARMS:
    print(f'  {a:10s} first {tot[a][0]}/{len(list(O.glob("*.json")))-1}  all {tot[a][1]}/{tot[a][2]}')
q = lambda v: f'median {statistics.median(v):.1f} p90 {statistics.quantiles(v, n=10)[8]:.1f}'
print('\nturn forecasts: dip below start clearance (mm) | largest translation (mm)')
for a in ARMS[1:]:
    print(f'  {a:10s} {q(dips[a])} | {q(trans[a])}')
print(f'\nwall fit residual: median {statistics.median(resid):.2f} mm, max {max(resid):.2f} mm')
