"""Render then replay the deadlocked episodes (stuck_list.json, >100 stuck decisions): the first 10 stuck decisions and
stuck decisions about 25, 50, 100, 200 and 300 decisions later."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
S = Path(__file__).parent
PY = sys.executable
O = Path('outputs/lewm/stuck_replay')
jobs = []
for e in json.loads((O/'stuck_list.json').read_text()):
    if e['stuck'] < 100:
        continue
    st = e['stuck_idx']
    pick = st[:10]+[next(i for i in st if i >= st[0]+k) for k in (25, 50, 100, 200, 300) if any(i >= st[0]+k for i in st)]
    plan = json.loads((Path(e['run'])/'planning.json').read_text())
    name = f"{e['cond']}_{Path(e['run']).name.split('_')[-2]}"
    jobs.append(dict(name=name, run=e['run'], pick=sorted(set(pick)), last=max(plan[i]['frame'] for i in pick)))
def render(j):
    subprocess.run([PY, str(S/'stuck_render.py'), j['run'], str(O/j['name']), str(j['last'])], check=True,
                   stdout=subprocess.DEVNULL, stderr=open(O/f"{j['name']}.render.log", 'w'))
    print('rendered', j['name'], flush=True)
O.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(4) as ex:
    list(ex.map(render, jobs))
for j in jobs:
    out = O/f"{j['name']}.json"
    if not out.exists():
        subprocess.run([PY, str(S/'stuck_forecast.py'), j['run'], str(O/j['name']), str(out), ','.join(map(str, j['pick']))],
                       check=True, stderr=subprocess.DEVNULL)
print('done', flush=True)
