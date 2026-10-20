"""counterfactual.py over the deadlocked episodes: the first stuck decision and the first stuck one >= 50 later."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
T = Path(__file__).parent
O = Path('outputs/lewm/stuck_counterfactual')
jobs = []
for e in json.loads(Path('outputs/lewm/stuck_replay/stuck_list.json').read_text()):
    if e['stuck'] >= 100:
        st = e['stuck_idx']
        pick = [st[0], next(i for i in st if i >= st[0]+50)]
        jobs.append((f"{e['cond']}_{Path(e['run']).name.split('_')[-2]}", e['run'], pick))
def one(job):
    name, run, pick = job
    with open(O/f'{name}.log', 'w') as log:
        subprocess.run([sys.executable, str(T/'counterfactual.py'), run, str(O/name), ','.join(map(str, pick))],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    print('done', name, flush=True)
O.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(4) as ex:
    list(ex.map(one, jobs))
