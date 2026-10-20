"""Re-collect every unit of a readout protocol at full resolution (fullres_unit.py per unit, in parallel processes).
Resumable: verified units are skipped; a failed unit is listed and retried on the next run.

    python tools/readout_protocol/fullres_collect.py PROTOCOL.json CORPUS OUT_ROOT [--workers 10]
"""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('protocol')
    p.add_argument('corpus')
    p.add_argument('out_root')
    p.add_argument('--workers', type=int, default=10)
    a = p.parse_args()
    units = json.loads(Path(a.protocol).read_text())['corpus_units']
    todo = [u for u in units if not (Path(a.out_root)/u['path']/'.verified').exists()]
    print(json.dumps(dict(units=len(units), todo=len(todo))), flush=True)
    logs = Path(a.out_root)/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    started, failed = time.time(), []

    def one(u):
        _, role, name = u['path'].split('/')
        maze, kind = name.split('_')
        with open(logs/f'{role}_{name}.log', 'w') as log:
            r = subprocess.run([sys.executable, '-W', 'ignore', str(HERE/'fullres_unit.py'), a.corpus, a.out_root, role,
                                str(int(maze)), kind], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            failed.append(u['path'])
        return u['path'], r.returncode
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (path, rc) in enumerate(ex.map(one, todo), 1):
            if k % 20 == 0 or rc:
                print(json.dumps(dict(done=k, of=len(todo), last=path, rc=rc, wall_s=round(time.time()-started))), flush=True)
    print(json.dumps(dict(finished=len(todo)-len(failed), failed=failed, wall_s=round(time.time()-started))), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
