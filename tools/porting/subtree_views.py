"""Executed views of every function below a root in the recorded call graphs (porting aid).

    python tools/porting/subtree_views.py ROOT_SUBSTRING OUT.txt [graphs...]

Lists the subtree's functions (with executed-line and call counts) and prints their executed views, file by file.
Methods of composed classes are included; use executed_view.py for MRO-ordered chains.
"""
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import executed_view as ev  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main(root, out, graphs):
    graphs = graphs or sorted(str(p) for p in (REPO/'outputs/porting/call_graph').glob('*.json'))
    calls, edges = defaultdict(int), defaultdict(set)
    for path in graphs:
        g = json.loads(Path(path).read_text())
        for k, n in g['calls'].items():
            calls[k] += n
        for a, b, _ in g['edges']:
            edges[a].add(b)
    roots = [k for k in calls if root in k]
    seen, stack = set(roots), list(roots)
    while stack:
        for c in edges[stack.pop()]:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    cov = ev.coverage()
    rows = []
    for k in seen:
        path, name, line = k.rsplit(':', 2)
        if name == '<module>' or '<locals>' in name:
            continue
        rng = ev.functions(path).get(name)
        if rng is None:
            continue
        executed = cov.get(path, (set(), set()))[0]
        rows.append((path, int(line), name, sum(1 for i in range(rng[0]+1, rng[1]+1) if i in executed), calls[k]))
    rows.sort()
    with open(out, 'w') as stream:
        stream.write(f'# subtree of {root}: {len(rows)} functions, {sum(r[3] for r in rows)} executed lines\n')
        for path, line, name, n, count in rows:
            stream.write(f'# {n:4d} {count:8d}  {path}:{line}  {name}\n')
        sys.stdout = stream
        for path, line, name, n, count in rows:
            ev.show(path, name, cov)
        sys.stdout = sys.__stdout__
    print(out, len(rows), 'functions')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
