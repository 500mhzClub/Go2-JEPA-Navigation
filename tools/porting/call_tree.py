"""Print the call tree below a function from one or more recorded call graphs (porting aid; see call_graph.py).

    python tools/porting/call_tree.py outputs/porting/call_graph/*.json --root '_plan' [--depth 6] [--all]

`--root` is a substring of "FILE:QUALNAME:LINE"; every matching function is a root. Each callee is printed once (later
occurrences are marked "(above)"); `--all` lists every recorded function with its call count instead.
"""
import argparse
from collections import Counter, defaultdict
import json


def load(paths):
    calls, edges = Counter(), defaultdict(Counter)
    for path in paths:
        graph = json.load(open(path))
        calls.update(graph['calls'])
        for caller, callee, n in graph['edges']:
            edges[caller][callee] += n
    return calls, edges


def short(key):
    path, name, line = key.rsplit(':', 2)
    return f'{name}  [{path}:{line}]'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('graphs', nargs='+')
    p.add_argument('--root')
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--all', action='store_true')
    a = p.parse_args()
    calls, edges = load(a.graphs)
    if a.all:
        for key, n in calls.most_common():
            print(f'{n:9d}  {short(key)}')
        return
    seen = set()

    def walk(key, depth):
        for callee, n in sorted(edges.get(key, {}).items(), key=lambda kv: kv[0].rsplit(':', 1)[0]):
            mark = '  (above)' if callee in seen else ''
            print(f"{'  '*depth}{n:7d}  {short(callee)}{mark}")
            if callee not in seen and depth < a.depth:
                seen.add(callee)
                walk(callee, depth+1)
    for root in sorted(k for k in calls if a.root in k):
        print(f'{calls[root]:7d}  {short(root)}')
        seen.add(root)
        walk(root, 1)
        print()


if __name__ == '__main__':
    main()
