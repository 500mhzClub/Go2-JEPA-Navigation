"""Draft a port of an original module: drop the functions and methods that no traced run executed (porting aid).

    python tools/porting/trim_module.py SOURCE.py --lines LINES.json [--lines ...] --out DRAFT.py [--rename OLD=NEW ...]

LINES.json files come from tools/porting/line_trace.py (executed lines per absolute file path; several are merged).
A function or method is kept if any line of its body executed. Classes, module-level code, constants and imports
are kept; `--rename` rewrites import prefixes (e.g. lewm_worlds.planning_grid=go2nav.corpus.grid). The draft is a
starting point for a hand-tidied port, verified by the stage's byte-identity check.
"""
import argparse
import ast
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('--lines', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rename', action='append', default=[])
    a = p.parse_args()
    source = a.source.resolve()
    executed = set()
    for path in a.lines:
        executed |= set(json.loads(path.read_text()).get(str(source), []))
    text = source.read_text()
    lines = text.splitlines(keepends=True)
    drop = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = set(range(node.body[0].lineno, node.end_lineno+1))
            if not body & executed:
                start = min([node.lineno]+[d.lineno for d in node.decorator_list])
                drop.append((start, node.end_lineno, node.name))
    removed = set()
    for start, end, _ in drop:
        removed |= set(range(start, end+1))
    out = ''.join(line for i, line in enumerate(lines, 1) if i not in removed)
    for rule in a.rename:
        old, new = rule.split('=', 1)
        out = out.replace(old, new)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out)
    names = sorted({name for _, _, name in drop})
    print(json.dumps(dict(source=str(a.source), lines=len(lines), kept=len(out.splitlines()), dropped=names)))


if __name__ == '__main__':
    main()
