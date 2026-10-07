"""Print the executed code of a composed runtime method, override by override (porting aid).

The original controllers are composed classes of 112-121 mixins; one method (say `_route`) is a chain of up to 45
overrides calling `super()`. This prints each override in MRO order with only the lines the P1 trace executed; runs
of statements that never ran are collapsed to `... N not run`. Module-level helpers can be printed by name.

    python tools/porting/executed_view.py C1_on _route
    python tools/porting/executed_view.py pose:self.model observe
    python tools/porting/executed_view.py --function lewm/clearance_lookahead_development.py:forecast_clearance
    python tools/porting/executed_view.py --file lewm/dev_harness_fixes_development.py

Coverage comes from the P1 trace (`p1_runtime_trace/merged/all.json`, every traced mission and offline step), source
from the frozen snapshot (`GO2NAV_SNAPSHOT`); `--mission NAME` restricts coverage to one traced mission's data.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re

import original

REPO = Path(__file__).resolve().parents[2]
TRACE = original.path('recovery', '.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001/p1_runtime_trace')
SNAPSHOT = Path(os.environ.get('GO2NAV_SNAPSHOT', Path.home()/'Workspace/LeWMQuad-v3-snapshot-2026-10-06'))
BLOCK_HEADER = re.compile(r'^\s*(else|try|finally)\s*:')


def coverage(mission=None):
    """{repository-relative path: (executed lines, missing lines or None)}."""
    if mission is None:
        files = json.loads((TRACE/'merged/all.json').read_text())['files']
        return {k: (set(v['executed_lines']), set(v['missing_lines'])) for k, v in files.items()}
    import sys
    sys.path.append(str(TRACE/'site'))
    from coverage import CoverageData
    out = {}
    for path in sorted((TRACE/'data'/mission).glob('cov*')):
        part = CoverageData(basename=str(path))
        part.read()
        for f in part.measured_files():
            out.setdefault(_relative(f), set()).update(part.lines(f) or ())
    return {k: (v, None) for k, v in out.items()}


def _relative(path):
    for marker in ('/LeWMQuad-v3/', '/LeWMQuad-v3-snapshot-2026-10-06/'):
        if marker in path:
            return path.split(marker, 1)[1]
    return path


def render(path, start, end, executed, missing):
    """Lines start..end of a snapshot file with unexecuted statement runs collapsed."""
    lines = (SNAPSHOT/path).read_text().splitlines()
    out, skipped, showing, indent = [], 0, True, ''
    for n in range(start, end+1):
        text = lines[n-1]
        if n in executed:
            showing = True
        elif missing is not None and n in missing or missing is None and _is_statement(text) and n not in executed:
            showing = False
        elif BLOCK_HEADER.match(text):
            showing = _next_statement_executed(lines, n, end, executed, missing)
        if showing:
            if skipped:
                out.append(f'{indent}... {skipped} not run')
                skipped = 0
            out.append(f'{n:5d}  {text}')
        elif text.strip() and (missing is not None and n in missing):
            if not skipped:
                indent = ' '*7+text[:len(text)-len(text.lstrip())]
            skipped += 1
    if skipped:
        out.append(f'{indent}... {skipped} not run')
    return out


def _is_statement(text):
    s = text.strip()
    return bool(s) and not s.startswith(('#', ')', ']', '}', '@')) and not BLOCK_HEADER.match(text)


def _next_statement_executed(lines, n, end, executed, missing):
    for m in range(n+1, end+1):
        if m in executed:
            return True
        if missing is not None and m in missing:
            return False
    return False


def functions(path):
    """{qualified name: (first line, last line)} for every function and method in a snapshot file."""
    tree = ast.parse((SNAPSHOT/path).read_text())
    out = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first = min([d.lineno for d in child.decorator_list]+[child.lineno])
                out[prefix+child.name] = (first, child.end_lineno)
                visit(child, prefix+child.name+'.')
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix+child.name+'.')
    visit(tree, '')
    return out


def method_chain(runtime, method):
    """`runtime` is a controller (C1_on: docs/porting/runtime_C1_on.json) or a worker object
    (pose:self.model: the object at that attribute path in docs/porting/worker_pose.json)."""
    if ':' in runtime:
        worker, path = runtime.split(':')
        record = json.loads((REPO/f'docs/porting/worker_{worker}.json').read_text())[path]
    else:
        record = json.loads((REPO/f'docs/porting/runtime_{runtime}.json').read_text())
    chain = record['methods'][method]['chain']
    return [(c.split(' (')[0], c.split(' (')[1].rstrip(')')) for c in chain]


def assignments(path):
    """{Class.attribute: (first line, last line)} for class-level assignments (methods bound by assignment)."""
    out = {}
    for node in ast.walk(ast.parse((SNAPSHOT/path).read_text())):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            out[f'{node.name}.{target.id}'] = (child.lineno, child.end_lineno)
    return out


def show(path, name, cov):
    executed, missing = cov.get(path, (set(), set()))
    if name not in functions(path):
        first, last = assignments(path)[name]
        lines = (SNAPSHOT/path).read_text().splitlines()[first-1:last]
        print(f'## {name}  ({path}:{first}; bound by assignment)')
        print('\n'.join(f'{first+i:5d}  {text}' for i, text in enumerate(lines)), end='\n\n')
        return
    first, last = functions(path)[name]
    run = [n for n in range(first, last+1) if n in executed]
    print(f'## {name}  ({path}:{first}; {len(run)} executed lines)')
    if not run:
        print('   (never executed)')
        return
    print('\n'.join(render(path, first, last, executed, missing)))
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runtime', nargs='?', help='e.g. C1_on, C3_off (docs/porting/runtime_*.json)')
    p.add_argument('method', nargs='?')
    p.add_argument('--function', action='append', default=[], help='FILE:QUALNAME')
    p.add_argument('--file', help='every executed function of one file')
    p.add_argument('--mission', help='coverage of one traced mission (data/<mission>)')
    a = p.parse_args()
    cov = coverage(a.mission)
    if a.runtime and a.method:
        for cls, path in method_chain(a.runtime, a.method):
            show(path, f'{cls}.{a.method}', cov)
    for spec in a.function:
        path, name = spec.split(':')
        show(path, name, cov)
    if a.file:
        for name in functions(a.file):
            if '.' not in name or name.count('.') == 1:
                executed = cov.get(a.file, (set(), set()))[0]
                first, last = functions(a.file)[name]
                if any(n in executed for n in range(first+1, last+1)):
                    show(a.file, name, cov)


if __name__ == '__main__':
    main()
