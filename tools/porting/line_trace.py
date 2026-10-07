"""Run an original script and record which lines of the given source trees it executes (porting aid).

    python tools/porting/line_trace.py --out LINES.json --root lewm_genesis --root lewm_worlds -- SCRIPT [ARGS...]

Run from the snapshot root with the original's environment. Lines are recorded per file (the first line of each
executed statement); LINES.json is merged with an existing file, so several runs accumulate. Only frames whose code
lies under a root are traced; other calls return at once.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import threading


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', action='append', required=True)
    p.add_argument('script')
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    roots = tuple(str(Path(r).resolve()) for r in a.root)
    lines = {}

    def local(frame, event, arg):
        if event == 'line':
            lines.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
        return local

    def tracer(frame, event, arg):
        if not frame.f_code.co_filename.startswith(roots):
            return None
        lines.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
        return local
    sys.argv = [a.script, *[x for x in a.args if x != '--']]
    sys.settrace(tracer)
    threading.settrace(tracer)
    try:
        runpy.run_path(a.script, run_name='__main__')
    except SystemExit:
        pass
    finally:
        sys.settrace(None)
        previous = json.loads(a.out.read_text()) if a.out.exists() else {}
        for f, executed in lines.items():
            previous[f] = sorted(set(previous.get(f, [])) | executed)
        a.out.write_text(json.dumps(previous))


if __name__ == '__main__':
    main()
