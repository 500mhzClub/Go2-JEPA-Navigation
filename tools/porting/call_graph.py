"""Record which repository functions call which during one original mission (porting aid).

The original controller is a composed class whose methods are also rewired with `bind()` (a function copy with some
globals replaced), so reading the source statically misleads. This runs one mission of the frozen snapshot exactly as
the v14 pinned launcher would (same entry, harness, fixes and dynamics), except that the four worker pools
(registration, mapping, pose, obstacles) run in-process, so one `sys.monitoring` hook sees every call. It writes, for
every snapshot function that ran, its call count and its nearest snapshot caller(s):

    {"calls": {"FILE:QUALNAME:LINE": n, ...}, "edges": [["CALLER", "CALLEE", n], ...]}

Run from the snapshot root in the programme environment:

    cd $GO2NAV_SNAPSHOT && PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 PYTHONPATH=.:lewm_genesis:lewm_worlds \\
        $PY /path/to/tools/porting/call_graph.py --out graph.json --harness reserve_exit_v2 [--dynamics patches3:0.3:marked] \\
        [--stall-stop-s 60] -- --controller C1 --set prelim_test --maze 30 --assignment dev_p2cg_C1_prelim_test30_ep0 --recovery on

The assignment's run folder is written under the capability root like any mission (name it dev_p2cg_*).
"""
import argparse
import atexit
from collections import Counter
from concurrent.futures import Future
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path.cwd().resolve()
PREFIX = str(ROOT)+'/'


class InlineExecutor:
    """A one-worker pool that runs its initializer and every task in the calling process."""

    def __init__(self, max_workers=1, mp_context=None, initializer=None):
        if initializer is not None:
            initializer()

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def shutdown(self, wait=True):
        pass


def key(code):
    return f'{code.co_filename[len(PREFIX):]}:{code.co_qualname}:{code.co_firstlineno}'


def install_monitor(out):
    calls, edges = Counter(), Counter()
    monitoring = sys.monitoring
    tool = monitoring.PROFILER_ID
    monitoring.use_tool_id(tool, 'go2nav-call-graph')

    def started(code, offset):
        if not code.co_filename.startswith(PREFIX):
            return monitoring.DISABLE
        frame = sys._getframe(1)
        callee = key(code)
        caller = frame.f_back
        while caller is not None and not caller.f_code.co_filename.startswith(PREFIX):
            caller = caller.f_back
        calls[callee] += 1
        edges[(key(caller.f_code) if caller is not None else '<root>', callee)] += 1

    monitoring.register_callback(tool, monitoring.events.PY_START, started)
    monitoring.set_events(tool, monitoring.events.PY_START)

    def save():
        monitoring.set_events(tool, 0)
        Path(out).write_text(json.dumps(dict(calls=dict(calls.most_common()),
                                             edges=[[a, b, n] for (a, b), n in edges.most_common()]), indent=0))
    atexit.register(save)
    return save


def main():
    argv = sys.argv[1:]
    split = argv.index('--')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--harness', default='reserve_exit_v2')
    p.add_argument('--dynamics')
    p.add_argument('--stall-stop-s', type=float)
    a = p.parse_args(argv[:split])
    mission = argv[split+1:]
    from scripts.launch_go2_dev_cohort_pinned_v14_development import pin
    os.environ['LEWM_LAUNCH_PIN'] = json.dumps(pin())
    os.environ['LEWM_HARNESS'] = a.harness
    if a.dynamics:
        os.environ['LEWM_DYNAMICS'] = a.dynamics
    if a.stall_stop_s:
        os.environ['LEWM_STALL_STOP_S'] = str(a.stall_stop_s)
    from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
    owner.ProcessPoolExecutor = InlineExecutor  # this process only; read when the entry copies or binds owner.run
    save = install_monitor(Path(a.out).resolve())
    entry = ROOT/'scripts/run_go2_dev_mission_pinned_v14_development.py'
    sys.argv = [str(entry)]+mission
    try:
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        save()


if __name__ == '__main__':
    main()
