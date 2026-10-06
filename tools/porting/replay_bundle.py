"""Replay bundles: everything that crosses a component boundary during one original mission (porting tool).

A bundle is the reference for the equivalence tests of the ported runtime (`tests/equivalence/`):
- `pool_<name>.stream`, one record per call to the registration, mapping, pose and obstacle worker pools:
  {function, args, kwargs, result} (the pools cross a process boundary, so everything is picklable, and the worker's
  state depends only on the sequence of its calls);
- `controller.stream`, the controller's inputs and outputs in call order:
  {kind: 'submit', packet} for each acquired frame, {kind: 'request', now_ns, result} for each 50 Hz dispatch request;
- `bundle.json`, the mission arguments, the launch pin and the run folder (which holds planning.json, requests.json,
  mission.json, model_calls.json and the native physics trace).

Recording does not change the mission: the pools are the original spawned process pools (wrapped in the main process),
and the controller methods are wrapped on the instance. Run from the snapshot root in the programme environment:

    cd $GO2NAV_SNAPSHOT && PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 PYTHONPATH=.:lewm_genesis:lewm_worlds \\
        $PY /path/to/tools/porting/replay_bundle.py record --out BUNDLE_DIR --harness reserve_exit_v2 [--dynamics ...] \\
        [--stall-stop-s S] -- --controller C1 --set prelim_test --maze 30 --assignment dev_p2rb_C1_prelim_test30_ep0 --recovery on
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import runpy
import struct
import sys
from threading import Lock
import zlib


class StreamWriter:
    """Append-only file of length-prefixed, zlib-compressed pickles."""

    def __init__(self, path):
        self.file, self.lock, self.count = open(path, 'xb'), Lock(), 0

    def write(self, record):
        data = zlib.compress(pickle.dumps(record, protocol=5), 1)
        with self.lock:
            self.file.write(struct.pack('<Q', len(data)))
            self.file.write(data)
            self.count += 1

    def close(self):
        self.file.close()


def read_stream(path):
    """Yield the records of a stream file in order."""
    with open(path, 'rb') as file:
        while header := file.read(8):
            yield pickle.loads(zlib.decompress(file.read(struct.unpack('<Q', header)[0])))


def function_name(function):
    function = getattr(function, 'func', function)  # functools.partial
    return f'{function.__module__}.{function.__qualname__}'


class RecordedFuture:
    """Records the call when its result is collected, in the caller's thread, before the caller can modify it."""

    def __init__(self, future, record, stream):
        self.future, self.record, self.stream = future, record, stream

    def result(self, timeout=None):
        try:
            value = self.future.result(timeout)
        except BaseException as error:
            self.stream.write(dict(self.record, result=None, error=repr(error)))
            raise
        self.stream.write(dict(self.record, result=value, error=None))
        return value


class RecordingPool:
    """The original one-worker spawn pool, recording each call's arguments and result."""

    def __init__(self, out, max_workers=1, mp_context=None, initializer=None):
        name = function_name(initializer).rsplit('.', 1)[-1].removeprefix('initialize_')
        self.pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context, initializer=initializer)
        self.stream, self.name = StreamWriter(Path(out)/f'pool_{name}.stream'), name

    def submit(self, function, *args, **kwargs):
        record = dict(function=function_name(function), args=pickle.dumps(args, protocol=5),
                      kwargs=pickle.dumps(kwargs, protocol=5))
        return RecordedFuture(self.pool.submit(function, *args, **kwargs), record, self.stream)

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self.pool.__exit__(*exc)
        finally:
            self.stream.close()


def record(a, mission):
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    from scripts.launch_go2_dev_cohort_pinned_v14_development import pin
    launch = pin()
    os.environ['LEWM_LAUNCH_PIN'] = json.dumps(launch)
    os.environ['LEWM_HARNESS'] = a.harness
    for name, value in (('LEWM_DYNAMICS', a.dynamics), ('LEWM_STALL_STOP_S', a.stall_stop_s and str(a.stall_stop_s))):
        if value:
            os.environ[name] = value
    from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
    controller_stream = StreamWriter(out/'controller.stream')
    original_cues = owner.install_task_cues

    def install_task_cues(controller, cues):
        """Called once, right after the controller is built: wrap its inputs and outputs on the instance."""
        original_cues(controller, cues)
        submit, request = controller.submit, controller.request

        def recorded_submit(packet):
            controller_stream.write(dict(kind='submit', packet=packet))
            return submit(packet)

        def recorded_request(*, now_ns):
            result = request(now_ns=now_ns)
            controller_stream.write(dict(kind='request', now_ns=now_ns, result=result))
            return result
        controller.submit, controller.request = recorded_submit, recorded_request
    owner.install_task_cues = install_task_cues  # read when the entry binds owner.run
    owner.ProcessPoolExecutor = lambda **kwargs: RecordingPool(out, **kwargs)
    entry = Path.cwd()/'scripts/run_go2_dev_mission_pinned_v14_development.py'
    assignment = mission[mission.index('--assignment')+1]
    capability_root = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
    (out/'bundle.json').write_text(json.dumps(dict(
        mission=mission, harness=a.harness, dynamics=a.dynamics, stall_stop_s=a.stall_stop_s, launch_pin=launch,
        run=str(capability_root/'runs'/assignment)), indent=1))
    sys.argv = [str(entry)]+mission
    try:
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        controller_stream.close()


def main():
    argv = sys.argv[1:]
    split = argv.index('--')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['record'])
    p.add_argument('--out', required=True)
    p.add_argument('--harness', default='reserve_exit_v2')
    p.add_argument('--dynamics')
    p.add_argument('--stall-stop-s', type=float)
    a = p.parse_args(argv[:split])
    record(a, argv[split+1:])


if __name__ == '__main__':
    main()
