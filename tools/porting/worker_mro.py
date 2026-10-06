"""Flatten the original worker objects: the pose tracker, mapper, obstacle observer and registration (porting aid).

Each mission runs four one-worker process pools whose initializers build one stateful object: the visual pose tracker
(`process_mapped_runtime_development._motion`), the routing mapper (`_mapper`), the depth-obstacle observer
(`independent_depth_process_development._observer`) and the floor registration (`_registration`). This runs the same
initializers in-process (as `run_go2_navigation_capability_completed_support_v4_development.run` passes them) and
writes each object's MRO and method ownership, as `runtime_mro.py` does for the controller.

Output: docs/porting/worker_<name>.json. Usage (from the snapshot root, programme environment):
  PYTHONPATH=.:lewm_genesis:lewm_worlds python $GO2NAV_REPO/tools/porting/worker_mro.py --out $GO2NAV_REPO/docs/porting
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_mro import executed_functions, flatten  # noqa: E402


def build_workers():
    """[(name, object)]: the four worker objects, built by the original initializers in this process."""
    sys.path.insert(0, '.')
    from functools import partial
    from lewm import decision_headroom_json_v42_development as output
    from lewm import independent_depth_process_development as obstacles
    from lewm import process_mapped_runtime_development as mapped
    from lewm import process_registered_round_trip_development as registered
    from lewm.navigation_capability_paired_floor_start_development import initialize_mapping
    from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
    source = owner.source
    output.install(Path(json.loads(owner.PROTOCOL.read_text())['output_root']))
    with tempfile.TemporaryDirectory() as scratch:
        for initializer in (source.previous.study.previous.reference.previous.initialize_registration, initialize_mapping,
                            partial(source.initialize_pose, scratch), source.previous.study.previous.reference.previous.initialize_obstacles):
            initializer()
    return [('pose', mapped._motion), ('mapping', mapped._mapper), ('obstacles', obstacles._observer),
            ('registration', registered._registration)]


def main(out, trace):
    executed = executed_functions(trace)
    out = Path(out)
    for name, value in build_workers():
        objects = {path: flatten(type(obj), executed) for path, obj in repository_objects(value)}
        (out/f'worker_{name}.json').write_text(json.dumps(objects, indent=1)+'\n')
        print(f'{name}: {type(value).__name__}')
        for path, flat in objects.items():
            print(f"  {path:40s} {flat['mro'][0]['cls']:40s} MRO {len(flat['mro']):3d}, {len(flat['methods']):3d} methods")


def repository_objects(root, path='self'):
    """(attribute path, object) for the root and every repository-defined object reachable through attributes,
    lists, tuples and dict values (each object once)."""
    seen, stack = set(), [(path, root)]
    while stack:
        path, obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, (list, tuple)):
            stack.extend((f'{path}[{i}]', v) for i, v in enumerate(obj[:4]))
            continue
        if isinstance(obj, dict):
            stack.extend((f'{path}[{k!r}]', v) for k, v in list(obj.items())[:4])
            continue
        if not type(obj).__module__.startswith(('lewm', 'scripts')):
            continue
        yield path, obj
        stack.extend((f'{path}.{k}', v) for k, v in sorted(getattr(obj, '__dict__', {}).items()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--trace', default='docs/go2_research_artifact_p1_runtime_trace_2026-10-06.json')
    a = p.parse_args()
    main(a.out, a.trace)
