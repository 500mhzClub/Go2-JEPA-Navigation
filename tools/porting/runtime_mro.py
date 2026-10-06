"""Flatten the original composed runtime: effective class order and method ownership per controller (porting aid).

The original mission runtime is a class built at run time: development fixes and harness-version mixins composed over a
base runtime that is itself a deep chain of versioned classes. Run against the frozen snapshot, this tool rebuilds the
class exactly as the v14 entry composes it and reports, per controller:
- the MRO (class, module);
- for every method name, the class providing the effective implementation and the number of overridden definitions
  further down the MRO, marking those that ran in the P1 trace.

Output: docs/porting/runtime_<arm>_<recovery>.json and a summary on stdout.

Usage (from the snapshot root, with the programme environment):
  PYTHONPATH=.:lewm_genesis:lewm_worlds python $GO2NAV_REPO/tools/porting/runtime_mro.py --out $GO2NAV_REPO/docs/porting
"""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys


def executed_functions(trace_path):
    files = json.loads(Path(trace_path).read_text())['files']
    return {(path, name) for path, v in files.items() for name in v['executed_functions']}


def module_path(cls):
    try:
        return str(Path(inspect.getfile(cls)).resolve()).split('snapshot-2026-10-06/', 1)[-1]
    except TypeError:
        return '<builtin>'


def flatten(cls, executed):
    mro = [c for c in cls.__mro__ if c is not object]
    names = sorted({n for c in mro for n, v in vars(c).items() if inspect.isfunction(v) or isinstance(v, (property, staticmethod, classmethod))})
    methods = {}
    for n in names:
        owners = [c for c in mro if n in vars(c)]
        top = owners[0]
        qual = f'{top.__name__}.{n}'
        methods[n] = dict(owner=top.__name__, module=module_path(top), overrides=len(owners)-1,
                          executed=(module_path(top), qual) in executed,
                          chain=[f'{c.__name__} ({module_path(c)})' for c in owners])
    return dict(mro=[dict(cls=c.__name__, module=module_path(c)) for c in mro], methods=methods)


def build(arm, recovery):
    from lewm.dev_harness_fixes_development import compose, fixes_for
    from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
    from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
    version = importlib.import_module('lewm.dev_harness_reserve_exit_v2_development')
    extra = ()
    base_arm = arm
    if arm == 'C1A':
        from lewm.dev_c1_adaptive_travel_v3_development import AdaptiveTravelMixin
        extra, base_arm = (AdaptiveTravelMixin,), 'C1'
    elif arm == 'C1R':
        from lewm.dev_c1_refit_development import RefitCommandModelMixin
        extra, base_arm = (RefitCommandModelMixin,), 'C1'
    mixin = compose(fixes_for(recovery), CompletedSupportRuntimeMixin, extra=extra+tuple(version.mixins_for(base_arm)))
    base = owner.source.DenseReactiveNavigationRuntime if base_arm == 'C2' else owner.source.DenseNavigationRuntime
    return type(f'Runtime_{arm}_{recovery}', (mixin, base), {})


def main(out, trace):
    sys.path.insert(0, '.')
    executed = executed_functions(trace)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for arm in ('C1', 'C1A', 'C1R', 'C2', 'C3', 'C4'):
        for recovery in ('off', 'on'):
            flat = flatten(build(arm, recovery), executed)
            (out/f'runtime_{arm}_{recovery}.json').write_text(json.dumps(flat, indent=1)+'\n')
            m = flat['methods']
            print(f"{arm:4s} recovery {recovery:3s}: MRO {len(flat['mro'])} classes in {len({c['module'] for c in flat['mro']})} modules; "
                  f"{len(m)} methods, {sum(v['executed'] for v in m.values())} executed, "
                  f"{sum(v['overrides'] > 0 for v in m.values())} overriding, max override depth {max(v['overrides'] for v in m.values())}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--trace', default='docs/go2_research_artifact_p1_runtime_trace_2026-10-06.json')
    a = p.parse_args()
    main(a.out, a.trace)
