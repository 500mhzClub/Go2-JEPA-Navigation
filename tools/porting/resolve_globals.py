"""Resolve which functions a composed method really calls, through `bind()`/`use()` global rewiring (porting aid).

The original code rewires helpers by copying a function with some globals replaced (`bind(f, register=...)`), often
several layers deep, so `register` inside a method body need not be the `register` of its own module. This walks the
live function objects of one worker object or controller method chain and prints, for every global name each
repository function reads, the object it resolves to, marking rebound functions (whose globals differ from their
module's) and recursing into repository functions.

    python tools/porting/resolve_globals.py pose:self.model _candidate observe [--depth 3]

Run from the snapshot root in the programme environment (it builds the workers as worker_mro.py does).
"""
import argparse
import inspect
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPOSITORY = ('lewm', 'scripts')


def describe(value):
    if isinstance(value, types.FunctionType):
        module = sys.modules.get(value.__module__)
        rebound = module is not None and value.__globals__ is not module.__dict__
        return f"{value.__module__}.{value.__qualname__}{'  [rebound]' if rebound else ''}"
    if isinstance(value, type):
        return f'class {value.__module__}.{value.__qualname__}'
    if isinstance(value, types.ModuleType):
        return f'module {value.__name__}'
    return f'{type(value).__name__} {repr(value)[:60]}'


def global_names(code):
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= global_names(const)
    return names


def walk(function, depth, seen, indent='  '):
    if function in seen:
        return
    seen.add(function)
    for name in sorted(global_names(function.__code__)):
        if name not in function.__globals__:
            continue
        value = function.__globals__[name]
        if isinstance(value, (types.FunctionType, type)) and getattr(value, '__module__', '').startswith(REPOSITORY):
            print(f'{indent}{name} -> {describe(value)}')
            if isinstance(value, types.FunctionType) and depth > 0:
                walk(value, depth-1, seen, indent+'    ')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('target', help='worker:attribute.path, e.g. pose:self.model')
    p.add_argument('methods', nargs='*')
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--constants', nargs='*', default=[], help='modules whose upper-case module constants to print, '
                   'as they are after every import and patch')
    a = p.parse_args()
    sys.path.insert(0, '.')
    import worker_mro
    worker, path = a.target.split(':')
    objects = dict(worker_mro.build_workers())
    obj = objects[worker]
    for part in path.split('.')[1:]:
        obj = getattr(obj, part)
    for name in a.constants:
        module = sys.modules[name] if name in sys.modules else __import__(name, fromlist=['_'])
        print(f'== constants of {name}')
        for key, value in sorted(vars(module).items()):
            if key.isupper() and not callable(value):
                print(f'  {key} = {value!r}'[:400])
    for method in a.methods:
        print(f'== {type(obj).__name__}.{method}')
        for cls in type(obj).__mro__:
            function = vars(cls).get(method)
            if isinstance(function, (staticmethod, classmethod)):
                function = function.__func__
            if isinstance(function, types.FunctionType):
                print(f'{cls.__name__}.{method}: {describe(function)}  ({inspect.getsourcefile(function).split("snapshot-2026-10-06/")[-1]}:{function.__code__.co_firstlineno})')
                walk(function, a.depth, set())


if __name__ == '__main__':
    main()
