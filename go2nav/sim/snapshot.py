"""Capture and restore the simulator at a decision boundary, for the C0 oracle's what-if branches.

Genesis' own scene state is not a complete physics snapshot, so the capture copies every mutable solver field (the
Quadrants fields and arrays reachable from the scene and its active solvers, except two immutable geometry tables),
plus the session's own state: time, executed command, rate-limit window, the sensor histories, the policy's last
action, the moving obstacles' walk state, and the Python, NumPy and torch random states. Restoring also resets the
renderer's caches.
"""
import copy
import random

import numpy as np

STATIC_FIELDS = ('._sdf_info.geoms_sdf_', '._support_field_info.support_')
SESSION_FIELDS = ('sim_ns', 'last_executed', 'window', 'window_start_command', 'body_sensor', 'history', 'fast_history',
                  'phase', 'guard', 'base_xy')
RECORD_LISTS = ('samples', 'contact_packets', 'contact_times', 'contact_events')


def solver_fields(scene):
    """[(path, field)] of the scene's mutable Quadrants fields, in a fixed walk order."""
    import quadrants as qd
    kinds = (qd.Field, qd.Ndarray)
    found, seen = [], set()

    def walk(obj, path, depth):
        if id(obj) in seen or depth > 4:
            return
        seen.add(id(obj))
        for name, value in sorted(getattr(obj, '__dict__', {}).items()):
            if name.startswith('__'):
                continue
            sub = f'{path}.{name}'
            if isinstance(value, kinds):
                if not any(marker in sub for marker in STATIC_FIELDS):
                    found.append((sub, value))
            elif hasattr(value, '__dict__') and not isinstance(value, (str, bytes, np.ndarray, type)):
                walk(value, sub, depth+1)
    walk(scene, type(scene).__name__, 0)
    for solver in scene.active_solvers:
        walk(solver, type(solver).__name__, 0)
    if not found or len({path for path, _ in found}) != len(found):
        raise ValueError('ambiguous solver field inventory')
    return found


def capture(session):
    import torch
    if session.sim_ns % 100_000_000 or session.policy.last_actions is None:
        raise ValueError('snapshots are taken at a 100-ms decision boundary after the first policy step')
    fields = session.solver_fields
    return dict(measured_ns=session.sim_ns, scene_step=int(session.world.scene.t),
                solver={path: field.to_numpy() for path, field in fields},
                session={name: copy.deepcopy(getattr(session, name)) for name in SESSION_FIELDS},
                policy_last_actions=np.array(session.policy.last_actions, copy=True),
                obstacles=session.obstacles.state() if getattr(session, 'obstacles', None) is not None else None,
                rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state().clone(),
                         torch_devices=[s.clone().cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []))


def restore(session, snapshot):
    import torch
    for path, field in session.solver_fields:
        value = snapshot['solver'].get(path)
        if value is not None:
            field.from_numpy(value)
    session.world.scene._t = snapshot['scene_step']
    for name, value in snapshot['session'].items():
        setattr(session, name, copy.deepcopy(value))
    session.policy.last_actions = np.array(snapshot['policy_last_actions'], copy=True)
    if snapshot.get('obstacles') is not None:
        session.obstacles.restore(snapshot['obstacles'])
    session.world.scene._visualizer.reset()
    rng = snapshot['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu'])
    if rng['torch_devices']:
        torch.cuda.set_rng_state_all(rng['torch_devices'])
