"""Dynamics perturbations used in the stage-1 and stage-2 experiments, and the recording-mission stall stop.

**Uniform friction** (stage 1): friction mu on the collision floor and all 27 robot geometries, set after the build.
Genesis combines a contact pair by the maximum of its two coefficients, so the walls (1.0) are unaffected.

**Low-friction strips** (stage 2, the original's placement v2 with patches v3). Placement uses only the maze layout and
the episode's home and beacon: the route is the breadth-first shortest cell path from the home cell to the beacon
cell; its *straight* cells (not a junction, dead end, endpoint or corner) form segments, and strips span the full
corridor width inside a segment, at least 0.5 m from a neighbouring junction, dead end or endpoint, at least 1.5 m
long, drawn with a seeded generator until they cover 20-40 % of the route length (lengths and offsets on the 12.5-cm
floor grid). Physics: every policy step, the floor's friction ratio is mu_p, and each leg's four calf geometries get
mu_p while its foot sphere is over a strip (1.0 otherwise). Marked strips tint the floor cells inside them
(r, g, b) -> (0.45 r, 0.70 g, b), keeping their texture. On strip missions the speed guard is raised to 0.40 m/s.

**Stall stop** (recording missions only): stop once the base has stayed within 10 cm of where it was 60 s earlier for
the whole 60 s (sampled every 100 ms).
"""
from collections import deque
from dataclasses import dataclass
import hashlib

import numpy as np

PITCH_M = 1.3
MARGIN_M, CORNER_MARGIN_M = .5, 0.
MIN_STRIP_M = 1.5
COVERAGE = (.20, .40)
QUAD_M = .125
TINT = (.45, .70, 1.00)
STRIP_SPEED_LIMIT_M_S = .40
STALL_WINDOW_S, STALL_MIN_TRANSLATION_M, STALL_SAMPLE_S = 60., .10, .1
STALL_REASON = 'DEV_RECORDING_STALL_STOP'


@dataclass
class Dynamics:
    friction_mu: float = None     # uniform friction (stage 1)
    strips: list = None           # strip rectangles [x0, x1, y0, y1] (stage 2; from `place_strips`)
    strip_mu: float = None
    marked: bool = False
    stall_stop_s: float = None    # recording missions


# -- strip placement --------------------------------------------------------------------------------------------------

def placement_seed(set_name, maze, episode):
    """The fixed seed rule: sha256 of "patches-v2:<set>:<maze>:<episode>", first 8 bytes (big-endian)."""
    return int.from_bytes(hashlib.sha256(f'patches-v2:{set_name}:{int(maze)}:{int(episode)}'.encode()).digest()[:8], 'big')


def route_cells(spec, episode):
    """Shortest cell path from the home cell to the beacon cell (breadth-first, neighbours in sorted order)."""
    layout = spec['evaluation_layout']
    cells = {tuple(c) for c in layout['cells']}
    neighbours = {c: [] for c in cells}
    for a, b in layout['edges']:
        neighbours[tuple(a)].append(tuple(b))
        neighbours[tuple(b)].append(tuple(a))
    cell = lambda xy: tuple(int(round(v/PITCH_M)) for v in xy[:2])
    start, goal = cell(episode['home_se2_world']), cell(episode['beacon_xy_world'])
    previous, queue = {start: None}, deque([start])
    while queue:
        c = queue.popleft()
        for n in sorted(neighbours[c]):
            if n not in previous:
                previous[n] = c
                queue.append(n)
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = previous[c]
    return path[::-1]


def cell_kinds(spec, route):
    degree = {tuple(c): 0 for c in spec['evaluation_layout']['cells']}
    for a, b in spec['evaluation_layout']['edges']:
        degree[tuple(a)] += 1
        degree[tuple(b)] += 1
    kinds = []
    for i, c in enumerate(route):
        if degree[c] >= 3:
            kinds.append('junction')
        elif degree[c] == 1:
            kinds.append('dead_end')
        elif i in (0, len(route)-1):
            kinds.append('endpoint')
        else:
            into, out = (c[0]-route[i-1][0], c[1]-route[i-1][1]), (route[i+1][0]-c[0], route[i+1][1]-c[1])
            kinds.append('straight' if into == out else 'corner')
    return kinds


def straight_segments(route, kinds):
    """Usable intervals of straight runs: axis, lateral centre and [start, end] along the axis, after the margins."""
    runs, current = [], []
    for i, kind in enumerate(kinds):
        if kind == 'straight':
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    margin = lambda kind: CORNER_MARGIN_M if kind == 'corner' else MARGIN_M
    out = []
    for run in runs:
        first, last = route[run[0]], route[run[-1]]
        step = (route[run[0]+1][0]-first[0], route[run[0]+1][1]-first[1])
        axis = 0 if step[0] else 1
        sign = step[axis]
        lo_cell, hi_cell = (first, last) if sign > 0 else (last, first)
        lo_neighbour = kinds[run[0]-1] if sign > 0 else kinds[run[-1]+1]
        hi_neighbour = kinds[run[-1]+1] if sign > 0 else kinds[run[0]-1]
        start = (lo_cell[axis]-.5)*PITCH_M+margin(lo_neighbour)
        end = (hi_cell[axis]+.5)*PITCH_M-margin(hi_neighbour)
        if end-start >= MIN_STRIP_M-1e-9:
            out.append(dict(axis=axis, lateral_m=first[1-axis]*PITCH_M, start_m=start, end_m=end, cells=[list(route[i]) for i in run]))
    return out


def place_strips(spec, episode, seed):
    """The strip placement of an episode (see the module docstring): dict with `strips` ([x0, x1, y0, y1] each)."""
    route = route_cells(spec, episode)
    kinds = cell_kinds(spec, route)
    usable = straight_segments(route, kinds)
    route_m = len(route)*PITCH_M
    base = dict(route=[list(c) for c in route], cell_kinds=kinds, seed=int(seed), usable_segments=usable)
    if not usable:
        return dict(base, strips=[], coverage=0.)
    rng = np.random.default_rng(seed)
    strips, covered = [], 0.
    for i in rng.permutation(len(usable)):
        if covered >= COVERAGE[0]*route_m:
            break
        s = usable[int(i)]
        available = s['end_m']-s['start_m']
        length = min(available, max(MIN_STRIP_M, COVERAGE[1]*route_m-covered))
        if length < MIN_STRIP_M-1e-9 or covered+length > max(COVERAGE[1]*route_m, MIN_STRIP_M)+1e-9:
            continue
        length = np.floor(length/QUAD_M+1e-9)*QUAD_M
        slack = available-length
        offset = np.floor(rng.uniform(0, slack)/QUAD_M)*QUAD_M if slack > 0 else 0.
        start, half = s['start_m']+offset, PITCH_M/2
        rect = ([start, start+length, s['lateral_m']-half, s['lateral_m']+half] if s['axis'] == 0
                else [s['lateral_m']-half, s['lateral_m']+half, start, start+length])
        strips.append(dict(rect=rect, length_m=float(length), segment=s['cells']))
        covered += length
    return dict(base, strips=strips, coverage=covered/route_m)


def on_strip(xy, rects):
    xy = np.asarray(xy, float).reshape(-1, 2)
    if not rects:
        return np.zeros(len(xy), bool)
    r = np.asarray(rects, float)
    return np.any((xy[:, None, 0] >= r[None, :, 0])&(xy[:, None, 0] <= r[None, :, 1])
                  & (xy[:, None, 1] >= r[None, :, 2])&(xy[:, None, 1] <= r[None, :, 3]), axis=1)


def tint_floor(mesh, rects):
    """A copy of the floor mesh with the cells inside the strips tinted (their own variation kept)."""
    vertices = np.asarray(mesh.vertices)
    inside = on_strip(vertices.reshape(-1, 4, 3).mean(axis=1)[:, :2], rects)
    colours = np.asarray(mesh.visual.vertex_colors).copy()
    mask = np.repeat(inside, 4)
    colours[mask, :3] = np.rint(colours[mask, :3].astype(float)*np.asarray(TINT)).astype(np.uint8)
    marked = mesh.copy()
    marked.visual.vertex_colors = colours
    return marked, int(inside.sum())


# -- physics ----------------------------------------------------------------------------------------------------------

class FrictionField:
    """Per-leg friction ratios over the strips, re-applied before every policy step."""

    def __init__(self, robot, floor, rects, mu):
        self.rects, self.mu = rects, float(mu)
        self.solver = robot._solver
        self.feet = [g for g in robot.geoms if g.link.name.endswith('_calf') and 'SPHERE' in repr(g.type)]
        self.legs = [[int(h.idx) for h in robot.geoms if h.link.name == g.link.name] for g in self.feet]
        self.floor = [int(g.idx) for g in floor.geoms]
        if len(self.feet) != 4 or any(len(leg) != 4 for leg in self.legs) or len(self.floor) != 1:
            raise ValueError('four foot spheres on four calf links and one collision floor required')
        self.ids = [i for leg in self.legs for i in leg]+self.floor

    def foot_xy(self):
        def xy(geom):
            p = geom.get_pos()
            return np.asarray(p.detach().cpu().numpy() if hasattr(p, 'detach') else p).reshape(-1)[:2]
        return np.stack([xy(g) for g in self.feet])

    def update(self):
        on = on_strip(self.foot_xy(), self.rects)
        self.solver.set_geoms_friction_ratio(np.asarray([self.mu if on[k] else 1. for k, leg in enumerate(self.legs) for _ in leg]
                                                        + [self.mu]*len(self.floor)), self.ids)
        return on


class StallMonitor:
    """True once the base stayed within 10 cm of its window-start position for the whole trailing 60 s."""

    def __init__(self, window_s=STALL_WINDOW_S, min_translation_m=STALL_MIN_TRANSLATION_M):
        self.window_s, self.min_translation_m = float(window_s), float(min_translation_m)
        self.track, self.stopped_at = deque(), None

    def update(self, t, xy):
        if self.track and t-self.track[-1][0] < STALL_SAMPLE_S-1e-9:
            return False
        self.track.append((float(t), np.asarray(xy, float)[:2].copy()))
        while len(self.track) > 1 and t-self.track[1][0] >= self.window_s:
            self.track.popleft()
        start_t, start_xy = self.track[0]
        if t-start_t < self.window_s-1e-9:
            return False
        if max(float(np.linalg.norm(p-start_xy)) for _, p in self.track) < self.min_translation_m:
            self.stopped_at = float(t)
            return True
        return False
