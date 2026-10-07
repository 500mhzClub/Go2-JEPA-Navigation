"""Routing on the observed map: where to go next, and which views to take to extend the map.

**Route** (`propose`). Only observed floor is traversable, minus cells within 0.45 m of an observed obstacle
(`grid.inflated_cells`). The robot connects to the nearest such cell within 1.25 m whose straight connector keeps the
0.45-m disk clear of the 1-cm obstacle cells. A breadth-first search over 4-connected cells then reaches the goal's
cell, or else the *frontier* cell (one next to an unobserved cell) minimising distance to the goal plus 0.1 x path
length. The path is then re-planned as a clearance-weighted A* (`grid.clearance_costs`) between the same ends, and,
when the goal cell is observed floor but not reachable this way, by an A* whose edges are checked directly against
the 1-cm cells (`fine_goal_route`).

**Look-ahead target** (`clear_route_target`): the furthest route point, up to 0.35 m away, whose straight connector
keeps the clearance the robot has now (at most 0.48 m).

**Views.** A frontier is approached until one of the cameras can see its unobserved neighbour: a viewpoint on the
route (or nearby observed floor) from which that cell projects fully into a camera image with no observed obstacle
in between (`CameraFrontierVisits`). The robot routes to the viewpoint, turns to face the cell, and re-plans once the
cell is observed (or tries another viewpoint). The same mechanism covers floor that a translation would sweep without
it being observed (`CoverageViews`). At the start, the robot turns through eight 45° views (`InitialPanorama`).
"""
from collections import deque
import math

import numpy as np

from go2nav.perception import cameras
from go2nav.planning import grid
from go2nav.planning.grid import NEIGHBOURS, centre, segment_cells

CONNECT_RADIUS_M = 1.25
LOOKAHEAD_M, LOOKAHEAD_CLEARANCE_M = .35, .48
VIEW_ALIGNMENT_RAD = .10
VIEWPOINT_RADIUS_M = .10


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def propose(floor, occupied, fine_occupied, position, goal, *, excluded_frontiers=(), radius_m=grid.FOOTPRINT_RADIUS_M):
    """The base route from a position to a goal (map frame): connector and breadth-first search (see the module
    docstring); `prefer_clearance` then re-plans its cells."""
    clearance = grid.fine_clearance(fine_occupied)
    floor, occupied = set(floor), set(occupied)
    p, g = np.asarray(position, float), np.asarray(goal, float)
    blocked = grid.inflated_cells(occupied, radius_m)
    available = floor-blocked
    candidates = sorted((c for c in available if np.linalg.norm(centre(c)-p) <= CONNECT_RADIUS_M),
                        key=lambda c: (float(np.linalg.norm(centre(c)-p)), c))
    if not clearance.clear(p, p, radius_m):
        candidates = []  # the start disk already touches an obstacle; no connector can clear it
    seed = connector = None
    for c in candidates:
        if clearance.clear(p, centre(c), radius_m):
            seed, connector = c, segment_cells(p, centre(c))
            break
    common = dict(nominal_radius_m=radius_m, goal_map_xy_m=g.tolist())
    if seed is None:
        return common | dict(status='ADDITIONAL_VIEW_REQUIRED', route_cells=[], unknown_connector_cells=[],
                             complete_route_floor_coverage=False)
    parent, queue, distance = {seed: None}, deque([seed]), {seed: 0}
    while queue:
        cell = queue.popleft()
        for dx, dy in NEIGHBOURS:
            nxt = (cell[0]+dx, cell[1]+dy)
            if nxt in available and nxt not in parent:
                parent[nxt], distance[nxt] = cell, distance[cell]+1
                queue.append(nxt)
    goal_cell = grid.cell_of(g)
    observed = floor|occupied
    frontier = [c for c in parent if c not in excluded_frontiers
                and any((c[0]+dx, c[1]+dy) not in observed for dx, dy in NEIGHBOURS)]
    if goal_cell in parent:
        target, status = goal_cell, 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    elif frontier:
        target = min(frontier, key=lambda c: (float(np.linalg.norm(centre(c)-g))+.1*grid.CELL_M*distance[c], c))
        status = 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    else:
        return common | dict(status='OBSERVED_COMPONENT_HAS_NO_FRONTIER', route_cells=[],
                             unknown_connector_cells=sorted(connector-floor), complete_route_floor_coverage=False)
    path, here = [], target
    while here is not None:
        path.append(here)
        here = parent[here]
    path.reverse()
    unknown = connector-floor
    return common | dict(status=status, route_cells=[list(c) for c in path], entry_map_xy_m=centre(seed).tolist(),
                         target_map_xy_m=centre(target).tolist(), unknown_connector_cells=[list(c) for c in sorted(unknown)],
                         complete_route_floor_coverage=not unknown, initial_connector_requires_observation=bool(unknown))


def prefer_clearance(route, floor, occupied):
    """Re-plan a route's cells, between the same ends, as a clearance-weighted A* over the uninflated floor."""
    cells = route['route_cells']
    if not cells:
        return route
    available = set(floor)-grid.inflated_cells(occupied, route['nominal_radius_m'])
    start, target = tuple(cells[0]), tuple(cells[-1])
    if start not in available or target not in available:
        raise ValueError('same observed traversable entry and target required')
    found = grid.astar(start, target, lambda c: c in available, grid.clearance_costs(frozenset(occupied)))
    if found is None:
        raise ValueError('original reachable target became unreachable')
    return route | dict(route_cells=[list(c) for c in found[0]])


def fine_goal_route(snapshot, position, goal, route):
    """When the goal cell is observed floor but the inflated grid does not reach it: an A* over observed floor whose
    edges (and the connectors from the robot and to the goal) keep the 0.45-m disk clear of the 1-cm cells."""
    if route['status'] == 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':
        return route
    target = grid.cell_of(goal)
    if target not in snapshot.floor:
        return route
    clearance = grid.fine_clearance(snapshot.fine_occupied)
    radius = route['nominal_radius_m']
    if not clearance.clear(position, position, radius) or not clearance.clear(centre(target), goal, radius):
        return route
    seed = next((c for c in _nearby_floor(snapshot.floor, position) if clearance.clear(position, centre(c), radius)), None)
    if seed is None:
        return route
    found = _fine_search(snapshot.floor, snapshot.occupied, snapshot.fine_occupied, radius, seed, target)
    if found is None:
        return route
    path, _, _ = found
    unknown = segment_cells(position, centre(seed))-snapshot.floor
    return route | dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL', route_cells=[list(c) for c in path],
                        entry_map_xy_m=centre(seed).tolist(), target_map_xy_m=centre(target).tolist(),
                        unknown_connector_cells=[list(c) for c in sorted(unknown)], complete_route_floor_coverage=not unknown,
                        initial_connector_requires_observation=bool(unknown), fine_goal_route=True)


def _nearby_floor(floor, position):
    """Floor cells within 1.25 m of a position, nearest first (ties by cell)."""
    position = np.asarray(position, float)
    rows = [(float(np.linalg.norm(centre(c)-position)), c) for c in floor]
    return [c for d, c in sorted(rows) if d <= CONNECT_RADIUS_M]


_FINE_SEARCHES = {}


def _fine_search(floor, occupied, fine_occupied, radius, seed, target):
    key = (floor, occupied, fine_occupied, radius, seed, target)
    if key not in _FINE_SEARCHES:
        if len(_FINE_SEARCHES) >= 8:
            _FINE_SEARCHES.pop(next(iter(_FINE_SEARCHES)))
        axis = grid.axis_clearance(fine_occupied)

        def edge_clear(a, b):
            d = axis.minimum(a, b)
            return d is None or d > radius+1e-12
        found = grid.astar(seed, target, lambda c: c in floor, grid.clearance_costs(occupied), edge_clear)
        _FINE_SEARCHES[key] = None if found is None else (tuple(found[0]), found[1], found[2])
    return _FINE_SEARCHES[key]


def clear_route_target(points, position, fine_occupied, clearance=None):
    """The look-ahead target on a route (see the module docstring). `clearance` replaces the stored-obstacle clearance
    (the cautious variants of `planning.caution`)."""
    position = np.asarray(position, float)
    points = [np.asarray(p, float) for p in points]
    clearance = clearance or grid.fine_clearance(fine_occupied)

    def clear_to(target):
        value = clearance.minimum(position, target)
        return value if value is not None else float('inf')
    required = min(LOOKAHEAD_CLEARANCE_M, clear_to(position))
    target = position.copy()
    for point in points:
        if clear_to(point)+1e-12 < required:
            break
        target = point
        if np.linalg.norm(point-position) >= LOOKAHEAD_M:
            break
    return target


# -- camera views ----------------------------------------------------------------------------------------------------

def floor_cell_projection(cell, position_map, rotation_map_from_body, floor_height):
    """Per camera: whether a floor cell's square projects fully into its image (depth 0.2–5 m), and the camera's
    origin (map xy)."""
    p, R = np.asarray(position_map, float), np.asarray(rotation_map_from_body, float)
    xy = (np.asarray(cell, int)+np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))*grid.CELL_M
    world = np.c_[xy, np.full(4, floor_height)]
    rows = []
    for mount in (cameras.PRIMARY, cameras.AUXILIARY):
        camera = ((world-p)@R-mount[:3, 3])@mount[:3, :3]
        z = camera[:, 2]
        uv = camera[:, :2]/np.maximum(z[:, None], 1e-12)*cameras.FOCAL+[319.5, 239.5]
        lo, hi = uv.min(0)-1e-9, uv.max(0)+1e-9
        visible = bool(np.all((z >= .2)&(z <= 5.)) and np.all(lo >= 0) and np.all(hi < [639, 479]))
        rows.append(dict(fully_projected=visible, camera_origin_map_xy_m=(p+R@mount[:3, 3])[:2]))
    return rows


def directed_rotation(rotation, position_xy, cell):
    """The rotation turned (about z) to face a cell's centre, and that heading."""
    R = np.asarray(rotation, float)
    direction = centre(cell)-position_xy
    heading = math.atan2(direction[1], direction[0])
    delta = heading-math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(delta), math.sin(delta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])@R, heading


def _seeing_cameras(snapshot, cell, position, rotation):
    return [row for row in floor_cell_projection(cell, position, rotation, snapshot.floor_height)
            if row['fully_projected'] and not (segment_cells(row['camera_origin_map_xy_m'], centre(cell))&snapshot.occupied)]


def route_viewpoint(snapshot, route, position, rotation, unknown_cell, *, excluded=()):
    """The route cell nearest its end from which a camera, facing the unobserved cell, would see it."""
    if route['status'] != 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER' or tuple(unknown_cell) in snapshot.floor|snapshot.occupied:
        return None
    p = np.asarray(position, float)
    for index in range(len(route['route_cells'])-1, -1, -1):
        cell = tuple(route['route_cells'][index])
        if cell in excluded:
            continue
        candidate = np.r_[centre(cell), p[2]]
        R, heading = directed_rotation(rotation, candidate[:2], unknown_cell)
        if _seeing_cameras(snapshot, unknown_cell, candidate, R):
            return dict(viewpoint_cell=list(cell), viewpoint_map_xy_m=candidate[:2].tolist(),
                        route_cells=route['route_cells'][:index+1], view_heading_rad=heading)
    return None


class CameraFrontierVisits:
    """Visits to frontier (or requested) cells until a camera has seen them (see the module docstring).

    `advance` returns ('route', None) to follow the (possibly redirected) route, ('view', visit) to turn to the visit's
    heading, or ('replan', None) when the visit ended."""

    def __init__(self):
        self.excluded = set()
        self.visit = None
        self.attempted = {}
        self.floor_count = None
        self.context = None  # (position, rotation) in the map frame

    def _finish(self, reason):
        self.visit = None

    def _choose(self, snapshot, route, p, R, unknown):
        tried = self.attempted.get(unknown, set())
        view = route_viewpoint(snapshot, route, p, R, unknown, excluded=tried)
        if view is not None:
            return view
        # The robot may be inside the cameras' near blind region: try nearby observed floor reachable by a route.
        available = snapshot.floor-grid.inflated_cells(snapshot.occupied)
        candidates = sorted((c for c in available if c not in tried and np.linalg.norm(centre(c)-centre(unknown)) <= 1.25),
                            key=lambda c: (float(np.linalg.norm(centre(c)-p[:2])), c))[:128]
        for cell in candidates:
            view = route_viewpoint(snapshot, route | dict(route_cells=[list(cell)]), p, R, unknown)
            if view is None:
                continue
            path = propose(snapshot.floor, snapshot.occupied, snapshot.fine_occupied, p[:2], centre(cell))
            if path['status'] != 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':
                continue
            return view | dict(route_cells=path['route_cells'])
        return None

    def _refresh(self, snapshot):
        if self.floor_count != (len(snapshot.floor), len(snapshot.occupied)):
            self.excluded.clear()  # new map evidence ends temporary exclusions
            self.floor_count = (len(snapshot.floor), len(snapshot.occupied))

    def advance(self, snapshot, position, heading, goal, now_ns, route):
        self._refresh(snapshot)
        p, R = self.context
        observed = snapshot.floor|snapshot.occupied
        if self.visit is not None:
            if tuple(self.visit['unknown_neighbour']) in observed:
                self._finish('REQUESTED_PATCH_OBSERVED')
                return 'replan', None
            if route['status'] == 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':
                self._finish('MISSION_GOAL_ROUTE_AVAILABLE')
                return 'route', None
        if self.visit is None:
            if route['status'] != 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER':
                return 'route', None
            target = tuple(route['route_cells'][-1])
            blocked = grid.inflated_cells(snapshot.occupied)
            unknowns = [(target[0]+dx, target[1]+dy) for dx, dy in NEIGHBOURS if (target[0]+dx, target[1]+dy) not in observed]
            unknowns.sort(key=lambda c: (c in blocked, float(np.linalg.norm(centre(c)-goal)), c))
            view = None
            for unknown in unknowns:
                view = self._choose(snapshot, route, p, R, unknown)
                if view is not None:
                    break
            if view is None:
                self.excluded.add(target)  # until new map evidence arrives
                return 'replan', None
            self.visit = dict(unknown_neighbour=list(unknown), started_ns=now_ns, aligned_ns=None,
                              heading_rad=view['view_heading_rad'], camera_viewpoint=view)
        v = self.visit
        unknown = tuple(v['unknown_neighbour'])
        waypoint = np.asarray(v['camera_viewpoint']['viewpoint_map_xy_m'])
        planned_R, desired = directed_rotation(R, p[:2], unknown)
        projected = floor_cell_projection(unknown, p, planned_R, snapshot.floor_height)
        if np.linalg.norm(position-waypoint) <= VIEWPOINT_RADIUS_M and any(row['fully_projected'] for row in projected):
            v['heading_rad'] = desired
            if abs(wrap(desired-heading)) <= VIEW_ALIGNMENT_RAD:
                actual = floor_cell_projection(unknown, p, R, snapshot.floor_height)
                if any(row['fully_projected'] for row in actual):
                    if v['aligned_ns'] is None:
                        v['aligned_ns'] = now_ns
                    if snapshot.measured_ns >= v['aligned_ns']:
                        # A fresh mapped view left the patch unknown: try another position.
                        tried = self.attempted.setdefault(unknown, set())
                        tried.update(c for c in snapshot.floor if np.linalg.norm(centre(c)-position) <= VIEWPOINT_RADIUS_M)
                        self._finish('FRESH_VIEW_PATCH_STILL_UNKNOWN')
                        return 'replan', None
                else:
                    v['aligned_ns'] = None
            else:
                v['aligned_ns'] = None
            return 'view', dict(v)
        v['aligned_ns'] = None
        path = propose(snapshot.floor, snapshot.occupied, snapshot.fine_occupied, p[:2], waypoint)
        if path['status'] != 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':
            self._finish('VIEWPOINT_ROUTE_NO_LONGER_AVAILABLE')
            return 'replan', None
        route.update(route_cells=path['route_cells'], target_map_xy_m=waypoint.tolist(),
                     status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER', entry_map_xy_m=path['entry_map_xy_m'],
                     unknown_connector_cells=path['unknown_connector_cells'],
                     complete_route_floor_coverage=path['complete_route_floor_coverage'],
                     initial_connector_requires_observation=path['initial_connector_requires_observation'])
        return 'route', None


class CommittedFrontierVisits(CameraFrontierVisits):
    """Frontier visits whose viewing turn, once started, is completed before the patch is re-assessed."""

    def advance(self, snapshot, position, heading, goal, now_ns, route):
        self._refresh(snapshot)
        if self.visit is None or not self.visit.get('directed_view_committed'):
            action, receipt = super().advance(snapshot, position, heading, goal, now_ns, route)
            if action == 'view':
                self.visit.update(directed_view_committed=True)
                receipt = dict(self.visit)
            return action, receipt
        v = self.visit
        unknown = tuple(v['unknown_neighbour'])
        if unknown in snapshot.floor|snapshot.occupied:
            self._finish('REQUESTED_PATCH_OBSERVED')
            return 'replan', None
        if route['status'] == 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':
            self._finish('MISSION_GOAL_ROUTE_AVAILABLE')
            return 'route', None
        p, R = self.context
        _, desired = directed_rotation(R, p[:2], unknown)
        v['heading_rad'] = desired
        if abs(wrap(desired-heading)) > VIEW_ALIGNMENT_RAD:
            v['aligned_ns'] = None
            return 'view', dict(v)
        visible = any(r['fully_projected'] for r in floor_cell_projection(unknown, p, R, snapshot.floor_height))
        if visible:
            if v['aligned_ns'] is None:
                v['aligned_ns'] = now_ns
            if snapshot.measured_ns < v['aligned_ns']:
                return 'view', dict(v)
        # Seen (or unseeable) from here yet still unobserved: try another viewpoint.
        tried = self.attempted.setdefault(unknown, set())
        tried.add(tuple(v['camera_viewpoint']['viewpoint_cell']))
        tried.update(c for c in snapshot.floor if np.linalg.norm(centre(c)-position) <= VIEWPOINT_RADIUS_M)
        self._finish('FRESH_VIEW_PATCH_STILL_UNKNOWN' if visible else 'ALIGNED_VIEW_PATCH_OUTSIDE_IMAGE')
        return 'replan', None


class CoverageViews(CameraFrontierVisits):
    """Views of floor a translation would sweep unobserved; taken from where the robot stands when possible."""

    def __init__(self):
        super().__init__()
        self.failed_current_positions = {}

    def _choose(self, snapshot, route, p, R, unknown):
        p = np.asarray(p, float)
        attempted = self.failed_current_positions.get(unknown, [])
        if unknown not in snapshot.floor|snapshot.occupied and not any(np.linalg.norm(p[:2]-old) <= .10 for old in attempted):
            directed, heading = directed_rotation(R, p[:2], unknown)
            if _seeing_cameras(snapshot, unknown, p, directed):
                return dict(viewpoint_cell=np.floor(p[:2]/grid.CELL_M).astype(int).tolist(), viewpoint_map_xy_m=p[:2].tolist(),
                            route_cells=[], view_heading_rad=heading, current_measured_position_view=True)
        return super()._choose(snapshot, route, p, R, unknown)

    def _finish(self, reason):
        if reason == 'FRESH_VIEW_PATCH_STILL_UNKNOWN' and self.visit['camera_viewpoint'].get('current_measured_position_view'):
            self.failed_current_positions.setdefault(tuple(self.visit['unknown_neighbour']), []).append(
                np.asarray(self.context[0][:2]).copy())
        super()._finish(reason)


class InitialPanorama:
    """Eight 45° turns at the start (nine headings), each completed when a map update follows its alignment."""

    def __init__(self):
        self.state = None

    @property
    def complete(self):
        return self.state is not None and self.state['complete']

    def advance(self, snapshot, position, heading, now_ns):
        if self.state is None:
            headings = [wrap(heading+i*math.pi/4) for i in range(9)]
            self.state = dict(headings_rad=headings, view_index=0, heading_rad=headings[0], aligned_ns=None, complete=False)
        v = self.state
        if not v['complete'] and abs(wrap(v['heading_rad']-heading)) <= .1:
            if v['aligned_ns'] is None:
                v['aligned_ns'] = now_ns
            if snapshot.measured_ns >= v['aligned_ns']:
                if v['view_index'] == 8:
                    v['complete'] = True
                else:
                    v['view_index'] += 1
                    v['heading_rad'] = v['headings_rad'][v['view_index']]
                    v['aligned_ns'] = None
        return dict(v)
