"""The maze family: a 4 x 4 grid of square cells joined by a random spanning tree.

Cells are integer coordinates (x, y) with x, y in -1..2. Neighbouring cells are joined by a passage or separated by a
wall; the outer boundary is always walled. The robot starts in cell START = (-1, 0), which always opens east into
(0, 0). The world position of a cell's centre is `cell * PITCH_M`.

A layout is the set of passages ("links"): 15 undirected edges forming a spanning tree, so every cell is reachable by
exactly one route.

Other grid sizes (`Shape`: columns along x, rows along y; cells x in -1..columns-2, y in -1..rows-2) keep the same
pitch, walls and start cell. Every registered evaluation set is 4 x 4 (`SQUARE`, the default everywhere); a training
set on another shape can never repeat an evaluation maze's topology or embedding (different cell count), whether or
not that maze is listed in an exclusion registry.
"""
from collections import deque
from dataclasses import dataclass

PITCH_M = 1.3
WALL_THICKNESS_M = 0.08
WALL_HEIGHT_M = 1.4
START = (-1, 0)
CELLS = tuple((x, y) for x in range(-1, 3) for y in range(-1, 3))
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
WORLD_BOUNDS_XY_M = ((-2.1, -2.1), (3.4, 3.4))
BOUNDS_MARGIN_M = .8  # world bounds: the outermost cell centres plus this margin


@dataclass(frozen=True)
class Shape:
    columns: int = 4
    rows: int = 4

    @property
    def cells(self):
        return tuple((x, y) for x in range(-1, self.columns-1) for y in range(-1, self.rows-1))

    @property
    def world_bounds_xy_m(self):
        if self == SQUARE:
            return WORLD_BOUNDS_XY_M
        return ((-2.1, -2.1), tuple(round((n-2)*PITCH_M+BOUNDS_MARGIN_M, 6) for n in (self.columns, self.rows)))


SQUARE = Shape()


def edge(a, b):
    """The undirected passage between cells a and b, in canonical (sorted) order."""
    return tuple(sorted((a, b)))


def random_spanning_tree(rng, shape=SQUARE):
    """A random spanning tree over CELLS that always contains the START-(0, 0) passage (randomised Kruskal).

    The candidate passages are every grid edge not touching START, shuffled once by `rng` (a `random.Random`), then
    accepted whenever they join two different components. Returns the passages (15 on the 4 x 4 grid) as a sorted tuple.
    """
    cells = shape.cells
    parent = {c: c for c in cells}

    def root(c):
        while parent[c] != c:
            c = parent[c]
        return c

    links = {edge(START, (0, 0))}
    parent[root(START)] = root((0, 0))
    choices = [edge(c, (c[0]+dx, c[1]+dy)) for c in cells for dx, dy in ((1, 0), (0, 1))
               if (c[0]+dx, c[1]+dy) in parent and START not in (c, (c[0]+dx, c[1]+dy))]
    rng.shuffle(choices)
    for a, b in choices:
        if root(a) != root(b):
            links.add(edge(a, b))
            parent[root(a)] = root(b)
    if len(links) != len(cells)-1 or len({root(c) for c in cells}) != 1:
        raise ValueError('a connected spanning tree over all cells is required')
    return tuple(sorted(links))


def farthest_route(links, shape=SQUARE):
    """The route from START to the farthest cell (ties broken by the smallest cell), as a list of cells.

    Breadth-first search over the passages, visiting neighbours in sorted order. This is the evaluator's task
    definition; it is never a controller input.
    """
    cells = shape.cells
    adjacency = {c: [] for c in cells}
    for a, b in links:
        adjacency[a].append(b)
        adjacency[b].append(a)
    parent, distance, queue = {START: None}, {START: 0}, deque([START])
    while queue:
        c = queue.popleft()
        for other in sorted(adjacency[c]):
            if other not in parent:
                parent[other], distance[other] = c, distance[c]+1
                queue.append(other)
    if len(parent) != len(cells):
        raise ValueError('every cell must be reachable')
    here = min(cells, key=lambda c: (-distance[c], c))
    route = []
    while here is not None:
        route.append(here)
        here = parent[here]
    return route[::-1]


def route_turns(route):
    """Number of direction changes along a cell route."""
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(route, route[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def wall_boxes(links, *, wall_id_prefix, shape=SQUARE):
    """Axis-aligned wall boxes for every unconnected neighbour pair, including the outer boundary.

    A wall sits at the midpoint between two cell centres. Its key (2x+dx, 2y+dy, |dx|) is shared by both cells, so each
    wall appears once. Boxes are returned sorted by key; each is a dict with wall_id, centre_xyz, size_xyz, yaw_rad and
    material_id.
    """
    walls = {}
    for x, y in shape.cells:
        for dx, dy in NEIGHBOURS:
            if edge((x, y), (x+dx, y+dy)) in links:
                continue
            mx, my = 2*x+dx, 2*y+dy
            key = (mx, my, abs(dx))
            walls[key] = dict(
                wall_id=f'{wall_id_prefix}_{mx}_{my}_{abs(dx)}',
                centre_xyz=[mx*PITCH_M/2, my*PITCH_M/2, WALL_HEIGHT_M/2],
                size_xyz=([WALL_THICKNESS_M, PITCH_M+WALL_THICKNESS_M, WALL_HEIGHT_M] if dx
                          else [PITCH_M+WALL_THICKNESS_M, WALL_THICKNESS_M, WALL_HEIGHT_M]),
                yaw_rad=0., material_id='NEUTRAL_WALL')
    return [walls[k] for k in sorted(walls)]
