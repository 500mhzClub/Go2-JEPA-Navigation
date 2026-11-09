"""Layout families (docs/dynamic/PLAN.md P13 item 4): how a candidate layout's passages and pitch are drawn.

Every family stays on the cell grid (`grid.py`): axis-aligned walls between unconnected neighbours, the robot's start
cell `START` opening east, a pitch per layout. Each draws from the generator's one seeded stream (`random.Random`).
- **tree** (family A, every registered set so far): a random spanning tree at `grid.PITCH_M`, the draws exactly as
  `grid.random_spanning_tree`, so the registered sets regenerate byte for byte.
- **loops_rooms** (family B, development): a spanning tree, then `EXTRA_PASSAGES` more passages (loops), then
  `ROOMS` 2 x 2 rooms (all four inner passages open, so no inner wall stands), at a pitch drawn from `LOOPS_ROOMS_PITCHES_M`.
- **offices** (family C, held out for good): on a grid four rows high, a corridor along row 0 with rooms off both
  sides, each opening onto it through one door (sometimes a second, `SECOND_DOOR`). Below, the rooms are 2 x 1; above,
  each pair of columns is one 2 x 2 room or an outer 2 x 1 room with an inner one behind it. The pitch is drawn from
  `OFFICES_PITCHES_M`.
"""
from go2nav.mazes import grid

TREE, LOOPS_ROOMS, OFFICES = 'tree', 'loops_rooms', 'offices'
FAMILIES = (TREE, LOOPS_ROOMS, OFFICES)
LOOPS_ROOMS_PITCHES_M = (1.1, 1.3, 1.6)
EXTRA_PASSAGES, ROOMS = (2, 3), (1, 2)
OFFICES_PITCHES_M = (1.2, 1.4)
SECOND_DOOR, ONE_BIG_ROOM = .3, .5


def interior_edges(shape):
    """Every passage between neighbouring cells of the grid, sorted."""
    cells = set(shape.cells)
    return sorted(grid.edge(c, (c[0]+dx, c[1]+dy)) for c in cells for dx, dy in ((1, 0), (0, 1))
                  if (c[0]+dx, c[1]+dy) in cells)


def room(x, y):
    """The four inner passages of the 2 x 2 room whose lowest corner cell is (x, y)."""
    return {grid.edge((x, y), (x+1, y)), grid.edge((x, y+1), (x+1, y+1)),
            grid.edge((x, y), (x, y+1)), grid.edge((x+1, y), (x+1, y+1))}


def loops_rooms(rng, shape):
    links = set(grid.random_spanning_tree(rng, shape))
    pitch = rng.choice(LOOPS_ROOMS_PITCHES_M)
    extra = sorted(set(interior_edges(shape))-links)
    rng.shuffle(extra)
    links |= set(extra[:rng.choice(EXTRA_PASSAGES)])
    corners = [(x, y) for x in range(-1, shape.columns-2) for y in range(-1, shape.rows-2)
               if grid.START not in {(x, y), (x+1, y), (x, y+1), (x+1, y+1)}]
    rng.shuffle(corners)
    wanted, used = rng.choice(ROOMS), set()
    for x, y in corners:
        block = {(x, y), (x+1, y), (x, y+1), (x+1, y+1)}
        if len(used) == 4*wanted:
            break
        if not block & used:
            used |= block
            links |= room(x, y)
    return tuple(sorted(links)), pitch


def offices(rng, shape):
    if shape.rows != 4 or shape.columns % 2:
        raise ValueError('offices need a grid four rows high with an even number of columns')
    xs = list(range(-1, shape.columns-1))
    links = {grid.edge((x, 0), (x+1, 0)) for x in xs[:-1]}            # the corridor, START at its west end

    def doors(cells, across):
        """One door from these room cells to the cells `across` (dy away), sometimes two."""
        order = list(cells)
        rng.shuffle(order)
        count = 2 if rng.random() < SECOND_DOOR else 1
        return {grid.edge(c, (c[0], c[1]+across)) for c in order[:count]}

    for x in xs[::2]:
        links.add(grid.edge((x, -1), (x+1, -1)))                       # the room below the corridor
        links |= doors([(x, -1), (x+1, -1)], 1)
        if rng.random() < ONE_BIG_ROOM:
            links |= room(x, 1)                                       # one 2 x 2 room above
            links |= doors([(x, 1), (x+1, 1)], -1)
        else:                                                         # an outer room, an inner one behind it
            links.add(grid.edge((x, 1), (x+1, 1)))
            links.add(grid.edge((x, 2), (x+1, 2)))
            links |= doors([(x, 1), (x+1, 1)], -1)
            links |= doors([(x, 2), (x+1, 2)], -1)
    return tuple(sorted(links)), rng.choice(OFFICES_PITCHES_M)


def layout(kind, rng, shape=grid.SQUARE):
    """(passages, pitch_m) of one candidate of this family."""
    if kind == TREE:
        return grid.random_spanning_tree(rng, shape), grid.PITCH_M
    if kind == LOOPS_ROOMS:
        return loops_rooms(rng, shape)
    if kind == OFFICES:
        return offices(rng, shape)
    raise ValueError(f'unknown layout family {kind!r}; known: {FAMILIES}')
