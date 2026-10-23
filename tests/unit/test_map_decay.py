import numpy as np

from go2nav.perception import mapping

FLOOR = 0.


def index(cell):
    return (cell[0]+mapping.HALF_CELLS)*2*mapping.HALF_CELLS+cell[1]+mapping.HALF_CELLS


def update(routing_map, monkeypatch, floor_cells, points):
    """One map update at the identity pose, the primary image seeing `floor_cells` as floor and `points` (body frame)
    above it; the auxiliary image sees nothing."""
    covered = np.zeros(len(mapping.GRID), bool)
    covered[[index(c) for c in floor_cells]] = True
    images = iter([(covered, np.asarray(points, float).reshape(-1, 3)), (np.zeros_like(covered), np.zeros((0, 3)))])
    current = {}

    def coverage(*args):
        current['image'] = next(images)
        return current['image'][0]
    monkeypatch.setattr(mapping, 'floor_coverage', coverage)
    monkeypatch.setattr(mapping, 'sampled_points', lambda *args: current['image'][1])
    monkeypatch.setattr(mapping.floor, 'paired_floor_points', lambda *args: ((None, None),))
    monkeypatch.setattr(mapping.floor, 'joint_plane', lambda *args: dict(available=True))
    return routing_map.update(4, 0, np.zeros(3), np.eye(3), (None, None), (None, None), None)


def started(clear):
    routing_map = mapping.RoutingMap(clear_seen_free=clear)
    routing_map.B, routing_map.floor_height = np.eye(3), FLOOR
    return routing_map


def test_cells_seen_free_are_cleared_only_when_enabled(monkeypatch):
    obstacle = [(1.025, .025, .3)]                              # in cell (20, 0)
    for clear in (False, True):
        routing_map = started(clear)
        first = update(routing_map, monkeypatch, [], obstacle)
        assert (20, 0) in first.occupied and (102, 2) in first.fine_occupied
        second = update(routing_map, monkeypatch, [(20, 0)], [])  # the obstacle has gone: the cell is seen as floor
        assert ((20, 0) in second.occupied) is not clear
        assert ((102, 2) in second.fine_occupied) is not clear
        assert second.cleared == (frozenset({(20, 0)}) if clear else frozenset())


def test_a_cell_still_holding_samples_is_kept(monkeypatch):
    routing_map = started(True)
    update(routing_map, monkeypatch, [], [(1.025, .025, .3), (1.5, .5, .3)])
    snapshot = update(routing_map, monkeypatch, [(20, 0), (30, 10)], [(1.025, .025, .3)])
    assert (20, 0) in snapshot.occupied and (30, 10) not in snapshot.occupied
    assert snapshot.cleared == frozenset({(30, 10)})
