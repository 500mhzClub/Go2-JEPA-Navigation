"""Generate maze layouts and episodes.

A **layout** is a random spanning tree on the grid (`grid.py`). Candidates are drawn from one seeded stream and accepted
in order when:
- the route from START to the farthest cell has at least MIN_ROUTE_CELLS cells and at least MIN_ROUTE_TURNS turns;
- its topology and its metric embedding (`identity.py`) differ from every excluded (earlier) maze;
- and from every layout already accepted in this run.

An **episode** is a home pose and a beacon position in one layout, chosen from a seeded permutation of ordered
(home cell, beacon cell) pairs. The first pair is used where:
- both ends are at least `min_endpoint_clearance_m` from any wall;
- a wall blocks the straight line between them;
- the beacon is more than `min_geodesic_m` away along the inflated-footprint shortest path (`reference.py`).
The home heading is then drawn uniformly. Every controller gets identical episodes.

The specification and episode dicts reproduce the registered set files exactly (key order included), so regenerating a
set with its recorded seeds reproduces its files byte for byte.
"""
from collections import Counter
from dataclasses import dataclass, field
import math
import random

import numpy as np

from go2nav.mazes import grid
from go2nav.mazes.identity import layout_identities
from go2nav.mazes.reference import ReferenceGeometry, segment_crosses_wall, walls_from_boxes

MIN_ROUTE_CELLS = 7
MIN_ROUTE_TURNS = 2
MAXIMUM_CANDIDATES = 10_000


@dataclass(frozen=True)
class Seeds:
    construction: int     # the candidate stream
    physics_base: int     # spec procedural_seed = physics_base + layout index
    appearance_base: int  # spec appearance_seed = appearance_base + layout index
    episode_base: int     # episode seed = episode_base + 2 * layout index + episode index


@dataclass(frozen=True)
class EpisodeRules:
    min_geodesic_m: float = 5.2
    min_endpoint_clearance_m: float = 0.5
    inflation_radius_m: float = 0.46
    additional_clearance_m: float = 0.005
    grid_resolution_m: float = 0.02


@dataclass(frozen=True)
class Role:
    """A block of consecutive layout indices sharing a role."""
    name: str
    count: int
    episodes: tuple = (0, 1)                        # episode indices built for each layout
    spec_fields: dict = field(default_factory=dict)  # extra specification fields (e.g. recording_split)


def identities(links):
    return layout_identities(links, pitch_m=grid.PITCH_M, wall_thickness_m=grid.WALL_THICKNESS_M,
                             wall_height_m=grid.WALL_HEIGHT_M)


def accepted_layouts(construction_seed, excluded_graphs, rejections, *, maximum_candidates=MAXIMUM_CANDIDATES):
    """Yield every layout the acceptance rules admit, in candidate order, as (candidate_index, links, identity).

    `excluded_graphs` is a list of {'name', 'edges'} for every earlier maze. Each rejected candidate is appended to
    `rejections` as {'candidate_index', 'reason', <identities>}.
    """
    previous = [identities(g['edges']) for g in excluded_graphs]
    old_topology = {p['abstract_topology_code'] for p in previous}
    old_metric = {p['metric_code'] for p in previous}
    topology, metric = set(), set()
    rng = random.Random(construction_seed)
    for number in range(maximum_candidates):
        links = grid.random_spanning_tree(rng)
        identity = identities(links)
        route = grid.farthest_route(links)
        if len(route) < MIN_ROUTE_CELLS or grid.route_turns(route) < MIN_ROUTE_TURNS:
            reason = 'insufficient_declared_route_structure'
        elif identity['abstract_topology_code'] in old_topology:
            reason = 'prior_abstract_topology'
        elif identity['metric_code'] in old_metric:
            reason = 'prior_grid_embedding'
        elif identity['abstract_topology_code'] in topology:
            reason = 'repeated_abstract_topology'
        elif identity['metric_code'] in metric:
            reason = 'repeated_grid_embedding'
        else:
            reason = None
        if reason:
            rejections.append(dict(candidate_index=number, reason=reason, **identity))
            continue
        topology.add(identity['abstract_topology_code'])
        metric.add(identity['metric_code'])
        yield number, links, identity


def make_spec(index, links, identity, candidate_index, *, seeds, scene_id_prefix, family, role,
              wall_id_prefix='independent_round_trip_wall', spec_fields=None):
    """The maze specification (schema v1: the format of every registered set)."""
    route = grid.farthest_route(links)
    spec = dict(
        scene_id=f'{scene_id_prefix}-{index:02d}', layout_index=index, family=family, data_role=role,
        procedural_seed=seeds.physics_base+index, appearance_seed=seeds.appearance_base+index,
        appearance_arm='distinctive', friction_mu=1., render_near_m=.005,
        visual_surface_contract='floor_first_variable_height_union_walls',
        geometry=dict(spawn_se2_world=[grid.START[0]*grid.PITCH_M, grid.START[1]*grid.PITCH_M, 0.],
                      wall_boxes=grid.wall_boxes(links, wall_id_prefix=wall_id_prefix)),
        evaluation_layout=dict(cells=[list(c) for c in grid.CELLS], edges=[[list(a), list(b)] for a, b in links],
                               goal_cell=list(route[-1]), shortest_outbound_route=[list(c) for c in route],
                               shortest_outbound_distance_m=(len(route)-1)*grid.PITCH_M, pitch_m=grid.PITCH_M, **identity),
        construction_candidate_index=candidate_index, native_execution=False, final_evaluation=False,
        navigation_qualified=False)
    return spec | (spec_fields or {})


def make_episode(spec, episode_index, *, seeds, rules=EpisodeRules()):
    """The episode packet: home pose, beacon, the mission instruction in the initial body frame, and the seeds."""
    index = spec['layout_index']
    seed = seeds.episode_base+2*index+episode_index
    rng = random.Random(seed)
    cells = [list(c) for c in grid.CELLS]
    pairs = [(a, b) for a in cells for b in cells if a != b]
    rng.shuffle(pairs)
    walls = walls_from_boxes(spec['geometry']['wall_boxes'])
    for a, b in pairs:
        start, target = np.asarray(a)*grid.PITCH_M, np.asarray(b)*grid.PITCH_M
        geometry = ReferenceGeometry(walls, grid.WORLD_BOUNDS_XY_M, target, radius_m=rules.inflation_radius_m,
                                     clearance_m=rules.additional_clearance_m, resolution_m=rules.grid_resolution_m)
        if float(min(geometry.footprint_clearance(start), geometry.footprint_clearance(target)))+geometry.radius \
                < rules.min_endpoint_clearance_m:
            continue
        if not segment_crosses_wall(start, target, walls):
            continue
        distance = geometry.distance_and_heading(start)
        if not distance['valid'] or distance['distance_m'] <= rules.min_geodesic_m:
            continue
        heading = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(heading), math.sin(heading)
        goal_body = np.array([[c, s], [-s, c]]) @ (target-start)
        return dict(
            schema='navigation_capability_episode.v1', maze_id=index, episode_index=episode_index,
            episode_id=f'{index:02d}/{episode_index}', role=spec['data_role'], seed=seed,
            simulation_seed=seeds.physics_base+2*index+episode_index,
            home_se2_world=[*start.tolist(), heading], beacon_xy_world=target.tolist(),
            mission=dict(goal_initial_body_xy_m=goal_body.tolist(), return_initial_body_xy_m=[0., 0.],
                         require_return_after_goal=True),
            shortest_outbound_m=distance['distance_m'], shortest_return_m=distance['distance_m'],
            initial_wall_occlusion=True, endpoint_clearance_qualified=True, runtime_executed=False,
            rendering_performed=False)
    raise ValueError(f'no home/beacon pair satisfies the episode rules in maze {index}')


def generate(roles, *, seeds, excluded_graphs, scene_id_prefix, family, qualify=None,
             maximum_candidates=MAXIMUM_CANDIDATES):
    """Generate a set: layouts take roles in index order; each gets its role's episodes.

    Without `qualify`, every accepted layout is used in order. With `qualify(spec, episode) -> bool`, a layout is used
    only if one of its role's episodes qualifies, and only that first qualifying episode is kept; indices count
    qualifying layouts only (the stage-2 selection).

    Returns dict(entries=[dict(spec=..., episodes=[...])], candidates_examined, rejection_counts).
    """
    plan = [role for role in roles for _ in range(role.count)]
    entries, rejections = [], []
    for candidate_index, links, identity in accepted_layouts(seeds.construction, excluded_graphs, rejections,
                                                             maximum_candidates=maximum_candidates):
        index = len(entries)
        role = plan[index]
        spec = make_spec(index, links, identity, candidate_index, seeds=seeds, scene_id_prefix=scene_id_prefix,
                         family=family, role=role.name, spec_fields=role.spec_fields)
        if qualify is None:
            entries.append(dict(spec=spec, episodes=[make_episode(spec, e, seeds=seeds) for e in role.episodes]))
        else:
            for e in role.episodes:
                episode = make_episode(spec, e, seeds=seeds)
                if qualify(spec, episode):
                    entries.append(dict(spec=spec, episodes=[episode]))
                    break
        if len(entries) == len(plan):
            break
    if len(entries) != len(plan):
        raise ValueError(f'only {len(entries)} of {len(plan)} layouts within {maximum_candidates} candidates')
    return dict(entries=entries, candidates_examined=entries[-1]['spec']['construction_candidate_index']+1,
                rejection_counts=dict(Counter(r['reason'] for r in rejections)))
