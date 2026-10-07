"""Topology labels derived directly from canonical scene manifests.

Two products:

- :func:`topology_summary` — coarse per-scene stats (node count, dead ends,
  cycle count). Written to ``topology.json`` next to each scene at corpus
  build time.
- :func:`local_graph_type_per_node` — per-node categorical label
  (corridor / turn / T-junction / crossroad / dead-end / open / unknown).
  Used by the offline derived-labels pass to tag every per-step record with
  the kind of place the agent currently occupies (data spec §5.3
  ``local_graph_type``).

Categorical labels are deliberately coarse: the downstream encoder isn't
expected to consume them directly — they exist as audit splits for the
visual-aliasing analysis ([docs/v3_topological_nav_plan.md](docs/v3_topological_nav_plan.md)
§4.3) and as multi-task heads for ablations.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

from go2nav.corpus.manifest import SceneManifest


# Per-spec §5.3 local_graph_type categorical set used as audit splits.
LOCAL_GRAPH_TYPES: tuple[str, ...] = (
    "dead_end",
    "corridor",
    "turn",
    "t_junction",
    "crossroad",
    "open",
    "unknown",
)

# Two consecutive corridor edges meeting at this angle (or straighter) count
# as "corridor"; below the threshold the node is classified as "turn". The
# default is 30°: a 30° bend is visually obvious enough that the encoder
# should treat the two segments as distinct local geometries.
_CORRIDOR_STRAIGHTNESS_THRESHOLD_RAD = math.radians(30.0)


def topology_summary(manifest: SceneManifest) -> dict[str, Any]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in manifest.graph_edges:
        if not edge.traversable:
            continue
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    dead_ends = sorted(
        node.node_id for node in manifest.graph_nodes if len(adjacency[node.node_id]) <= 1
    )
    component_count = _component_count([node.node_id for node in manifest.graph_nodes], adjacency)
    edge_count = sum(1 for edge in manifest.graph_edges if edge.traversable)
    node_count = len(manifest.graph_nodes)
    cycle_count = max(0, edge_count - node_count + component_count)

    return {
        "scene_id": manifest.scene_id,
        "node_count": node_count,
        "edge_count": edge_count,
        "component_count": component_count,
        "dead_end_node_ids": dead_ends,
        "cycle_count": cycle_count,
        "landmark_object_ids": [obj.object_id for obj in manifest.landmarks],
    }


def local_graph_type_per_node(manifest: SceneManifest) -> dict[int, str]:
    """Return one categorical label per graph node (data spec §5.3).

    Classification rule (degree of traversable neighbours):

    - 0 → ``unknown`` (orphan node — shouldn't happen on a well-formed
      generator but we keep the bucket so the histogram never silently
      under-reports).
    - 1 → ``dead_end``.
    - 2 → ``corridor`` if the two neighbours sit on a near-straight line
      through the node (turn angle < ``_CORRIDOR_STRAIGHTNESS_THRESHOLD_RAD``
      off π), otherwise ``turn``.
    - 3 → ``t_junction``.
    - 4 → ``crossroad``.
    - ≥ 5 → ``open`` (open room / atypical degree; rare on grid graphs).
    """

    adjacency: dict[int, list[int]] = {node.node_id: [] for node in manifest.graph_nodes}
    for edge in manifest.graph_edges:
        if not edge.traversable:
            continue
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)

    by_id = {node.node_id: node for node in manifest.graph_nodes}
    labels: dict[int, str] = {}
    for node in manifest.graph_nodes:
        neighbours = adjacency.get(node.node_id, [])
        degree = len(neighbours)
        if degree == 0:
            labels[node.node_id] = "unknown"
        elif degree == 1:
            labels[node.node_id] = "dead_end"
        elif degree == 2:
            a = by_id[neighbours[0]].center_xy_m
            b = by_id[neighbours[1]].center_xy_m
            c = node.center_xy_m
            angle = _interior_angle(a, c, b)
            if abs(math.pi - angle) <= _CORRIDOR_STRAIGHTNESS_THRESHOLD_RAD:
                labels[node.node_id] = "corridor"
            else:
                labels[node.node_id] = "turn"
        elif degree == 3:
            labels[node.node_id] = "t_junction"
        elif degree == 4:
            labels[node.node_id] = "crossroad"
        else:
            labels[node.node_id] = "open"
    return labels


def local_graph_type_histogram(manifest: SceneManifest) -> dict[str, int]:
    """Return ``{label: count}`` over the manifest's graph nodes."""

    histogram: dict[str, int] = {label: 0 for label in LOCAL_GRAPH_TYPES}
    for label in local_graph_type_per_node(manifest).values():
        histogram[label] = histogram.get(label, 0) + 1
    return histogram


def _interior_angle(
    a: tuple[float, float], pivot: tuple[float, float], b: tuple[float, float]
) -> float:
    """Return the interior angle (radians) of triangle ``a-pivot-b`` at pivot."""

    ax = a[0] - pivot[0]
    ay = a[1] - pivot[1]
    bx = b[0] - pivot[0]
    by = b[1] - pivot[1]
    dot = ax * bx + ay * by
    norm_a = math.hypot(ax, ay)
    norm_b = math.hypot(bx, by)
    if norm_a == 0.0 or norm_b == 0.0:
        return math.pi
    cos_angle = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return math.acos(cos_angle)


def _component_count(node_ids: list[int], adjacency: dict[int, set[int]]) -> int:
    unseen = set(node_ids)
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return count
