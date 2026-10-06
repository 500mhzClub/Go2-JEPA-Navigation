"""Layout identity: is a new maze genuinely different from every maze built before?

Two identities are used, and a new layout must differ from all earlier layouts in both:
- **topology**: the exact canonical form of the passage graph, independent of how it is embedded in the grid;
- **metric embedding**: the passage set up to translation, reflection and right-angle rotation, together with the
  pitch, wall thickness and wall height.
"""
import hashlib
import json


def grid_edges(edges):
    """Validate unit cardinal grid passages; return (sorted canonical edges, adjacency dict). The graph must be connected."""
    if not isinstance(edges, (list, tuple)) or not edges:
        raise ValueError('non-empty passage list required')
    result = []
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            raise ValueError('each passage needs two endpoints')
        a, b = (tuple(p) for p in e)
        if any(len(p) != 2 or any(type(v) is not int for v in p) for p in (a, b)):
            raise ValueError('integer grid coordinates required')
        if sum(abs(x-y) for x, y in zip(a, b)) != 1:
            raise ValueError('passages must join cardinal neighbours')
        result.append(tuple(sorted((a, b))))
    if len(set(result)) != len(result):
        raise ValueError('duplicate passage')
    result = tuple(sorted(result))
    nodes = sorted({p for e in result for p in e})
    adjacency = {p: set() for p in nodes}
    for a, b in result:
        adjacency[a].add(b)
        adjacency[b].add(a)
    reached = {nodes[0]}
    while True:
        expanded = reached | {q for p in reached for q in adjacency[p]}
        if expanded == reached:
            break
        reached = expanded
    if reached != set(nodes):
        raise ValueError('connected layout required')
    return result, adjacency


def canonical_graph(adjacency, *, maximum_states=100_000):
    """Exact canonical form of a small connected graph (vertices 0..n-1, degree 1-4).

    Colour refinement by neighbour counts narrows the candidates; the canonical form is the lexicographically smallest
    upper-triangle adjacency string over every individualisation of the remaining ties (exhaustive, bounded by
    `maximum_states`; exceeding the bound is an error, never an approximation).
    """
    n = len(adjacency)
    if not 2 <= n <= 24:
        raise ValueError('2-24 vertices supported')
    graph = [frozenset(row) for row in adjacency]
    if any(i not in graph[j] for i, row in enumerate(graph) for j in row):
        raise ValueError('undirected graph required')

    def refine(partition):
        while True:
            refined, sets = [], [set(c) for c in partition]
            for cell in partition:
                groups = {}
                for v in cell:
                    groups.setdefault(tuple(len(graph[v] & s) for s in sets), []).append(v)
                refined.extend(tuple(groups[k]) for k in sorted(groups))
            if len(refined) == len(partition):
                return tuple(refined)
            partition = tuple(refined)

    states, best, best_order = 0, None, None

    def visit(partition):
        nonlocal states, best, best_order
        states += 1
        if states > maximum_states:
            raise ValueError('canonical search budget exhausted; identity unresolved')
        partition = refine(partition)
        tied = [(len(c), i) for i, c in enumerate(partition) if len(c) > 1]
        if not tied:
            order = tuple(c[0] for c in partition)
            code = ''.join('1' if order[j] in graph[order[i]] else '0' for i in range(n) for j in range(i+1, n))
            if best is None or code < best:
                best, best_order = code, order
            return
        _, at = min(tied)
        cell = partition[at]
        for v in cell:
            visit(partition[:at] + ((v,), tuple(w for w in cell if w != v)) + partition[at+1:])

    visit(tuple(tuple(i for i in range(n) if len(graph[i]) == d) for d in sorted({len(g) for g in graph})))
    code = f'undirected-v1:{n}:{best}'
    return dict(code=code, sha256=hashlib.sha256(code.encode()).hexdigest(),
                canonical_vertex_order=list(best_order), search_states=states)


def topology_identity(edges):
    _, adjacency = grid_edges(edges)
    nodes = sorted(adjacency)
    index = {p: i for i, p in enumerate(nodes)}
    return canonical_graph([sorted(index[q] for q in adjacency[p]) for p in nodes])


def metric_identity(edges, *, pitch_m, wall_thickness_m, wall_height_m):
    """Passages up to translation, reflection and right-angle rotation, plus the physical dimensions."""
    edges, _ = grid_edges(edges)
    variants = []
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                def transform(p):
                    return (sx*p[1], sy*p[0]) if swap else (sx*p[0], sy*p[1])
                pairs = [(transform(a), transform(b)) for a, b in edges]
                low = tuple(min(p[i] for e in pairs for p in e) for i in range(2))
                variants.append(tuple(sorted(tuple(sorted(tuple(p[i]-low[i] for i in range(2)) for p in e)) for e in pairs)))
    value = dict(edges=min(variants), pitch_m=float(pitch_m).hex(), wall_thickness_m=float(wall_thickness_m).hex(),
                 wall_height_m=float(wall_height_m).hex())
    code = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return dict(code=code, sha256=hashlib.sha256(code.encode()).hexdigest())


def layout_identities(links, *, pitch_m, wall_thickness_m, wall_height_m):
    """Both identities of a layout, in the form recorded in every maze specification."""
    top = topology_identity(links)
    metric = metric_identity(links, pitch_m=pitch_m, wall_thickness_m=wall_thickness_m, wall_height_m=wall_height_m)
    return dict(abstract_topology_sha256=top['sha256'], abstract_topology_code=top['code'],
                exact_canonicalization_states=top['search_states'], metric_sha256=metric['sha256'],
                metric_code=metric['code'])
