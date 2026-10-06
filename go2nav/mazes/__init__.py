"""Maze and episode generation.

- `grid`: the maze family (4 x 4 cells of 1.3 m, a random spanning tree of passages, walls elsewhere).
- `identity`: topology and metric-embedding identities, used to keep every new maze distinct from all earlier ones.
- `reference`: shortest paths on the true geometry for a disk footprint (episode selection, SPL).
- `generate`: layout acceptance, specifications and episodes.
- `sets`: maze sets on disk with hash-binding registries; `MazeSet` loads and verifies them.
- `cli`: generate a set from a config in configs/mazes/.

The registered sets ship in data/mazes; regenerating them from their configs reproduces their files byte for byte.
"""
