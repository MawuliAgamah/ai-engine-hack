"""
World grid — the spatial substrate for the simulation.

Each cell is a pixel with rich metadata: terrain type, walkability, movement cost,
hazard level, pheromone deposits, elevation, and neighbor references. This is what
enables fully local decision-making by agents.

Design principles:
- NumPy arrays for field data (terrain costs, hazard levels, pheromones) → fast vectorized ops
- Cell objects only materialized on demand for rich queries
- Neighbor lookups are O(1) via pre-computed offset tables
- PropertyLayer pattern (inspired by Mesa) for overlay fields
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter, maximum_filter


class Terrain(enum.IntEnum):
    """Terrain types that define the base character of each cell."""

    OPEN = 0          # Open space (plaza, field)
    CORRIDOR = 1      # Indoor corridor
    ROAD = 2          # Vehicle road
    SIDEWALK = 3      # Pedestrian sidewalk
    STAIRS = 4        # Stairway (higher cost)
    DOOR = 5          # Doorway (bottleneck)
    EXIT = 6          # Exit / evacuation target
    WALL = 7          # Impassable wall
    WATER = 8         # Water body (flood modeling)
    BUILDING = 9      # Building interior
    OBSTACLE = 10     # Impassable obstacle (furniture, barrier)
    GRASS = 11        # Grass / park (walkable, slightly slower)


# Base movement cost multipliers per terrain type
TERRAIN_COSTS: dict[Terrain, float] = {
    Terrain.OPEN: 1.0,
    Terrain.CORRIDOR: 1.0,
    Terrain.ROAD: 0.8,
    Terrain.SIDEWALK: 1.0,
    Terrain.STAIRS: 2.5,
    Terrain.DOOR: 1.5,
    Terrain.EXIT: 0.5,
    Terrain.WALL: np.inf,
    Terrain.WATER: np.inf,
    Terrain.BUILDING: np.inf,
    Terrain.OBSTACLE: np.inf,
    Terrain.GRASS: 1.3,
}

# Whether terrain is walkable by default
TERRAIN_WALKABLE: dict[Terrain, bool] = {
    Terrain.OPEN: True,
    Terrain.CORRIDOR: True,
    Terrain.ROAD: True,
    Terrain.SIDEWALK: True,
    Terrain.STAIRS: True,
    Terrain.DOOR: True,
    Terrain.EXIT: True,
    Terrain.WALL: False,
    Terrain.WATER: False,
    Terrain.BUILDING: False,
    Terrain.OBSTACLE: False,
    Terrain.GRASS: True,
}


@dataclass(frozen=True, slots=True)
class Position:
    """Integer grid position."""

    x: int
    y: int

    def __add__(self, other: Position) -> Position:
        return Position(self.x + other.x, self.y + other.y)

    def manhattan_distance(self, other: Position) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def euclidean_distance(self, other: Position) -> float:
        return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))

    def chebyshev_distance(self, other: Position) -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

# 6-connected neighborhood offsets for hex grids (pointy-topped)
HEX_OFFSETS: list[Position] = [
    Position(-1, 1), Position(1, 1),
    Position(-1, 0), Position(1, 0),
    Position(-1, -1), Position(1, -1)
]


@dataclass
class Cell:
    """
    Rich metadata for a single pixel/cell in the world.

    This is the 'what it's on, what's in it, what's next to it' abstraction.
    Materialized on demand from the underlying NumPy arrays for detailed queries.
    """

    pos: Position
    terrain: Terrain
    walkable: bool
    cost: float
    max_occupancy: int = 100
    hazard_level: float = 0.0
    elevation: float = 0.0
    agent_ids: list[int] = field(default_factory=list)

    @property
    def is_passable(self) -> bool:
        """Can an agent move through this cell right now?"""
        return self.walkable and self.hazard_level < 1.0 and self.occupancy < self.max_occupancy

    @property
    def effective_cost(self) -> float:
        """Movement cost including hazard penalty."""
        if not self.is_passable:
            return np.inf
        # Hazard increases cost exponentially — agents strongly avoid danger
        return self.cost * (1.0 + 10.0 * self.hazard_level**2)

    @property
    def is_occupied(self) -> bool:
        return len(self.agent_ids) > 0

    @property
    def occupancy(self) -> int:
        return len(self.agent_ids)


class PropertyLayer:
    """
    A named 2D float array overlaying the world grid.

    Used for pheromones, gradient fields, danger maps, etc.
    Supports decay, diffusion, and clamping operations.
    Inspired by Mesa's PropertyLayer pattern.
    """

    def __init__(self, name: str, width: int, height: int, default: float = 0.0):
        self.name = name
        self.width = width
        self.height = height
        self.data: NDArray[np.float64] = np.full((height, width), default, dtype=np.float64)

    def get(self, x: int, y: int) -> float:
        return float(self.data[y, x])

    def set(self, x: int, y: int, value: float) -> None:
        self.data[y, x] = value

    def add(self, x: int, y: int, value: float) -> None:
        self.data[y, x] += value

    def decay(self, rate: float) -> None:
        """Multiplicative decay: value *= (1 - rate). For pheromone evaporation."""
        self.data *= 1.0 - rate

    def diffuse(self, rate: float) -> None:
        """
        Diffuse values to neighbors via convolution.
        Each cell shares `rate` fraction of its value equally with 4 neighbors.
        """
        if rate <= 0:
            return

        diffused = uniform_filter(self.data, size=3, mode="constant", cval=0.0)
        self.data = (1.0 - rate) * self.data + rate * diffused

    def clamp(self, low: float = 0.0, high: float = float("inf")) -> None:
        np.clip(self.data, low, high, out=self.data)

    def reset(self) -> None:
        self.data.fill(0.0)

    def max_pos(self) -> Position:
        """Position of maximum value."""
        idx = np.unravel_index(np.argmax(self.data), self.data.shape)
        return Position(int(idx[1]), int(idx[0]))

    def sum(self) -> float:
        return float(np.sum(self.data))


class World:
    """
    The spatial grid world.

    Core data is stored in dense NumPy arrays for performance:
    - terrain_grid: int array of Terrain enum values
    - cost_grid: float array of movement costs
    - walkable_grid: bool array
    - hazard_grid: float array [0, 1]
    - elevation_grid: float array
    - occupancy_grid: int array of agent counts per cell

    Property layers are named overlays for pheromones, gradients, etc.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Core grids
        self.terrain_grid: NDArray[np.int8] = np.full(
            (height, width), Terrain.OPEN, dtype=np.int8
        )
        self.cost_grid: NDArray[np.float64] = np.ones((height, width), dtype=np.float64)
        self.walkable_grid: NDArray[np.bool_] = np.ones((height, width), dtype=np.bool_)
        self.hazard_grid: NDArray[np.float64] = np.zeros((height, width), dtype=np.float64)
        self.elevation_grid: NDArray[np.float64] = np.zeros((height, width), dtype=np.float64)
        self.occupancy_grid: NDArray[np.int32] = np.zeros((height, width), dtype=np.int32)

        # Agent tracking: maps (x, y) → set of agent IDs
        self._agents_at: dict[tuple[int, int], set[int]] = {}

        # Named property layers (pheromones, gradients, etc.)
        self._layers: dict[str, PropertyLayer] = {}

        # Exit positions for pathfinding
        self.exits: list[Position] = []

    # ── Grid construction ───────────────────────────────────────────

    def set_terrain(self, x: int, y: int, terrain: Terrain) -> None:
        """Set terrain type for a cell, auto-updating cost and walkability."""
        self.terrain_grid[y, x] = terrain
        self.cost_grid[y, x] = TERRAIN_COSTS[terrain]
        self.walkable_grid[y, x] = TERRAIN_WALKABLE[terrain]
        if terrain == Terrain.EXIT:
            pos = Position(x, y)
            if pos not in self.exits:
                self.exits.append(pos)

    def set_terrain_rect(
        self, x: int, y: int, w: int, h: int, terrain: Terrain
    ) -> None:
        """Set terrain for a rectangular region."""
        for dy in range(h):
            for dx in range(w):
                if self.in_bounds(x + dx, y + dy):
                    self.set_terrain(x + dx, y + dy, terrain)

    def set_elevation(self, x: int, y: int, elevation: float) -> None:
        self.elevation_grid[y, x] = elevation

    # ── Bounds and neighbor queries ──────────────────────────────────

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(
        self, x: int, y: int, walkable_only: bool = True
    ) -> list[Position]:
        """Get neighboring positions."""
        offsets = HEX_OFFSETS # potentially could use Moore or Von Neumann
        result = []
        for off in offsets:
            nx, ny = x + off.x, y + off.y
            if self.in_bounds(nx, ny):
                if walkable_only and not self.walkable_grid[ny, nx]:
                    continue
                result.append(Position(nx, ny))
        return result

    def walkable_neighbors(self, x: int, y: int, moore: bool = True) -> list[Position]:
        """Get walkable neighboring positions (also checks hazard passability)."""
        offsets = HEX_OFFSETS # potentially could use Moore or Von Neumann
        result = []
        for off in offsets:
            nx, ny = x + off.x, y + off.y
            if (
                self.in_bounds(nx, ny)
                and self.walkable_grid[ny, nx]
                and self.hazard_grid[ny, nx] < 1.0
            ):
                result.append(Position(nx, ny))
        return result

    # ── Cell materialization ─────────────────────────────────────────

    def get_cell(self, x: int, y: int) -> Cell:
        """Materialize a full Cell object for detailed queries."""
        pos = Position(x, y)
        agent_ids = list(self._agents_at.get((x, y), set()))
        return Cell(
            pos=pos,
            terrain=Terrain(self.terrain_grid[y, x]),
            walkable=bool(self.walkable_grid[y, x]),
            cost=float(self.cost_grid[y, x]),
            hazard_level=float(self.hazard_grid[y, x]),
            elevation=float(self.elevation_grid[y, x]),
            agent_ids=agent_ids,
        )

    # ── Agent placement ──────────────────────────────────────────────

    def place_agent(self, agent_id: int, x: int, y: int) -> None:
        key = (x, y)
        if key not in self._agents_at:
            self._agents_at[key] = set()
        self._agents_at[key].add(agent_id)
        self.occupancy_grid[y, x] += 1

    def remove_agent(self, agent_id: int, x: int, y: int) -> None:
        key = (x, y)
        if key in self._agents_at:
            self._agents_at[key].discard(agent_id)
            if not self._agents_at[key]:
                del self._agents_at[key]
        self.occupancy_grid[y, x] = max(0, self.occupancy_grid[y, x] - 1)

    def move_agent(
        self, agent_id: int, from_x: int, from_y: int, to_x: int, to_y: int
    ) -> None:
        self.remove_agent(agent_id, from_x, from_y)
        self.place_agent(agent_id, to_x, to_y)

    def agents_at(self, x: int, y: int) -> set[int]:
        return self._agents_at.get((x, y), set())

    # ── Hazard management ────────────────────────────────────────────

    def set_hazard(self, x: int, y: int, level: float) -> None:
        """Set hazard level [0, 1] for a cell. 1.0 = impassable."""
        self.hazard_grid[y, x] = np.clip(level, 0.0, 1.0)

    def spread_hazard(self, rate: float = 0.1) -> None:
        """
        Spread hazards to neighboring cells via diffusion.
        Models fire spread, flood expansion, etc.
        """
        # Hazard spreads to neighbors but is blocked by walls
        expanded = maximum_filter(self.hazard_grid, size=3)
        # Only spread where walkable (fire doesn't spread through concrete walls... much)
        mask = self.walkable_grid.astype(np.float64)
        self.hazard_grid = np.clip(
            self.hazard_grid + rate * (expanded - self.hazard_grid) * mask,
            0.0,
            1.0,
        )

    # ── Property layers ──────────────────────────────────────────────

    def add_layer(self, name: str, default: float = 0.0) -> PropertyLayer:
        layer = PropertyLayer(name, self.width, self.height, default)
        self._layers[name] = layer
        return layer

    def get_layer(self, name: str) -> PropertyLayer:
        return self._layers[name]

    def has_layer(self, name: str) -> bool:
        return name in self._layers

    @property
    def layers(self) -> dict[str, PropertyLayer]:
        return self._layers

    # ── Effective cost (combines base cost + hazard + occupancy) ─────

    def effective_cost_grid(self) -> NDArray[np.float64]:
        """
        Compute the effective movement cost grid combining:
        - Base terrain cost
        - Hazard penalty (exponential)
        - Occupancy penalty (crowding)
        """
        hazard_penalty = 1.0 + 10.0 * self.hazard_grid**2
        crowd_penalty = 1.0 + 0.5 * self.occupancy_grid
        costs = self.cost_grid * hazard_penalty * crowd_penalty
        # Impassable cells get inf cost
        costs[~self.walkable_grid] = np.inf
        costs[self.hazard_grid >= 1.0] = np.inf
        return costs

    # ── Iteration helpers ────────────────────────────────────────────

    def all_positions(self) -> Iterator[Position]:
        for y in range(self.height):
            for x in range(self.width):
                yield Position(x, y)

    def walkable_positions(self) -> Iterator[Position]:
        for y in range(self.height):
            for x in range(self.width):
                if self.walkable_grid[y, x]:
                    yield Position(x, y)

    def __repr__(self) -> str:
        return f"World({self.width}x{self.height}, exits={len(self.exits)}, layers={list(self._layers.keys())})"
