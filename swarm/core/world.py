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
import math
from dataclasses import dataclass, field
from typing import Iterator, Union
from collections import deque

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter, maximum_filter
import pandas as pd
import h3

LANDUSE = {
    'allotments', 'apiary', 'brownfield', 'cemetery', 'commercial', 'construction', 'depot', 
    'farmland', 'farmyard', 'flowerbed', 'forest', 'grass', 'industrial', 'meadow', 'military', 
    'mixed', 'orchard', 'platform', 'railway', 'recreation_ground', 'religious', 'residential', 
    'retail', 'traffic_island', 'village_green'
}

LEISURE = {
    'bandstand', 'bleachers', 'dance', 'dog_park', 'fitness_centre', 'fitness_station', 'garden', 
    'golf_course', 'marina', 'nature_reserve', 'nets', 'outdoor_seating', 'park', 'pitch', 
    'playground', 'schoolyard', 'slipway', 'sports_centre', 'stadium', 'swimming_pool', 
    'tanning_salon', 'track'
}

NATURE = {
    'beach', 'grass', 'mud', 'scru', 'shrubbery', 'tree_row', 'water', 'wetland', 'wood'
}

ROADS = {
    'cycleway', 'footway', 'primary', 'residential', 'secondary', 'service', 'tertiary', 'unclassified'
}

class Terrain(enum.IntEnum):
    """Terrain types that define the base character of each cell."""
    OPEN = 0  # Default open terrain (sidewalks, parks, etc.)
    NATURE = 10 # Color of any cell that has park, nature_reserve
    EXIT = 20  # Special terrain type for exits/goals
    OBSTACLE = 30  # Special terrain type for impassable objects
    WATER = 40  # Color of any cell that has water, wetland

TRAVERSABILITY_RATINGS = {
    "LANDUSE": {
        "allotments": 0.40,
        "apiary": 0.55,
        "brownfield": 0.65,
        "cemetery": 0.25,
        "commercial": 0.20,
        "construction": 0.90,
        "depot": 0.70,
        "farmland": 0.50,
        "farmyard": 0.45,
        "flowerbed": 0.85,
        "forest": 0.75,
        "grass": 0.15,
        "industrial": 0.60,
        "meadow": 0.20,
        "military": 1.00,
        "mixed": 0.40,
        "orchard": 0.45,
        "platform": 0.10,
        "railway": 0.95,
        "recreation_ground": 0.15,
        "religious": 0.20,
        "residential": 0.20,
        "retail": 0.20,
        "traffic_island": 0.80,
        "village_green": 0.10
    },
    "LEISURE": {
        "bandstand": 0.70,
        "bleachers": 0.85,
        "dance": 0.10,
        "dog_park": 0.20,
        "fitness_centre": 0.15,
        "fitness_station": 0.35,
        "garden": 0.40,
        "golf_course": 0.35,
        "marina": 0.95,
        "nature_reserve": 0.65,
        "nets": 0.60,
        "outdoor_seating": 0.45,
        "park": 0.10,
        "pitch": 0.20,
        "playground": 0.35,
        "schoolyard": 0.20,
        "slipway": 0.55,
        "sports_centre": 0.20,
        "stadium": 0.50,
        "swimming_pool": 1.00,
        "tanning_salon": 0.10,
        "track": 0.05
    },
    "NATURE": {
        "beach": 0.45,
        "grass": 0.10,
        "mud": 0.80,
        "scru": 0.70,
        "shrubbery": 0.85,
        "tree_row": 0.60,
        "water": 1.00,
        "wetland": 0.95,
        "wood": 0.75
    }
}

# Cost lookup for programmatic terrain changes (set_terrain)
TERRAIN_COSTS: dict[Terrain, float] = {
    Terrain.OPEN: 0.2,
    Terrain.NATURE: 0.15,
    Terrain.EXIT: 0.0,
    Terrain.OBSTACLE: 1.0,
    Terrain.WATER: 1.0,
}


def compute_terrain_cost(osm_dict: dict) -> float:
    """Compute traversability cost from OSM metadata."""
    landuse = list(osm_dict['summary']['landuse'])
    leisure = list(osm_dict['summary']['leisure'])
    nature = list(osm_dict['summary']['natural'])
    if not landuse and not leisure and not nature:
        return 0.5
    parts = (
        [TRAVERSABILITY_RATINGS['LANDUSE'][l] for l in landuse]
        + [TRAVERSABILITY_RATINGS['LEISURE'][l] for l in leisure]
        + [TRAVERSABILITY_RATINGS['NATURE'][l] for l in nature]
    )
    return float(np.mean(parts)) if parts else 0.5


# Tags that indicate "nature / green" terrain
_NATURE_TAGS = {
    'nature_reserve', 'park'
}

# Tags that indicate "water" terrain
_WATER_TAGS = {'water', 'wetland'}


def compute_terrain_type(osm_dict: dict) -> Terrain:
    """Derive terrain type from OSM metadata."""
    landuse = list(osm_dict['summary']['landuse'])
    leisure = list(osm_dict['summary']['leisure'])
    natural = list(osm_dict['summary']['natural'])
    all_tags = set(landuse + leisure + natural)
    if not all_tags:
        return Terrain.OPEN
    # Pure water cell
    if all_tags <= _WATER_TAGS:
        return Terrain.WATER
    # Any nature/green tag → colour as nature
    if all_tags & _NATURE_TAGS:
        return Terrain.NATURE
    return Terrain.OPEN


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

# 8-connected neighborhood offsets (Moore neighborhood)
MOORE_OFFSETS: list[Position] = [
    Position(-1, -1), Position(0, -1), Position(1, -1),
    Position(-1, 0),                   Position(1, 0),
    Position(-1, 1),  Position(0, 1),  Position(1, 1),
]

# 4-connected neighborhood offsets (Von Neumann neighborhood)
VON_NEUMANN_OFFSETS: list[Position] = [
    Position(0, -1),
    Position(-1, 0), Position(1, 0),
    Position(0, 1),
]

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
        return self.hazard_level < 1.0 and self.occupancy < self.max_occupancy

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
    - cost_grid: float array of movement costs
    - walkable_grid: bool array
    - hazard_grid: float array [0, 1]
    - elevation_grid: float array
    - occupancy_grid: int array of agent counts per cell

    Property layers are named overlays for pheromones, gradients, etc.
    """

    def __init__(self, data: Union[str, pd.DataFrame]):
        if isinstance(data, str):
            if data.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.endswith('.json'):
                df = pd.read_json(data)
            elif 'parquet' in data:
                df = pd.read_parquet(data)
            else:
                raise ValueError(f"Unsupported file format for World constructor: {data.split('.')[-1]}")
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("World constructor requires a file path or DataFrame")

        df = quantise_grid(df)[['h3_index', 'latitude', 'longitude', 'osm_structured_json_dict', 'i', 'j']]
        df['terrain_cost'] = df['osm_structured_json_dict'].apply(compute_terrain_cost)
        df['walkable'] = df['terrain_cost'] < 0.95  # threshold for walkability
        df['terrain_type'] = df['osm_structured_json_dict'].apply(compute_terrain_type)

        # Shift i/j to 0-based coordinates
        i_min = int(df['i'].min())
        j_min = int(df['j'].min())
        df['i'] = df['i'] - i_min
        df['j'] = df['j'] - j_min

        self.width = int(df['i'].max() + 1)
        self.height = int(df['j'].max() + 1)

        # Store lat/lon lookup for each (i, j) cell
        self._cell_metadata: dict[tuple[int, int], dict] = {}
        for _, row in df.iterrows():
            ci, cj = int(row['i']), int(row['j'])
            self._cell_metadata[(ci, cj)] = {
                'h3_index': row['h3_index'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'description': row.get('description', ''),
            }

        # Track which (i,j) cells actually have data (not all grid positions do)
        self._valid_cells: set[tuple[int, int]] = set()
        for _, row in df.iterrows():
            self._valid_cells.add((int(row['i']), int(row['j'])))

        # Core grids — fill NaN positions as obstacle/unwalkable
        # sort_index ascending so grid[y,x] = grid[j,i] → Position(x,y) = (i,j)
        terrain_pivot = df.pivot_table(index='j', columns='i', values='terrain_type')
        terrain_pivot = terrain_pivot.reindex(
            index=range(self.height), columns=range(self.width)
        ).fillna(Terrain.OBSTACLE.value)
        self.terrain_grid: NDArray[np.int32] = terrain_pivot.sort_index().to_numpy(dtype=np.int32).copy()

        cost_pivot = df.pivot_table(index='j', columns='i', values='terrain_cost')
        cost_pivot = cost_pivot.reindex(
            index=range(self.height), columns=range(self.width)
        ).fillna(1.0)
        self.cost_grid: NDArray[np.float64] = cost_pivot.sort_index().to_numpy(dtype=np.float64).copy()

        walkable_pivot = df.pivot_table(index='j', columns='i', values='walkable')
        walkable_pivot = walkable_pivot.reindex(
            index=range(self.height), columns=range(self.width)
        ).fillna(False)
        self.walkable_grid: NDArray[np.bool_] = walkable_pivot.sort_index().to_numpy(dtype=np.bool_).copy()

        self.hazard_grid: NDArray[np.float64] = np.zeros((self.height, self.width), dtype=np.float64)
        self.elevation_grid: NDArray[np.float64] = np.zeros((self.height, self.width), dtype=np.float64)
        self.occupancy_grid: NDArray[np.int32] = np.zeros((self.height, self.width), dtype=np.int32)

        # Agent tracking: maps (x, y) → set of agent IDs
        self._agents_at: dict[tuple[int, int], set[int]] = {}

        # Named property layers (pheromones, gradients, etc.)
        self._layers: dict[str, PropertyLayer] = {}

        # Exit positions for pathfinding — border cells of the valid hex region
        self.exits: list[Position] = []
        self._border_cells: list[tuple[int, int]] = self._find_border_cells()
        self._init_border_exits()
        

    def _find_border_cells(self) -> list[tuple[int, int]]:
        """Return walkable border cells (cells with at least one invalid neighbour)."""
        border: list[tuple[int, int]] = []
        for (ci, cj) in sorted(self._valid_cells):
            for off in HEX_OFFSETS:
                ni, nj = ci + off.x, cj + off.y
                if (ni, nj) not in self._valid_cells:
                    if self.walkable_grid[cj, ci]:
                        border.append((ci, cj))
                    break
        return border

    def _init_border_exits(self, num_exits: int | None = None) -> None:
        """Place exits evenly around the border.

        Parameters
        ----------
        num_exits :
            Desired number of exits.  When ``None`` (default) every 5th
            border cell is used (legacy behaviour).  When specified, the
            exits are evenly spaced around the perimeter.
        """
        border_cells = self._border_cells
        if not border_cells:
            return

        if num_exits is None:
            # Legacy: every-5th border cell
            indices = [i for i in range(len(border_cells)) if i % 5 == 0]
        else:
            n = max(1, min(num_exits, len(border_cells)))
            step = len(border_cells) / n
            indices = [int(i * step) for i in range(n)]

        for idx in indices:
            ci, cj = border_cells[idx]
            pos = Position(ci, cj)
            self.exits.append(pos)
            self.terrain_grid[cj, ci] = Terrain.EXIT.value

    def configure_exits(self, num_exits: int) -> None:
        """Re-initialise exits with a specific count.

        Clears existing exits, then places *num_exits* exits evenly
        spaced around the border of the valid hex region.
        """
        # Clear existing exits
        for pos in self.exits:
            if self.terrain_grid[pos.y, pos.x] == Terrain.EXIT.value:
                self.terrain_grid[pos.y, pos.x] = Terrain.OPEN.value
                self.cost_grid[pos.y, pos.x] = TERRAIN_COSTS[Terrain.OPEN]
        self.exits.clear()
        self._init_border_exits(num_exits=num_exits)

    # ── Grid construction ───────────────────────────────────────────

    def set_terrain(self, x: int, y: int, terrain: Terrain) -> None:
        """Set terrain type for a cell, auto-updating cost and walkability."""
        self.terrain_grid[y, x] = terrain.value
        cost = TERRAIN_COSTS.get(terrain, 0.5)
        self.cost_grid[y, x] = cost
        self.walkable_grid[y, x] = cost < 0.95
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

    def walkable_neighbors(self, x: int, y: int) -> list[Position]:
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
        terrain_val = int(self.terrain_grid[y, x])
        try:
            terrain = Terrain(terrain_val)
        except ValueError:
            terrain = Terrain.OPEN
        return Cell(
            pos=pos,
            terrain=terrain,
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


### h3 grid quantisation helpers (for converting lat/lon to grid coordinates)

# Bearing -> (di, dj) offset bins for H3 res-9 in London
BEARING_OFFSETS = [
    (  0,  60, +1, +1),   # NE  ~31.3
    ( 60, 110, +1,  0),   # E   ~87.4
    (110, 180, +1, -1),   # SE  ~130.4
    (290, 360, -1, +1),   # NW  ~310.4
    (240, 290, -1,  0),   # W   ~267.4
    (180, 240, -1, -1),   # SW  ~211.3
]

def geo_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def bearing_to_offset(b):
    for lo, hi, di, dj in BEARING_OFFSETS:
        if lo <= b < hi:
            return di, dj
    raise ValueError(f"Bearing {b:.1f} not matched by any bin")

def neighbour_offsets(h3_ind):
    """Return list of (neighbour_cell, di, dj) for all 6 ring-1 neighbours."""
    clat, clon = h3.cell_to_latlng(h3_ind)
    result = []
    for nb in h3.grid_ring(h3_ind, 1):
        nlat, nlon = h3.cell_to_latlng(nb)
        b = geo_bearing(clat, clon, nlat, nlon)
        di, dj = bearing_to_offset(b)
        result.append((nb, di, dj))
    return result

def quantise_grid(df):

    # ---------------------------------------------------------------------------
    # Find origin: min longitude, tiebreak min latitude
    origin_row = df.sort_values(["longitude", "latitude"]).iloc[0]
    origin_cell = origin_row["h3_index"]

    

    # Bounding box (with buffer) to constrain BFS expansion.
    # Without this the BFS expands infinitely across all H3 cells globally.
    # A one-cell buffer (~0.005 deg) ensures relay nodes just outside the data
    # extent are still reachable to bridge gaps between dataset cells.
    BUFFER = 0.005
    lat_min = df.latitude.min() - BUFFER
    lat_max = df.latitude.max() + BUFFER
    lon_min = df.longitude.min() - BUFFER
    lon_max = df.longitude.max() + BUFFER

    def in_bounds(cell):
        lat, lon = h3.cell_to_latlng(cell)
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

    # BFS - traverses through ALL in-bounds cells as relay nodes.
    # (i, j) is only recorded for cells that exist in valid_cells.
    cell_to_ij = {origin_cell: (0, 0)}
    visited = {origin_cell}
    queue = deque([origin_cell])

    count = 0
    while queue:
        
        current = queue.pop()
        ci, cj = cell_to_ij[current]

        for nb_cell, di, dj in neighbour_offsets(current):
            if nb_cell in visited:
                continue
            if not in_bounds(nb_cell):     # stop expanding beyond the dataset extent
                continue
            visited.add(nb_cell)

            nb_ij = (ci + di, cj + dj)
            cell_to_ij[nb_cell] = nb_ij   # store for ALL cells so relay nodes work
            queue.append(nb_cell)          # traverse through non-dataset cells too
            count += 1

    # Write (i, j) to dataframe - only for cells in the dataset
    df["i"] = df["h3_index"].map(lambda c: cell_to_ij.get(c, (None, None))[0]).astype("Int64")
    df["j"] = df["h3_index"].map(lambda c: cell_to_ij.get(c, (None, None))[1]).astype("Int64")
    return df

# Legacy aliases removed — use compute_terrain_cost / compute_terrain_type above.