"""
Agent perception system.

Agents perceive a local neighborhood around their current position.  The perception
module extracts structured information from the world grid that the decision
modules can reason about.

**Topology-agnostic design**: the awareness scan uses a BFS flood-fill via
``world.neighbors()`` — the same connectivity (Moore-8, Von-Neumann-4, Hex-6, …)
the world defines.  This keeps perception consistent with the hazard spread model.

**Multi-hazard awareness**: instead of collapsing all nearby hazard cells into
a single scalar + single direction vector, we now also emit per-quadrant hazard
summaries (``HazardLobe``) and a local hazard-rate-of-change estimate so that
agents can reason about *which* direction(s) danger is arriving from and how
quickly.

Design: perception is a PULL operation — each tick, the agent reads its local
environment. This keeps the interface clean and deterministic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from swarm.core.world import Position, World


# ── Hazard lobe descriptor ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HazardLobe:
    """
    A directional cluster of hazard sensed in a quadrant around the agent.

    Agents receive a small list of these (one per quadrant that contains
    hazard) so they can differentiate a fire spreading from the north-east
    and a flood rising from the south.

    Attributes:
        direction: Unit vector (dx, dy) pointing from the agent *toward*
                   the centroid of hazard cells in this lobe.
        mean_intensity: Average hazard level of contributing cells.
        max_intensity: Peak hazard level in the lobe.
        cell_count: Number of hazardous cells in this quadrant.
    """

    direction: tuple[float, float]
    mean_intensity: float
    max_intensity: float
    cell_count: int


# ── Main percept ─────────────────────────────────────────────────────


@dataclass
class LocalPercept:
    """
    Structured perception of an agent's local environment.

    Everything an agent can see / sense in its awareness radius.
    This is the input to all decision modules.
    """

    # Agent state
    position: Position
    tick: int

    # Spatial context (topology-agnostic)
    walkable_neighbors: list[Position]
    neighbor_costs: dict[Position, float]  # pos → effective cost
    neighbor_hazards: dict[Position, float]  # pos → hazard level
    neighbor_occupancy: dict[Position, int]  # pos → agent count

    # Pheromone readings (layer_name → {pos → value})
    pheromone_readings: dict[str, dict[Position, float]] = field(
        default_factory=dict
    )

    # Nearby agents (within awareness radius)
    nearby_agent_positions: list[Position] = field(default_factory=list)
    nearby_agent_count: int = 0

    # ── Scalar hazard summary (backward-compatible) ─────────────────
    max_local_hazard: float = 0.0
    hazard_direction: tuple[float, float] = (0.0, 0.0)  # avg direction of danger

    # ── Multi-hazard perception (new) ───────────────────────────────
    hazard_lobes: list[HazardLobe] = field(default_factory=list)
    hazard_rate_of_change: float = 0.0  # estimated ∂hazard/∂t at agent position

    # Goal info
    nearest_exit_direction: tuple[float, float] | None = None
    nearest_exit_distance: float | None = None

    # Flow field reading (if available)
    flow_direction: tuple[float, float] | None = None


# ── Helpers ──────────────────────────────────────────────────────────


def _bfs_scan(
    world: World, origin_x: int, origin_y: int, radius: int
) -> list[tuple[int, int, float]]:
    """
    Topology-agnostic BFS from *(origin_x, origin_y)* up to Euclidean
    *radius*.  Uses ``world.neighbors(x, y, walkable_only=False)`` so
    the connectivity matches whatever grid the world defines.

    Returns:
        List of ``(x, y, euclidean_distance_from_origin)`` tuples.
    """
    if not world.in_bounds(origin_x, origin_y):
        return []

    visited: dict[tuple[int, int], float] = {(origin_x, origin_y): 0.0}
    queue: deque[tuple[int, int]] = deque([(origin_x, origin_y)])
    r2 = float(radius * radius)

    while queue:
        cx, cy = queue.popleft()
        for nb in world.neighbors(cx, cy, walkable_only=False):
            key = (nb.x, nb.y)
            if key in visited:
                continue
            dx, dy = nb.x - origin_x, nb.y - origin_y
            dist_sq = float(dx * dx + dy * dy)
            if dist_sq > r2:
                continue
            dist = float(np.sqrt(dist_sq))
            visited[key] = dist
            queue.append(key)

    return [(x, y, d) for (x, y), d in visited.items()]


def _quadrant_index(dx: int, dy: int) -> int:
    """Map an offset to one of 4 quadrants (0=NE, 1=SE, 2=SW, 3=NW).
    Uses screen-coord convention: +x=East, +y=South."""
    if dx >= 0 and dy <= 0:
        return 0  # NE (includes North and East axes)
    elif dx >= 0 and dy > 0:
        return 1  # SE
    elif dx < 0 and dy > 0:
        return 2  # SW
    else:
        return 3  # NW


def _build_hazard_lobes(
    origin_x: int,
    origin_y: int,
    cells: list[tuple[int, int, float]],
    hazard_grid,
) -> list[HazardLobe]:
    """
    Cluster hazardous cells into 4 quadrant lobes and return a
    ``HazardLobe`` for each non-empty quadrant.
    """
    # Per-quadrant accumulators: sum_intensity, max_intensity, count, sum_dx, sum_dy
    quads: list[list[float]] = [[0.0, 0.0, 0, 0.0, 0.0] for _ in range(4)]

    for cx, cy, dist in cells:
        hz = float(hazard_grid[cy, cx])
        if hz <= 1e-6:
            continue
        dx, dy = cx - origin_x, cy - origin_y
        qi = _quadrant_index(dx, dy)
        quads[qi][0] += hz           # sum_intensity
        quads[qi][1] = max(quads[qi][1], hz)  # max_intensity
        quads[qi][2] += 1            # count
        quads[qi][3] += dx * hz      # weighted dx
        quads[qi][4] += dy * hz      # weighted dy

    lobes: list[HazardLobe] = []
    for q in quads:
        count = int(q[2])
        if count == 0:
            continue
        sum_int, max_int = q[0], q[1]
        wdx, wdy = q[3], q[4]
        mag = float(np.sqrt(wdx * wdx + wdy * wdy))
        if mag > 1e-8:
            direction = (wdx / mag, wdy / mag)
        else:
            direction = (0.0, 0.0)
        lobes.append(
            HazardLobe(
                direction=direction,
                mean_intensity=sum_int / count,
                max_intensity=max_int,
                cell_count=count,
            )
        )
    return lobes


# ── Main entry point ─────────────────────────────────────────────────


def perceive(
    world: World,
    agent_id: int,
    position: Position,
    awareness_radius: float,
    tick: int,
) -> LocalPercept:
    """
    Generate a ``LocalPercept`` for an agent at *position*.

    The scan uses **BFS flood-fill** via ``world.neighbors()`` so it
    is agnostic to whether the underlying grid is square, hex, or
    any other topology.

    Multi-hazard: hazard cells within the awareness radius are
    clustered into quadrant-based ``HazardLobe`` objects so agents
    can distinguish multiple simultaneous hazard sources.

    Hazard rate-of-change is estimated from the ``"hazard_prev"``
    property layer (if present); the engine should snapshot
    ``hazard_grid`` each tick for this to work.
    """
    x, y = position.x, position.y
    radius = int(awareness_radius)

    # ── Immediate walkable neighbors (for movement choices) ─────────
    walkable = world.walkable_neighbors(x, y)

    # ── BFS scan of awareness area (topology-agnostic) ──────────────
    scanned_cells = _bfs_scan(world, x, y, radius)

    neighbor_costs: dict[Position, float] = {}
    neighbor_hazards: dict[Position, float] = {}
    neighbor_occupancy: dict[Position, int] = {}
    nearby_positions: list[Position] = []
    nearby_count = 0

    # Pheromone accumulators
    pheromone_readings: dict[str, dict[Position, float]] = {
        name: {} for name in world.layers
    }

    hazard_dx, hazard_dy = 0.0, 0.0
    max_hazard = 0.0

    effective_costs = world.effective_cost_grid()

    for nx, ny, dist in scanned_cells:
        key = Position(nx, ny)
        neighbor_costs[key] = float(effective_costs[ny, nx])
        neighbor_hazards[key] = float(world.hazard_grid[ny, nx])
        neighbor_occupancy[key] = int(world.occupancy_grid[ny, nx])

        # Track nearby agents
        agents_here = world.agents_at(nx, ny)
        agent_count = len(agents_here - {agent_id})
        if agent_count > 0:
            nearby_positions.append(key)
            nearby_count += agent_count

        # Aggregate hazard gradient (scalar summary, backward-compatible)
        hz = float(world.hazard_grid[ny, nx])
        ddx, ddy = nx - x, ny - y
        if hz > 1e-6 and (ddx != 0 or ddy != 0):
            d = max(1.0, float(np.sqrt(ddx * ddx + ddy * ddy)))
            hazard_dx += (ddx / d) * hz
            hazard_dy += (ddy / d) * hz
        max_hazard = max(max_hazard, hz)

        # Read pheromone layers
        for layer_name, layer in world.layers.items():
            val = layer.get(nx, ny)
            if val > 1e-6:
                pheromone_readings[layer_name][key] = val

    # ── Normalize scalar hazard direction ───────────────────────────
    mag = float(np.sqrt(hazard_dx ** 2 + hazard_dy ** 2))
    if mag > 1e-8:
        hazard_direction = (hazard_dx / mag, hazard_dy / mag)
    else:
        hazard_direction = (0.0, 0.0)

    # ── Build multi-hazard lobes ────────────────────────────────────
    hazard_lobes = _build_hazard_lobes(x, y, scanned_cells, world.hazard_grid)

    # ── Hazard rate-of-change at agent position ─────────────────────
    hazard_roc = 0.0
    if world.has_layer("hazard_prev"):
        prev = world.get_layer("hazard_prev").get(x, y)
        curr = float(world.hazard_grid[y, x])
        hazard_roc = curr - prev  # positive = worsening

    # ── Find nearest exit ───────────────────────────────────────────
    nearest_exit_dir = None
    nearest_exit_dist = None
    if world.exits:
        best_dist = float("inf")
        best_dx, best_dy = 0.0, 0.0
        for ex in world.exits:
            edist = position.euclidean_distance(ex)
            if edist < best_dist:
                best_dist = edist
                ddx = ex.x - x
                ddy = ex.y - y
                d = max(1.0, float(np.sqrt(ddx ** 2 + ddy ** 2)))
                best_dx = ddx / d
                best_dy = ddy / d
        nearest_exit_dir = (best_dx, best_dy)
        nearest_exit_dist = best_dist

    # ── Read flow field if available ────────────────────────────────
    flow_dir = None
    if world.has_layer("flow_x") and world.has_layer("flow_y"):
        fx = world.get_layer("flow_x").get(x, y)
        fy = world.get_layer("flow_y").get(x, y)
        if abs(fx) > 1e-6 or abs(fy) > 1e-6:
            flow_dir = (fx, fy)

    return LocalPercept(
        position=position,
        tick=tick,
        walkable_neighbors=walkable,
        neighbor_costs=neighbor_costs,
        neighbor_hazards=neighbor_hazards,
        neighbor_occupancy=neighbor_occupancy,
        pheromone_readings=pheromone_readings,
        nearby_agent_positions=nearby_positions,
        nearby_agent_count=nearby_count,
        max_local_hazard=max_hazard,
        hazard_direction=hazard_direction,
        hazard_lobes=hazard_lobes,
        hazard_rate_of_change=hazard_roc,
        nearest_exit_direction=nearest_exit_dir,
        nearest_exit_distance=nearest_exit_dist,
        flow_direction=flow_dir,
    )
