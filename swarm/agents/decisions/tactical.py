"""
Tactical decision layer — local pathfinding.

Runs every N ticks (or on demand). Computes a short-range path from the agent's
current position to a nearby waypoint using A* on the local cost grid.

The tactical layer provides a GOAL DIRECTION to the reactive layer, which then
uses social forces to execute movement toward that goal while avoiding collisions.

Also supports gradient descent on pre-computed distance fields for fast routing
when available.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from swarm.core.world import Position, World


@dataclass
class TacticalPlan:
    """
    A short-range plan: a sequence of waypoints to follow.

    Attributes:
        waypoints: Ordered list of positions to visit.
        goal: Final destination of this plan segment.
        computed_at_tick: When this plan was computed.
        valid: Whether the plan is still viable.
    """

    waypoints: list[Position]
    goal: Position
    computed_at_tick: int
    valid: bool = True

    @property
    def next_waypoint(self) -> Position | None:
        return self.waypoints[0] if self.waypoints else None

    def advance(self) -> Position | None:
        """Pop and return the next waypoint."""
        if self.waypoints:
            return self.waypoints.pop(0)
        return None

    @property
    def is_complete(self) -> bool:
        return len(self.waypoints) == 0


def astar_local(
    world: World,
    start: Position,
    goal: Position,
    max_steps: int = 200,
) -> list[Position] | None:
    """
    A* pathfinding on the world grid.

    Uses effective cost (terrain + hazard + crowding) for edge weights.
    Returns a list of positions from start to goal, or None if no path found.

    Args:
        world: The world grid.
        start: Starting position.
        goal: Target position.
        max_steps: Maximum nodes to expand before giving up (bounds compute time).

    Returns:
        Path as list of positions (excluding start), or None if unreachable.
    """
    costs = world.effective_cost_grid()

    # Priority queue: (f_score, counter, position)
    counter = 0
    open_set: list[tuple[float, int, Position]] = []
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: dict[tuple[int, int], Position] = {}
    g_score: dict[tuple[int, int], float] = {start.as_tuple(): 0.0}

    steps = 0

    while open_set and steps < max_steps:
        steps += 1
        _, _, current = heapq.heappop(open_set)

        if current.x == goal.x and current.y == goal.y:
            # Reconstruct path
            path: list[Position] = []
            pos = goal
            while pos.as_tuple() in came_from:
                path.append(pos)
                pos = came_from[pos.as_tuple()]
            path.reverse()
            return path

        for neighbor in world.walkable_neighbors(current.x, current.y, moore=True):
            # Movement cost: effective cost of the destination cell
            move_cost = float(costs[neighbor.y, neighbor.x])
            if move_cost == float("inf"):
                continue

            # Diagonal moves cost sqrt(2) more
            dx = abs(neighbor.x - current.x)
            dy = abs(neighbor.y - current.y)
            dist_mult = 1.414 if (dx + dy) == 2 else 1.0

            tentative_g = g_score[current.as_tuple()] + move_cost * dist_mult

            nkey = neighbor.as_tuple()
            if nkey not in g_score or tentative_g < g_score[nkey]:
                came_from[nkey] = current
                g_score[nkey] = tentative_g
                # Heuristic: octile distance
                h = octile_heuristic(neighbor, goal)
                f = tentative_g + h
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbor))

    return None  # No path found within budget


def octile_heuristic(a: Position, b: Position) -> float:
    """Octile distance heuristic for 8-connected grid."""
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    return max(dx, dy) + (1.414 - 1.0) * min(dx, dy)


def compute_goal_distance_field(
    world: World,
    goals: list[Position],
) -> NDArray[np.float64]:
    """
    Compute a distance field from multiple goal positions using Dijkstra.

    Each cell gets the minimum cost-weighted distance to any goal.
    Agents can follow the gradient of this field to reach any exit efficiently.

    This is computed once (or when the world changes) and cached as a property layer.
    """
    dist = np.full((world.height, world.width), np.inf, dtype=np.float64)
    costs = world.effective_cost_grid()

    # Priority queue: (distance, x, y)
    pq: list[tuple[float, int, int]] = []

    for goal in goals:
        dist[goal.y, goal.x] = 0.0
        heapq.heappush(pq, (0.0, goal.x, goal.y))

    while pq:
        d, x, y = heapq.heappop(pq)
        if d > dist[y, x]:
            continue

        for neighbor in world.walkable_neighbors(x, y, moore=True):
            nx, ny = neighbor.x, neighbor.y
            move_cost = float(costs[ny, nx])
            if move_cost == float("inf"):
                continue

            dx_abs = abs(nx - x)
            dy_abs = abs(ny - y)
            dist_mult = 1.414 if (dx_abs + dy_abs) == 2 else 1.0

            new_dist = d + move_cost * dist_mult
            if new_dist < dist[ny, nx]:
                dist[ny, nx] = new_dist
                heapq.heappush(pq, (new_dist, nx, ny))

    return dist


def compute_flow_field(
    world: World,
    distance_field: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute a flow field (gradient) from a distance field.

    Each cell gets a (dx, dy) vector pointing downhill toward the nearest goal.
    Agents can follow this field for fast goal-directed movement without A* per agent.

    Returns:
        Tuple of (flow_x, flow_y) arrays.
    """
    # Compute gradient (negative = downhill toward goals)
    # Replace inf with large finite value to avoid NaN in gradient
    safe_field = np.where(np.isinf(distance_field), 1e10, distance_field)
    grad_y, grad_x = np.gradient(safe_field)

    # Normalize
    mag = np.sqrt(grad_x**2 + grad_y**2)
    mag = np.where(mag < 1e-8, 1.0, mag)

    flow_x = -grad_x / mag  # Negative because we want to go TOWARD goals
    flow_y = -grad_y / mag

    # Zero out non-walkable cells
    flow_x[~world.walkable_grid] = 0.0
    flow_y[~world.walkable_grid] = 0.0

    # Zero out cells at infinite distance (unreachable)
    unreachable = distance_field == np.inf
    flow_x[unreachable] = 0.0
    flow_y[unreachable] = 0.0

    return flow_x, flow_y


def gradient_direction_at(
    world: World, pos: Position
) -> tuple[float, float] | None:
    """
    Get the flow field direction at a position.

    Returns (dx, dy) unit vector toward nearest goal, or None if no flow field.
    """
    if not world.has_layer("flow_x") or not world.has_layer("flow_y"):
        return None

    fx = world.get_layer("flow_x").get(pos.x, pos.y)
    fy = world.get_layer("flow_y").get(pos.x, pos.y)

    if abs(fx) < 1e-8 and abs(fy) < 1e-8:
        return None

    return (fx, fy)
