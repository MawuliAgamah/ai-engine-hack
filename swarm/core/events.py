"""
Event and hazard scheduling system.

Defines hazard types (fire, flood, obstruction, gas) and their temporal evolution.
Hazards are applied to the world grid and can spread, intensify, or recede over time.

Spread is **cell-structure agnostic**: instead of iterating a rectangular bounding
box, we BFS from the origin using ``world.neighbors()`` — so the same code works
for square (Moore / Von Neumann), hexagonal, or any other topology the world
provides.

Directional spread is controlled via ``spread_direction``:
  - ``None`` (default) → uniform radial (isotropic) spread.
  - A list of ``(angle_degrees, strength)`` tuples → directional lobes.
    * ``angle_degrees`` is measured **clockwise from North**
      (0°=N, 90°=E, 180°=S, 270°=W).
    * ``strength`` in [0, 1] controls how focused the lobe is:
      0 = no bias (uniform), 1 = fully directional (zero behind).
  - Multiple tuples create independent lobes; the **maximum** weight across
    lobes is used at each cell — so two lobes don't over-count.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from swarm.core.world import Position, World


@dataclass
class ScenarioEvent:
    """
    A discrete scenario event (e.g., alarm sounds, door opens, road closes).

    Attributes:
        name: Human-readable event name.
        tick: When the event fires.
        action: Callable that mutates the world state.
    """

    name: str
    tick: int
    action: Callable[[World], None]


class HazardType(enum.Enum):
    FIRE = "fire"
    FLOOD = "flood"
    OBSTRUCTION = "obstruction"  # e.g., construction, road closure
    GAS = "gas"                  # toxic gas / smoke


@dataclass
class HazardEvent:
    """
    A hazard that evolves over time on the world grid.

    The spread algorithm is **topology-agnostic**: it performs a BFS from
    *origin* using the world's own neighbour connectivity, then applies an
    intensity that is the product of:

        intensity × distance_falloff × directional_weight

    Attributes:
        hazard_type: Type of hazard.
        origin: Starting position (grid coordinates).
        start_tick: Simulation tick when the hazard begins.
        intensity: Peak hazard level at origin [0, 1].
        spread_direction:
            Controls *directional* spread.  ``None`` (default) gives uniform
            radial (isotropic) spread.  Otherwise a list of
            ``(angle_degrees, strength)`` tuples where:
              - *angle_degrees* is measured **clockwise from North**
                (0 = North / -Y, 90 = East / +X, 180 = South / +Y,
                270 = West / -X).
              - *strength* ∈ [0, 1] blends between uniform (0) and
                fully-directional (1).
            Multiple directions create independent lobes — the
            **maximum** weight across lobes is taken for each cell.
        spread_rate: Speed of radius expansion per tick
                     (current_radius = elapsed × spread_rate × 10, capped).
        spread_radius: Maximum Euclidean radius from origin.
        duration: How many ticks the hazard lasts (0 = permanent).
    """

    hazard_type: HazardType
    origin: Position
    start_tick: int
    intensity: float = 0.8
    spread_direction: list[tuple[float, float]] | None = None
    spread_rate: float = 0.05
    spread_radius: int = 20
    duration: int = 0  # 0 = permanent

    # ── helpers ──────────────────────────────────────────────────────

    def _directional_weight(self, dx: int, dy: int) -> float:
        """
        Compute a [0, 1] directional weight for a cell at offset *(dx, dy)*
        from the origin.

        * Uniform mode (``spread_direction is None``): always returns 1.0.
        * Directional mode: for each ``(angle, strength)`` lobe, blends
          between uniform (strength = 0) and cosine-gated (strength = 1).
          Returns the **max** weight across all lobes.

        Angle convention (clockwise from North):
            North = 0°  → (dx=0,  dy=-1)
            East  = 90° → (dx=+1, dy=0)
            South = 180° → (dx=0,  dy=+1)
            West  = 270° → (dx=-1, dy=0)
        """
        if self.spread_direction is None or len(self.spread_direction) == 0:
            return 1.0  # uniform / isotropic

        if dx == 0 and dy == 0:
            return 1.0  # origin always receives full intensity

        # Cell angle from origin, clockwise from North.
        # Grid convention: +x = East, +y = South (screen coords).
        # atan2(east_component, north_component) where north = -dy.
        cell_angle = float(np.degrees(np.arctan2(dx, -dy))) % 360.0

        best: float = 0.0
        for dir_angle, strength in self.spread_direction:
            strength = float(np.clip(strength, 0.0, 1.0))
            dir_angle = float(dir_angle) % 360.0

            # Signed angular difference in [-180, 180]
            diff = (cell_angle - dir_angle + 180.0) % 360.0 - 180.0
            cos_factor = float(np.cos(np.radians(diff)))

            # Blend: strength=0 → weight=1 (uniform), strength=1 → weight=cos
            # Clamp the cosine term so cells *behind* the lobe get 0, not negative.
            w = (1.0 - strength) + strength * max(0.0, cos_factor)
            best = max(best, w)

        return best

    def _affected_cells(
        self, world: World, current_radius: float
    ) -> list[tuple[int, int, float]]:
        """
        BFS flood-fill from *origin* to discover every reachable cell within
        *current_radius* (Euclidean).

        Uses ``world.neighbors(x, y, walkable_only=False)`` so the iteration
        automatically follows whatever connectivity the world defines
        (Moore 8-connected, Von Neumann 4-connected, hex 6-connected, …).

        Returns:
            List of ``(x, y, euclidean_distance_from_origin)`` tuples.
        """
        ox, oy = self.origin.x, self.origin.y
        if not world.in_bounds(ox, oy):
            return []

        visited: dict[tuple[int, int], float] = {(ox, oy): 0.0}
        queue: deque[tuple[int, int]] = deque([(ox, oy)])

        while queue:
            x, y = queue.popleft()
            for nb in world.neighbors(x, y, walkable_only=False):
                key = (nb.x, nb.y)
                if key in visited:
                    continue
                dist = np.sqrt(float((nb.x - ox) ** 2 + (nb.y - oy) ** 2))
                if dist > current_radius:
                    continue
                visited[key] = dist
                queue.append((nb.x, nb.y))

        return [(x, y, d) for (x, y), d in visited.items()]

    # ── main entry point ─────────────────────────────────────────────

    def apply(self, world: World, current_tick: int) -> None:
        """
        Apply / update this hazard on the world grid for *current_tick*.

        1. Compute the current spread radius from elapsed time.
        2. BFS from origin to discover all cells within that radius
           (cell-structure agnostic).
        3. For each cell, compute ``intensity × distance_falloff ×
           directional_weight`` and write the hazard level.
        """
        if current_tick < self.start_tick:
            return

        elapsed = current_tick - self.start_tick

        # Check if hazard has expired
        if self.duration > 0 and elapsed > self.duration:
            return

        # Compute current spread radius (grows with time, capped)
        current_radius = min(
            elapsed * self.spread_rate * 10.0, float(self.spread_radius)
        )

        # Discover affected cells via topology-agnostic BFS
        ox, oy = self.origin.x, self.origin.y
        affected = self._affected_cells(world, current_radius)

        for cx, cy, dist in affected:
            # Intensity falls off linearly with Euclidean distance
            falloff = max(0.0, 1.0 - dist / max(current_radius, 1.0))

            # Directional weighting (uniform when spread_direction is None)
            dx, dy = cx - ox, cy - oy
            dir_weight = self._directional_weight(dx, dy)

            level = self.intensity * falloff * dir_weight

            # Hazard accumulates (doesn't replace lower values)
            current = float(world.hazard_grid[cy, cx])
            world.set_hazard(cx, cy, max(current, level))

    @property
    def end_tick(self) -> int | None:
        if self.duration == 0:
            return None
        return self.start_tick + self.duration
    

class EventScheduler:
    """
    Manages all hazards and scenario events, applying them at the right ticks.
    """

    def __init__(self) -> None:
        self.hazards: list[HazardEvent] = []
        self.events: list[ScenarioEvent] = []

    def add_hazard(self, hazard: HazardEvent) -> None:
        self.hazards.append(hazard)

    def add_event(self, event: ScenarioEvent) -> None:
        self.events.append(event)

    def process_tick(self, world: World, tick: int) -> list[str]:
        """
        Process all hazards and events for the current tick.

        Returns:
            Names of events that fired this tick.
        """
        fired: list[str] = []

        # Apply hazards
        for hazard in self.hazards:
            hazard.apply(world, tick)

        # Fire discrete events
        for event in self.events:
            if event.tick == tick:
                event.action(world)
                fired.append(event.name)

        return fired

    def active_hazards(self, tick: int) -> list[HazardEvent]:
        """Get hazards that are currently active."""
        active = []
        for h in self.hazards:
            if tick < h.start_tick:
                continue
            if h.duration > 0 and tick > h.start_tick + h.duration:
                continue
            active.append(h)
        return active