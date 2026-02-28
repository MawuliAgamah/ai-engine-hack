"""
Gradient and potential fields — pre-computed spatial guidance.

Fields provide global navigation information that agents can read locally:
- Goal distance fields: Dijkstra-computed cost-to-nearest-exit per cell
- Flow fields: Gradient of distance fields (direction to go)
- Danger potential fields: Repulsive fields around hazards

These are recomputed periodically (not every tick) and cached as PropertyLayers.
"""

from __future__ import annotations

import numpy as np

from swarm.agents.decisions.tactical import (
    compute_flow_field,
    compute_goal_distance_field,
)
from swarm.core.world import World


class FieldManager:
    """
    Manages pre-computed spatial fields for the simulation.

    Updates fields periodically when the world state changes significantly
    (new hazards, blocked routes, etc.).
    """

    def __init__(self, world: World, update_interval: int = 20):
        self.world = world
        self.update_interval = update_interval
        self._last_update_tick = -1
        self._hazard_hash: float = 0.0

    def update(self, tick: int, force: bool = False) -> bool:
        """
        Recompute fields if necessary.

        Returns True if fields were updated.
        """
        if not force and not self._should_update(tick):
            return False

        self._compute_goal_fields()
        self._last_update_tick = tick
        self._hazard_hash = float(np.sum(self.world.hazard_grid))
        return True

    def _should_update(self, tick: int) -> bool:
        """Determine if fields need recomputation."""
        # First computation
        if self._last_update_tick < 0:
            return True

        # Periodic update
        if tick - self._last_update_tick >= self.update_interval:
            return True

        # Environment changed significantly (hazard shifted)
        current_hazard = float(np.sum(self.world.hazard_grid))
        if abs(current_hazard - self._hazard_hash) > 1.0:
            return True

        return False

    def _compute_goal_fields(self) -> None:
        """Compute distance and flow fields toward exits."""
        if not self.world.exits:
            return

        # Distance field
        dist_field = compute_goal_distance_field(self.world, self.world.exits)

        # Store as property layer
        if not self.world.has_layer("goal_distance"):
            self.world.add_layer("goal_distance")
        self.world.get_layer("goal_distance").data = dist_field

        # Flow field
        flow_x, flow_y = compute_flow_field(self.world, dist_field)

        if not self.world.has_layer("flow_x"):
            self.world.add_layer("flow_x")
        if not self.world.has_layer("flow_y"):
            self.world.add_layer("flow_y")

        self.world.get_layer("flow_x").data = flow_x
        self.world.get_layer("flow_y").data = flow_y
