"""
Pheromone system — stigmergic indirect communication.

Agents communicate by depositing chemical-like signals on the grid that
other agents can sense. Pheromones decay over time and diffuse spatially,
creating dynamic information fields.

Inspired by Ant Colony Optimization (Dorigo, 1992) and adapted for
spatial navigation scenarios.

Pheromone types:
- "visited": Marks explored cells (negative signal — spread out)
- "exit_path": Trail toward exits left by successful agents (positive — follow)
- "danger": Warning about hazards detected ahead (negative — avoid)
- "congestion": Marks crowded areas (negative — seek alternatives)
"""

from __future__ import annotations

from dataclasses import dataclass

from swarm.core.world import World


@dataclass
class PheromoneConfig:
    """Configuration for a pheromone type."""

    name: str
    decay_rate: float = 0.05     # Multiplicative decay per tick
    diffusion_rate: float = 0.1  # Spatial diffusion rate per tick
    max_value: float = 100.0     # Clamp maximum
    min_value: float = 0.0       # Clamp minimum (usually 0)


# Default pheromone configurations
DEFAULT_PHEROMONES: list[PheromoneConfig] = [
    PheromoneConfig(name="visited", decay_rate=0.02, diffusion_rate=0.05, max_value=50.0),
    PheromoneConfig(name="exit_path", decay_rate=0.03, diffusion_rate=0.15, max_value=100.0),
    PheromoneConfig(name="danger", decay_rate=0.08, diffusion_rate=0.2, max_value=100.0),
    PheromoneConfig(name="congestion", decay_rate=0.1, diffusion_rate=0.1, max_value=50.0),
]


class PheromoneSystem:
    """
    Manages all pheromone layers on the world grid.

    Each tick:
    1. Agents deposit pheromones (handled by Agent._deposit_pheromones)
    2. This system decays and diffuses all pheromone layers
    3. Agents read pheromone values during perception

    The result is emergent collective navigation: if many agents find an exit,
    the pheromone trail to that exit gets reinforced, guiding others.
    """

    def __init__(self, world: World, configs: list[PheromoneConfig] | None = None):
        self.world = world
        self.configs = configs or DEFAULT_PHEROMONES
        self._layers: dict[str, PheromoneConfig] = {}

        # Initialize pheromone layers on the world
        for config in self.configs:
            world.add_layer(config.name, default=0.0)
            self._layers[config.name] = config

    def update(self) -> None:
        """
        Update all pheromone layers: decay and diffuse.

        Should be called once per tick after all agents have deposited.
        """
        for name, config in self._layers.items():
            layer = self.world.get_layer(name)

            # Decay
            layer.decay(config.decay_rate)

            # Diffuse
            layer.diffuse(config.diffusion_rate)

            # Clamp
            layer.clamp(config.min_value, config.max_value)

    def deposit(self, name: str, x: int, y: int, amount: float) -> None:
        """Directly deposit pheromone at a position."""
        if name in self._layers:
            self.world.get_layer(name).add(x, y, amount)

    def read(self, name: str, x: int, y: int) -> float:
        """Read pheromone value at a position."""
        if name in self._layers:
            return self.world.get_layer(name).get(x, y)
        return 0.0

    def get_total(self, name: str) -> float:
        """Get total pheromone deposited across the grid."""
        if name in self._layers:
            return self.world.get_layer(name).sum()
        return 0.0

    def reset(self, name: str | None = None) -> None:
        """Reset one or all pheromone layers."""
        if name is not None:
            if name in self._layers:
                self.world.get_layer(name).reset()
        else:
            for n in self._layers:
                self.world.get_layer(n).reset()

    def summary(self) -> dict[str, float]:
        """Summary of total pheromone quantities per layer."""
        return {name: self.get_total(name) for name in self._layers}
