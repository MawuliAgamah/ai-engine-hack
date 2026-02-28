from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .mesa_compat import Agent

if TYPE_CHECKING:
    from .model import SouthwarkModel


class GiraffeAgent(Agent):
    """Single roaming giraffe with NetLogo-equivalent energy dynamics."""

    def __init__(self, unique_id: int, model: "SouthwarkModel") -> None:
        self._init_agent_base(unique_id=unique_id, model=model)
        self.energy: float = 100.0
        self.distance_travelled: int = 0
        self.start_patch_x: int = -1
        self.start_patch_y: int = -1
        self.total_green_consumed: float = 0.0

    def _init_agent_base(self, unique_id: int, model: "SouthwarkModel") -> None:
        # Mesa Agent signature changed across major versions; handle both styles.
        try:
            super().__init__(unique_id=unique_id, model=model)
            return
        except TypeError:
            pass

        try:
            super().__init__(unique_id, model)
            return
        except TypeError:
            pass

        super().__init__(model)
        self.unique_id = unique_id

    @property
    def rng(self) -> Any:
        return getattr(self, "random", self.model.random)

    def step(self) -> None:
        if self.pos is None:
            return

        moved = False
        neighbors = self.model.get_traversable_neighbors(self.pos)
        if neighbors:
            target = self.rng.choice(neighbors)
            self.model.grid.move_agent(self, target)
            self.distance_travelled += 1
            moved = True

        self.energy -= self.model.move_cost

        green_consumed = self.model.consume_green_resource(self.pos, self.model.green_energy_gain)
        self.energy += green_consumed
        self.total_green_consumed += green_consumed

        if self.energy > 100:
            self.energy = 100

        self.model.record_agent_step(
            self,
            moved=moved,
            green_consumed=green_consumed,
            action="move" if moved else "stay",
            tick_value=self.model.tick + 1,
        )

        if self.energy <= 0:
            self.model.remove_giraffe(self, reason="energy_depleted", tick=self.model.tick + 1)
