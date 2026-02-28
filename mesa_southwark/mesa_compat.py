from __future__ import annotations

import random
from itertools import product
from typing import Any

try:
    from mesa import Agent as MesaAgent
    from mesa import Model as MesaModel
    from mesa.space import MultiGrid as MesaMultiGrid

    Agent = MesaAgent
    Model = MesaModel
    MultiGrid = MesaMultiGrid
    MESA_BACKEND = "mesa"
except ImportError:
    MESA_BACKEND = "fallback"

    class Model:  # type: ignore[no-redef]
        def __init__(self, seed: int | None = None) -> None:
            self.random = random.Random(seed)
            self.running = True

    class Agent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if "model" in kwargs and "unique_id" in kwargs:
                self.unique_id = kwargs["unique_id"]
                self.model = kwargs["model"]
            elif len(args) == 2:
                self.unique_id = args[0]
                self.model = args[1]
            elif len(args) == 1:
                self.model = args[0]
                self.unique_id = kwargs.get("unique_id", 0)
            else:
                raise TypeError("Fallback Agent expects (unique_id, model) or (model)")
            self.pos: tuple[int, int] | None = None

        @property
        def random(self) -> random.Random:
            return self.model.random

    class MultiGrid:  # type: ignore[no-redef]
        def __init__(self, width: int, height: int, torus: bool = False) -> None:
            self.width = width
            self.height = height
            self.torus = torus
            self._cells: dict[tuple[int, int], list[Agent]] = {}

        def place_agent(self, agent: Agent, pos: tuple[int, int]) -> None:
            self._cells.setdefault(pos, []).append(agent)
            agent.pos = pos

        def move_agent(self, agent: Agent, pos: tuple[int, int]) -> None:
            if agent.pos is None:
                self.place_agent(agent, pos)
                return
            current = self._cells.get(agent.pos, [])
            if agent in current:
                current.remove(agent)
            if not current and agent.pos in self._cells:
                self._cells.pop(agent.pos, None)
            self._cells.setdefault(pos, []).append(agent)
            agent.pos = pos

        def remove_agent(self, agent: Agent) -> None:
            if agent.pos is None:
                return
            current = self._cells.get(agent.pos, [])
            if agent in current:
                current.remove(agent)
            if not current and agent.pos in self._cells:
                self._cells.pop(agent.pos, None)
            agent.pos = None

        def get_neighborhood(
            self,
            pos: tuple[int, int],
            moore: bool = True,
            include_center: bool = False,
            radius: int = 1,
        ) -> list[tuple[int, int]]:
            x, y = pos
            neighbors: list[tuple[int, int]] = []

            for dx, dy in product(range(-radius, radius + 1), repeat=2):
                if dx == 0 and dy == 0 and not include_center:
                    continue
                if not moore and abs(dx) + abs(dy) > radius:
                    continue

                nx = x + dx
                ny = y + dy

                if self.torus:
                    nx %= self.width
                    ny %= self.height
                elif nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue

                neighbors.append((nx, ny))

            return neighbors
