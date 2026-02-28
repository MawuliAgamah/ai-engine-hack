"""
Swarm manager — creates, manages, and orchestrates the agent population.

Handles:
- Batch agent creation with stochastic personalities
- Agent placement on the world grid
- Coordinated stepping of all agents
- Population-level statistics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.agents.base import Agent, AgentState
from swarm.agents.personality import PersonalityDistribution
from swarm.core.world import Position, World


@dataclass
class SwarmStats:
    """Population-level statistics for the swarm."""

    total: int
    active: int
    evacuated: int
    dead: int
    stuck: int
    panicking: int
    mean_speed: float
    mean_progress: float


class Swarm:
    """
    Swarm manager — the container for all agents.

    Creates agents with stochastic personalities, places them on the world,
    and coordinates their per-tick execution.
    """

    def __init__(self, seed: int = 42):
        self.agents: dict[int, Agent] = {}
        self._next_id = 0
        self.rng = np.random.default_rng(seed)

    def spawn_agents(
        self,
        world: World,
        count: int,
        distribution: PersonalityDistribution,
        positions: list[Position] | None = None,
        spawn_area: tuple[int, int, int, int] | None = None,
    ) -> list[Agent]:
        """
        Spawn a batch of agents with stochastic personalities.

        Args:
            world: The world to place agents in.
            count: Number of agents to create.
            distribution: Personality distribution to sample from.
            positions: Explicit positions (must match count). If None, random placement.
            spawn_area: (x, y, w, h) rectangle to spawn agents in randomly.

        Returns:
            List of created agents.
        """
        personalities = distribution.sample_batch(count, self.rng)

        # Determine positions
        if positions is not None:
            assert len(positions) == count
            spawn_positions = positions
        elif spawn_area is not None:
            sx, sy, sw, sh = spawn_area
            spawn_positions = self._random_positions_in_rect(
                world, count, sx, sy, sw, sh
            )
        else:
            spawn_positions = self._random_walkable_positions(world, count)

        created: list[Agent] = []
        for i in range(count):
            agent = Agent(
                agent_id=self._next_id,
                position=spawn_positions[i],
                personality=personalities[i],
                seed=int(self.rng.integers(0, 2**31)),
            )
            self.agents[agent.id] = agent
            world.place_agent(agent.id, agent.position.x, agent.position.y)
            self._next_id += 1
            created.append(agent)

        return created

    # TODO: make async version of this for parallel stepping
    def step_all(self, world: World, tick: int) -> None:
        """
        Step all active agents for one tick.

        Agents are stepped in shuffled order to avoid systematic bias
        from sequential processing.
        """
        active_ids = [
            aid for aid, agent in self.agents.items() if agent.is_active
        ]
        self.rng.shuffle(active_ids)

        for aid in active_ids:
            self.agents[aid].step(world, tick)

    def get_stats(self) -> SwarmStats:
        """Compute population-level statistics."""
        agents = list(self.agents.values())
        total = len(agents)
        active = sum(1 for a in agents if a.is_active)
        evacuated = sum(1 for a in agents if a.state == AgentState.EVACUATED)
        dead = sum(1 for a in agents if a.state == AgentState.DEAD)
        stuck = sum(1 for a in agents if a.state == AgentState.STUCK)
        panicking = sum(1 for a in agents if a.state == AgentState.PANIC)

        speeds = [
            np.sqrt(a.velocity[0] ** 2 + a.velocity[1] ** 2)
            for a in agents
            if a.is_active
        ]
        mean_speed = float(np.mean(speeds)) if speeds else 0.0

        progress = [
            float(a.memory.ticks_since_progress) for a in agents if a.is_active
        ]
        mean_progress = float(np.mean(progress)) if progress else 0.0

        return SwarmStats(
            total=total,
            active=active,
            evacuated=evacuated,
            dead=dead,
            stuck=stuck,
            panicking=panicking,
            mean_speed=mean_speed,
            mean_progress=mean_progress,
        )

    def _random_walkable_positions(
        self, world: World, count: int
    ) -> list[Position]:
        """Pick random walkable positions on the grid."""
        walkable = list(world.walkable_positions())
        if len(walkable) < count:
            raise ValueError(
                f"Not enough walkable cells ({len(walkable)}) for {count} agents"
            )
        indices = self.rng.choice(len(walkable), size=count, replace=False)
        return [walkable[i] for i in indices]

    def _random_positions_in_rect(
        self, world: World, count: int, x: int, y: int, w: int, h: int
    ) -> list[Position]:
        """Pick random walkable positions within a rectangle."""
        candidates = []
        for dy in range(h):
            for dx in range(w):
                px, py = x + dx, y + dy
                if world.in_bounds(px, py) and world.walkable_grid[py, px]:
                    candidates.append(Position(px, py))
        if len(candidates) < count:
            raise ValueError(
                f"Not enough walkable cells in spawn area ({len(candidates)}) for {count} agents"
            )
        indices = self.rng.choice(len(candidates), size=count, replace=False)
        return [candidates[i] for i in indices]

    @property
    def active_agents(self) -> list[Agent]:
        return [a for a in self.agents.values() if a.is_active]

    @property
    def all_agents(self) -> list[Agent]:
        return list(self.agents.values())

    def __len__(self) -> int:
        return len(self.agents)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"Swarm(total={stats.total}, active={stats.active}, "
            f"evacuated={stats.evacuated}, dead={stats.dead})"
        )
