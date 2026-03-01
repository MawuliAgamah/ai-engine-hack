"""
LLM-driven swarm orchestrator.

``LLMSwarm`` manages a population of ``LLMAgent`` instances and drives the
three-phase simulation loop:

1. **Perceive** — all active agents observe their local environment.
2. **Decide**   — all observations are sent to the LLM (potentially in a
   batch) and responses are parsed into ranked-action decisions.
3. **Execute**  — movements are applied to the world grid and agent memory
   is updated.

This phased design means *all* agents perceive the same world snapshot before
any agent moves, eliminating order-dependent bias.

``LLMSwarm`` exposes the same public interface as the algorithmic ``Swarm``
(``step_all``, ``get_stats``, ``active_agents``, etc.) so it is a drop-in
replacement inside ``SimulationEngine``.
"""

from __future__ import annotations

import numpy as np

from swarm.agents.base import AgentState
from swarm.agents.llm_agent import LLMAgent
from swarm.agents.swarm import SwarmStats
from swarm.core.world import Position, World
from swarm.llm.client import LLMClient


class LLMSwarm:
    """Swarm of LLM-driven agents with phased, order-independent stepping."""

    def __init__(self, client: LLMClient, seed: int = 42):
        self.client = client
        self.agents: dict[int, LLMAgent] = {}
        self._next_id = 0
        self.rng = np.random.default_rng(seed)

    # ── Spawning ──────────────────────────────────────────────────

    def spawn_agent(
        self,
        world: World,
        position: Position,
        goal: str,
        personality: str,
        scenario: str,
        awareness_radius: float = 8.0,
    ) -> LLMAgent:
        """Create and place a single LLM-driven agent."""
        agent = LLMAgent(
            agent_id=self._next_id,
            position=position,
            goal=goal,
            personality=personality,
            scenario=scenario,
            client=self.client,
            awareness_radius=awareness_radius,
            seed=int(self.rng.integers(0, 2**31)),
        )
        self.agents[agent.id] = agent
        world.place_agent(agent.id, position.x, position.y)
        self._next_id += 1
        return agent

    def spawn_batch(
        self,
        world: World,
        count: int,
        goal: str,
        personality: str,
        scenario: str,
        awareness_radius: float = 8.0,
        spawn_area: tuple[int, int, int, int] | None = None,
    ) -> list[LLMAgent]:
        """Spawn *count* agents with the same goal/personality text.

        For heterogeneous populations, call ``spawn_agent`` individually
        with per-agent personality strings.
        """
        positions = self._get_spawn_positions(world, count, spawn_area)
        return [
            self.spawn_agent(
                world, pos, goal, personality, scenario, awareness_radius
            )
            for pos in positions
        ]

    # ── Phased stepping ───────────────────────────────────────────

    def step_all(self, world: World, tick: int) -> None:
        """Execute one simulation tick with phased ordering.

        Phase 1 — **Perceive**: every active agent reads the *same*
        world snapshot.

        Phase 2 — **Decide**: each perception is sent to the LLM.
        (Override ``_decide_batch`` for true parallel / batch API calls.)

        Phase 3 — **Execute**: chosen moves are applied to the world
        grid, agent state and memory are updated, pheromones deposited.
        """
        active = [a for a in self.agents.values() if a.is_active]
        # Shuffle to avoid consistent bias in execution order
        self.rng.shuffle(active)

        # Phase 1: Perceive
        agent_percepts: dict[int, tuple[LLMAgent, object]] = {}
        for agent in active:
            percept = agent.perceive_env(world, tick)
            if percept is not None:
                agent_percepts[agent.id] = (agent, percept)

        # Phase 2: Decide
        decisions = self._decide_batch(agent_percepts, tick)

        # Phase 3: Execute
        for aid, (agent, _percept) in agent_percepts.items():
            if aid in decisions:
                agent.execute(world, decisions[aid], tick)
                agent.deposit_pheromones(world)

    def _decide_batch(
        self,
        agent_percepts: dict[int, tuple[LLMAgent, object]],
        tick: int,
    ) -> dict[int, object]:
        """Build all prompts, call the LLM in a single batch, parse results.

        This gathers messages from every agent first, sends them all
        through ``client.complete_batch`` (which may execute concurrently
        on a GPU inference server), and then parses all responses.
        """
        from swarm.agents.perception import LocalPercept  # avoid circular

        # ── Build all messages (CPU-only) ─────────────────────────
        ordered_ids: list[int] = []
        all_messages: list[list[dict[str, str]]] = []
        all_available: list[list[Position]] = []
        all_fallbacks: list[Position] = []

        for aid, (agent, percept) in agent_percepts.items():
            assert isinstance(percept, LocalPercept)
            messages, available = agent.build_messages(percept, tick)
            ordered_ids.append(aid)
            all_messages.append(messages)
            all_available.append(available)
            all_fallbacks.append(percept.position)

        if not ordered_ids:
            return {}

        # ── Batch LLM call (GPU / network) ────────────────────────
        responses = self.client.complete_batch(all_messages)

        # ── Parse all responses (CPU-only) ────────────────────────
        decisions: dict[int, object] = {}
        for aid, resp, avail, fb in zip(
            ordered_ids, responses, all_available, all_fallbacks
        ):
            agent = agent_percepts[aid][0]
            decisions[aid] = agent.parse_decision(resp, avail, fb)

        return decisions

    # ── Statistics (engine-compatible) ────────────────────────────

    def get_stats(self) -> SwarmStats:
        agents = list(self.agents.values())
        total = len(agents)
        active = sum(1 for a in agents if a.is_active)
        evacuated = sum(1 for a in agents if a.state == AgentState.EVACUATED)
        dead = sum(1 for a in agents if a.state == AgentState.DEAD)
        stuck = sum(1 for a in agents if a.state == AgentState.STUCK)
        panicking = sum(1 for a in agents if a.state == AgentState.PANIC)

        speeds = [
            float(np.sqrt(a.velocity[0] ** 2 + a.velocity[1] ** 2))
            for a in agents
            if a.is_active
        ]
        progress = [
            float(a.memory.ticks_since_progress)
            for a in agents
            if a.is_active
        ]

        return SwarmStats(
            total=total,
            active=active,
            evacuated=evacuated,
            dead=dead,
            stuck=stuck,
            panicking=panicking,
            mean_speed=float(np.mean(speeds)) if speeds else 0.0,
            mean_progress=float(np.mean(progress)) if progress else 0.0,
        )

    # ── Accessors ─────────────────────────────────────────────────

    @property
    def active_agents(self) -> list[LLMAgent]:
        return [a for a in self.agents.values() if a.is_active]

    @property
    def all_agents(self) -> list[LLMAgent]:
        return list(self.agents.values())

    # ── Internal ──────────────────────────────────────────────────

    def _get_spawn_positions(
        self,
        world: World,
        count: int,
        spawn_area: tuple[int, int, int, int] | None,
    ) -> list[Position]:
        if spawn_area:
            sx, sy, sw, sh = spawn_area
            candidates = [
                Position(sx + dx, sy + dy)
                for dy in range(sh)
                for dx in range(sw)
                if world.in_bounds(sx + dx, sy + dy)
                and world.walkable_grid[sy + dy, sx + dx]
            ]
        else:
            candidates = list(world.walkable_positions())

        if len(candidates) < count:
            raise ValueError(
                f"Not enough walkable cells ({len(candidates)}) for {count} agents"
            )
        indices = self.rng.choice(len(candidates), size=count, replace=False)
        return [candidates[i] for i in indices]

    def __len__(self) -> int:
        return len(self.agents)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"LLMSwarm(total={stats.total}, active={stats.active}, "
            f"evacuated={stats.evacuated}, dead={stats.dead})"
        )
