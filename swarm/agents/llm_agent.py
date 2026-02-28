"""
LLM-driven agent — replaces the three-layer decision stack with a single LLM call.

Each tick the agent:

1. **Perceives** its local environment (BFS scan via ``perceive()``).
2. **Builds a prompt** containing perception, available moves, journey history,
   and prior reasoning.
3. **Calls the LLM** and parses the response into a ranked list of positions.
4. **Executes** the top-ranked valid move and updates memory.

Personality and goals are expressed in **natural language** (not numeric traits),
so the LLM can reason about them in a human-interpretable way.

The ``LLMAgent`` exposes the same ``step(world, tick)`` convenience method as
the algorithmic ``Agent``, plus split ``perceive → decide → execute`` methods
for use by ``LLMSwarm``'s phased stepping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from swarm.agents.base import AgentState  # reuse existing enum
from swarm.agents.perception import LocalPercept, perceive
from swarm.core.world import Position, World
from swarm.llm.client import LLMClient
from swarm.llm.parser import LLMDecision, parse_llm_response
from swarm.llm.prompt import (
    build_system_prompt,
    build_user_message,
    get_available_moves,
)


# ── Memory ───────────────────────────────────────────────────────────


@dataclass
class LLMAgentMemory:
    """Stores journey history, reasoning history, and decision records."""

    journey: list[tuple[int, Position]] = field(default_factory=list)
    reasoning: list[tuple[int, str]] = field(default_factory=list)
    visited: set[Position] = field(default_factory=set)
    ticks_since_progress: int = 0
    decisions: list[LLMDecision] = field(default_factory=list)

    # How many entries to include in each prompt
    JOURNEY_WINDOW: int = 20
    REASONING_WINDOW: int = 5


# ── Agent ────────────────────────────────────────────────────────────


class LLMAgent:
    """An autonomous agent whose decisions are delegated to an LLM.

    Parameters
    ----------
    agent_id:
        Unique integer identifier.
    position:
        Starting grid position.
    goal:
        Natural-language description of what this agent is trying to do.
    personality:
        Natural-language description of the agent's personality / biases.
    scenario:
        Natural-language description of the simulation context.
    client:
        An ``LLMClient`` instance (shared across agents).
    awareness_radius:
        BFS perception radius in grid cells (numeric, controls ``perceive``).
    seed:
        RNG seed (used only for tie-breaking; decisions come from the LLM).
    """

    def __init__(
        self,
        agent_id: int,
        position: Position,
        goal: str,
        personality: str,
        scenario: str,
        client: LLMClient,
        awareness_radius: float = 8.0,
        seed: int | None = None,
    ):
        self.id = agent_id
        self.position = position
        self.goal = goal
        self.personality = personality
        self.scenario = scenario
        self.client = client
        self.awareness_radius = awareness_radius
        self.rng = np.random.default_rng(seed)

        self.state = AgentState.NAVIGATING
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.memory = LLMAgentMemory()

        # System prompt is static for the life of the agent — build once.
        self._system_prompt = build_system_prompt(
            agent_id=agent_id,
            scenario=scenario,
            personality=personality,
            goal=goal,
        )

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.state in (
            AgentState.NAVIGATING,
            AgentState.PANIC,
            AgentState.STUCK,
        )

    # ── Phase 1: PERCEIVE ─────────────────────────────────────────

    def perceive_env(self, world: World, tick: int) -> LocalPercept | None:
        """Run the perception pipeline and check death / evacuation.

        Returns ``None`` if agent is no longer active.
        """
        if not self.is_active:
            return None

        # Death check
        if world.hazard_grid[self.position.y, self.position.x] >= 0.95:
            self.state = AgentState.DEAD
            return None

        # Evacuation check
        for exit_pos in world.exits:
            if self.position == exit_pos:
                self.state = AgentState.EVACUATED
                return None

        return perceive(
            world=world,
            agent_id=self.id,
            position=self.position,
            awareness_radius=self.awareness_radius,
            tick=tick,
        )

    # ── Phase 2: DECIDE (LLM call) ───────────────────────────────

    def decide(self, percept: LocalPercept, tick: int) -> LLMDecision:
        """Build the prompt, call the LLM, and parse the response."""

        available_labeled = get_available_moves(percept)
        available_positions = [pos for _, pos in available_labeled]

        user_message = build_user_message(
            percept=percept,
            available_moves=available_labeled,
            tick=tick,
            state=self.state.value,
            journey_history=self.memory.journey,
            reasoning_history=self.memory.reasoning,
            max_journey=self.memory.JOURNEY_WINDOW,
            max_reasoning=self.memory.REASONING_WINDOW,
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]

        response_text = self.client.complete(messages, agent_id=self.id)

        return parse_llm_response(
            response_text=response_text,
            available=available_positions,
            fallback=percept.position,
        )

    # ── Phase 3: EXECUTE + UPDATE ─────────────────────────────────

    def execute(
        self,
        world: World,
        decision: LLMDecision,
        tick: int,
    ) -> Position | None:
        """Apply the chosen move, update world grids, and record memory.

        Returns the new position if the agent moved, else ``None``.
        """
        chosen = decision.chosen_position
        old_pos = self.position

        if chosen != self.position:
            world.move_agent(self.id, old_pos.x, old_pos.y, chosen.x, chosen.y)
            self.position = chosen
            self.velocity = (
                float(chosen.x - old_pos.x),
                float(chosen.y - old_pos.y),
            )
        else:
            self.velocity = (0.0, 0.0)

        # ── Memory update ─────────────────────────────────────────
        self.memory.journey.append((tick, self.position))
        self.memory.reasoning.append((tick, decision.reasoning))
        self.memory.decisions.append(decision)

        if self.position in self.memory.visited:
            self.memory.ticks_since_progress += 1
        else:
            self.memory.ticks_since_progress = 0
        self.memory.visited.add(self.position)

        # ── State update ──────────────────────────────────────────
        if self.memory.ticks_since_progress > 30:
            self.state = AgentState.STUCK
        else:
            self.state = AgentState.NAVIGATING

        return self.position if chosen != old_pos else None

    # ── Convenience: single-call lifecycle ────────────────────────

    def step(self, world: World, tick: int) -> Position | None:
        """Full tick: perceive → decide (LLM) → execute.

        Compatible with ``Swarm.step_all`` / ``SimulationEngine``.
        """
        percept = self.perceive_env(world, tick)
        if percept is None:
            return None
        decision = self.decide(percept, tick)
        return self.execute(world, decision, tick)

    # ── Pheromone deposit (for engine compatibility) ──────────────

    def deposit_pheromones(self, world: World) -> None:
        x, y = self.position.x, self.position.y
        if world.has_layer("visited"):
            world.get_layer("visited").add(x, y, 1.0)
        if world.has_layer("danger"):
            hz = world.hazard_grid[y, x]
            if hz > 0.2:
                world.get_layer("danger").add(x, y, hz * 3.0)

    # ── Repr ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"LLMAgent(id={self.id}, pos=({self.position.x},{self.position.y}), "
            f"state={self.state.value})"
        )
