"""
Mesa-compatible model wrapping the swarm simulation engine.

Loads the world from real H3 hex-grid reference data (Southwark, London),
spawns LLM-driven agents, and advances the simulation tick-by-tick.
Exposes helpers that the web viewer queries for state snapshots.

Supports two modes:

1. **SimConfig** — programmatic config with a flat personality string.
2. **Scenario YAML** — rich config loaded via ``swarm.scenarios.loader``
   with per-group personality archetypes, hazard events, blackboard
   alerts, and discrete terrain events.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from swarm.agents.llm_agent import LLMAgent
from swarm.agents.llm_swarm import LLMSwarm
from swarm.core.clock import Clock
from swarm.core.events import EventScheduler
from swarm.core.world import Terrain, World
from swarm.llm.client import LLMClient, MockClient
from swarm.scenarios.loader import (
    BlackboardPost,
    ScenarioConfig,
    build_event_scheduler,
    load_scenario,
)
from swarm.shared.blackboard import Blackboard
from swarm.shared.pheromones import PheromoneSystem

load_dotenv()  # Load .env file from project root

logger = logging.getLogger(__name__)

# Default path to the reference data (relative to project root)
DEFAULT_DATA_PATH = "reference_data/southwark_reference_data_table.parquet.gzip"


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class SimConfig:
    """Core simulation parameters (flat / programmatic mode)."""

    data_path: str = DEFAULT_DATA_PATH  # path to parquet/csv reference data
    num_agents: int = 20
    steps: int = 120
    seed: int | None = 42
    dt: float = 0.1
    scenario: str = "Business as usual."
    goal: str = "Navigate the environment."
    personality: str = ""
    awareness_radius: float = 5.0
    use_llm: bool = False
    interval_ms: int = 250

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_path": self.data_path,
            "num_agents": self.num_agents,
            "steps": self.steps,
            "seed": self.seed,
            "scenario": self.scenario,
            "awareness_radius": self.awareness_radius,
            "use_llm": self.use_llm,
        }


# ── Model ────────────────────────────────────────────────────────────


class SwarmModel:
    """Top-level simulation model — wraps the swarm engine for the web viewer.

    Accepts *either* a flat ``SimConfig`` or a ``scenario_path`` to a YAML
    file.  When a scenario path is provided, its agent groups, hazards,
    events, and blackboard posts are used instead of :class:`SimConfig`
    defaults.
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        scenario_path: str | Path | None = None,
    ) -> None:
        # ── Resolve config ────────────────────────────────────────
        self._scenario: ScenarioConfig | None = None
        self._bb_posts: list[BlackboardPost] = []

        if scenario_path is not None:
            sc = load_scenario(scenario_path)
            self._scenario = sc
            self._bb_posts = list(sc.blackboard_posts)

            # Build a SimConfig merging YAML values with any CLI overrides.
            total_agents = sum(g.count for g in sc.agent_groups) or 20
            config = SimConfig(
                data_path=sc.data_path,
                num_agents=total_agents,
                steps=sc.steps,
                seed=sc.seed,
                dt=sc.dt,
                scenario=sc.scenario_text,
                goal=sc.goal,
                personality="",  # unused — per-agent personality below
                awareness_radius=sc.awareness_radius,
                use_llm=sc.use_llm,
                interval_ms=sc.interval_ms,
            )
        elif config is None:
            config = SimConfig()

        self.config = config
        self.rng = random.Random(config.seed)
        self.tick = 0
        self.running = True

        # Build world from reference data
        data_path = config.data_path
        if not Path(data_path).is_absolute():
            # Try relative to cwd, then relative to project root
            if not Path(data_path).exists():
                project_root = Path(__file__).resolve().parent.parent
                candidate = project_root / data_path
                if candidate.exists():
                    data_path = str(candidate)
        logger.info("Loading world from: %s", data_path)
        self.world = World(data_path)

        # Configure exits from scenario (if specified)
        if self._scenario and self._scenario.num_exits is not None:
            self.world.configure_exits(self._scenario.num_exits)
            logger.info("Configured %d exits around periphery", self._scenario.num_exits)

        # Build LLM client
        self.client: LLMClient = self._make_client(config)

        # Build swarm
        seed = config.seed or 42
        self.swarm = LLMSwarm(
            client=self.client,
            seed=seed,
        )

        # ── Spawn agents ──────────────────────────────────────────
        if self._scenario and self._scenario.agent_groups:
            # Per-group spawning using YAML group descriptions
            for group in self._scenario.agent_groups:
                spawn_area = group.spawn_area
                group_goal = group.goal or config.goal
                personality_text = group.description or config.personality
                positions = self.swarm._get_spawn_positions(
                    self.world, group.count, spawn_area,
                )
                for pos in positions:
                    self.swarm.spawn_agent(
                        world=self.world,
                        position=pos,
                        goal=group_goal,
                        personality=personality_text,
                        scenario=config.scenario,
                        awareness_radius=config.awareness_radius,
                    )
        else:
            # Flat-mode batch spawn
            self.swarm.spawn_batch(
                world=self.world,
                count=config.num_agents,
                goal=config.goal,
                personality=config.personality,
                scenario=config.scenario,
                awareness_radius=config.awareness_radius,
            )

        # ── Engine subsystems ─────────────────────────────────────
        self.clock = Clock(dt=config.dt, max_ticks=config.steps)

        # Use pheromone configs from scenario if available
        phero_cfgs = None
        if self._scenario and self._scenario.pheromone_configs:
            phero_cfgs = self._scenario.pheromone_configs
        self.pheromones = PheromoneSystem(self.world, configs=phero_cfgs)
        self.blackboard = Blackboard()

        # Event scheduler — populated from YAML or empty
        if self._scenario:
            self.events = build_event_scheduler(self._scenario)
        else:
            self.events = EventScheduler()

        # Initialise fields
        if not self.world.has_layer("hazard_prev"):
            self.world.add_layer("hazard_prev", default=0.0)

    # ── Step ──────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the simulation by one tick."""
        if not self.running:
            return

        # Snapshot hazard for rate-of-change
        if self.world.has_layer("hazard_prev"):
            self.world.get_layer("hazard_prev").data[:] = self.world.hazard_grid

        self.clock.advance()
        tick = self.clock.tick

        # ── Post scheduled blackboard alerts ──────────────────────
        for post in self._bb_posts:
            if post.tick == tick:
                self.blackboard.set(
                    post.key, post.value, tick, ttl=post.ttl,
                )
                logger.debug("BB post @t=%d: %s = %s", tick, post.key, post.value)

        # Step events / hazards
        self.events.process_tick(self.world, tick)

        # Step all agents (perceive → decide → execute)
        self.swarm.step_all(self.world, tick, blackboard=self.blackboard)

        # Update pheromones
        self.pheromones.update()

        self.tick = tick

        # Check termination
        stats = self.swarm.get_stats()
        if tick >= self.config.steps or stats.active == 0:
            self.running = False

    @staticmethod
    def _make_client(config: SimConfig) -> LLMClient:
        if config.use_llm:
            import os
            api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key:
                from swarm.llm.client import OpenAIClient
                logger.info("Using OpenAIClient (model=%s", OpenAIClient.__init__.__defaults__[0])
                return OpenAIClient()
            else:
                logger.warning("use_llm=True but no API_KEY or OPENAI_API_KEY found — falling back to MockClient")
        else:
            logger.info("use_llm=False — using MockClient (strategy=first_shuffled_move)")
        return MockClient(strategy="first_shuffled_move")

    # ── Queries (for web viewer) ─────────────────────────────────

    @property
    def agents(self) -> list[LLMAgent]:
        return self.swarm.all_agents

    def get_stats_dict(self) -> dict[str, Any]:
        stats = self.swarm.get_stats()
        return {
            "tick": self.tick,
            "total": stats.total,
            "active": stats.active,
            "evacuated": stats.evacuated,
            "dead": stats.dead,
            "stuck": stats.stuck,
            "panicking": stats.panicking,
            "mean_speed": round(stats.mean_speed, 3),
        }

    def get_terrain_grid(self) -> list[dict[str, Any]]:
        """Return a list of patch dicts for every cell in the world."""
        patches: list[dict[str, Any]] = []
        for y in range(self.world.height):
            for x in range(self.world.width):
                terrain_val = int(self.world.terrain_grid[y, x])
                try:
                    t = Terrain(terrain_val)
                except ValueError:
                    t = Terrain.OPEN
                is_valid = (x, y) in self.world._valid_cells
                patches.append({
                    "x": x,
                    "y": y,
                    "terrain": t.name.lower(),
                    "walkable": bool(self.world.walkable_grid[y, x]),
                    "cost": float(self.world.cost_grid[y, x]) if np.isfinite(self.world.cost_grid[y, x]) else 999,
                    "hazard": float(self.world.hazard_grid[y, x]),
                    "is_exit": t == Terrain.EXIT,
                    "valid": is_valid,
                })
        return patches

    def get_agent_list(self) -> list[dict[str, Any]]:
        """Return serialisable agent state for the web viewer."""
        out: list[dict[str, Any]] = []
        for agent in self.agents:
            out.append({
                "id": agent.id,
                "x": agent.position.x,
                "y": agent.position.y,
                "state": agent.state.value,
                "reasoning": (
                    agent.memory.reasoning[-1][1]
                    if agent.memory.reasoning
                    else ""
                ),
            })
        return out