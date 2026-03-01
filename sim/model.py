"""
Mesa-compatible model wrapping the swarm simulation engine.

Creates a randomised world with diverse terrain, spawns LLM-driven agents,
and advances the simulation tick-by-tick.  Exposes helpers that the web
viewer queries for state snapshots.

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


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class SimConfig:
    """Core simulation parameters (flat / programmatic mode)."""

    width: int = 40
    height: int = 40
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
    # World generation
    num_exits: int = 4
    wall_density: float = 0.10
    building_density: float = 0.08
    grass_density: float = 0.12
    water_density: float = 0.03

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "num_agents": self.num_agents,
            "steps": self.steps,
            "seed": self.seed,
            "scenario": self.scenario,
            "awareness_radius": self.awareness_radius,
            "use_llm": self.use_llm,
        }


# ── Random world generation ─────────────────────────────────────────


def generate_random_world(
    width: int = 40,
    height: int = 40,
    rng: random.Random = None,
    num_exits: int = 4,
    wall_density: float = 0.10,
    building_density: float = 0.08,
    grass_density: float = 0.12,
    water_density: float = 0.03,
) -> World:
    """Build a random world with mixed terrain for demonstration.

    Layout strategy:
    - Border walls on all edges
    - Random exit positions along the border
    - Interior mix of open, road, sidewalk, grass, buildings, obstacles, water
    """
    world = World(width, height)

    # 1. Border walls
    for x in range(width):
        world.set_terrain(x, 0, Terrain.WALL)
        world.set_terrain(x, height - 1, Terrain.WALL)
    for y in range(height):
        world.set_terrain(0, y, Terrain.WALL)
        world.set_terrain(width - 1, y, Terrain.WALL)

    # 2. Place exits along the border (replacing wall cells)
    border_cells: list[tuple[int, int]] = []
    for x in range(2, width - 2):
        border_cells.append((x, 0))
        border_cells.append((x, height - 1))
    for y in range(2, height - 2):
        border_cells.append((0, y))
        border_cells.append((width - 1, y))
    rng.shuffle(border_cells)
    for i in range(min(num_exits, len(border_cells))):
        ex, ey = border_cells[i]
        world.set_terrain(ex, ey, Terrain.EXIT)

    # 3. Interior terrain
    interior: list[tuple[int, int]] = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            interior.append((x, y))

    rng.shuffle(interior)
    n_interior = len(interior)

    # Carve a few "road" corridors (horizontal and vertical)
    road_y = [height // 3, 2 * height // 3]
    road_x = [width // 3, 2 * width // 3]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if y in road_y or x in road_x:
                world.set_terrain(x, y, Terrain.ROAD)

    # Sidewalk next to roads
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if world.terrain_grid[y, x] != Terrain.ROAD:
                # Check if adjacent to road
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if world.terrain_grid[ny, nx] == Terrain.ROAD:
                            world.set_terrain(x, y, Terrain.SIDEWALK)
                            break

    # Place random terrain on remaining OPEN cells
    open_cells = [
        (x, y)
        for x, y in interior
        if world.terrain_grid[y, x] == Terrain.OPEN
    ]
    rng.shuffle(open_cells)

    idx = 0
    n_walls = int(wall_density * len(open_cells))
    n_buildings = int(building_density * len(open_cells))
    n_grass = int(grass_density * len(open_cells))
    n_water = int(water_density * len(open_cells))

    for _ in range(n_walls):
        if idx >= len(open_cells):
            break
        x, y = open_cells[idx]
        world.set_terrain(x, y, Terrain.WALL)
        idx += 1

    for _ in range(n_buildings):
        if idx >= len(open_cells):
            break
        x, y = open_cells[idx]
        world.set_terrain(x, y, Terrain.BUILDING)
        idx += 1

    for _ in range(n_grass):
        if idx >= len(open_cells):
            break
        x, y = open_cells[idx]
        world.set_terrain(x, y, Terrain.GRASS)
        idx += 1

    for _ in range(n_water):
        if idx >= len(open_cells):
            break
        x, y = open_cells[idx]
        world.set_terrain(x, y, Terrain.WATER)
        idx += 1

    # Scatter a few doors and stairs
    remaining = [
        (x, y)
        for x, y in open_cells[idx:]
        if world.terrain_grid[y, x] == Terrain.OPEN
    ]
    rng.shuffle(remaining)
    n_doors = max(2, len(remaining) // 40)
    n_stairs = max(1, len(remaining) // 60)
    for i in range(n_doors):
        if i < len(remaining):
            x, y = remaining[i]
            world.set_terrain(x, y, Terrain.DOOR)
    for i in range(n_stairs):
        j = n_doors + i
        if j < len(remaining):
            x, y = remaining[j]
            world.set_terrain(x, y, Terrain.STAIRS)

    return world


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
                width=sc.width,
                height=sc.height,
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
                num_exits=sc.num_exits,
                wall_density=sc.wall_density,
                building_density=sc.building_density,
                grass_density=sc.grass_density,
                water_density=sc.water_density,
            )
        elif config is None:
            config = SimConfig()

        self.config = config
        self.rng = random.Random(config.seed)
        self.tick = 0
        self.running = True

        # Build world
        self.world = generate_random_world(
            width=config.width,
            height=config.height,
            rng=self.rng,
            num_exits=config.num_exits,
            wall_density=config.wall_density,
            building_density=config.building_density,
            grass_density=config.grass_density,
            water_density=config.water_density,
        )

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
                t = Terrain(self.world.terrain_grid[y, x])
                patches.append({
                    "x": x,
                    "y": y,
                    "terrain": t.name.lower(),
                    "walkable": bool(self.world.walkable_grid[y, x]),
                    "cost": float(self.world.cost_grid[y, x]) if np.isfinite(self.world.cost_grid[y, x]) else 999,
                    "hazard": float(self.world.hazard_grid[y, x]),
                    "is_exit": t == Terrain.EXIT,
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