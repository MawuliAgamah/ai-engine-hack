"""
Mesa-compatible model wrapping the swarm simulation engine.

Creates a randomised world with diverse terrain, spawns LLM-driven agents,
and advances the simulation tick-by-tick.  Exposes helpers that the web
viewer queries for state snapshots.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from swarm.agents.base import AgentState
from swarm.agents.llm_agent import LLMAgent
from swarm.agents.llm_swarm import LLMSwarm
from swarm.core.clock import Clock
from swarm.core.engine import SimulationEngine
from swarm.core.events import EventScheduler
from swarm.core.world import Position, Terrain, World
from swarm.llm.client import LLMClient, MockClient
from swarm.shared.blackboard import Blackboard
from swarm.shared.fields import FieldManager
from swarm.shared.pheromones import PheromoneSystem


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class SimConfig:
    """Core simulation parameters."""

    width: int = 40
    height: int = 40
    num_agents: int = 20
    steps: int = 120
    seed: int | None = 2026
    scenario: str = "A fire has broken out in the building. Evacuate immediately."
    goal: str = "Reach the nearest exit and evacuate safely."
    personality: str = "You are a cautious pedestrian who prefers safe, uncrowded routes."
    awareness_radius: float = 8.0
    use_llm: bool = False  # If True and API_KEY is set, use real LLM

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "num_agents": self.num_agents,
            "steps": self.steps,
            "seed": self.seed,
            "scenario": self.scenario,
            "goal": self.goal,
            "awareness_radius": self.awareness_radius,
            "use_llm": self.use_llm,
        }


# ── Random world generation ─────────────────────────────────────────


def generate_random_world(
    width: int,
    height: int,
    rng: random.Random,
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
    """Top-level simulation model — wraps the swarm engine for the web viewer."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.tick = 0
        self.running = True

        # Build world
        self.world = generate_random_world(
            config.width, config.height, self.rng,
        )

        # Build LLM client
        self.client: LLMClient = self._make_client(config)

        # Build swarm
        self.swarm = LLMSwarm(
            client=self.client,
            seed=config.seed or 42,
        )

        # Spawn agents
        self.swarm.spawn_batch(
            world=self.world,
            count=config.num_agents,
            goal=config.goal,
            personality=config.personality,
            scenario=config.scenario,
            awareness_radius=config.awareness_radius,
        )

        # Build engine subsystems
        self.clock = Clock(dt=0.1, max_ticks=config.steps)
        self.pheromones = PheromoneSystem(self.world)
        self.fields = FieldManager(self.world)
        self.blackboard = Blackboard()
        self.events = EventScheduler()

        # Initialise fields
        if not self.world.has_layer("hazard_prev"):
            self.world.add_layer("hazard_prev", default=0.0)
        self.fields.update(0, force=True)

    @staticmethod
    def _make_client(config: SimConfig) -> LLMClient:
        if config.use_llm:
            import os
            api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key:
                from swarm.llm.client import OpenAIClient
                return OpenAIClient()
        return MockClient(strategy="first_move")

    # ── Step ──────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the simulation by one tick."""
        if not self.running:
            return

        # Snapshot hazard for rate-of-change
        if self.world.has_layer("hazard_prev"):
            self.world.get_layer("hazard_prev").data[:] = self.world.hazard_grid

        self.clock.advance()
        tick = self.clock.tick

        # Step events / hazards
        self.events.process_tick(self.world, tick)

        # Update fields periodically
        self.fields.update(tick)

        # Step all agents
        self.swarm.step_all(self.world, tick)

        # Update pheromones
        self.pheromones.update()

        self.tick = tick

        # Check termination
        stats = self.swarm.get_stats()
        if tick >= self.config.steps or stats.active == 0:
            self.running = False

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
