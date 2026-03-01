"""
Configuration loader — reads YAML scenario files and builds simulation objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from swarm.agents.swarm import Swarm
from swarm.core.clock import Clock
from swarm.core.events import EventScheduler, HazardEvent, HazardType
from swarm.core.world import Position, Terrain, World


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_world(config: dict[str, Any]) -> World:
    """Build a World from configuration."""
    world_cfg = config["world"]
    width = world_cfg["width"]
    height = world_cfg["height"]
    world = World(width, height)

    # Apply terrain features
    if "features" in world_cfg:
        for feature in world_cfg["features"]:
            terrain = Terrain[feature["terrain"].upper()]
            if "rect" in feature:
                r = feature["rect"]
                world.set_terrain_rect(r["x"], r["y"], r["w"], r["h"], terrain)
            elif "pos" in feature:
                p = feature["pos"]
                world.set_terrain(p["x"], p["y"], terrain)

    # Apply exits
    if "exits" in world_cfg:
        for exit_cfg in world_cfg["exits"]:
            if "rect" in exit_cfg:
                r = exit_cfg["rect"]
                world.set_terrain_rect(r["x"], r["y"], r["w"], r["h"], Terrain.EXIT)
            elif "pos" in exit_cfg:
                p = exit_cfg["pos"]
                world.set_terrain(p["x"], p["y"], Terrain.EXIT)

    return world


def build_clock(config: dict[str, Any]) -> Clock:
    """Build a Clock from configuration."""
    sim_cfg = config.get("simulation", {})
    return Clock(
        dt=sim_cfg.get("dt", 0.1),
        max_ticks=sim_cfg.get("max_ticks", 5000),
    )


def build_swarm(config: dict[str, Any], world: World) -> Swarm:
    """Build and populate a Swarm from configuration."""
    agents_cfg = config.get("agents", {})
    seed = agents_cfg.get("seed", 42)
    swarm = Swarm(seed=seed)

    # Agent groups
    groups = agents_cfg.get("groups", [agents_cfg])
    for group in groups:
        count = group.get("count", 100)

        # Spawn area
        spawn_cfg = group.get("spawn_area")
        spawn_area = None
        if spawn_cfg:
            spawn_area = (
                spawn_cfg["x"],
                spawn_cfg["y"],
                spawn_cfg["w"],
                spawn_cfg["h"],
            )

        swarm.spawn_agents(
            world=world,
            count=count,
            spawn_area=spawn_area,
        )

    return swarm


def build_events(config: dict[str, Any]) -> EventScheduler:
    """Build an EventScheduler from configuration."""
    scheduler = EventScheduler()

    for hazard_cfg in config.get("hazards", []):
        # Parse optional spread_direction: list of {angle, strength} dicts
        raw_dirs = hazard_cfg.get("spread_direction", None)
        spread_direction: list[tuple[float, float]] | None = None
        if raw_dirs is not None:
            spread_direction = [
                (float(d["angle"]), float(d["strength"])) for d in raw_dirs
            ]

        hazard = HazardEvent(
            hazard_type=HazardType(hazard_cfg["type"]),
            origin=Position(hazard_cfg["origin"]["x"], hazard_cfg["origin"]["y"]),
            start_tick=hazard_cfg.get("start_tick", 0),
            intensity=hazard_cfg.get("intensity", 0.8),
            spread_direction=spread_direction,
            spread_rate=hazard_cfg.get("spread_rate", 0.05),
            spread_radius=hazard_cfg.get("spread_radius", 20),
            duration=hazard_cfg.get("duration", 0),
        )
        scheduler.add_hazard(hazard)

    return scheduler


def build_simulation(config_path: str | Path):
    """
    Build a complete simulation from a YAML config file.

    Returns:
        SimulationEngine ready to run.
    """
    from swarm.core.engine import SimulationEngine
    from swarm.shared.blackboard import Blackboard
    from swarm.shared.pheromones import PheromoneSystem

    config = load_config(config_path)
    world = build_world(config)
    clock = build_clock(config)
    swarm = build_swarm(config, world)
    events = build_events(config)

    pheromones = PheromoneSystem(world)
    blackboard = Blackboard()

    engine = SimulationEngine(
        world=world,
        swarm=swarm,
        clock=clock,
        event_scheduler=events,
        pheromone_system=pheromones,
        blackboard=blackboard,
    )

    return engine
