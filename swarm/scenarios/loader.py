"""
Scenario loader — parse YAML scenario configs into runtime objects.

A scenario YAML defines:

* **scenario** — top-level metadata (name, description, goal text)
* **agents.groups** — one or more agent groups, each with count, spawn_area,
  personality distribution, and optional description override
* **hazards** — list of hazard events (fire, flood, gas, obstruction) with
  origin, timing, spread parameters, and optional spread_direction
* **events** — list of discrete one-shot events (alarms, door closures, …)
  that fire at a specific tick and may post to the blackboard
* **blackboard_alerts** — global information posted to the shared blackboard
  at specific ticks (e.g. fire alarm at tick 50)
* **simulation** — engine parameters (dt, max_ticks, field_update_interval)

The loader converts all of this into:

1. ``ScenarioConfig`` — a flat data object the model can read
2. Populated ``EventScheduler`` with ``HazardEvent`` and ``ScenarioEvent``
3. A list of ``(tick, key, value, ttl)`` blackboard posts
4. Per-group ``PersonalityArchetype`` objects for heterogeneous spawning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from swarm.core.events import EventScheduler, HazardEvent, HazardType, ScenarioEvent
from swarm.core.world import Position, Terrain, World
from swarm.shared.pheromones import PheromoneConfig

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────


@dataclass
class AgentGroupConfig:
    """One group of agents with shared personality distribution."""

    count: int
    name: str = ""  # human-readable group label (e.g. "commuter")
    description: str = ""  # personality text passed to the LLM
    spawn_area: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    goal: str | None = None  # per-group override; falls back to scenario goal


@dataclass
class BlackboardPost:
    """A blackboard entry to be posted at a specific tick."""

    tick: int
    key: str
    value: Any
    ttl: int | None = None  # None = permanent



@dataclass
class ScenarioConfig:
    """Fully-parsed scenario configuration ready for the model."""

    # Metadata
    name: str = "baseline"  # scenario name
    description: str = ""  # scenario description
    scenario_text: str = "Business as usual — no hazards."  # scenario text
    goal: str = "Navigate the environment."  # default agent goal

    # Simulation params
    width: int = 40
    height: int = 40
    steps: int = 120  # simulation steps
    seed: int | None = 42  # simulation seed
    dt: float = 0.1  # time per tick in seconds
    awareness_radius: float = 5.0  # perception radius for LLM agents
    use_llm: bool = False  # use real LLM client
    interval_ms: int = 250  # web viewer poll / step interval

    # World generation
    num_exits: int = 4
    wall_density: float = 0.10
    building_density: float = 0.08
    grass_density: float = 0.12
    water_density: float = 0.03

    # Agent groups
    agent_groups: list[AgentGroupConfig] = field(default_factory=list)

    # Pheromone layer configs
    pheromone_configs: list[PheromoneConfig] = field(default_factory=list)

    # Events / hazards
    hazard_events: list[HazardEvent] = field(default_factory=list)
    scenario_events: list[ScenarioEvent] = field(default_factory=list)
    blackboard_posts: list[BlackboardPost] = field(default_factory=list)


# ── YAML parsing helpers ─────────────────────────────────────────


def _parse_position(d: dict[str, int]) -> Position:
    return Position(int(d["x"]), int(d["y"]))


def _parse_spawn_area(d: dict[str, int] | None) -> tuple[int, int, int, int] | None:
    if d is None:
        return None
    return (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))


def _parse_hazard_type(s: str) -> HazardType:
    return HazardType(s.lower())


def _parse_spread_direction(
    dirs: list[dict[str, float]] | None,
) -> list[tuple[float, float]] | None:
    if not dirs:
        return None
    return [(float(d["angle"]), float(d["strength"])) for d in dirs]


def _parse_hazard(d: dict[str, Any]) -> HazardEvent:
    return HazardEvent(
        hazard_type=_parse_hazard_type(d["type"]),
        origin=_parse_position(d["origin"]),
        start_tick=int(d.get("start_tick", 0)),
        intensity=float(d.get("intensity", 0.8)),
        spread_direction=_parse_spread_direction(d.get("spread_direction")),
        spread_rate=float(d.get("spread_rate", 0.05)),
        spread_radius=int(d.get("spread_radius", 20)),
        duration=int(d.get("duration", 0)),
    )


def _parse_agent_group(d: dict[str, Any], default_goal: str) -> AgentGroupConfig:
    return AgentGroupConfig(
        count=int(d.get("count", 10)),
        name=d.get("name", ""),
        description=d.get("description", ""),
        spawn_area=_parse_spawn_area(d.get("spawn_area")),
        goal=d.get("goal", default_goal),
    )


def _parse_pheromone_config(d: dict[str, Any]) -> PheromoneConfig:
    return PheromoneConfig(
        name=str(d["name"]),
        decay_rate=float(d.get("decay_rate", 0.05)),
        diffusion_rate=float(d.get("diffusion_rate", 0.1)),
        max_value=float(d.get("max_value", 100.0)),
        min_value=float(d.get("min_value", 0.0)),
    )


def _build_terrain_event(d: dict[str, Any]) -> ScenarioEvent:
    """Build a ScenarioEvent that mutates terrain at a specific tick."""
    tick = int(d["tick"])
    name = d.get("name", f"event_t{tick}")
    terrain_type = Terrain[d["terrain"].upper()]
    cells: list[tuple[int, int]] = []
    for c in d.get("cells", []):
        cells.append((int(c["x"]), int(c["y"])))
    if "area" in d:
        a = d["area"]
        for dy in range(int(a["h"])):
            for dx in range(int(a["w"])):
                cells.append((int(a["x"]) + dx, int(a["y"]) + dy))

    def action(world: World, _cells=cells, _t=terrain_type) -> None:
        for x, y in _cells:
            if world.in_bounds(x, y):
                world.set_terrain(x, y, _t)

    return ScenarioEvent(name=name, tick=tick, action=action)


def _parse_blackboard_post(d: dict[str, Any]) -> BlackboardPost:
    return BlackboardPost(
        tick=int(d["tick"]),
        key=str(d["key"]),
        value=d.get("value", True),
        ttl=d.get("ttl"),
    )


# ── Main loader ──────────────────────────────────────────────────


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load a YAML scenario file and return a ``ScenarioConfig``.

    Parameters
    ----------
    path :
        Path to a ``.yaml`` scenario file.

    Returns
    -------
    ScenarioConfig
        Fully-parsed configuration ready to drive a simulation.
    """
    path = Path(path)
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}


    # ── Metadata ──────────────────────────────────────────────────
    scenario_sec = raw.get("scenario", {})
    name = scenario_sec.get("name", path.stem)
    description = scenario_sec.get("description", "")
    scenario_text = scenario_sec.get("scenario_text", description or "Business as usual.")
    goal = scenario_sec.get("goal", "Navigate the environment safely.")

    # ── Simulation params ─────────────────────────────────────────
    sim_sec = raw.get("simulation", {})
    steps = int(sim_sec.get("steps", sim_sec.get("max_ticks", 120)))
    seed = sim_sec.get("seed")
    if seed is not None:
        seed = int(seed)
    dt = float(sim_sec.get("dt", 0.1))
    awareness_radius = float(sim_sec.get("awareness_radius", 5.0))
    use_llm = bool(sim_sec.get("use_llm", False))
    interval_ms = int(sim_sec.get("interval_ms", 250))

    # ── World generation params ───────────────────────────────────
    world_sec = raw.get("world", {})
    width = int(world_sec.get("width", 40))
    height = int(world_sec.get("height", 40))
    num_exits = int(world_sec.get("num_exits", 4))
    wall_density = float(world_sec.get("wall_density", 0.10))
    building_density = float(world_sec.get("building_density", 0.08))
    grass_density = float(world_sec.get("grass_density", 0.12))
    water_density = float(world_sec.get("water_density", 0.03))

    # ── Agent groups ──────────────────────────────────────────────
    agents_sec = raw.get("agents", {})
    groups: list[AgentGroupConfig] = []
    for gd in agents_sec.get("groups", []):
        groups.append(_parse_agent_group(gd, default_goal=goal))

    # ── Pheromone configs ─────────────────────────────────────────
    pheromone_cfgs: list[PheromoneConfig] = []
    for pd in raw.get("pheromones", []) or []:
        pheromone_cfgs.append(_parse_pheromone_config(pd))

    # ── Hazards ───────────────────────────────────────────────────
    hazards: list[HazardEvent] = []
    for hd in raw.get("hazards", []) or []:
        hazards.append(_parse_hazard(hd))

    # ── Discrete events (terrain changes, door openings, etc.) ───
    events_list: list[ScenarioEvent] = []
    for ed in raw.get("events", []) or []:
        events_list.append(_build_terrain_event(ed))

    # ── Blackboard alerts ─────────────────────────────────────────
    bb_posts: list[BlackboardPost] = []
    for bd in raw.get("blackboard", []) or []:
        bb_posts.append(_parse_blackboard_post(bd))

    config = ScenarioConfig(
        name=name,
        description=description,
        scenario_text=scenario_text,
        goal=goal,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        dt=dt,
        awareness_radius=awareness_radius,
        use_llm=use_llm,
        interval_ms=interval_ms,
        num_exits=num_exits,
        wall_density=wall_density,
        building_density=building_density,
        grass_density=grass_density,
        water_density=water_density,
        agent_groups=groups,
        pheromone_configs=pheromone_cfgs,
        hazard_events=hazards,
        scenario_events=events_list,
        blackboard_posts=bb_posts,
    )

    logger.info(
        "Loaded scenario '%s': %d groups, %d hazards, %d events, %d bb, %d pheromones",
        name,
        len(groups),
        len(hazards),
        len(events_list),
        len(bb_posts),
        len(pheromone_cfgs),
    )
    return config


# ── Wire into EventScheduler ────────────────────────────────────


def build_event_scheduler(config: ScenarioConfig) -> EventScheduler:
    """Create a fully-populated ``EventScheduler`` from scenario config."""
    scheduler = EventScheduler()
    for h in config.hazard_events:
        scheduler.add_hazard(h)
    for e in config.scenario_events:
        scheduler.add_event(e)
    return scheduler
