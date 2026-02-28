"""
Base agent — the autonomous entity that navigates the world.

Each agent has:
- A unique ID
- A position in the world
- Personality traits (stochastic)
- A three-layer decision system (reactive / tactical / strategic)
- Local memory (visited cells, reasoning chain, last plan, current state)
- An RNG for stochastic behavior

The agent lifecycle per tick:
1. PERCEIVE — read local environment
2. DECIDE — three-layer decision fusion
3. ACT — execute movement
4. DEPOSIT — leave pheromone traces
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from swarm.agents.decisions.reactive import (
    compute_reactive_forces,
    force_to_position,
)
from swarm.agents.decisions.strategic import (
    StrategicDecision,
    evaluate_exits,
    should_replan,
    simulated_annealing_replan,
)
from swarm.agents.decisions.tactical import (
    TacticalPlan,
    astar_local,
    gradient_direction_at,
)
from swarm.agents.perception import perceive
from swarm.agents.personality import PersonalityTraits
from swarm.core.world import Position, World


class AgentState(enum.Enum):
    NAVIGATING = "navigating"  # Normal movement toward goal
    EVACUATED = "evacuated"    # Reached exit, done
    PANIC = "panic"            # High-danger panic mode
    STUCK = "stuck"            # Can't make progress
    DEAD = "dead"              # Caught by hazard


@dataclass
class AgentMemory:
    """Agent's local memory — what it remembers about the world."""

    visited: set[Position] = field(default_factory=set)
    last_positions: list[Position] = field(default_factory=list)  # Recent positions
    ticks_at_current: int = 0
    ticks_since_progress: int = 0  # Ticks since making meaningful progress
    last_plan_tick: int = 0


class Agent:
    """
    An autonomous spatial agent with stochastic personality and three-layer decisions.

    This is the core entity in the simulation. Each agent perceives locally,
    makes decisions through a reactive→tactical→strategic pipeline, and deposits
    pheromone traces for indirect communication.
    """

    # Tactical replanning interval (ticks between A* recomputes)
    TACTICAL_INTERVAL = 10
    # Maximum positions to remember
    MEMORY_LENGTH = 50

    def __init__(
        self,
        agent_id: int,
        position: Position,
        personality: PersonalityTraits,
        seed: int | None = None,
    ):
        self.id = agent_id
        self.position = position
        self.personality = personality
        self.rng = np.random.default_rng(seed)

        # State
        self.state = AgentState.NAVIGATING
        self.velocity: tuple[float, float] = (0.0, 0.0)

        # Decision layers
        self.strategic_goal: StrategicDecision | None = None
        self.tactical_plan: TacticalPlan | None = None

        # Memory
        self.memory = AgentMemory()

    @property
    def is_active(self) -> bool:
        return self.state in (AgentState.NAVIGATING, AgentState.PANIC, AgentState.STUCK)

    def step(self, world: World, tick: int) -> Position | None:
        """
        Execute one tick of the agent lifecycle.

        Returns:
            New position if agent moved, None if inactive.
        """
        if not self.is_active:
            return None

        # ── 1. PERCEIVE ─────────────────────────────────────────────
        percept = perceive(
            world=world,
            agent_id=self.id,
            position=self.position,
            awareness_radius=self.personality.awareness_radius,
            tick=tick,
        )

        # ── Check death condition (hazard at current position) ──────
        my_hazard = world.hazard_grid[self.position.y, self.position.x]
        if my_hazard >= 0.95:
            self.state = AgentState.DEAD
            return None

        # ── Check if at exit ────────────────────────────────────────
        for exit_pos in world.exits:
            if self.position.x == exit_pos.x and self.position.y == exit_pos.y:
                self.state = AgentState.EVACUATED
                return None

        # ── Update state based on perception ────────────────────────
        if percept.max_local_hazard > self.personality.panic_threshold:
            self.state = AgentState.PANIC
        elif self.memory.ticks_since_progress > self.personality.patience * 2:
            self.state = AgentState.STUCK
        else:
            self.state = AgentState.NAVIGATING

        # ── 2. STRATEGIC LAYER (infrequent) ─────────────────────────
        if (
            self.strategic_goal is None
            or should_replan(percept, self.personality, self.memory.ticks_since_progress)
        ):
            self.strategic_goal = evaluate_exits(
                world, self.position, self.personality, percept, self.rng
            )
            self.memory.ticks_since_progress = 0  # Reset after replan

            # If stuck, try simulated annealing for intermediate waypoint
            if self.state == AgentState.STUCK and self.strategic_goal is not None:
                waypoint = simulated_annealing_replan(
                    world, self.position, self.strategic_goal.target_goal,
                    self.personality, self.rng,
                )
                self.tactical_plan = TacticalPlan(
                    waypoints=[waypoint, self.strategic_goal.target_goal],
                    goal=self.strategic_goal.target_goal,
                    computed_at_tick=tick,
                )

        # ── 3. TACTICAL LAYER (periodic) ────────────────────────────
        goal_direction: tuple[float, float] | None = None

        # Try flow field first (cheapest)
        flow_dir = gradient_direction_at(world, self.position)
        if flow_dir is not None:
            goal_direction = flow_dir
        elif self.strategic_goal is not None:
            # A* pathfinding to goal (recompute periodically)
            if (
                self.tactical_plan is None
                or self.tactical_plan.is_complete
                or tick - self.memory.last_plan_tick > self.TACTICAL_INTERVAL
            ):
                path = astar_local(
                    world, self.position, self.strategic_goal.target_goal
                )
                if path:
                    self.tactical_plan = TacticalPlan(
                        waypoints=path,
                        goal=self.strategic_goal.target_goal,
                        computed_at_tick=tick,
                    )
                    self.memory.last_plan_tick = tick

            # Get direction to next waypoint
            if self.tactical_plan and self.tactical_plan.next_waypoint:
                wp = self.tactical_plan.next_waypoint
                dx = wp.x - self.position.x
                dy = wp.y - self.position.y
                dist = max(0.1, np.sqrt(dx * dx + dy * dy))
                goal_direction = (dx / dist, dy / dist)

                # Pop waypoint if reached
                if self.position.manhattan_distance(wp) <= 1:
                    self.tactical_plan.advance()

        # ── 4. REACTIVE LAYER (every tick) ──────────────────────────
        speed_mult = 1.5 if self.state == AgentState.PANIC else 1.0
        force = compute_reactive_forces(
            percept=percept,
            personality=self.personality,
            current_velocity=self.velocity,
            goal_direction=goal_direction,
        )

        new_pos = force_to_position(
            current=self.position,
            force=force,
            speed=self.personality.speed * speed_mult,
            walkable_neighbors=percept.walkable_neighbors,
            rng=self.rng,
        )

        # ── 5. ACT — execute movement ──────────────────────────────
        old_pos = self.position
        if new_pos != self.position:
            world.move_agent(self.id, old_pos.x, old_pos.y, new_pos.x, new_pos.y)
            self.position = new_pos
            # Update velocity
            self.velocity = (
                float(new_pos.x - old_pos.x),
                float(new_pos.y - old_pos.y),
            )
            self.memory.ticks_at_current = 0
        else:
            self.velocity = (0.0, 0.0)
            self.memory.ticks_at_current += 1

        # ── Track progress ──────────────────────────────────────────
        pos_key = self.position
        if pos_key in self.memory.visited:
            self.memory.ticks_since_progress += 1
        else:
            self.memory.ticks_since_progress = 0
        self.memory.visited.add(pos_key)

        # Keep recent position history
        self.memory.last_positions.append(self.position)
        if len(self.memory.last_positions) > self.MEMORY_LENGTH:
            self.memory.last_positions.pop(0)

        # ── 6. DEPOSIT — leave pheromone traces ─────────────────────
        self._deposit_pheromones(world)

        return new_pos

    def _deposit_pheromones(self, world: World) -> None:
        """
        Deposit pheromone traces for indirect communication.

        - "visited" pheromone: marks cells as explored (helps others spread out)
        - "exit_path" pheromone: if agent is making good progress toward exit,
          leave a trail others can follow (ACO-inspired)
        - "danger" pheromone: if agent detects hazard, warn others
        """
        x, y = self.position.x, self.position.y

        # Visited pheromone (negative signal — avoid retracing)
        if world.has_layer("visited"):
            world.get_layer("visited").add(x, y, 1.0)

        # Exit path pheromone (positive signal if making progress)
        if world.has_layer("exit_path") and self.strategic_goal is not None:
            dist_to_goal = self.position.euclidean_distance(
                self.strategic_goal.target_goal
            )
            if dist_to_goal < 30 and self.state == AgentState.NAVIGATING:
                # Stronger deposit when closer to exit
                strength = max(0.0, 1.0 - dist_to_goal / 30.0) * 2.0
                world.get_layer("exit_path").add(x, y, strength)

        # Danger pheromone
        if world.has_layer("danger"):
            my_hazard = world.hazard_grid[y, x]
            if my_hazard > 0.2:
                world.get_layer("danger").add(x, y, my_hazard * 3.0)

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.id}, pos=({self.position.x},{self.position.y}), "
            f"state={self.state.value})"
        )
