"""
Reactive decision layer — Social Force Model.

This is the fastest layer, running every tick. It computes immediate forces
on the agent based on:
- Repulsion from walls and obstacles
- Repulsion from other agents (collision avoidance)
- Repulsion from hazards (danger avoidance)
- Attraction toward goal direction
- Herding forces (alignment with nearby agents' movement)

Based on Helbing & Molnár (1995) Social Force Model, extended with
hazard and pheromone forces.

The output is a force vector that gets combined with tactical/strategic
decisions via priority arbitration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.agents.perception import LocalPercept
from swarm.agents.personality import PersonalityTraits
from swarm.core.world import Position


@dataclass
class Force:
    """A 2D force vector."""

    dx: float
    dy: float

    @property
    def magnitude(self) -> float:
        return float(np.sqrt(self.dx**2 + self.dy**2))

    def normalized(self) -> Force:
        mag = self.magnitude
        if mag < 1e-8:
            return Force(0.0, 0.0)
        return Force(self.dx / mag, self.dy / mag)

    def scaled(self, factor: float) -> Force:
        return Force(self.dx * factor, self.dy * factor)

    def __add__(self, other: Force) -> Force:
        return Force(self.dx + other.dx, self.dy + other.dy)

    def __repr__(self) -> str:
        return f"Force({self.dx:.3f}, {self.dy:.3f})"


def compute_reactive_forces(
    percept: LocalPercept,
    personality: PersonalityTraits,
    current_velocity: tuple[float, float],
    goal_direction: tuple[float, float] | None = None,
) -> Force:
    """
    Compute the total reactive force on an agent.

    Combines multiple social forces:
    1. Goal attraction (desired direction)
    2. Agent repulsion (collision avoidance)
    3. Hazard repulsion (danger avoidance, modulated by risk_tolerance)
    4. Wall/obstacle repulsion (boundary avoidance)
    5. Herding force (alignment with crowd flow, modulated by herding_tendency)
    6. Pheromone attraction (follow "exit_path" pheromones)

    Returns:
        Combined force vector to be applied to agent movement.
    """
    total = Force(0.0, 0.0)
    pos = percept.position

    # ── 1. Goal attraction force ────────────────────────────────────
    if goal_direction is not None:
        desired_speed = personality.speed
        goal_force = Force(
            goal_direction[0] * desired_speed,
            goal_direction[1] * desired_speed,
        )
        # Subtract current velocity to get correction force
        correction = Force(
            goal_force.dx - current_velocity[0],
            goal_force.dy - current_velocity[1],
        )
        total = total + correction.scaled(2.0)  # Strong goal-seeking
    elif percept.nearest_exit_direction is not None:
        # Default: head toward nearest exit
        ex_dir = percept.nearest_exit_direction
        desired_speed = personality.speed
        goal_force = Force(ex_dir[0] * desired_speed, ex_dir[1] * desired_speed)
        correction = Force(
            goal_force.dx - current_velocity[0],
            goal_force.dy - current_velocity[1],
        )
        total = total + correction.scaled(1.5)

    # ── 2. Agent repulsion (collision avoidance) ────────────────────
    for agent_pos in percept.nearby_agent_positions:
        dx = pos.x - agent_pos.x
        dy = pos.y - agent_pos.y
        dist = max(0.1, np.sqrt(dx * dx + dy * dy))
        # Exponential repulsion: stronger when closer
        strength = 3.0 * np.exp(-dist / 1.5)
        total = total + Force(dx / dist * strength, dy / dist * strength)

    # ── 3. Hazard repulsion ─────────────────────────────────────────
    if percept.max_local_hazard > 0.01:
        # Flee from danger, modulated by risk tolerance
        # Low risk tolerance = strong repulsion from hazard
        hazard_sensitivity = 5.0 * (1.0 - personality.risk_tolerance)
        hdir = percept.hazard_direction
        # Repel AWAY from hazard (negate direction toward hazard)
        total = total + Force(
            -hdir[0] * hazard_sensitivity * percept.max_local_hazard,
            -hdir[1] * hazard_sensitivity * percept.max_local_hazard,
        )

    # ── 4. Wall/obstacle repulsion ──────────────────────────────────
    for dy_off in range(-2, 3):
        for dx_off in range(-2, 3):
            if dx_off == 0 and dy_off == 0:
                continue
            key = Position(pos.x + dx_off, pos.y + dy_off)
            cost = percept.neighbor_costs.get(key, 0.0)
            if cost == float("inf"):
                # Wall! Repel.
                dist = max(0.5, np.sqrt(dx_off**2 + dy_off**2))
                strength = 2.0 / (dist * dist)
                total = total + Force(-dx_off / dist * strength, -dy_off / dist * strength)

    # ── 5. Herding force (follow the crowd) ─────────────────────────
    if percept.nearby_agent_count > 0 and personality.herding_tendency > 0.1:
        # Pull toward the center of nearby agents
        centroid_x = np.mean([p.x for p in percept.nearby_agent_positions]) if percept.nearby_agent_positions else pos.x
        centroid_y = np.mean([p.y for p in percept.nearby_agent_positions]) if percept.nearby_agent_positions else pos.y
        dx = centroid_x - pos.x
        dy = centroid_y - pos.y
        dist = max(0.1, np.sqrt(dx * dx + dy * dy))
        herding_strength = personality.herding_tendency * 0.5
        total = total + Force(dx / dist * herding_strength, dy / dist * herding_strength)

    # ── 6. Pheromone attraction (follow "exit_path" trails) ─────────
    if "exit_path" in percept.pheromone_readings:
        phero = percept.pheromone_readings["exit_path"]
        if phero:
            # Compute weighted centroid of pheromone deposits
            total_weight = 0.0
            wx, wy = 0.0, 0.0
            for ppos, val in phero.items():
                total_weight += val
                wx += ppos.x * val
                wy += ppos.y * val
            if total_weight > 0:
                wx /= total_weight
                wy /= total_weight
                dx = wx - pos.x
                dy = wy - pos.y
                dist = max(0.1, np.sqrt(dx * dx + dy * dy))
                pheromone_strength = 0.8
                total = total + Force(
                    dx / dist * pheromone_strength, dy / dist * pheromone_strength
                )

    return total


def force_to_position(
    current: Position,
    force: Force,
    speed: float,
    walkable_neighbors: list[Position],
    rng: np.random.Generator,
) -> Position:
    """
    Convert a force vector into a discrete grid movement decision.

    The force indicates desired direction; we pick the walkable neighbor
    that best aligns with the force direction. Stochastic tie-breaking
    via softmax selection adds realistic variation.

    Args:
        current: Current position.
        force: Computed force vector.
        speed: Agent's movement speed (determines probability of moving).
        walkable_neighbors: Available positions to move to.
        rng: Random number generator for stochastic decisions.

    Returns:
        Chosen next position (may be current position if staying put).
    """
    if not walkable_neighbors:
        return current  # Stuck

    if force.magnitude < 0.01:
        # No significant force — stay put or random walk
        if rng.random() < 0.3:
            return rng.choice(walkable_neighbors)  # type: ignore
        return current

    # Score each neighbor by alignment with force direction
    scores: list[float] = []
    fnorm = force.normalized()

    for neighbor in walkable_neighbors:
        dx = neighbor.x - current.x
        dy = neighbor.y - current.y
        dist = max(0.1, np.sqrt(dx * dx + dy * dy))
        # Dot product = alignment with force
        alignment = (dx / dist) * fnorm.dx + (dy / dist) * fnorm.dy
        scores.append(alignment)

    # Softmax selection with temperature
    scores_arr = np.array(scores)
    temperature = 0.5  # Lower = more deterministic
    exp_scores = np.exp((scores_arr - np.max(scores_arr)) / temperature)
    total_exp = np.sum(exp_scores)

    # Handle edge cases where exp_scores are all zero or contain NaN
    if total_exp <= 0 or not np.isfinite(total_exp):
        probs = np.ones(len(walkable_neighbors)) / len(walkable_neighbors)
    else:
        probs = exp_scores / total_exp
        # Fix any remaining NaN
        if not np.all(np.isfinite(probs)):
            probs = np.ones(len(walkable_neighbors)) / len(walkable_neighbors)

    # Speed determines probability of actually moving (vs staying)
    if rng.random() > min(speed, 1.0):
        return current  # Too slow this tick

    idx = rng.choice(len(walkable_neighbors), p=probs)
    return walkable_neighbors[idx]
