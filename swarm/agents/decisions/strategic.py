"""
Strategic decision layer — high-level replanning.

This layer fires infrequently: when an agent gets stuck, when the environment
changes significantly (new hazard, blocked route), or on a periodic timer.

Uses optimization techniques to find alternative routes:
- Simulated Annealing for finding non-obvious route alternatives
- Monte Carlo sampling of waypoints to evaluate multiple potential paths
- Game-theoretic reasoning about congested exits (Nash equilibrium-inspired)

The output is a new goal or waypoint sequence that overrides the tactical layer.

**Perception-driven design**: ``evaluate_exits`` reasons exclusively over the
agent's ``LocalPercept`` — it does NOT read ``world.occupancy_grid`` or
``world.hazard_grid`` directly.  Exits outside the awareness radius incur an
*uncertainty penalty* modulated by the agent's ``exploration`` trait, modeling
bounded rationality.

**Personality-modulated SA**: ``simulated_annealing_replan`` derives its
hyper-parameters (temperature, cooling rate, iteration budget, hazard
discount) from ``PersonalityTraits`` so that exploratory, risk-tolerant, or
patient agents produce qualitatively different escape routes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from swarm.agents.perception import LocalPercept
from swarm.agents.personality import PersonalityTraits
from swarm.core.world import Position, World


@dataclass
class StrategicDecision:
    """
    A strategic-level decision about WHERE to go.

    Attributes:
        target_goal: The chosen destination (e.g., which exit to aim for).
        reason: Why this decision was made.
        confidence: How confident the agent is in this choice [0, 1].
    """

    target_goal: Position
    reason: str
    confidence: float = 1.0


# ── Helpers ──────────────────────────────────────────────────────────


def _perception_occupancy_near(
    percept: LocalPercept,
    center: Position,
    radius: int = 3,
) -> tuple[float, int, int]:
    """Return (total_occupancy, observed_cells, total_cells) near *center*.

    Only cells present in *percept.neighbor_occupancy* count as observed.
    """
    total = 0.0
    observed = 0
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            pos = Position(center.x + dx, center.y + dy)
            count += 1
            if pos in percept.neighbor_occupancy:
                total += percept.neighbor_occupancy[pos]
                observed += 1
    return total, observed, count


def _perception_hazard_near(
    percept: LocalPercept,
    center: Position,
    radius: int = 2,
) -> tuple[float, int, int]:
    """Return (total_hazard, observed_cells, total_cells) near *center*.

    Only cells present in *percept.neighbor_hazards* count as observed.
    """
    total = 0.0
    observed = 0
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            pos = Position(center.x + dx, center.y + dy)
            count += 1
            if pos in percept.neighbor_hazards:
                total += percept.neighbor_hazards[pos]
                observed += 1
    return total, observed, count


def _lobe_interference(
    percept: LocalPercept,
    agent_pos: Position,
    exit_pos: Position,
    dist: float,
) -> float:
    """Penalty if a perceived hazard lobe lies *between* the agent and the exit.

    Computes the dot product of each lobe's direction with the agent→exit
    unit vector.  Positive dot means the lobe is roughly in the direction the
    agent needs to travel — penalise proportionally.
    """
    if dist < 1e-3 or not percept.hazard_lobes:
        return 0.0

    exit_dx = (exit_pos.x - agent_pos.x) / dist
    exit_dy = (exit_pos.y - agent_pos.y) / dist

    penalty = 0.0
    for lobe in percept.hazard_lobes:
        dot = exit_dx * lobe.direction[0] + exit_dy * lobe.direction[1]
        if dot > 0.3:  # lobe is roughly in the way
            penalty -= dot * lobe.mean_intensity * lobe.cell_count * 0.5
    return penalty


# ── Exit evaluation ──────────────────────────────────────────────────


def evaluate_exits(
    world: World,
    agent_pos: Position,
    personality: PersonalityTraits,
    percept: LocalPercept,
    rng: np.random.Generator,
) -> StrategicDecision | None:
    """
    Evaluate all known exits and choose the best one.

    Uses a utility function that considers:

    * **Distance** — shorter is better, amplified when hazard is worsening.
    * **Congestion** — estimated from ``percept.neighbor_occupancy``; exits
      outside the perception range receive an uncertainty-based estimate
      modulated by ``personality.exploration``.
    * **Hazard near exit** — estimated from ``percept.neighbor_hazards``;
      discounted by ``personality.risk_tolerance``.
    * **Hazard-lobe interference** — if a perceived hazard cluster lies
      *between* the agent and a candidate exit, that exit is penalised.
    * **Hazard rate-of-change** — when local danger is worsening, farther
      exits are penalised more heavily (urgency pressure).
    * **Exploration noise** — stochastic offset scaled by
      ``personality.exploration``, modelling game-theoretic exit diversity.

    This is inspired by game-theoretic reasoning: if everyone goes to the
    nearest exit, it becomes congested.  Strategic agents may choose a farther
    but less congested exit.
    """
    if not world.exits:
        return None

    scores: list[float] = []

    for exit_pos in world.exits:
        # ── Distance component ────────────────────────────────────
        dist = agent_pos.euclidean_distance(exit_pos)
        if dist < 0.1:
            return StrategicDecision(
                target_goal=exit_pos, reason="at_exit", confidence=1.0
            )

        dist_score = -dist  # closer is better

        # ── Hazard rate-of-change urgency ─────────────────────────
        # When local hazard is worsening, amplify distance penalty
        # (farther exits become less attractive under time pressure).
        roc = percept.hazard_rate_of_change
        if roc > 0.01:
            dist_score *= 1.0 + roc * 0.5

        # ── Congestion (perception-based) ─────────────────────────
        occ_total, occ_obs, occ_cells = _perception_occupancy_near(
            percept, exit_pos, radius=3
        )
        if occ_obs > 0:
            # Extrapolate observed congestion to full area
            congestion = occ_total * (occ_cells / occ_obs)
        else:
            # Exit outside awareness — uncertain; exploratory agents
            # are less scared of the unknown.
            congestion = 3.0 * (1.0 - personality.exploration * 0.5)

        congestion_penalty = -congestion * (2.0 / max(personality.patience, 1.0))

        # ── Hazard near exit (perception-based) ───────────────────
        haz_total, haz_obs, haz_cells = _perception_hazard_near(
            percept, exit_pos, radius=2
        )
        if haz_obs > 0:
            hazard_near = haz_total * (haz_cells / haz_obs)
        elif dist > personality.awareness_radius:
            # Exit fully outside perception — mild uncertainty
            hazard_near = 0.2 * (1.0 - personality.exploration * 0.3)
        else:
            hazard_near = 0.0

        hazard_penalty = -hazard_near * (3.0 * (1.0 - personality.risk_tolerance))

        # ── Hazard-lobe interference ──────────────────────────────
        lobe_penalty = _lobe_interference(percept, agent_pos, exit_pos, dist)

        # ── Exploration noise ─────────────────────────────────────
        noise = rng.normal(0, personality.exploration * 5.0)

        total = (
            dist_score
            + congestion_penalty
            + hazard_penalty
            + lobe_penalty
            + noise
        )
        scores.append(total)

    if not scores:
        return None

    # Pick the best exit
    best_idx = int(np.argmax(scores))
    best_exit = world.exits[best_idx]
    best_score = scores[best_idx]

    # Confidence based on margin over second-best
    if len(scores) > 1:
        sorted_scores = sorted(scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        confidence = min(1.0, margin / (abs(sorted_scores[0]) + 1e-8))
    else:
        confidence = 1.0

    return StrategicDecision(
        target_goal=best_exit,
        reason=f"exit_{best_idx}_score_{best_score:.1f}",
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


# ── Simulated annealing replan ───────────────────────────────────────


def simulated_annealing_replan(
    world: World,
    agent_pos: Position,
    current_goal: Position,
    personality: PersonalityTraits,
    rng: np.random.Generator,
) -> Position:
    """
    Use simulated annealing to find an alternative waypoint when stuck.

    All SA hyper-parameters are derived from ``PersonalityTraits``:

    * **exploration** → ``initial_temp`` (higher ⇒ bolder jumps) and
      perturbation radius.
    * **patience** → ``iterations`` (more patient ⇒ longer search) and
      ``cooling_rate`` (slower cooling ⇒ more thorough search).
    * **risk_tolerance** → hazard weight in the evaluation function (risk-
      tolerant agents discount hazard at candidate waypoints).

    Returns:
        A waypoint position to route through.
    """
    # ── Personality-derived hyper-parameters ──────────────────────
    initial_temp = 5.0 + 15.0 * personality.exploration
    iterations = max(20, int(personality.patience * 2.5))
    # Patient agents cool slower → explore longer
    cooling_rate = 0.85 + 0.1 * min(1.0, personality.patience / 40.0)

    # Weight of hazard in the cost function.  Risk-tolerant agents
    # treat hazard as less costly, enabling shortcuts through danger.
    hazard_weight = 5.0 * (1.0 - personality.risk_tolerance)

    costs = world.effective_cost_grid()

    def evaluate(waypoint: Position) -> float:
        """Score a candidate waypoint. Lower = better."""
        if not world.in_bounds(waypoint.x, waypoint.y):
            return float("inf")
        if not world.walkable_grid[waypoint.y, waypoint.x]:
            return float("inf")

        d1 = agent_pos.euclidean_distance(waypoint)
        d2 = waypoint.euclidean_distance(current_goal)
        local_cost = float(costs[waypoint.y, waypoint.x])
        congestion = float(world.occupancy_grid[waypoint.y, waypoint.x])
        hazard = float(world.hazard_grid[waypoint.y, waypoint.x])

        return (
            d1
            + d2
            + local_cost * 2.0
            + congestion * 3.0
            + hazard * hazard_weight
        )

    # Initialize: midpoint between agent and goal
    mid_x = (agent_pos.x + current_goal.x) // 2
    mid_y = (agent_pos.y + current_goal.y) // 2
    current = Position(
        np.clip(mid_x, 0, world.width - 1),
        np.clip(mid_y, 0, world.height - 1),
    )
    current_score = evaluate(current)
    best = current
    best_score = current_score

    temp = initial_temp

    for _ in range(iterations):
        # Perturbation radius scales with temperature AND exploration
        radius = max(1, int(temp * (0.5 + 0.5 * personality.exploration)))
        new_x = current.x + rng.integers(-radius, radius + 1)
        new_y = current.y + rng.integers(-radius, radius + 1)
        new_x = int(np.clip(new_x, 0, world.width - 1))
        new_y = int(np.clip(new_y, 0, world.height - 1))
        candidate = Position(new_x, new_y)

        new_score = evaluate(candidate)
        delta = new_score - current_score

        # Accept if better, or probabilistically if worse
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 0.01)):
            current = candidate
            current_score = new_score

        if current_score < best_score:
            best = current
            best_score = current_score

        temp *= cooling_rate

    return best


# ── Replan trigger ───────────────────────────────────────────────────


def should_replan(
    percept: LocalPercept,
    personality: PersonalityTraits,
    ticks_since_progress: int,
) -> bool:
    """
    Determine if the agent should trigger strategic replanning.

    Conditions (any triggers ``True``):

    * Stuck for longer than ``personality.patience`` allows.
    * Perceived peak hazard exceeds ``personality.panic_threshold``.
    * Hazard detected in 3+ quadrants (surrounded).
    * Hazard rate-of-change exceeds 0.2 (rapidly worsening environment).
    * High local congestion (> 8 neighbours).
    """
    # Stuck too long
    if ticks_since_progress > personality.patience:
        return True

    # Panic trigger
    if percept.max_local_hazard > personality.panic_threshold:
        return True

    # Surrounded by hazard from multiple directions
    if len(percept.hazard_lobes) >= 3:
        return True

    # Rapidly worsening danger
    if percept.hazard_rate_of_change > 0.2:
        return True

    # High local congestion
    if percept.nearby_agent_count > 8:
        return True

    return False
