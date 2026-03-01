"""
Prompt construction for LLM-driven agents.

Two functions build the two parts of every API call:

* ``build_system_prompt``  — static per agent for the whole simulation.
* ``build_user_message``   — changes every tick (perception + available
  actions + journey history + reasoning history).

Together they define the *contract* between the simulation and the LLM.

**Output contract** — the LLM must reply with exactly::

    <reasoning text> | <integer index>

where the index refers to one of the numbered actions listed in the user
message.  This is deliberately minimal: no JSON, no markdown — just a pipe-
separated pair that is cheap to generate and trivial to parse.
"""

from __future__ import annotations

import random

from swarm.agents.perception import LocalPercept
from swarm.core.world import Position


# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are Agent #{agent_id} in a spatial grid simulation.

## Scenario
{scenario}

## Your Personality
{personality}

## Your Goal
{goal}

## How the World Works
- The world is a 2D grid.  You occupy one cell at a time.
- Each tick you choose ONE action from the numbered list you are given.
- Every cell has a terrain cost (higher = harder), a hazard level \
(0.0–1.0, ≥0.95 is lethal), and an occupancy count (agents present).
- Hazards spread over time.  Conditions change between ticks.

## How to Respond
Reply with EXACTLY one line in this format:

<your reasoning> | <action index>

- <your reasoning>: 1-2 sentences explaining your thought process.
- |               : a literal pipe character separating reasoning from choice.
- <action index>  : the integer index of your chosen action (e.g. 0, 1, 2 …).

Example response:
The exit is to the east and the hazard is approaching from the north, so I move east. | 3

Rules:
- Output ONLY one line.  No extra text, no markdown, no JSON.
- The index MUST match one of the options shown under "Available Actions".
- Pick the SINGLE best action considering cost, hazard, occupancy, and your goal.
- If all options are bad, pick the least bad one and explain why.

## Learnings from Previous Attempts
{learnings}
"""


def build_system_prompt(
    agent_id: int,
    scenario: str,
    personality: str,
    goal: str,
    learnings: str = "",
) -> str:
    """Build the static system prompt for an LLM-driven agent.

    This prompt is set once at agent creation and does NOT include the
    available actions (those change every tick and go in the user message).
    If 'learnings' is non-empty, include the section in the prompt.
    """
    base_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        agent_id=agent_id,
        scenario=scenario,
        personality=personality,
        goal=goal,
        learnings=""  # placeholder, will be replaced below
    )
    if learnings and learnings.strip():
        learnings_section = f"\n## Learnings from Previous Attempts\n{learnings.strip()}\n"
        # Insert before the final newline (or at the end)
        # Remove the placeholder line if present
        base_prompt = base_prompt.replace("##\u00A0Learnings from Previous Attempts\n", "")
        base_prompt += learnings_section
    return base_prompt


# ── Available-moves list ─────────────────────────────────────────────


def get_available_moves(
    percept: LocalPercept,
) -> list[tuple[str, Position]]:
    """Return ``[(label, Position), ...]`` for STAY + each walkable neighbour.

    The first entry is always STAY at the agent's current position.
    """
    moves: list[tuple[str, Position]] = [("STAY at", percept.position)]
    for nb in percept.walkable_neighbors:
        if nb != percept.position:
            moves.append(("MOVE to", nb))
    random.shuffle(moves)  # randomize order to avoid positional bias in LLM choices
    return moves


# ── User (per-tick) message ──────────────────────────────────────────


def build_user_message(
    percept: LocalPercept,
    available_moves: list[tuple[str, Position]],
    tick: int,
    state: str,
    journey_history: list[tuple[int, Position]],
    reasoning_history: list[tuple[int, str]],
    max_journey: int = 20,
    max_reasoning: int = 5,
) -> str:
    """Build the per-tick user message from perception + memory.

    Sections included:
    1. **Current State** — tick, position, agent state.
    2. **Local Perception** — exit direction/distance, hazards (scalar +
       lobes + trend), nearby agents, flow suggestion.
    3. **Available Actions** — numbered list with cost/hazard/occupancy per
       cell, plus warning flags.
    4. **Journey History** — last *max_journey* (tick, position) pairs so
       the LLM can detect loops / backtracking.
    5. **Previous Reasoning** — last *max_reasoning* reasoning strings so
       the LLM can maintain decision continuity.
    """

    x, y = percept.position.x, percept.position.y

    # ── Perception section ────────────────────────────────────────
    perception_lines: list[str] = []

    # Nearest exit
    if percept.nearest_exit_direction and percept.nearest_exit_distance is not None:
        ed = percept.nearest_exit_direction
        perception_lines.append(
            f"- Nearest exit: direction ({ed[0]:.2f}, {ed[1]:.2f}), "
            f"distance {percept.nearest_exit_distance:.1f}"
        )
    else:
        perception_lines.append("- No exits detected in awareness range")

    # Hazard at current position
    current_hz = percept.neighbor_hazards.get(percept.position, 0.0)
    perception_lines.append(f"- Hazard at your position: {current_hz:.2f}")
    perception_lines.append(f"- Peak hazard nearby: {percept.max_local_hazard:.2f}")

    # Hazard direction
    if percept.max_local_hazard > 0.01:
        hd = percept.hazard_direction
        perception_lines.append(f"- Danger direction: ({hd[0]:.2f}, {hd[1]:.2f})")

    # Hazard lobes (multi-directional awareness)
    for i, lobe in enumerate(percept.hazard_lobes):
        perception_lines.append(
            f"- Hazard cluster {i + 1}: direction ({lobe.direction[0]:.2f}, "
            f"{lobe.direction[1]:.2f}), avg intensity {lobe.mean_intensity:.2f} "
            f"(peak {lobe.max_intensity:.2f}), {lobe.cell_count} cells"
        )

    # Hazard trend
    roc = percept.hazard_rate_of_change
    if roc > 0.1:
        trend = "RAPIDLY WORSENING"
    elif roc > 0.01:
        trend = "worsening"
    elif roc < -0.01:
        trend = "improving"
    else:
        trend = "stable"
    perception_lines.append(f"- Hazard trend: {trend} (Δ={roc:+.3f}/tick)")

    # Nearby agents
    perception_lines.append(f"- Nearby agents: {percept.nearby_agent_count}")

    # Flow field suggestion
    if percept.flow_direction:
        fd = percept.flow_direction
        perception_lines.append(f"- Suggested flow: ({fd[0]:.2f}, {fd[1]:.2f})")

    perception_text = "\n".join(perception_lines)

    # ── Available actions ─────────────────────────────────────────
    action_lines: list[str] = []
    for i, (label, pos) in enumerate(available_moves):
        cost = percept.neighbor_costs.get(pos, 1.0)
        hazard = percept.neighbor_hazards.get(pos, 0.0)
        occ = percept.neighbor_occupancy.get(pos, 0)

        flags: list[str] = []
        if hazard >= 0.5:
            flags.append("DANGEROUS")
        elif hazard > 0.1:
            flags.append("hazardous")
        if occ >= 3:
            flags.append("crowded")

        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        action_lines.append(
            f"{i}: {label} ({pos.x}, {pos.y}) — "
            f"cost={cost:.1f}, hazard={hazard:.2f}, agents={occ}"
            f"{flag_str}"
        )
    actions_text = "\n".join(action_lines)

    # ── Journey history ───────────────────────────────────────────
    recent_journey = journey_history[-max_journey:]
    if recent_journey:
        journey_lines = [f"  t={t}: ({p.x}, {p.y})" for t, p in recent_journey]
        journey_text = "\n".join(journey_lines)
    else:
        journey_text = "  (first move — no history yet)"

    # ── Reasoning history ─────────────────────────────────────────
    recent_reasoning = reasoning_history[-max_reasoning:]
    if recent_reasoning:
        reasoning_lines = [f"  t={t}: {r}" for t, r in recent_reasoning]
        reasoning_text = "\n".join(reasoning_lines)
    else:
        reasoning_text = "  (no previous reasoning)"

    return (
        f"## Current State\n"
        f"- Tick: {tick}\n"
        f"- Position: ({x}, {y})\n"
        f"- State: {state}\n"
        f"\n"
        f"## Local Perception\n"
        f"{perception_text}\n"
        f"\n"
        f"## Available Actions\n"
        f"{actions_text}\n"
        f"\n"
        f"## Journey History (last {len(recent_journey)} steps)\n"
        f"{journey_text}\n"
        f"\n"
        f"## Previous Reasoning\n"
        f"{reasoning_text}\n"
        f"\n"
        f"Respond: <reasoning> | <action index>"
    )
