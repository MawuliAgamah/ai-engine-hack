"""
Generational learning — summarise a completed simulation and persist insights.

At the end of each generation (simulation run), we:

1. **Gather** every agent's journey, reasoning trace, and final outcome.
2. **Call the LLM** with a meta-prompt asking it to distil reusable
   navigation insights from the collective experience.
3. **Save** the resulting summary to a ``.txt`` file so the *next*
   generation can load it into every agent's system prompt as
   "Learnings from Previous Attempts".

The learnings file is scenario-specific: ``learnings/<scenario_name>.txt``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from swarm.agents.base import AgentState
from swarm.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Directory under the project root where learnings are persisted.
LEARNINGS_DIR = Path(__file__).resolve().parent.parent.parent / "learnings"

# ── Meta-prompt for the summarisation call ───────────────────────────

_SUMMARISE_SYSTEM = """\
You are an analyst reviewing the movement logs of autonomous agents in a \
spatial grid simulation.  Your task is to produce a concise set of \
**reusable navigation insights** that future agents can follow to perform \
better.

Rules:
- Write 5-15 bullet points.
- Each bullet should be a concrete, actionable lesson (e.g. "Avoid cells \
north of row 20 after tick 50 — fire spreads there quickly").
- Reference spatial directions, approximate grid regions, hazard timing, \
and exit locations where relevant.
- Do NOT repeat raw coordinates verbatim — generalise into directional \
guidance.
- Be concise.  No preamble, no closing remarks — just the bullet list.
"""

# ── Public API ───────────────────────────────────────────────────────


def gather_agent_summaries(agents: list) -> str:
    """Build a text block summarising every agent's experience.

    Parameters
    ----------
    agents :
        List of ``LLMAgent`` instances (after the simulation has finished).

    Returns
    -------
    str
        A multi-section text ready to include in the summarisation prompt.
    """
    from swarm.agents.llm_agent import LLMAgent  # deferred to avoid circular

    sections: list[str] = []

    for agent in agents:
        assert isinstance(agent, LLMAgent)

        # Final outcome
        outcome = agent.state.value
        start_pos = agent.memory.journey[0][1] if agent.memory.journey else agent.position
        end_pos = agent.position
        ticks_lived = len(agent.memory.journey)

        lines = [
            f"### Agent #{agent.id} — {outcome}",
            f"- Personality: {agent.personality[:120]}",
            f"- Goal: {agent.goal[:120]}",
            f"- Start: ({start_pos.x}, {start_pos.y})  →  End: ({end_pos.x}, {end_pos.y})",
            f"- Ticks survived: {ticks_lived}",
        ]

        # Journey (sampled — keep it short)
        journey = agent.memory.journey
        if len(journey) > 20:
            # Sample: first 5, middle 5, last 10
            sampled = journey[:5] + journey[len(journey) // 2 - 2 : len(journey) // 2 + 3] + journey[-10:]
        else:
            sampled = journey
        path_str = " → ".join(f"t{t}:({p.x},{p.y})" for t, p in sampled)
        lines.append(f"- Path (sampled): {path_str}")

        # Last 5 reasoning entries
        recent_reasoning = agent.memory.reasoning[-5:]
        if recent_reasoning:
            lines.append("- Recent reasoning:")
            for t, r in recent_reasoning:
                lines.append(f"    t={t}: {r[:200]}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def summarise_generation(
    client: LLMClient,
    agents: list,
    scenario_text: str,
    stats_dict: dict | None = None,
) -> str:
    """Make a final LLM call to distil learnings from the completed run.

    Parameters
    ----------
    client :
        The LLM client to use for the summarisation call.
    agents :
        All agents from the completed simulation.
    scenario_text :
        The scenario description (for context).
    stats_dict :
        Optional final stats (total, evacuated, dead, …).

    Returns
    -------
    str
        The LLM's bullet-point summary of reusable insights.
    """
    agent_summaries = gather_agent_summaries(agents)

    stats_block = ""
    if stats_dict:
        stats_block = (
            "\n## Final Statistics\n"
            + "\n".join(f"- {k}: {v}" for k, v in stats_dict.items())
            + "\n"
        )

    user_message = (
        f"## Scenario\n{scenario_text}\n"
        f"{stats_block}\n"
        f"## Agent Logs\n{agent_summaries}\n\n"
        "Now produce the bullet-point learnings."
    )

    messages = [
        {"role": "system", "content": _SUMMARISE_SYSTEM},
        {"role": "user", "content": user_message},
    ]

    logger.info("Calling LLM for generational summary (%d agents)…", len(agents))
    response = client.complete(messages)
    logger.info("Generational summary received (%d chars)", len(response))

    # Post-process (change "Agents" to "People")    
    response = response.strip()
    response = response.replace("Agents", "People")

    return response


def save_learnings(scenario_name: str, text: str) -> Path:
    """Persist learnings to ``learnings/<scenario_name>.txt``.

    Returns the path to the saved file.
    """
    LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LEARNINGS_DIR / f"{scenario_name}.txt"
    path.write_text(text, encoding="utf-8")
    logger.info("Saved learnings → %s (%d chars)", path, len(text))
    return path


def load_learnings(scenario_name: str) -> str:
    """Load previously saved learnings for a scenario.

    Returns an empty string if no file exists yet (first generation).
    """
    path = LEARNINGS_DIR / f"{scenario_name}.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        logger.info("Loaded learnings from %s (%d chars)", path, len(text))
        return text
    logger.info("No prior learnings found for scenario '%s'", scenario_name)
    return ""
