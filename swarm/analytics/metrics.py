"""
Metrics collector — tracks aggregate KPIs per tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarm.core.engine import SimulationState


@dataclass
class TickMetrics:
    """Per-tick metrics snapshot."""

    tick: int
    time: float
    active_agents: int
    evacuated_agents: int
    dead_agents: int
    panicking_agents: int
    stuck_agents: int
    active_hazards: int
    avg_speed: float = 0.0
    throughput: float = 0.0  # evacuations per second


class MetricsCollector:
    """Collects and stores per-tick metrics."""

    def __init__(self):
        self.history: list[TickMetrics] = []
        self._prev_evacuated = 0

    def record(self, state: "SimulationState") -> TickMetrics:
        """Record metrics for a tick."""
        stats = state.swarm_stats

        # Throughput: new evacuations this tick
        new_evac = stats.evacuated - self._prev_evacuated
        self._prev_evacuated = stats.evacuated

        metrics = TickMetrics(
            tick=state.tick,
            time=state.time,
            active_agents=stats.active,
            evacuated_agents=stats.evacuated,
            dead_agents=stats.dead,
            panicking_agents=stats.panicking,
            stuck_agents=stats.stuck,
            active_hazards=state.active_hazards,
            throughput=new_evac,
        )
        self.history.append(metrics)
        return metrics

    def evacuation_curve(self) -> list[tuple[int, int]]:
        """Return (tick, evacuated_count) pairs."""
        return [(m.tick, m.evacuated_agents) for m in self.history]

    def survival_curve(self) -> list[tuple[int, float]]:
        """Return (tick, survival_rate) pairs."""
        if not self.history:
            return []
        first = self.history[0]
        total = first.active_agents + first.evacuated_agents + first.dead_agents
        if total == 0:
            return []
        return [(m.tick, (total - m.dead_agents) / total) for m in self.history]

    def peak_panic(self) -> int:
        """Return the maximum number of panicking agents at any tick."""
        if not self.history:
            return 0
        return max(m.panicking_agents for m in self.history)

    def mean_evacuation_time(self) -> float:
        """Estimate mean evacuation time from the curve."""
        curve = self.evacuation_curve()
        if not curve or curve[-1][1] == 0:
            return float("inf")
        # Find tick at 50% evacuation
        target = curve[-1][1] / 2
        for tick, evac in curve:
            if evac >= target:
                return float(tick)
        return float(curve[-1][0])

    def summary_dict(self) -> dict:
        """Return summary statistics."""
        if not self.history:
            return {}
        final = self.history[-1]
        total = final.active_agents + final.evacuated_agents + final.dead_agents
        return {
            "total_ticks": final.tick,
            "total_time": final.time,
            "total_agents": total,
            "evacuated": final.evacuated_agents,
            "dead": final.dead_agents,
            "stuck": final.stuck_agents,
            "peak_panic": self.peak_panic(),
            "mean_evacuation_time": self.mean_evacuation_time(),
            "final_evacuation_rate": final.evacuated_agents / max(total, 1),
        }
