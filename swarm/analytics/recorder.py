"""
Trajectory recorder — stores per-agent position history for analysis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from swarm.core.world import Position


class TrajectoryRecorder:
    """Records agent positions over time for post-hoc analysis."""

    def __init__(self):
        # agent_id -> list of (tick, x, y)
        self._trajectories: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    def record_position(self, tick: int, agent_id: int, position: Position) -> None:
        """Record an agent's position at a given tick."""
        self._trajectories[agent_id].append((tick, position.x, position.y))

    def get_trajectory(self, agent_id: int) -> list[tuple[int, int, int]]:
        """Get full trajectory for an agent: [(tick, x, y), ...]."""
        return self._trajectories.get(agent_id, [])

    def all_positions(self) -> list[tuple[int, int]]:
        """Get all recorded positions (flattened, for heatmaps)."""
        result = []
        for traj in self._trajectories.values():
            for _, x, y in traj:
                result.append((x, y))
        return result

    def positions_at_tick(self, tick: int) -> list[tuple[int, int]]:
        """Get all agent positions at a specific tick."""
        result = []
        for traj in self._trajectories.values():
            for t, x, y in traj:
                if t == tick:
                    result.append((x, y))
        return result

    def agent_ids(self) -> list[int]:
        """Get list of all recorded agent IDs."""
        return list(self._trajectories.keys())

    def path_length(self, agent_id: int) -> float:
        """Compute total path length for an agent."""
        traj = self._trajectories.get(agent_id, [])
        if len(traj) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(traj)):
            _, x1, y1 = traj[i - 1]
            _, x2, y2 = traj[i]
            total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return total

    def mean_path_length(self) -> float:
        """Mean path length across all agents."""
        lengths = [self.path_length(aid) for aid in self._trajectories]
        return sum(lengths) / max(len(lengths), 1)

    def to_dict(self) -> dict[int, list[tuple[int, int, int]]]:
        """Export all trajectories as a dict."""
        return dict(self._trajectories)
