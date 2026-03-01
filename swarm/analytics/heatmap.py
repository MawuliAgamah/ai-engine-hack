"""
Spatial heatmap generation for trajectory and density analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def trajectory_heatmap(
    width: int,
    height: int,
    positions: list[tuple[int, int]],
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Generate a smoothed heatmap from position samples.

    Args:
        width: Grid width.
        height: Grid height.
        positions: List of (x, y) positions visited.
        sigma: Gaussian blur sigma for smoothing.

    Returns:
        2D array of visit density.
    """
    heatmap = np.zeros((height, width), dtype=np.float64)
    for x, y in positions:
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1.0

    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap


def congestion_heatmap(
    width: int,
    height: int,
    occupancy_snapshots: list[np.ndarray],
) -> np.ndarray:
    """
    Average occupancy over multiple tick snapshots to find congestion hotspots.

    Args:
        width: Grid width.
        height: Grid height.
        occupancy_snapshots: List of 2D occupancy arrays per tick.

    Returns:
        2D array of mean occupancy.
    """
    if not occupancy_snapshots:
        return np.zeros((height, width))
    return np.mean(occupancy_snapshots, axis=0)


def danger_exposure_map(
    width: int,
    height: int,
    hazard_snapshots: list[np.ndarray],
) -> np.ndarray:
    """
    Cumulative hazard exposure map.

    Returns the maximum hazard intensity observed at each cell.
    """
    if not hazard_snapshots:
        return np.zeros((height, width))
    return np.max(hazard_snapshots, axis=0)


def flow_rate_at_exits(
    exit_positions: list[tuple[int, int]],
    agent_positions_per_tick: dict[int, list[tuple[int, int]]],
    neighborhood: int = 3,
) -> list[tuple[int, int]]:
    """
    Compute per-tick flow rate at exit regions.

    Returns list of (tick, agents_near_exits) for throughput analysis.
    """
    flow = []
    for tick in sorted(agent_positions_per_tick.keys()):
        positions = agent_positions_per_tick[tick]
        count = 0
        for px, py in positions:
            for ex, ey in exit_positions:
                if abs(px - ex) <= neighborhood and abs(py - ey) <= neighborhood:
                    count += 1
                    break
        flow.append((tick, count))
    return flow
