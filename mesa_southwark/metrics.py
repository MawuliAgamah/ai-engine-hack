from __future__ import annotations

from statistics import mean
from typing import Any, Iterable

from .environment import PatchCell, patch_is_green


def compute_static_environment_metrics(cells: Iterable[PatchCell]) -> dict[str, Any]:
    cells_list = list(cells)

    occupied_cells = [cell for cell in cells_list if bool(cell.attrs["occupied?"])]
    zone_counts: dict[str, int] = {}

    green_patches = 0
    water_patches = 0
    super_urban_patches = 0
    residential_patches = 0
    total_buildings = 0

    for cell in occupied_cells:
        zone_type = cell.attrs["zone-type"]
        zone_counts[zone_type] = zone_counts.get(zone_type, 0) + 1

        if patch_is_green(cell.attrs):
            green_patches += 1
        if zone_type == "water":
            water_patches += 1
        if zone_type == "super-urban":
            super_urban_patches += 1
        if zone_type == "residential":
            residential_patches += 1

        total_buildings += int(cell.attrs["building-count"])

    return {
        "occupied_patches": len(occupied_cells),
        "green_patches": green_patches,
        "water_patches": water_patches,
        "super_urban_patches": super_urban_patches,
        "residential_patches": residential_patches,
        "total_buildings": total_buildings,
        "zone_type_counts": dict(sorted(zone_counts.items())),
    }


def collect_tick_metrics(model: Any, replicate: int) -> dict[str, Any]:
    giraffes = model.giraffes
    energies = [float(agent.energy) for agent in giraffes]
    distances = [int(agent.distance_travelled) for agent in giraffes]

    mean_energy = mean(energies) if energies else 0.0
    avg_distance = mean(distances) if distances else 0.0
    giraffes_on_green = sum(1 for agent in giraffes if model.is_patch_green(agent.pos))
    green_summary = model.green_land_summary() if hasattr(model, "green_land_summary") else {}
    population_summary = model.giraffe_population_summary() if hasattr(model, "giraffe_population_summary") else {}

    return {
        "replicate": replicate,
        "tick": model.tick,
        "giraffes_alive": len(giraffes),
        "mean_energy": mean_energy,
        "avg_distance": avg_distance,
        "giraffes_on_green": giraffes_on_green,
        "giraffes_dead": int(population_summary.get("dead", 0)),
        "giraffes_spawned": int(population_summary.get("total_spawned", len(giraffes))),
        "green_resource_initial": float(green_summary.get("green_resource_initial", 0.0)),
        "green_resource_remaining": float(green_summary.get("green_resource_remaining", 0.0)),
        "green_resource_consumed": float(green_summary.get("green_resource_consumed", 0.0)),
        "green_resource_remaining_pct": float(green_summary.get("green_resource_remaining_pct", 0.0)),
        "green_resource_consumed_pct": float(green_summary.get("green_resource_consumed_pct", 0.0)),
    }


def survival_curve_area(tick_rows: list[dict[str, Any]]) -> float:
    if not tick_rows:
        return 0.0

    sorted_rows = sorted(tick_rows, key=lambda row: int(row["tick"]))
    if len(sorted_rows) == 1:
        return float(sorted_rows[0]["giraffes_alive"])

    area = 0.0
    for left, right in zip(sorted_rows[:-1], sorted_rows[1:]):
        t0 = float(left["tick"])
        t1 = float(right["tick"])
        y0 = float(left["giraffes_alive"])
        y1 = float(right["giraffes_alive"])
        area += ((y0 + y1) / 2.0) * (t1 - t0)
    return area


def summarize_replicate(
    replicate: int,
    seed: int | None,
    tick_rows: list[dict[str, Any]],
    static_metrics: dict[str, Any],
) -> dict[str, Any]:
    last_row = sorted(tick_rows, key=lambda row: int(row["tick"]))[-1] if tick_rows else {
        "tick": 0,
        "giraffes_alive": 0,
        "mean_energy": 0.0,
        "avg_distance": 0.0,
        "giraffes_on_green": 0,
    }

    green_consumed_values = [float(row.get("green_resource_consumed", 0.0)) for row in tick_rows]
    final_green_consumed = green_consumed_values[-1] if green_consumed_values else 0.0

    return {
        "replicate": replicate,
        "seed": seed,
        "final_tick": int(last_row["tick"]),
        "final_population": int(last_row["giraffes_alive"]),
        "final_mean_energy": float(last_row["mean_energy"]),
        "final_avg_distance": float(last_row["avg_distance"]),
        "final_giraffes_on_green": int(last_row["giraffes_on_green"]),
        "final_dead_giraffes": int(last_row.get("giraffes_dead", 0)),
        "survival_curve_area": survival_curve_area(tick_rows),
        "green_resource_consumed": float(final_green_consumed),
        "green_resource_remaining": float(last_row.get("green_resource_remaining", 0.0)),
        "green_resource_remaining_pct": float(last_row.get("green_resource_remaining_pct", 0.0)),
        "green_patches": int(static_metrics["green_patches"]),
        "water_patches": int(static_metrics["water_patches"]),
        "super_urban_patches": int(static_metrics["super_urban_patches"]),
        "residential_patches": int(static_metrics["residential_patches"]),
        "total_buildings": int(static_metrics["total_buildings"]),
    }


def aggregate_batch_summaries(replicate_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not replicate_summaries:
        return {
            "replicates": 0,
            "final_population_mean": 0.0,
            "final_mean_energy_mean": 0.0,
            "survival_curve_area_mean": 0.0,
        }

    populations = [float(row["final_population"]) for row in replicate_summaries]
    energies = [float(row["final_mean_energy"]) for row in replicate_summaries]
    survival_areas = [float(row["survival_curve_area"]) for row in replicate_summaries]
    green_consumed = [float(row.get("green_resource_consumed", 0.0)) for row in replicate_summaries]

    return {
        "replicates": len(replicate_summaries),
        "final_population_mean": mean(populations),
        "final_mean_energy_mean": mean(energies),
        "survival_curve_area_mean": mean(survival_areas),
        "green_resource_consumed_mean": mean(green_consumed) if green_consumed else 0.0,
    }


def mean_abs_percent_diff(current: float, baseline: float) -> float:
    denominator = abs(baseline) if abs(baseline) > 1e-9 else 1.0
    return abs(current - baseline) / denominator


def compare_to_baseline(
    current_aggregate: dict[str, Any],
    baseline_aggregate: dict[str, Any],
) -> dict[str, float]:
    population_diff = mean_abs_percent_diff(
        float(current_aggregate["final_population_mean"]),
        float(baseline_aggregate["final_population_mean"]),
    )
    energy_diff = mean_abs_percent_diff(
        float(current_aggregate["final_mean_energy_mean"]),
        float(baseline_aggregate["final_mean_energy_mean"]),
    )
    area_diff = mean_abs_percent_diff(
        float(current_aggregate["survival_curve_area_mean"]),
        float(baseline_aggregate["survival_curve_area_mean"]),
    )

    return {
        "final_population_mapd": population_diff,
        "final_mean_energy_mapd": energy_diff,
        "survival_curve_area_mapd": area_diff,
    }
