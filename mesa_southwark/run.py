from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import RunConfig
from .metrics import (
    aggregate_batch_summaries,
    collect_tick_metrics,
    summarize_replicate,
)
from .model import SouthwarkModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Southwark Mesa ABM in headless mode.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-giraffes", type=int, default=20)
    parser.add_argument("--move-cost", type=float, default=1.0)
    parser.add_argument("--green-energy-gain", type=float, default=10.0)
    parser.add_argument(
        "--depleting-green",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, green land is consumed and energy gain is limited by remaining resource.",
    )
    parser.add_argument("--green-regrowth-per-tick", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument(
        "--water-is-barrier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, water patches are non-traversable.",
    )
    return parser.parse_args()


def run_single_replicate(
    config: RunConfig,
    replicate: int,
    seed: int | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    model = SouthwarkModel(
        num_giraffes=config.num_giraffes,
        move_cost=config.move_cost,
        green_energy_gain=config.green_energy_gain,
        seed=seed,
        water_is_barrier=config.water_is_barrier,
        depleting_green=config.depleting_green,
        green_regrowth_per_tick=config.green_regrowth_per_tick,
        csv_path=config.csv_path,
    )

    tick_rows: list[dict[str, Any]] = [collect_tick_metrics(model, replicate=replicate)]

    while model.tick < config.steps and model.giraffes:
        model.step()
        tick_rows.append(collect_tick_metrics(model, replicate=replicate))

    model.finalize_outcomes(end_reason="max_steps_or_exhausted")
    route_rows = model.get_giraffe_routes()
    route_rows_long = model.get_giraffe_routes_long()
    outcome_rows = model.get_giraffe_outcomes()
    tick_matrix_rows = model.get_giraffe_tick_matrix(max_tick=config.steps)

    summary = summarize_replicate(
        replicate=replicate,
        seed=seed,
        tick_rows=tick_rows,
        static_metrics=model.static_metrics,
    )
    return tick_rows, summary, model.static_metrics, route_rows, route_rows_long, outcome_rows, tick_matrix_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch(config: RunConfig) -> dict[str, Any]:
    config.out.mkdir(parents=True, exist_ok=True)

    all_tick_rows: list[dict[str, Any]] = []
    replicate_summaries: list[dict[str, Any]] = []
    all_route_rows: list[dict[str, Any]] = []
    all_route_rows_long: list[dict[str, Any]] = []
    all_giraffe_outcomes: list[dict[str, Any]] = []
    all_agent_tick_rows: list[dict[str, Any]] = []
    static_metrics: dict[str, Any] | None = None

    for replicate in range(config.replicates):
        replicate_seed = None if config.seed is None else config.seed + replicate
        (
            tick_rows,
            summary,
            run_static_metrics,
            route_rows,
            route_rows_long,
            outcome_rows,
            tick_matrix_rows,
        ) = run_single_replicate(
            config=config,
            replicate=replicate,
            seed=replicate_seed,
        )

        all_tick_rows.extend(tick_rows)
        replicate_summaries.append(summary)
        for row in route_rows:
            row_copy = dict(row)
            row_copy["replicate"] = replicate
            all_route_rows.append(row_copy)
        for row in route_rows_long:
            row_copy = dict(row)
            row_copy["replicate"] = replicate
            all_route_rows_long.append(row_copy)
        for row in outcome_rows:
            row_copy = dict(row)
            row_copy["replicate"] = replicate
            all_giraffe_outcomes.append(row_copy)
        for row in tick_matrix_rows:
            row_copy = dict(row)
            row_copy["replicate"] = replicate
            all_agent_tick_rows.append(row_copy)
        if static_metrics is None:
            static_metrics = run_static_metrics

    aggregate = aggregate_batch_summaries(replicate_summaries)

    ticks_csv = config.out / "metrics_ticks.csv"
    replicate_csv = config.out / "replicate_summary.csv"
    routes_csv = config.out / "giraffe_routes.csv"
    routes_long_csv = config.out / "giraffe_routes_long.csv"
    outcomes_csv = config.out / "giraffe_outcomes.csv"
    agent_tick_csv = config.out / "agent_tick_matrix.csv"
    summary_json = config.out / "summary.json"

    write_csv(ticks_csv, all_tick_rows)
    write_csv(replicate_csv, replicate_summaries)
    write_csv(routes_csv, all_route_rows)
    write_csv(routes_long_csv, all_route_rows_long)
    write_csv(outcomes_csv, all_giraffe_outcomes)
    write_csv(agent_tick_csv, all_agent_tick_rows)

    payload = {
        "config": config.to_dict(),
        "static_metrics": static_metrics or {},
        "aggregate": aggregate,
        "replicate_summaries": replicate_summaries,
        "giraffe_outcomes_count": len(all_giraffe_outcomes),
        "giraffe_route_rows_count": len(all_route_rows),
        "giraffe_route_rows_long_count": len(all_route_rows_long),
        "agent_tick_rows_count": len(all_agent_tick_rows),
        "artifacts": {
            "ticks_csv": str(ticks_csv),
            "replicate_csv": str(replicate_csv),
            "routes_csv": str(routes_csv),
            "routes_long_csv": str(routes_long_csv),
            "outcomes_csv": str(outcomes_csv),
            "agent_tick_csv": str(agent_tick_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return payload


def main() -> None:
    args = parse_args()

    config = RunConfig(
        steps=args.steps,
        num_giraffes=args.num_giraffes,
        move_cost=args.move_cost,
        green_energy_gain=args.green_energy_gain,
        depleting_green=args.depleting_green,
        green_regrowth_per_tick=args.green_regrowth_per_tick,
        seed=args.seed,
        replicates=args.replicates,
        out=args.out,
        csv_path=args.csv_path,
        water_is_barrier=args.water_is_barrier,
    )

    payload = run_batch(config)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
