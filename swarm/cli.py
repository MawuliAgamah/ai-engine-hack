"""
CLI entry point — Click-based command-line interface.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="swarm")
def cli():
    """SWARM — Spatial Waypoint Agent Routing Machine.

    Simulate agent-based spatial navigation and evacuation scenarios.
    """
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--ticks", "-t", default=None, type=int, help="Override max ticks.")
@click.option("--no-progress", is_flag=True, help="Disable progress bar.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output metrics JSON.")
@click.option("--viz", is_flag=True, help="Enable live visualization.")
def run(config_path: str, ticks: int | None, no_progress: bool, output: str | None, viz: bool):
    """Run a simulation from a YAML config file."""
    from swarm.config import build_simulation

    console.print(f"[bold blue]Loading scenario:[/bold blue] {config_path}")
    engine = build_simulation(config_path)

    # Optionally attach visualization
    renderer = None
    if viz:
        try:
            from swarm.viz.renderer import PygameRenderer

            renderer = PygameRenderer(engine.world)
            engine.on_tick(lambda state: renderer.draw(engine.world, engine.swarm, state))
        except ImportError:
            console.print("[yellow]Pygame not available — running headless.[/yellow]")

    console.print(
        f"[bold green]Starting simulation:[/bold green] "
        f"{engine.world.width}x{engine.world.height} world, "
        f"{engine.swarm.get_stats().total} agents"
    )

    history = engine.run(
        max_ticks=ticks,
        show_progress=not no_progress,
    )

    if renderer:
        renderer.close()

    # Print summary
    summary = engine.summary()
    _print_summary(summary)

    # Save metrics
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_data = {
            "summary": summary,
            "tick_count": len(history),
            "history": [
                {
                    "tick": s.tick,
                    "time": s.time,
                    "active": s.swarm_stats.active,
                    "evacuated": s.swarm_stats.evacuated,
                    "dead": s.swarm_stats.dead,
                    "panicking": s.swarm_stats.panicking,
                    "active_hazards": s.active_hazards,
                }
                for s in history
            ],
        }
        out_path.write_text(json.dumps(metrics_data, indent=2))
        console.print(f"[dim]Metrics saved to {output}[/dim]")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--param", "-p", multiple=True, help="Parameter to sweep: path=start:stop:step")
@click.option("--runs", "-r", default=5, type=int, help="Runs per configuration.")
@click.option("--output", "-o", default="sweep_results.json", help="Output file.")
def sweep(config_path: str, param: tuple[str], runs: int, output: str):
    """Run parameter sweeps over a scenario config."""
    import numpy as np

    from swarm.config import build_simulation, load_config

    console.print(f"[bold blue]Parameter sweep:[/bold blue] {config_path}")

    base_config = load_config(config_path)

    # Parse parameter ranges
    sweeps = []
    for p in param:
        path, range_str = p.split("=")
        parts = range_str.split(":")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        values = np.arange(start, stop + step / 2, step).tolist()
        sweeps.append((path.split("."), values))

    if not sweeps:
        console.print("[red]No parameters specified. Use -p path=start:stop:step[/red]")
        return

    results = []
    total = 1
    for _, values in sweeps:
        total *= len(values)
    total *= runs

    console.print(f"[dim]Total runs: {total}[/dim]")

    # Simple grid sweep (first param only for simplicity)
    param_path, param_values = sweeps[0]
    for val in param_values:
        run_results = []
        for run_idx in range(runs):
            # Deep copy config and set parameter
            import copy

            cfg = copy.deepcopy(base_config)
            _set_nested(cfg, param_path, val)

            engine = build_simulation.__wrapped__(cfg) if hasattr(build_simulation, "__wrapped__") else _build_from_dict(cfg)

            engine.run(show_progress=False)
            summary = engine.summary()
            run_results.append(summary)

        avg_evac_rate = sum(r["evacuation_rate"] for r in run_results) / len(run_results)
        avg_ticks = sum(r["ticks"] for r in run_results) / len(run_results)

        results.append({
            "param": ".".join(param_path),
            "value": val,
            "avg_evacuation_rate": avg_evac_rate,
            "avg_ticks": avg_ticks,
            "runs": run_results,
        })

        console.print(f"  {'.'.join(param_path)}={val:.2f} → evac={avg_evac_rate:.1%} ticks={avg_ticks:.0f}")

    Path(output).write_text(json.dumps(results, indent=2))
    console.print(f"[bold green]Sweep complete.[/bold green] Results → {output}")


@cli.command()
@click.argument("metrics_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Save plot to file.")
def plot(metrics_path: str, output: str | None):
    """Plot simulation metrics from a JSON output file."""
    import matplotlib.pyplot as plt

    data = json.loads(Path(metrics_path).read_text())
    history = data["history"]

    ticks = [h["tick"] for h in history]
    active = [h["active"] for h in history]
    evacuated = [h["evacuated"] for h in history]
    dead = [h["dead"] for h in history]
    panicking = [h["panicking"] for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(ticks, active, label="Active", color="blue")
    axes[0].plot(ticks, evacuated, label="Evacuated", color="green")
    axes[0].plot(ticks, dead, label="Dead", color="red")
    axes[0].set_ylabel("Agent Count")
    axes[0].legend()
    axes[0].set_title("Agent Population Over Time")

    axes[1].plot(ticks, panicking, label="Panicking", color="orange")
    axes[1].fill_between(ticks, 0, panicking, alpha=0.3, color="orange")
    axes[1].set_ylabel("Panicking Count")
    axes[1].set_xlabel("Tick")
    axes[1].legend()

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        console.print(f"[dim]Plot saved to {output}[/dim]")
    else:
        plt.show()


def _print_summary(summary: dict):
    """Pretty-print simulation summary."""
    table = Table(title="Simulation Results", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Ticks", str(summary["ticks"]))
    table.add_row("Simulated Time", f'{summary["time_seconds"]:.1f}s')
    table.add_row("Total Agents", str(summary["total_agents"]))
    table.add_row("Evacuated", str(summary["evacuated"]))
    table.add_row("Dead", str(summary["dead"]))
    table.add_row("Stuck", str(summary["stuck"]))
    table.add_row("Evacuation Rate", f'{summary["evacuation_rate"]:.1%}')
    table.add_row("Survival Rate", f'{summary["survival_rate"]:.1%}')

    console.print(table)


def _set_nested(d: dict, keys: list[str], value):
    """Set a nested dictionary value by key path."""
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def _build_from_dict(config: dict):
    """Build simulation from an already-loaded config dict."""
    from swarm.core.engine import SimulationEngine
    from swarm.config import build_world, build_clock, build_swarm, build_events
    from swarm.shared.blackboard import Blackboard
    from swarm.shared.fields import FieldManager
    from swarm.shared.pheromones import PheromoneSystem

    world = build_world(config)
    clock = build_clock(config)
    swarm = build_swarm(config, world)
    events = build_events(config)

    return SimulationEngine(
        world=world,
        swarm=swarm,
        clock=clock,
        event_scheduler=events,
        pheromone_system=PheromoneSystem(world),
        field_manager=FieldManager(world),
        blackboard=Blackboard(),
    )


def main():
    cli()


if __name__ == "__main__":
    main()
