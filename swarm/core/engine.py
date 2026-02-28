"""
Main simulation engine — orchestrates world, agents, hazards, fields, and pheromones.

The engine is the top-level coordinator that runs the simulation loop:
1. Advance clock
2. Process hazard/event schedule
3. Update spatial fields (periodically)
4. Step all agents
5. Update pheromone system
6. Collect metrics
7. Render (if visualization enabled)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.progress import Progress

from swarm.agents.swarm import Swarm, SwarmStats
from swarm.analytics.metrics import MetricsCollector
from swarm.analytics.recorder import TrajectoryRecorder
from swarm.core.clock import Clock
from swarm.core.events import EventScheduler
from swarm.core.world import World
from swarm.shared.blackboard import Blackboard
from swarm.shared.fields import FieldManager
from swarm.shared.pheromones import PheromoneSystem


@dataclass
class SimulationState:
    """Snapshot of simulation state at a given tick."""

    tick: int
    time: float
    swarm_stats: SwarmStats
    active_hazards: int
    events_fired: list[str]
    pheromone_totals: dict[str, float]


class SimulationEngine:
    """
    Top-level simulation orchestrator.

    Wires together all subsystems and runs the main tick loop.
    """

    def __init__(
        self,
        world: World,
        swarm: Swarm,
        clock: Clock,
        event_scheduler: EventScheduler | None = None,
        pheromone_system: PheromoneSystem | None = None,
        field_manager: FieldManager | None = None,
        blackboard: Blackboard | None = None,
    ):
        self.world = world
        self.swarm = swarm
        self.clock = clock
        self.events = event_scheduler or EventScheduler()
        self.pheromones = pheromone_system or PheromoneSystem(world)
        self.fields = field_manager or FieldManager(world)
        self.blackboard = blackboard or Blackboard()

        self.metrics = MetricsCollector()
        self.recorder = TrajectoryRecorder()

        # Callbacks
        self._tick_callbacks: list[Callable[[SimulationState], None]] = []
        self._done_callbacks: list[Callable[[SimulationState], None]] = []

    def on_tick(self, callback: Callable[[SimulationState], None]) -> None:
        """Register a callback to run after each tick."""
        self._tick_callbacks.append(callback)

    def on_done(self, callback: Callable[[SimulationState], None]) -> None:
        """Register a callback to run when simulation completes."""
        self._done_callbacks.append(callback)

    def initialize(self) -> None:
        """Initialize all subsystems before running."""
        # Compute initial fields
        self.fields.update(0, force=True)

        # Create hazard_prev layer for rate-of-change estimation in perception
        if not self.world.has_layer("hazard_prev"):
            self.world.add_layer("hazard_prev", default=0.0)

    def step(self) -> SimulationState:
        """Execute one simulation tick."""
        # 0. Snapshot hazard grid *before* this tick's hazard updates
        if self.world.has_layer("hazard_prev"):
            self.world.get_layer("hazard_prev").data[:] = self.world.hazard_grid

        # 1. Advance clock
        clock_events = self.clock.advance()
        tick = self.clock.tick

        # 2. Process hazards and events
        event_names = self.events.process_tick(self.world, tick)
        event_names.extend(clock_events)

        # 3. Update spatial fields (periodically)
        self.fields.update(tick)

        # 4. Step all agents
        self.swarm.step_all(self.world, tick)

        # 5. Update pheromone system
        self.pheromones.update()

        # 6. Collect metrics
        stats = self.swarm.get_stats()
        active_hazards = len(self.events.active_hazards(tick))

        state = SimulationState(
            tick=tick,
            time=self.clock.time,
            swarm_stats=stats,
            active_hazards=active_hazards,
            events_fired=event_names,
            pheromone_totals=self.pheromones.summary(),
        )

        self.metrics.record(state)

        # Record trajectories
        for agent in self.swarm.active_agents:
            self.recorder.record_position(tick, agent.id, agent.position)

        # Fire callbacks
        for cb in self._tick_callbacks:
            cb(state)

        return state

    def run(
        self,
        max_ticks: int | None = None,
        stop_when_evacuated: bool = True,
        show_progress: bool = True,
    ) -> list[SimulationState]:
        """
        Run the simulation until completion.

        Args:
            max_ticks: Override clock's max_ticks.
            stop_when_evacuated: Stop when all agents are evacuated/dead.
            show_progress: Show a progress bar.

        Returns:
            List of SimulationState snapshots for every tick.
        """
        if max_ticks is not None:
            self.clock.max_ticks = max_ticks

        self.initialize()
        history: list[SimulationState] = []

        console = Console()
        total = self.clock.max_ticks if self.clock.max_ticks > 0 else None

        if show_progress and total:
            with Progress(console=console) as progress:
                task = progress.add_task("Simulating...", total=total)
                while not self._should_stop(stop_when_evacuated):
                    state = self.step()
                    history.append(state)
                    progress.update(task, completed=self.clock.tick)
        else:
            while not self._should_stop(stop_when_evacuated):
                state = self.step()
                history.append(state)
                if self.clock.tick % 100 == 0:
                    stats = state.swarm_stats
                    console.print(
                        f"[dim]t={self.clock.tick:>5} | "
                        f"active={stats.active:>4} | "
                        f"evac={stats.evacuated:>4} | "
                        f"dead={stats.dead:>3} | "
                        f"panic={stats.panicking:>3}[/dim]"
                    )

        # Final state
        final = history[-1] if history else self.step()

        # Fire done callbacks
        for cb in self._done_callbacks:
            cb(final)

        return history

    def _should_stop(self, stop_when_evacuated: bool) -> bool:
        """Check if simulation should end."""
        if self.clock.is_done:
            return True

        if stop_when_evacuated:
            stats = self.swarm.get_stats()
            if stats.active == 0:
                return True

        return False

    def summary(self) -> dict:
        """Get a summary of the simulation results."""
        stats = self.swarm.get_stats()
        return {
            "ticks": self.clock.tick,
            "time_seconds": self.clock.time,
            "total_agents": stats.total,
            "evacuated": stats.evacuated,
            "dead": stats.dead,
            "stuck": stats.stuck,
            "evacuation_rate": stats.evacuated / max(stats.total, 1),
            "survival_rate": (stats.total - stats.dead) / max(stats.total, 1),
        }
