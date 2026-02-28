"""
Simulation clock — manages tick progression, time scaling, and scheduled callbacks.

The clock is the heartbeat of the simulation. Each tick represents a discrete time step.
Events can be scheduled for specific ticks (hazard onset, scenario triggers, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ScheduledEvent:
    """An event scheduled to fire at a specific tick."""

    tick: int
    name: str
    callback: Callable[[], None]
    recurring_interval: int | None = None  # If set, reschedule after firing


class Clock:
    """
    Discrete-tick simulation clock with event scheduling.

    Attributes:
        tick: Current simulation tick (starts at 0).
        dt: Time represented per tick in seconds (for physics scaling).
        max_ticks: Maximum ticks before simulation ends (0 = unlimited).
    """

    def __init__(self, dt: float = 0.1, max_ticks: int = 0):
        self.tick: int = 0
        self.dt: float = dt
        self.max_ticks: int = max_ticks
        self._scheduled: list[ScheduledEvent] = []

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self.tick * self.dt

    @property
    def is_done(self) -> bool:
        """Whether the simulation has reached its time limit."""
        return self.max_ticks > 0 and self.tick >= self.max_ticks

    def schedule(
        self,
        tick: int,
        name: str,
        callback: Callable[[], None],
        recurring: int | None = None,
    ) -> None:
        """Schedule an event for a specific tick."""
        self._scheduled.append(
            ScheduledEvent(
                tick=tick, name=name, callback=callback, recurring_interval=recurring
            )
        )
        self._scheduled.sort(key=lambda e: e.tick)

    def advance(self) -> list[str]:
        """
        Advance clock by one tick and fire any scheduled events.

        Returns:
            List of event names that fired this tick.
        """
        self.tick += 1
        fired: list[str] = []
        reschedule: list[ScheduledEvent] = []

        remaining = []
        for event in self._scheduled:
            if event.tick <= self.tick:
                event.callback()
                fired.append(event.name)
                if event.recurring_interval is not None:
                    reschedule.append(
                        ScheduledEvent(
                            tick=self.tick + event.recurring_interval,
                            name=event.name,
                            callback=event.callback,
                            recurring_interval=event.recurring_interval,
                        )
                    )
            else:
                remaining.append(event)

        self._scheduled = remaining + reschedule
        self._scheduled.sort(key=lambda e: e.tick)
        return fired

    def reset(self) -> None:
        self.tick = 0
        self._scheduled.clear()

    def __repr__(self) -> str:
        return f"Clock(tick={self.tick}, time={self.time:.1f}s, pending={len(self._scheduled)})"
