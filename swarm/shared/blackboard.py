"""
Blackboard — global shared key-value store for system-level state.

The blackboard is a centralized communication channel for system-wide events
that all agents can read. Unlike pheromones (which are spatial), the blackboard
is non-spatial and global.

Examples:
- "fire_alarm": True/False — whether the fire alarm is active
- "blocked_exits": [list of blocked exit IDs]
- "evacuation_mode": True/False
- "crowd_density_warning": float — global congestion level
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlackboardEntry:
    """A single entry in the blackboard with metadata."""

    key: str
    value: Any
    set_at_tick: int
    expires_at_tick: int | None = None  # None = permanent

    def is_expired(self, current_tick: int) -> bool:
        if self.expires_at_tick is None:
            return False
        return current_tick > self.expires_at_tick


class Blackboard:
    """
    Global shared key-value store for system-level events.

    Agents and the simulation engine can read/write to the blackboard.
    Entries can have expiration times for transient events.
    """

    def __init__(self) -> None:
        self._entries: dict[str, BlackboardEntry] = {}
        self._history: list[tuple[int, str, Any]] = []  # (tick, key, value)

    def set(
        self,
        key: str,
        value: Any,
        tick: int,
        ttl: int | None = None,
    ) -> None:
        """
        Set a value on the blackboard.

        Args:
            key: Entry name.
            value: Entry value (any type).
            tick: Current simulation tick.
            ttl: Time-to-live in ticks (None = permanent).
        """
        expires = tick + ttl if ttl is not None else None
        self._entries[key] = BlackboardEntry(
            key=key, value=value, set_at_tick=tick, expires_at_tick=expires
        )
        self._history.append((tick, key, value))

    def get(self, key: str, default: Any = None, tick: int = 0) -> Any:
        """Read a value from the blackboard. Returns default if expired or missing."""
        entry = self._entries.get(key)
        if entry is None:
            return default
        if entry.is_expired(tick):
            return default
        return entry.value

    def has(self, key: str, tick: int = 0) -> bool:
        """Check if a key exists and is not expired."""
        entry = self._entries.get(key)
        if entry is None:
            return False
        return not entry.is_expired(tick)

    def remove(self, key: str) -> None:
        self._entries.pop(key, None)

    def cleanup(self, tick: int) -> None:
        """Remove all expired entries."""
        expired = [k for k, v in self._entries.items() if v.is_expired(tick)]
        for k in expired:
            del self._entries[k]

    def all_entries(self, tick: int = 0) -> dict[str, Any]:
        """Get all non-expired entries."""
        return {
            k: v.value
            for k, v in self._entries.items()
            if not v.is_expired(tick)
        }

    @property
    def history(self) -> list[tuple[int, str, Any]]:
        return self._history

    def __repr__(self) -> str:
        return f"Blackboard({len(self._entries)} entries)"
