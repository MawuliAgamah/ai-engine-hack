from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SimulationConfig:
    """Core simulation parameters."""

    steps: int = 100
    num_giraffes: int = 20
    move_cost: float = 1.0
    green_energy_gain: float = 10.0
    seed: int | None = 42
    water_is_barrier: bool = True
    depleting_green: bool = False
    green_regrowth_per_tick: float = 0.0
    csv_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.csv_path is not None:
            data["csv_path"] = str(self.csv_path)
        return data


@dataclass
class RunConfig(SimulationConfig):
    """Batch run parameters for the headless CLI."""

    replicates: int = 1
    out: Path = Path("outputs")

    def to_dict(self) -> dict[str, Any]:
        data = SimulationConfig.to_dict(self)
        data["replicates"] = self.replicates
        data["out"] = str(self.out)
        return data
