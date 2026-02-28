from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .agents import GiraffeAgent
from .data_loader import NCOLS, NROWS, LoadedPatch, default_csv_path, load_patch_data_from_csv
from .environment import PatchCell, build_patch_cells, build_patch_index, patch_is_green
from .mesa_compat import Model, MultiGrid
from .metrics import compute_static_environment_metrics


class SouthwarkModel(Model):
    """Mesa model equivalent of the Southwark NetLogo environment."""

    def __init__(
        self,
        num_giraffes: int = 20,
        move_cost: float = 1.0,
        green_energy_gain: float = 10.0,
        seed: int | None = 42,
        water_is_barrier: bool = True,
        csv_path: str | Path | None = None,
        depleting_green: bool = False,
        green_regrowth_per_tick: float = 0.0,
    ) -> None:
        self._init_model_base(seed=seed)

        self.width = NCOLS
        self.height = NROWS
        self.grid = MultiGrid(self.width, self.height, torus=False)

        self.num_giraffes = int(num_giraffes)
        self.move_cost = float(move_cost)
        self.green_energy_gain = float(green_energy_gain)
        self.water_is_barrier = bool(water_is_barrier)
        self.csv_path = Path(csv_path) if csv_path is not None else default_csv_path()
        self.depleting_green = bool(depleting_green)
        self.green_regrowth_per_tick = max(0.0, float(green_regrowth_per_tick))

        self.tick = 0
        self._next_agent_id = 0
        self._giraffes: dict[int, GiraffeAgent] = {}

        self.loaded_patches, self.bounds = self._load_patches()
        self.patch_cells = build_patch_cells(self.loaded_patches)
        self.patch_index = build_patch_index(self.patch_cells)
        self.traversable_positions = [
            (cell.x, cell.y) for cell in self.patch_cells if cell.traversable
        ]

        self.static_metrics = compute_static_environment_metrics(self.patch_cells)

        self.patch_green_capacity: dict[tuple[int, int], float] = {}
        self.patch_green_resource: dict[tuple[int, int], float] = {}
        self.patch_green_consumed: dict[tuple[int, int], float] = {}
        self.total_green_resource_initial = 0.0
        self.total_green_resource_remaining = 0.0
        self.total_green_consumed = 0.0
        self._initialize_green_resources()

        self.agent_routes: dict[int, list[dict[str, Any]]] = {}
        self.agent_outcomes: dict[int, dict[str, Any]] = {}

        self._spawn_giraffes(self.num_giraffes)
        self.running = bool(self._giraffes)

    def _init_model_base(self, seed: int | None) -> None:
        # Mesa model constructor signature can vary by version.
        try:
            super().__init__(seed=seed)
        except TypeError:
            super().__init__()

        if not hasattr(self, "random"):
            import random

            self.random = random.Random(seed)
        elif seed is not None:
            self.random.seed(seed)

    def _load_patches(self) -> tuple[list[LoadedPatch], object]:
        return load_patch_data_from_csv(
            csv_path=self.csv_path,
            water_is_barrier=self.water_is_barrier,
            ncols=self.width,
            nrows=self.height,
        )

    @property
    def giraffes(self) -> list[GiraffeAgent]:
        return list(self._giraffes.values())

    def _new_agent_id(self) -> int:
        identifier = self._next_agent_id
        self._next_agent_id += 1
        return identifier

    def _patch_info(self, pos: tuple[int, int] | None) -> dict[str, Any]:
        if pos is None:
            return {
                "x": None,
                "y": None,
                "hex_id": "",
                "location_id": "",
                "zone_type": "",
            }

        patch = self.get_patch(pos)
        if patch is None:
            return {
                "x": int(pos[0]),
                "y": int(pos[1]),
                "hex_id": "",
                "location_id": "",
                "zone_type": "",
            }

        return {
            "x": int(pos[0]),
            "y": int(pos[1]),
            "hex_id": str(patch.attrs.get("hex-id", "")),
            "location_id": str(patch.attrs.get("location-id", "")),
            "zone_type": str(patch.attrs.get("zone-type", "")),
        }

    def _initial_green_capacity(self, cell: PatchCell) -> float:
        attrs = cell.attrs
        if not patch_is_green(attrs):
            return 0.0

        # Weighted by natural/landuse signals so woodland-like patches carry more resource.
        capacity = 20.0
        if bool(attrs.get("landuse-grass?", False)):
            capacity += 15.0
        if bool(attrs.get("landuse-meadow?", False)):
            capacity += 15.0
        if bool(attrs.get("landuse-allotments?", False)):
            capacity += 12.0
        if bool(attrs.get("landuse-recreation-ground?", False)):
            capacity += 12.0
        if bool(attrs.get("leisure-park?", False)):
            capacity += 14.0
        if bool(attrs.get("leisure-garden?", False)):
            capacity += 10.0
        if bool(attrs.get("natural-tree-row?", False)):
            capacity += 15.0
        if bool(attrs.get("natural-scrub?", False)):
            capacity += 18.0
        if bool(attrs.get("natural-wood?", False)):
            capacity += 35.0

        return min(100.0, capacity)

    def _initialize_green_resources(self) -> None:
        self.patch_green_capacity = {}
        self.patch_green_resource = {}
        self.patch_green_consumed = {}

        for cell in self.patch_cells:
            pos = (cell.x, cell.y)
            capacity = self._initial_green_capacity(cell)
            self.patch_green_capacity[pos] = capacity
            self.patch_green_resource[pos] = capacity
            self.patch_green_consumed[pos] = 0.0

        self.total_green_resource_initial = float(sum(self.patch_green_capacity.values()))
        self.total_green_resource_remaining = float(sum(self.patch_green_resource.values()))
        self.total_green_consumed = 0.0

    def _init_agent_tracking(self, agent: GiraffeAgent) -> None:
        info = self._patch_info(agent.pos)
        self.agent_routes[agent.unique_id] = []
        self.agent_outcomes[agent.unique_id] = {
            "agent_id": int(agent.unique_id),
            "status": "alive",
            "death_reason": "",
            "death_tick": None,
            "spawn_tick": int(self.tick),
            "spawn_x": info["x"],
            "spawn_y": info["y"],
            "spawn_hex_id": info["hex_id"],
            "spawn_location_id": info["location_id"],
            "spawn_zone_type": info["zone_type"],
            "final_tick": int(self.tick),
            "final_x": info["x"],
            "final_y": info["y"],
            "final_hex_id": info["hex_id"],
            "final_location_id": info["location_id"],
            "final_zone_type": info["zone_type"],
            "distance_travelled": 0,
            "energy": float(agent.energy),
            "route_records": 0,
            "unique_hex_count": 1 if info["hex_id"] else 0,
            "total_green_consumed": 0.0,
        }
        self.record_agent_step(agent, moved=False, green_consumed=0.0, action="spawn", tick_value=self.tick)

    def _refresh_agent_outcome(self, agent: GiraffeAgent, tick_value: int) -> None:
        info = self._patch_info(agent.pos)
        outcome = self.agent_outcomes.setdefault(
            agent.unique_id,
            {
                "agent_id": int(agent.unique_id),
                "status": "alive",
                "death_reason": "",
                "death_tick": None,
                "spawn_tick": int(tick_value),
            },
        )

        outcome["final_tick"] = int(tick_value)
        outcome["final_x"] = info["x"]
        outcome["final_y"] = info["y"]
        outcome["final_hex_id"] = info["hex_id"]
        outcome["final_location_id"] = info["location_id"]
        outcome["final_zone_type"] = info["zone_type"]
        outcome["distance_travelled"] = int(agent.distance_travelled)
        outcome["energy"] = float(agent.energy)
        outcome["total_green_consumed"] = float(getattr(agent, "total_green_consumed", 0.0))

    def _spawn_giraffes(self, count: int) -> None:
        if not self.traversable_positions:
            return

        for _ in range(max(0, count)):
            spawn_pos = self.random.choice(self.traversable_positions)
            agent = GiraffeAgent(unique_id=self._new_agent_id(), model=self)
            self.grid.place_agent(agent, spawn_pos)
            agent.start_patch_x = spawn_pos[0]
            agent.start_patch_y = spawn_pos[1]
            self._giraffes[agent.unique_id] = agent
            self._init_agent_tracking(agent)

    def get_patch(self, pos: tuple[int, int]) -> PatchCell | None:
        return self.patch_index.get(pos)

    def is_traversable(self, pos: tuple[int, int]) -> bool:
        patch = self.get_patch(pos)
        if patch is None:
            return False
        return patch.traversable

    def is_patch_green(self, pos: tuple[int, int] | None) -> bool:
        if pos is None:
            return False
        patch = self.get_patch(pos)
        if patch is None:
            return False

        if not patch_is_green(patch.attrs):
            return False

        if not self.depleting_green:
            return True

        return self.get_patch_green_resource(pos) > 0

    def get_patch_green_resource(self, pos: tuple[int, int] | None) -> float:
        if pos is None:
            return 0.0
        return float(self.patch_green_resource.get(pos, 0.0))

    def get_patch_green_fraction(self, pos: tuple[int, int] | None) -> float:
        if pos is None:
            return 0.0
        remaining = float(self.patch_green_resource.get(pos, 0.0))
        capacity = float(self.patch_green_capacity.get(pos, 0.0))
        if capacity <= 0:
            return 0.0
        return max(0.0, min(1.0, remaining / capacity))

    def consume_green_resource(self, pos: tuple[int, int] | None, desired_amount: float) -> float:
        if desired_amount <= 0:
            return 0.0
        if pos is None:
            return 0.0

        patch = self.get_patch(pos)
        if patch is None or not patch_is_green(patch.attrs):
            return 0.0

        if not self.depleting_green:
            return float(desired_amount)

        available = float(self.patch_green_resource.get(pos, 0.0))
        if available <= 0:
            return 0.0

        consumed = min(float(desired_amount), available)
        self.patch_green_resource[pos] = available - consumed
        self.patch_green_consumed[pos] = float(self.patch_green_consumed.get(pos, 0.0)) + consumed

        self.total_green_consumed += consumed
        self.total_green_resource_remaining = max(0.0, self.total_green_resource_remaining - consumed)
        return consumed

    def _apply_green_regrowth(self) -> None:
        if not self.depleting_green or self.green_regrowth_per_tick <= 0:
            return

        for pos, capacity in self.patch_green_capacity.items():
            if capacity <= 0:
                continue
            current = float(self.patch_green_resource.get(pos, 0.0))
            if current >= capacity:
                continue
            updated = min(capacity, current + self.green_regrowth_per_tick)
            self.patch_green_resource[pos] = updated

        self.total_green_resource_remaining = float(sum(self.patch_green_resource.values()))
        self.total_green_consumed = max(0.0, self.total_green_resource_initial - self.total_green_resource_remaining)

    def get_traversable_neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors = self.grid.get_neighborhood(pos, moore=True, include_center=False, radius=1)
        return [neighbor for neighbor in neighbors if self.is_traversable(neighbor)]

    def record_agent_step(
        self,
        agent: GiraffeAgent,
        moved: bool,
        green_consumed: float,
        action: str,
        tick_value: int | None = None,
    ) -> None:
        tick_record = int(self.tick if tick_value is None else tick_value)
        info = self._patch_info(agent.pos)

        route_row = {
            "agent_id": int(agent.unique_id),
            "tick": tick_record,
            "action": action,
            "moved": bool(moved),
            "x": info["x"],
            "y": info["y"],
            "hex_id": info["hex_id"],
            "location_id": info["location_id"],
            "zone_type": info["zone_type"],
            "energy": float(agent.energy),
            "distance_travelled": int(agent.distance_travelled),
            "green_consumed_tick": float(green_consumed),
            "patch_green_remaining": self.get_patch_green_resource(agent.pos),
            "patch_green_fraction": self.get_patch_green_fraction(agent.pos),
        }
        self.agent_routes.setdefault(agent.unique_id, []).append(route_row)

        self._refresh_agent_outcome(agent, tick_value=tick_record)

        outcome = self.agent_outcomes[agent.unique_id]
        outcome["route_records"] = len(self.agent_routes.get(agent.unique_id, []))

        # Keep unique-hex count up to date for easy analysis/export.
        route_hexes = {
            row["hex_id"] for row in self.agent_routes.get(agent.unique_id, []) if row.get("hex_id")
        }
        outcome["unique_hex_count"] = len(route_hexes)

    def remove_giraffe(
        self,
        agent: GiraffeAgent,
        reason: str = "removed",
        tick: int | None = None,
    ) -> None:
        if agent.unique_id not in self._giraffes:
            return

        tick_value = int(self.tick if tick is None else tick)
        self._refresh_agent_outcome(agent, tick_value=tick_value)
        outcome = self.agent_outcomes.setdefault(agent.unique_id, {"agent_id": int(agent.unique_id)})
        outcome["status"] = "dead"
        outcome["death_reason"] = str(reason)
        outcome["death_tick"] = tick_value

        if agent.pos is not None:
            try:
                self.grid.remove_agent(agent)
            except Exception:
                pass

        remove_method = getattr(agent, "remove", None)
        if callable(remove_method):
            try:
                remove_method()
            except Exception:
                pass

        self._giraffes.pop(agent.unique_id, None)

    def iter_giraffes(self) -> Iterable[GiraffeAgent]:
        return self._giraffes.values()

    def finalize_outcomes(self, end_reason: str = "simulation_end") -> None:
        for agent in self._giraffes.values():
            outcome = self.agent_outcomes.setdefault(agent.unique_id, {"agent_id": int(agent.unique_id)})
            outcome["status"] = "alive"
            outcome["death_reason"] = ""
            outcome["death_tick"] = None
            outcome["end_reason"] = end_reason
            self._refresh_agent_outcome(agent, tick_value=self.tick)

    def green_land_summary(self) -> dict[str, float]:
        initial = float(self.total_green_resource_initial)
        remaining = float(self.total_green_resource_remaining)
        consumed = float(self.total_green_consumed)
        consumed_pct = (consumed / initial) if initial > 0 else 0.0
        remaining_pct = (remaining / initial) if initial > 0 else 0.0

        depleted_patches = sum(
            1
            for pos, capacity in self.patch_green_capacity.items()
            if capacity > 0 and self.patch_green_resource.get(pos, 0.0) <= 0
        )
        active_green_patches = sum(
            1
            for pos, capacity in self.patch_green_capacity.items()
            if capacity > 0 and self.patch_green_resource.get(pos, 0.0) > 0
        )

        return {
            "green_resource_initial": initial,
            "green_resource_remaining": remaining,
            "green_resource_consumed": consumed,
            "green_resource_consumed_pct": consumed_pct,
            "green_resource_remaining_pct": remaining_pct,
            "green_patches_depleted": float(depleted_patches),
            "green_patches_with_resource": float(active_green_patches),
        }

    def giraffe_population_summary(self) -> dict[str, Any]:
        self.finalize_outcomes(end_reason="state_snapshot")

        alive_agents = self.giraffes
        alive_ids = {agent.unique_id for agent in alive_agents}
        all_outcomes = list(self.agent_outcomes.values())

        total_spawned = len(all_outcomes)
        alive_count = len(alive_ids)
        dead_count = total_spawned - alive_count

        alive_energies = [float(agent.energy) for agent in alive_agents]
        alive_distances = [int(agent.distance_travelled) for agent in alive_agents]

        total_distance_all = float(sum(float(outcome.get("distance_travelled", 0.0)) for outcome in all_outcomes))
        total_green_consumed_by_giraffes = float(
            sum(float(outcome.get("total_green_consumed", 0.0)) for outcome in all_outcomes)
        )

        return {
            "total_spawned": total_spawned,
            "alive": alive_count,
            "dead": dead_count,
            "mean_energy_alive": mean(alive_energies) if alive_energies else 0.0,
            "mean_distance_alive": mean(alive_distances) if alive_distances else 0.0,
            "total_distance_all": total_distance_all,
            "total_green_consumed_by_giraffes": total_green_consumed_by_giraffes,
        }

    def get_giraffe_routes_wide(self) -> list[dict[str, Any]]:
        self.finalize_outcomes(end_reason="state_snapshot")

        rows: list[dict[str, Any]] = []
        for agent_id in sorted(self.agent_routes):
            outcome = self.agent_outcomes.get(agent_id, {})
            start_location_id = str(outcome.get("spawn_location_id", ""))
            final_location_id = str(outcome.get("final_location_id", ""))
            start_hex_id = str(outcome.get("spawn_hex_id", ""))
            final_hex_id = str(outcome.get("final_hex_id", ""))

            for row in self.agent_routes[agent_id]:
                enriched = dict(row)
                enriched["start_location_id"] = start_location_id
                enriched["current_location_id"] = str(row.get("location_id", ""))
                enriched["final_location_id"] = final_location_id
                enriched["start_hex_id"] = start_hex_id
                enriched["current_hex_id"] = str(row.get("hex_id", ""))
                enriched["final_hex_id"] = final_hex_id
                rows.append(enriched)

        return rows

    def get_giraffe_routes_long(self) -> list[dict[str, Any]]:
        """Return long-format route rows (skinny schema).

        One base route event expands to 3 rows with `location_role` in:
        `start`, `current`, `final`.
        """
        self.finalize_outcomes(end_reason="state_snapshot")

        wide_rows = self.get_giraffe_routes_wide()
        long_rows: list[dict[str, Any]] = []

        for row in wide_rows:
            agent_id = int(row["agent_id"])
            outcome = self.agent_outcomes.get(agent_id, {})

            role_payloads = [
                {
                    "location_role": "start",
                    "location_id": str(row.get("start_location_id", "")),
                    "hex_id": str(row.get("start_hex_id", "")),
                    "x": outcome.get("spawn_x"),
                    "y": outcome.get("spawn_y"),
                    "zone_type": outcome.get("spawn_zone_type", ""),
                },
                {
                    "location_role": "current",
                    "location_id": str(row.get("current_location_id", "")),
                    "hex_id": str(row.get("current_hex_id", "")),
                    "x": row.get("x"),
                    "y": row.get("y"),
                    "zone_type": row.get("zone_type", ""),
                },
                {
                    "location_role": "final",
                    "location_id": str(row.get("final_location_id", "")),
                    "hex_id": str(row.get("final_hex_id", "")),
                    "x": outcome.get("final_x"),
                    "y": outcome.get("final_y"),
                    "zone_type": outcome.get("final_zone_type", ""),
                },
            ]

            for payload in role_payloads:
                long_rows.append(
                    {
                        "agent_id": row.get("agent_id"),
                        "tick": row.get("tick"),
                        "location_role": payload["location_role"],
                        "location_id": payload["location_id"],
                        "hex_id": payload["hex_id"],
                        "x": payload["x"],
                        "y": payload["y"],
                        "zone_type": payload["zone_type"],
                        "action": row.get("action"),
                        "moved": row.get("moved"),
                        "energy": row.get("energy"),
                        "distance_travelled": row.get("distance_travelled"),
                        "green_consumed_tick": row.get("green_consumed_tick"),
                        "patch_green_remaining": row.get("patch_green_remaining"),
                        "patch_green_fraction": row.get("patch_green_fraction"),
                    }
                )

        return long_rows

    def get_giraffe_routes(self) -> list[dict[str, Any]]:
        """Default route export (wide schema, aligned with latest_run)."""
        return self.get_giraffe_routes_wide()

    def get_giraffe_outcomes(self) -> list[dict[str, Any]]:
        self.finalize_outcomes(end_reason="state_snapshot")
        rows: list[dict[str, Any]] = []
        for agent_id in sorted(self.agent_outcomes):
            rows.append(dict(self.agent_outcomes[agent_id]))
        return rows

    def get_giraffe_tick_matrix(self, max_tick: int | None = None) -> list[dict[str, Any]]:
        """Return a full per-agent/per-tick matrix up to current model tick.

        Guarantees one row per agent per tick (0..horizon_tick), carrying
        start/current/final location and hex ids on every row.
        """
        self.finalize_outcomes(end_reason="state_snapshot")

        routes = self.get_giraffe_routes_wide()
        route_by_agent_tick: dict[tuple[int, int], dict[str, Any]] = {}
        latest_route_by_agent: dict[int, dict[str, Any]] = {}

        for route in routes:
            agent_id = int(route["agent_id"])
            tick_value = int(route["tick"])
            route_by_agent_tick[(agent_id, tick_value)] = route

            previous = latest_route_by_agent.get(agent_id)
            if previous is None or int(previous["tick"]) < tick_value:
                latest_route_by_agent[agent_id] = route

        matrix_rows: list[dict[str, Any]] = []
        simulated_tick = int(self.tick)
        horizon_tick = simulated_tick if max_tick is None else max(int(max_tick), simulated_tick)

        for agent_id in sorted(self.agent_outcomes):
            outcome = self.agent_outcomes[agent_id]
            death_tick_raw = outcome.get("death_tick")
            death_tick = int(death_tick_raw) if death_tick_raw not in (None, "") else None
            last_route = latest_route_by_agent.get(agent_id)

            if last_route is None:
                continue

            for tick_value in range(0, horizon_tick + 1):
                route_row = route_by_agent_tick.get((agent_id, tick_value))

                if route_row is not None:
                    row = dict(route_row)
                    if tick_value > simulated_tick:
                        row["tick_status"] = "not_simulated"
                    else:
                        row["tick_status"] = "active" if (death_tick is None or tick_value <= death_tick) else "dead"
                else:
                    row = dict(last_route)
                    row["tick"] = tick_value
                    row["action"] = "not-simulated" if tick_value > simulated_tick else "no-step"
                    row["moved"] = False
                    row["green_consumed_tick"] = 0.0
                    if tick_value > simulated_tick:
                        row["tick_status"] = "not_simulated"
                    elif death_tick is not None and tick_value > death_tick:
                        row["tick_status"] = "dead"
                        row["energy"] = 0.0
                    else:
                        row["tick_status"] = "alive_carry"

                row["status"] = outcome.get("status", "")
                row["death_reason"] = outcome.get("death_reason", "")
                row["death_tick"] = outcome.get("death_tick")
                row["spawn_tick"] = outcome.get("spawn_tick")
                row["final_tick"] = outcome.get("final_tick")
                row["simulated_until_tick"] = simulated_tick
                row["horizon_tick"] = horizon_tick
                matrix_rows.append(row)

        return matrix_rows

    def step(self) -> bool:
        if not self._giraffes:
            self.running = False
            return False

        agent_order = list(self._giraffes.values())
        self.random.shuffle(agent_order)

        for agent in agent_order:
            if agent.unique_id not in self._giraffes:
                continue
            agent.step()

        self._apply_green_regrowth()

        self.tick += 1
        self.running = bool(self._giraffes)

        if not self.running:
            self.finalize_outcomes(end_reason="all_dead")

        return self.running
