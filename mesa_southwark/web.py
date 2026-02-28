from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from .config import SimulationConfig
from .environment import patch_is_green
from .metrics import collect_tick_metrics
from .model import SouthwarkModel

HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Southwark Mesa Viewer</title>
  <style>
    :root {
      --panel: #ffffff;
      --border: #d2dae8;
      --ink: #1d2430;
      --muted: #637083;
      --accent: #1e6fd9;
    }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      background: linear-gradient(140deg, #edf2f9 0%, #f6fbf2 100%);
      color: var(--ink);
    }
    .layout {
      display: grid;
      gap: 16px;
      padding: 16px;
      grid-template-columns: minmax(360px, 2fr) minmax(340px, 1fr);
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 6px 16px rgba(17, 24, 39, 0.06);
    }
    h1 {
      margin: 0 0 10px 0;
      font-size: 18px;
    }
    .controls {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      margin-bottom: 8px;
    }
    button {
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--ink);
      border-radius: 8px;
      padding: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
    }
    button:hover { border-color: var(--accent); }
    #world {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #f3f4f6;
      image-rendering: pixelated;
    }
    .small {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .stats {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-bottom: 10px;
      font-size: 14px;
    }
    .stat {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
    }
    .k { color: var(--muted); display: block; font-size: 12px; }
    .v { font-weight: 700; font-size: 16px; }
    .inputs {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-bottom: 10px;
    }
    label {
      display: grid;
      gap: 4px;
      font-size: 12px;
      color: var(--muted);
    }
    input {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px;
      font-size: 13px;
    }
    .legend {
      display: grid;
      gap: 6px;
      margin-top: 10px;
      font-size: 12px;
    }
    .row { display: flex; align-items: center; gap: 8px; }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid #adb8c8;
      display: inline-block;
      flex-shrink: 0;
    }
    pre {
      margin: 8px 0 0 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #f7f9fc;
      padding: 8px;
      font-size: 12px;
      line-height: 1.35;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
    }
    #export_status { min-height: 20px; color: var(--muted); font-size: 12px; margin-top: 6px; }
    @media (max-width: 1020px) {
      .layout { grid-template-columns: 1fr; }
      .controls { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class=\"layout\">
    <section class=\"card\">
      <h1>Southwark Live Simulation</h1>
      <div class=\"controls\">
        <button id=\"start\">Start</button>
        <button id=\"pause\">Pause</button>
        <button id=\"step\">Step</button>
        <button id=\"reset\">Reset</button>
        <button id=\"export\">Export</button>
      </div>
      <canvas id=\"world\"></canvas>
      <div class=\"small\">Green patches fade as resource is consumed. Water remains non-traversable.</div>
      <div id=\"export_status\"></div>
    </section>

    <aside class=\"card\">
      <div class=\"stats\">
        <div class=\"stat\"><span class=\"k\">Tick</span><span class=\"v\" id=\"tick\">0</span></div>
        <div class=\"stat\"><span class=\"k\">Running</span><span class=\"v\" id=\"running\">No</span></div>
        <div class=\"stat\"><span class=\"k\">Alive</span><span class=\"v\" id=\"alive\">0</span></div>
        <div class=\"stat\"><span class=\"k\">Dead</span><span class=\"v\" id=\"dead\">0</span></div>
        <div class=\"stat\"><span class=\"k\">Mean Energy</span><span class=\"v\" id=\"energy\">0</span></div>
        <div class=\"stat\"><span class=\"k\">Total Distance</span><span class=\"v\" id=\"distance_total\">0</span></div>
        <div class=\"stat\"><span class=\"k\">Green Consumed</span><span class=\"v\" id=\"green_consumed\">0</span></div>
        <div class=\"stat\"><span class=\"k\">Green Remaining %</span><span class=\"v\" id=\"green_remaining_pct\">0%</span></div>
      </div>

      <div class=\"inputs\">
        <label>Num Giraffes <input type=\"number\" id=\"num_giraffes\" min=\"0\" step=\"1\" value=\"20\"></label>
        <label>Max Steps <input type=\"number\" id=\"steps\" min=\"1\" step=\"1\" value=\"120\"></label>
        <label>Move Cost <input type=\"number\" id=\"move_cost\" min=\"0\" step=\"0.1\" value=\"1.0\"></label>
        <label>Green Energy Gain <input type=\"number\" id=\"green_energy_gain\" min=\"0\" step=\"0.1\" value=\"10.0\"></label>
        <label>Seed <input type=\"number\" id=\"seed\" step=\"1\" value=\"42\"></label>
        <label>Interval (ms) <input type=\"number\" id=\"interval_ms\" min=\"40\" step=\"10\" value=\"250\"></label>
      </div>

      <div class=\"legend\" id=\"legend\"></div>
      <pre id=\"giraffe_summary\">Giraffe outcomes will appear here.</pre>
    </aside>
  </div>

  <script>
    const zoneColors = {
      'water': '#3f82e0',
      'green-space': '#2f8a3b',
      'park-urban': '#66b86b',
      'industrial': '#7a7a7a',
      'super-urban': '#5b5b5b',
      'commercial': '#f29e3d',
      'residential': '#f4d35e',
      'mixed': '#a57548',
      'outside': '#d7dce5'
    };

    const staticData = { width: 0, height: 0, patches: [] };
    let currentState = null;

    const canvas = document.getElementById('world');
    const ctx = canvas.getContext('2d');

    function hexToRgb(hex) {
      const h = hex.replace('#', '');
      return {
        r: parseInt(h.substring(0, 2), 16),
        g: parseInt(h.substring(2, 4), 16),
        b: parseInt(h.substring(4, 6), 16)
      };
    }

    function rgbToHex(r, g, b) {
      const f = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
      return `#${f(r)}${f(g)}${f(b)}`;
    }

    function coalesce(value, fallback) {
      return (value === null || value === undefined) ? fallback : value;
    }

    function mix(base, other, t) {
      const a = hexToRgb(base);
      const b = hexToRgb(other);
      return rgbToHex(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t);
    }

    function energyColor(energy) {
      const clamped = Math.max(0, Math.min(100, energy));
      const t = clamped / 100;
      const r = Math.round(220 * (1 - t) + 34 * t);
      const g = Math.round(48 * (1 - t) + 170 * t);
      const b = Math.round(48 * (1 - t) + 84 * t);
      return `rgb(${r}, ${g}, ${b})`;
    }

    function drawLegend() {
      const legend = document.getElementById('legend');
      const rows = [];
      for (const [zone, color] of Object.entries(zoneColors)) {
        rows.push(`<div class=\"row\"><span class=\"swatch\" style=\"background:${color}\"></span>${zone}</div>`);
      }
      rows.push('<div class=\"row\"><span class=\"swatch\" style=\"background:linear-gradient(90deg,#e8e4d5,#2f8a3b)\"></span>green resource left</div>');
      rows.push('<div class=\"row\"><span class=\"swatch\" style=\"background:linear-gradient(90deg,#dc3030,#22aa54)\"></span>giraffe energy</div>');
      legend.innerHTML = rows.join('');
    }

    function patchColor(patch) {
      if (!patch.display_occupied) return zoneColors.outside;
      let color = zoneColors[patch.zone_type] || zoneColors.mixed;

      if (patch.is_green && currentState && currentState.patch_green_fraction) {
        const key = `${patch.x},${patch.y}`;
        const fraction = Number(coalesce(currentState.patch_green_fraction[key], 1.0));
        const depletion = 1 - Math.max(0, Math.min(1, fraction));
        color = mix(color, '#e8e4d5', depletion * 0.82);
      }
      return color;
    }

    function drawWorld() {
      if (!staticData.width || !staticData.height) return;
      const cell = Math.max(10, Math.floor((window.innerHeight - 180) / staticData.height));
      canvas.width = staticData.width * cell;
      canvas.height = staticData.height * cell;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const patch of staticData.patches) {
        const drawY = (staticData.height - 1 - patch.y) * cell;
        ctx.fillStyle = patchColor(patch);
        ctx.fillRect(patch.x * cell, drawY, cell, cell);
      }

      if (currentState && currentState.agents) {
        for (const agent of currentState.agents) {
          const cx = agent.x * cell + cell / 2;
          const cy = (staticData.height - 1 - agent.y) * cell + cell / 2;
          ctx.beginPath();
          ctx.arc(cx, cy, Math.max(2, Math.floor(cell * 0.28)), 0, Math.PI * 2);
          ctx.fillStyle = energyColor(agent.energy);
          ctx.fill();
          ctx.strokeStyle = '#121721';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      ctx.strokeStyle = 'rgba(30, 35, 45, 0.08)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= staticData.width; x++) {
        ctx.beginPath();
        ctx.moveTo(x * cell, 0);
        ctx.lineTo(x * cell, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y <= staticData.height; y++) {
        ctx.beginPath();
        ctx.moveTo(0, y * cell);
        ctx.lineTo(canvas.width, y * cell);
        ctx.stroke();
      }
    }

    function renderGiraffeSummary(outcomes) {
      if (!outcomes || outcomes.length === 0) {
        document.getElementById('giraffe_summary').textContent = 'No giraffes yet.';
        return;
      }

      const sorted = [...outcomes].sort((a, b) => Number(a.agent_id) - Number(b.agent_id));
      const lines = ['agent_id | status | death_reason | distance | green_consumed | final_location_id'];
      for (const o of sorted) {
        lines.push(
          `${o.agent_id} | ${o.status || ''} | ${o.death_reason || '-'} | ${Number(o.distance_travelled || 0).toFixed(0)} | ${Number(o.total_green_consumed || 0).toFixed(2)} | ${o.final_location_id || ''}`
        );
      }
      document.getElementById('giraffe_summary').textContent = lines.join('\\n');
    }

    function updateStats() {
      if (!currentState) return;
      const pop = currentState.giraffe_population_summary || {};
      const green = currentState.green_land_summary || {};

      document.getElementById('tick').textContent = currentState.tick;
      document.getElementById('running').textContent = currentState.running ? 'Yes' : 'No';
      document.getElementById('alive').textContent = coalesce(pop.alive, currentState.metrics.giraffes_alive);
      document.getElementById('dead').textContent = coalesce(pop.dead, coalesce(currentState.metrics.giraffes_dead, 0));
      document.getElementById('energy').textContent = Number(currentState.metrics.mean_energy).toFixed(2);
      document.getElementById('distance_total').textContent = Number(coalesce(pop.total_distance_all, 0)).toFixed(1);
      document.getElementById('green_consumed').textContent = Number(coalesce(green.green_resource_consumed, 0)).toFixed(2);
      document.getElementById('green_remaining_pct').textContent = `${(Number(coalesce(green.green_resource_remaining_pct, 0)) * 100).toFixed(1)}%`;

      renderGiraffeSummary(currentState.giraffe_outcomes || []);
    }

    async function postAction(endpoint, body = null) {
      const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(endpoint, opts);
      if (!res.ok) {
        const msg = await res.text();
        alert(msg);
        throw new Error(msg);
      }
      return await res.json();
    }

    async function fetchStatic() {
      const res = await fetch('/api/static');
      const data = await res.json();
      staticData.width = data.width;
      staticData.height = data.height;
      staticData.patches = data.patches;
      drawWorld();
    }

    async function pollState() {
      const res = await fetch('/api/state');
      currentState = await res.json();
      updateStats();
      drawWorld();
      setTimeout(pollState, 220);
    }

    function readConfigInputs() {
      return {
        num_giraffes: Number(document.getElementById('num_giraffes').value),
        steps: Number(document.getElementById('steps').value),
        move_cost: Number(document.getElementById('move_cost').value),
        green_energy_gain: Number(document.getElementById('green_energy_gain').value),
        seed: Number(document.getElementById('seed').value),
        interval_ms: Number(document.getElementById('interval_ms').value),
        depleting_green: true,
        green_regrowth_per_tick: 0,
      };
    }

    document.getElementById('start').onclick = () => postAction('/api/start', readConfigInputs());
    document.getElementById('pause').onclick = () => postAction('/api/pause');
    document.getElementById('step').onclick = () => postAction('/api/step');

    document.getElementById('reset').onclick = async () => {
      const payload = readConfigInputs();
      await postAction('/api/reset', payload);
      document.getElementById('export_status').textContent = 'Simulation reset.';
      await fetchStatic();
    };

    document.getElementById('export').onclick = async () => {
      document.getElementById('export_status').textContent = 'Running simulation to completion and exporting...';
      const data = await postAction('/api/run-and-export', readConfigInputs());
      let msg = `Exported: ${data.paths.summary_json}`;
      if (data.warnings && data.warnings.length > 0) {
        msg += ' | Warnings: ' + data.warnings.join('; ');
      }
      document.getElementById('export_status').textContent = msg;
    };

    window.addEventListener('resize', drawWorld);
    drawLegend();
    fetchStatic().then(pollState);
  </script>
</body>
</html>
"""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _neighbors4(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append((nx, ny))
    return neighbors


def _neighbors8(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
    return neighbors


def _infer_display_holes(
    occupied_zone_by_pos: dict[tuple[int, int], str],
    width: int,
    height: int,
) -> dict[tuple[int, int], str]:
    all_positions = {(x, y) for x in range(width) for y in range(height)}
    occupied_positions = set(occupied_zone_by_pos)
    outside_positions = all_positions - occupied_positions

    exterior: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        for y in (0, height - 1):
            pos = (x, y)
            if pos in outside_positions and pos not in exterior:
                exterior.add(pos)
                queue.append(pos)
    for y in range(height):
        for x in (0, width - 1):
            pos = (x, y)
            if pos in outside_positions and pos not in exterior:
                exterior.add(pos)
                queue.append(pos)

    while queue:
        x, y = queue.popleft()
        for neighbor in _neighbors4(x, y, width, height):
            if neighbor in outside_positions and neighbor not in exterior:
                exterior.add(neighbor)
                queue.append(neighbor)

    hole_positions = outside_positions - exterior
    inferred_zone_by_pos: dict[tuple[int, int], str] = {}
    unresolved = set(hole_positions)

    while unresolved:
        newly_resolved: dict[tuple[int, int], str] = {}
        for x, y in unresolved:
            zone_votes: list[str] = []
            for nx, ny in _neighbors8(x, y, width, height):
                neighbor = (nx, ny)
                if neighbor in occupied_zone_by_pos:
                    zone_votes.append(occupied_zone_by_pos[neighbor])
                elif neighbor in inferred_zone_by_pos:
                    zone_votes.append(inferred_zone_by_pos[neighbor])

            if zone_votes:
                newly_resolved[(x, y)] = Counter(zone_votes).most_common(1)[0][0]

        if not newly_resolved:
            break

        inferred_zone_by_pos.update(newly_resolved)
        unresolved -= set(newly_resolved)

    if unresolved:
        occupied_positions_list = list(occupied_positions)
        for x, y in unresolved:
            if not occupied_positions_list:
                inferred_zone_by_pos[(x, y)] = "park-urban"
                continue
            nearest = min(
                occupied_positions_list,
                key=lambda pos: (pos[0] - x) ** 2 + (pos[1] - y) ** 2,
            )
            inferred_zone_by_pos[(x, y)] = occupied_zone_by_pos[nearest]

    return inferred_zone_by_pos


class SimulationSession:
    def __init__(
        self,
        config: SimulationConfig,
        interval_ms: int = 250,
    ) -> None:
        self.config = config
        self.max_steps = config.steps
        self.interval_ms = max(40, int(interval_ms))

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._running = False

        self.model = self._build_model(config)
        self.history: list[dict[str, Any]] = [collect_tick_metrics(self.model, replicate=0)]

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _build_model(self, config: SimulationConfig) -> SouthwarkModel:
        return SouthwarkModel(
            num_giraffes=config.num_giraffes,
            move_cost=config.move_cost,
            green_energy_gain=config.green_energy_gain,
            seed=config.seed,
            water_is_barrier=config.water_is_barrier,
            csv_path=config.csv_path,
            depleting_green=config.depleting_green,
            green_regrowth_per_tick=config.green_regrowth_per_tick,
        )

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.interval_ms / 1000.0)
            with self._lock:
                if not self._running:
                    continue
                self._advance_once_locked()

    def _advance_once_locked(self) -> None:
        if self.model.tick >= self.max_steps or not self.model.giraffes:
            self.model.finalize_outcomes(end_reason="max_steps_or_exhausted")
            self._running = False
            return

        self.model.step()
        self.history.append(collect_tick_metrics(self.model, replicate=0))

        if self.model.tick >= self.max_steps or not self.model.giraffes:
            self.model.finalize_outcomes(end_reason="max_steps_or_exhausted")
            self._running = False

    def start(self) -> None:
        with self._lock:
            self._running = True

    def pause(self) -> None:
        with self._lock:
            self._running = False

    def step_once(self) -> None:
        with self._lock:
            self._advance_once_locked()

    def reset(self, config: SimulationConfig, interval_ms: int | None = None) -> None:
        with self._lock:
            self._running = False
            self.config = config
            self.max_steps = config.steps
            if interval_ms is not None:
                self.interval_ms = max(40, int(interval_ms))
            self.model = self._build_model(config)
            self.history = [collect_tick_metrics(self.model, replicate=0)]

    def run_to_completion(self, config: SimulationConfig | None = None, interval_ms: int | None = None) -> None:
        """Run simulation to max_steps synchronously within the lock.

        If *config* is provided the session is reset first so a fresh run
        is executed with those parameters.
        """
        with self._lock:
            if config is not None:
                self._running = False
                self.config = config
                self.max_steps = config.steps
                if interval_ms is not None:
                    self.interval_ms = max(40, int(interval_ms))
                self.model = self._build_model(config)
                self.history = [collect_tick_metrics(self.model, replicate=0)]

            self._running = False  # disable background stepping
            while self.model.tick < self.max_steps and self.model.giraffes:
                self.model.step()
                self.history.append(collect_tick_metrics(self.model, replicate=0))
            self.model.finalize_outcomes(end_reason="max_steps_or_exhausted")

    def _static_payload_locked(self) -> dict[str, Any]:
        occupied_cell_by_pos = {
            (cell.x, cell.y): cell for cell in self.model.patch_cells if bool(cell.attrs["occupied?"])
        }
        occupied_zone_by_pos = {
            pos: str(cell.attrs["zone-type"]) for pos, cell in occupied_cell_by_pos.items()
        }
        inferred_zone_by_pos = _infer_display_holes(
            occupied_zone_by_pos=occupied_zone_by_pos,
            width=self.model.width,
            height=self.model.height,
        )

        patches = []
        for pos, cell in occupied_cell_by_pos.items():
            patches.append(
                {
                    "x": pos[0],
                    "y": pos[1],
                    "occupied": bool(cell.attrs["occupied?"]),
                    "display_occupied": True,
                    "inferred": False,
                    "traversable": bool(cell.attrs["traversable"]),
                    "zone_type": str(cell.attrs["zone-type"]),
                    "is_green": patch_is_green(cell.attrs),
                    "hex_id": str(cell.attrs.get("hex-id", "")),
                    "location_id": str(cell.attrs.get("location-id", "")),
                }
            )
        for pos, zone_type in inferred_zone_by_pos.items():
            patches.append(
                {
                    "x": pos[0],
                    "y": pos[1],
                    "occupied": False,
                    "display_occupied": True,
                    "inferred": True,
                    "traversable": False,
                    "zone_type": zone_type,
                    "is_green": zone_type in ("green-space", "park-urban"),
                    "hex_id": "",
                    "location_id": "",
                }
            )

        return {
            "width": self.model.width,
            "height": self.model.height,
            "patches": patches,
            "inferred_hole_count": len(inferred_zone_by_pos),
            "static_metrics": self.model.static_metrics,
        }

    def static_payload(self) -> dict[str, Any]:
        with self._lock:
            return self._static_payload_locked()

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            agents = []
            for agent in self.model.giraffes:
                if agent.pos is None:
                    continue
                agents.append(
                    {
                        "id": int(agent.unique_id),
                        "x": int(agent.pos[0]),
                        "y": int(agent.pos[1]),
                        "energy": float(agent.energy),
                        "distance_travelled": int(agent.distance_travelled),
                        "total_green_consumed": float(getattr(agent, "total_green_consumed", 0.0)),
                    }
                )

            patch_green_fraction: dict[str, float] = {}
            for cell in self.model.patch_cells:
                if not bool(cell.attrs["occupied?"]):
                    continue
                key = f"{cell.x},{cell.y}"
                patch_green_fraction[key] = self.model.get_patch_green_fraction((cell.x, cell.y))

            current_metrics = self.history[-1] if self.history else collect_tick_metrics(self.model, replicate=0)

            return {
                "tick": int(self.model.tick),
                "running": bool(self._running),
                "max_steps": int(self.max_steps),
                "metrics": current_metrics,
                "history_tail": self.history[-200:],
                "agents": agents,
                "green_land_summary": self.model.green_land_summary(),
                "giraffe_population_summary": self.model.giraffe_population_summary(),
                "giraffe_outcomes": self.model.get_giraffe_outcomes(),
                "patch_green_fraction": patch_green_fraction,
            }

    def export_run(self, out_dir: Path | None = None) -> dict[str, Any]:
        with self._lock:
            export_root = out_dir if out_dir is not None else Path("outputs") / "web_exports"
            run_dir = export_root / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)

            tick_rows = list(self.history)
            route_rows = self.model.get_giraffe_routes()
            route_rows_long = self.model.get_giraffe_routes_long()
            outcome_rows = self.model.get_giraffe_outcomes()
            agent_tick_rows = self.model.get_giraffe_tick_matrix(max_tick=self.max_steps)

            route_rows = [{**row, "replicate": 0} for row in route_rows]
            route_rows_long = [{**row, "replicate": 0} for row in route_rows_long]
            outcome_rows = [{**row, "replicate": 0} for row in outcome_rows]
            agent_tick_rows = [{**row, "replicate": 0} for row in agent_tick_rows]
            tick_rows = [{**row, "replicate": 0} for row in tick_rows]

            ticks_csv = run_dir / "metrics_ticks.csv"
            routes_csv = run_dir / "giraffe_routes.csv"
            routes_long_csv = run_dir / "giraffe_routes_long.csv"
            outcomes_csv = run_dir / "giraffe_outcomes.csv"
            agent_tick_csv = run_dir / "agent_tick_matrix.csv"
            summary_json = run_dir / "summary.json"

            write_csv(ticks_csv, tick_rows)
            write_csv(routes_csv, route_rows)
            write_csv(routes_long_csv, route_rows_long)
            write_csv(outcomes_csv, outcome_rows)
            write_csv(agent_tick_csv, agent_tick_rows)

            summary = {
                "config": self.config.to_dict(),
                "tick": int(self.model.tick),
                "running": bool(self._running),
                "static_metrics": self.model.static_metrics,
                "green_land_summary": self.model.green_land_summary(),
                "giraffe_population_summary": self.model.giraffe_population_summary(),
                "configured_num_giraffes": int(self.config.num_giraffes),
                "recorded_unique_agents_routes": len({int(row["agent_id"]) for row in route_rows}),
                "recorded_unique_agents_matrix": len({int(row["agent_id"]) for row in agent_tick_rows}),
                "giraffe_outcomes_count": len(outcome_rows),
                "giraffe_route_rows_count": len(route_rows),
                "giraffe_route_rows_long_count": len(route_rows_long),
                "agent_tick_rows_count": len(agent_tick_rows),
                "artifacts": {
                    "ticks_csv": str(ticks_csv),
                    "routes_csv": str(routes_csv),
                    "routes_long_csv": str(routes_long_csv),
                    "outcomes_csv": str(outcomes_csv),
                    "agent_tick_csv": str(agent_tick_csv),
                    "summary_json": str(summary_json),
                },
            }
            summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

            return {
                "ok": True,
                "paths": summary["artifacts"],
                "counts": {
                    "configured_num_giraffes": summary["configured_num_giraffes"],
                    "giraffe_outcomes_count": summary["giraffe_outcomes_count"],
                    "recorded_unique_agents_routes": summary["recorded_unique_agents_routes"],
                    "recorded_unique_agents_matrix": summary["recorded_unique_agents_matrix"],
                },
            }


class AppContainer:
    def __init__(self, session: SimulationSession) -> None:
        self.session = session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Southwark Mesa live web viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--num-giraffes", type=int, default=20)
    parser.add_argument("--move-cost", type=float, default=1.0)
    parser.add_argument("--green-energy-gain", type=float, default=10.0)
    parser.add_argument("--depleting-green", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--green-regrowth-per-tick", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument(
        "--water-is-barrier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, water patches are non-traversable.",
    )
    return parser.parse_args()


def build_sim_config(payload: dict[str, Any], fallback: SimulationConfig) -> SimulationConfig:
    return SimulationConfig(
        steps=int(payload.get("steps", fallback.steps)),
        num_giraffes=int(payload.get("num_giraffes", fallback.num_giraffes)),
        move_cost=float(payload.get("move_cost", fallback.move_cost)),
        green_energy_gain=float(payload.get("green_energy_gain", fallback.green_energy_gain)),
        depleting_green=bool(payload.get("depleting_green", fallback.depleting_green)),
        green_regrowth_per_tick=float(payload.get("green_regrowth_per_tick", fallback.green_regrowth_per_tick)),
        seed=(None if payload.get("seed") in (None, "", "null") else int(payload.get("seed"))),
        water_is_barrier=bool(payload.get("water_is_barrier", fallback.water_is_barrier)),
        csv_path=fallback.csv_path,
    )


def _export_warnings(session: SimulationSession) -> list[str]:
    warnings: list[str] = []
    tick = session.model.tick
    max_steps = session.max_steps
    configured = session.config.num_giraffes
    actual_outcomes = len(session.model.agent_outcomes)

    if tick == 0:
        warnings.append("Simulation has not been run. Export contains only spawn data.")
    elif tick < max_steps and session.model.giraffes:
        warnings.append(f"Simulation not complete (tick {tick} of {max_steps}).")

    if configured != actual_outcomes:
        warnings.append(f"Agent count mismatch: configured {configured}, recorded {actual_outcomes}.")

    return warnings


def create_app(container: AppContainer) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(HTML_PAGE, mimetype="text/html")

    @app.get("/api/static")
    def api_static() -> Response:
        return jsonify(container.session.static_payload())

    @app.get("/api/state")
    def api_state() -> Response:
        return jsonify(container.session.state_payload())

    @app.post("/api/start")
    def api_start() -> Response:
        payload = request.get_json(force=True, silent=True) or {}
        if payload:
            config = build_sim_config(payload=payload, fallback=container.session.config)
            interval_ms = int(payload.get("interval_ms", container.session.interval_ms))
            # Only reset if config actually changed, to avoid destroying in-progress runs
            if config.to_dict() != container.session.config.to_dict():
                container.session.reset(config=config, interval_ms=interval_ms)
            elif interval_ms != container.session.interval_ms:
                container.session.interval_ms = max(40, interval_ms)
        container.session.start()
        return jsonify(
            {
                "ok": True,
                "config": container.session.config.to_dict(),
                "tick": container.session.model.tick,
                "num_giraffes": len(container.session.model.giraffes),
            }
        )

    @app.post("/api/pause")
    def api_pause() -> Response:
        container.session.pause()
        return jsonify({"ok": True})

    @app.post("/api/step")
    def api_step() -> Response:
        container.session.step_once()
        return jsonify({"ok": True})

    @app.post("/api/reset")
    def api_reset() -> Response:
        payload = request.get_json(force=True, silent=True) or {}
        config = build_sim_config(payload=payload, fallback=container.session.config)
        interval_ms = int(payload.get("interval_ms", container.session.interval_ms))
        container.session.reset(config=config, interval_ms=interval_ms)
        return jsonify({"ok": True, "config": config.to_dict()})

    @app.post("/api/export")
    def api_export() -> Response:
        payload = request.get_json(force=True, silent=True) or {}
        out_dir_value = payload.get("out_dir")
        out_dir = Path(out_dir_value) if out_dir_value else None
        result = container.session.export_run(out_dir=out_dir)
        result["warnings"] = _export_warnings(container.session)
        return jsonify(result)

    @app.post("/api/run-and-export")
    def api_run_and_export() -> Response:
        payload = request.get_json(force=True, silent=True) or {}
        config = build_sim_config(payload=payload, fallback=container.session.config)
        interval_ms = int(payload.get("interval_ms", container.session.interval_ms))
        container.session.run_to_completion(config=config, interval_ms=interval_ms)
        result = container.session.export_run()
        result["warnings"] = _export_warnings(container.session)
        return jsonify(result)

    @app.get("/api/config")
    def api_config() -> Response:
        return Response(json.dumps(container.session.config.to_dict()), mimetype="application/json")

    @app.teardown_appcontext
    def _teardown(_: Any) -> None:
        return None

    return app


def main() -> None:
    args = parse_args()

    config = SimulationConfig(
        steps=args.steps,
        num_giraffes=args.num_giraffes,
        move_cost=args.move_cost,
        green_energy_gain=args.green_energy_gain,
        depleting_green=args.depleting_green,
        green_regrowth_per_tick=args.green_regrowth_per_tick,
        seed=args.seed,
        water_is_barrier=args.water_is_barrier,
        csv_path=args.csv_path,
    )

    session = SimulationSession(config=config, interval_ms=args.interval_ms)
    container = AppContainer(session=session)
    app = create_app(container)

    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        session.close()


if __name__ == "__main__":
    main()
