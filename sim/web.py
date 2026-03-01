"""
Flask web viewer for the swarm simulation.

Serves an interactive HTML page with a canvas-based grid viewer,
agent visualisation, real-time stats, and start/pause/step/reset controls.

Run with::

    python -m sim.web --host 127.0.0.1 --port 8765 --config baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Any

from flask import Flask, Response, jsonify, request

from pathlib import Path

from .model import SimConfig, SwarmModel

# ── HTML page ────────────────────────────────────────────────────────

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Swarm Simulation Viewer</title>
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
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(140deg, #edf2f9 0%, #f6fbf2 100%);
      color: var(--ink);
    }
    .layout {
      display: grid;
      gap: 16px;
      padding: 16px;
      grid-template-columns: minmax(400px, 2fr) minmax(320px, 1fr);
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 6px 16px rgba(17, 24, 39, 0.06);
    }
    h1 { margin: 0 0 10px 0; font-size: 18px; }
    .controls {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
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
    .small { margin-top: 8px; font-size: 12px; color: var(--muted); }
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
      max-height: 300px;
      overflow: auto;
      white-space: pre-wrap;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .controls { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="layout">
    <section class="card">
      <h1>Swarm Live Simulation</h1>
      <div class="controls">
        <button id="start">Start</button>
        <button id="pause">Pause</button>
        <button id="step">Step</button>
        <button id="reset">Reset</button>
      </div>
      <canvas id="world"></canvas>
      <div class="small">Terrain colours show cell type. Circles are agents coloured by state.</div>
    </section>

    <aside class="card">
      <div class="inputs">
        <label>Agents<input id="num_agents" type="number" value="20"></label>
        <label>Steps<input id="steps" type="number" value="120"></label>
        <label>Seed<input id="seed" type="number" value="42"></label>
        <label>Interval ms<input id="interval_ms" type="number" value="250"></label>
      </div>

      <div class="stats">
        <div class="stat"><span class="k">Tick</span><span class="v" id="tick">0</span></div>
        <div class="stat"><span class="k">Running</span><span class="v" id="running">No</span></div>
        <div class="stat"><span class="k">Active</span><span class="v" id="active">0</span></div>
        <div class="stat"><span class="k">Mean Speed</span><span class="v" id="mean_speed">0</span></div>
      </div>

      <div class="legend" id="legend"></div>
      <pre id="agent_summary">Agent info will appear here.</pre>
    </aside>
  </div>

  <script>
    const terrainColors = {
      'open':      '#e8dcc8',
      'nature':    '#7bc47f',
      'water':     '#5b9bd5',
      'exit':      '#f5c542',
      'obstacle':  '#6b6b6b'
    };

    const stateColors = {
      'navigating': '#2266dd',
      'panic':      '#dd2222',
      'stuck':      '#ddaa22',
      'evacuated':  '#22bb44',
      'dead':       '#333333'
    };

    let staticData = { width: 0, height: 0, patches: [] };
    let currentState = null;

    const canvas = document.getElementById('world');
    const ctx = canvas.getContext('2d');

    function drawLegend() {
      const legend = document.getElementById('legend');
      const rows = [];
      for (const [name, color] of Object.entries(terrainColors)) {
        rows.push(`<div class="row"><span class="swatch" style="background:${color}"></span>${name}</div>`);
      }
      rows.push('<hr style="width:100%;border-color:var(--border)">');
      for (const [name, color] of Object.entries(stateColors)) {
        rows.push(`<div class="row"><span class="swatch" style="background:${color};border-radius:50%"></span>agent: ${name}</div>`);
      }
      legend.innerHTML = rows.join('');
    }

    function drawWorld() {
      if (!staticData.width || !staticData.height) return;

      // Hex cell sizing — pointy-top hexagons
      const maxH = window.innerHeight - 160;
      const maxW = window.innerWidth * 0.6;
      // hex geometry: width = sqrt(3)*size, height = 2*size
      // Total canvas: W = sqrt(3)*size*(width) + sqrt(3)/2*size, H = 1.5*size*(height) + 0.5*size
      const sizeByH = maxH / (1.5 * staticData.height + 0.5);
      const sizeByW = maxW / (Math.sqrt(3) * (staticData.width + 0.5));
      const size = Math.max(4, Math.min(sizeByH, sizeByW));
      const hexW = Math.sqrt(3) * size;
      const hexH = 2 * size;

      canvas.width = Math.ceil(hexW * (staticData.width + 0.5));
      canvas.height = Math.ceil(size * (1.5 * staticData.height + 0.5));

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      function hexCenter(gx, gy) {
        // gy=0 is south (bottom), gy=height-1 is north (top)
        // Flip for screen: screen_row = height-1-gy
        const screenRow = staticData.height - 1 - gy;
        const xOff = (screenRow % 2 === 1) ? hexW / 2 : 0;
        const cx = gx * hexW + hexW / 2 + xOff;
        const cy = screenRow * size * 1.5 + size;
        return [cx, cy];
      }

      function drawHex(cx, cy) {
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
          const angle = Math.PI / 180 * (60 * i - 30);
          const hx = cx + size * Math.cos(angle);
          const hy = cy + size * Math.sin(angle);
          if (i === 0) ctx.moveTo(hx, hy);
          else ctx.lineTo(hx, hy);
        }
        ctx.closePath();
      }

      // Draw patches
      for (const patch of staticData.patches) {
        // Skip invalid (empty) cells — draw them as dark void
        const [cx, cy] = hexCenter(patch.x, patch.y);
        let color;
        if (!patch.valid) {
          color = '#1a1a2e';
        } else {
          color = terrainColors[patch.terrain] || terrainColors['open'];
        }
        // Tint hazard
        if (patch.valid && currentState && currentState.hazards) {
          const key = `${patch.x},${patch.y}`;
          const hz = currentState.hazards[key] || 0;
          if (hz > 0.01) {
            const t = Math.min(1, hz);
            const r = parseInt(color.slice(1,3), 16);
            const g = parseInt(color.slice(3,5), 16);
            const b = parseInt(color.slice(5,7), 16);
            const nr = Math.round(r + (220 - r) * t);
            const ng = Math.round(g + (40 - g) * t);
            const nb = Math.round(b + (40 - b) * t);
            color = `rgb(${nr},${ng},${nb})`;
          }
        }
        drawHex(cx, cy);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = 'rgba(30, 35, 45, 0.12)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }

      // Draw agents
      if (currentState && currentState.agents) {
        for (const agent of currentState.agents) {
          const [cx, cy] = hexCenter(agent.x, agent.y);
          ctx.beginPath();
          ctx.arc(cx, cy, Math.max(2, Math.floor(size * 0.35)), 0, Math.PI * 2);
          ctx.fillStyle = stateColors[agent.state] || '#2266dd';
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    function renderAgentSummary(agents) {
      if (!agents || agents.length === 0) {
        document.getElementById('agent_summary').textContent = 'No agents.';
        return;
      }
      const sorted = [...agents].sort((a, b) => a.id - b.id);
      const lines = ['id | state     | pos        | reasoning'];
      for (const a of sorted) {
        const r = (a.reasoning || '').substring(0, 60);
        lines.push(`${String(a.id).padStart(2)} | ${a.state.padEnd(10)} | (${String(a.x).padStart(2)},${String(a.y).padStart(2)}) | ${r}`);
      }
      document.getElementById('agent_summary').textContent = lines.join('\n');
    }

    function updateStats() {
      if (!currentState) return;
      const s = currentState.stats || {};
      document.getElementById('tick').textContent       = s.tick ?? 0;
      document.getElementById('running').textContent    = currentState.running ? 'Yes' : 'No';
      document.getElementById('active').textContent     = s.active ?? 0;
      document.getElementById('mean_speed').textContent = Number(s.mean_speed ?? 0).toFixed(3);

      renderAgentSummary(currentState.agents);
    }

    async function postAction(endpoint, body) {
      const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(endpoint, opts);
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
      try {
        const res = await fetch('/api/state');
        currentState = await res.json();
        updateStats();
        drawWorld();
      } catch(e) {}
      setTimeout(pollState, 250);
    }

    function readInputs() {
      return {
        num_agents:  Number(document.getElementById('num_agents').value),
        steps:       Number(document.getElementById('steps').value),
        seed:        Number(document.getElementById('seed').value),
        interval_ms: Number(document.getElementById('interval_ms').value),
      };
    }

    document.getElementById('start').onclick = () => postAction('/api/start', readInputs());
    document.getElementById('pause').onclick = () => postAction('/api/pause');
    document.getElementById('step').onclick  = () => postAction('/api/step');
    document.getElementById('reset').onclick = async () => {
      await postAction('/api/reset', readInputs());
      await fetchStatic();
    };

    window.addEventListener('resize', drawWorld);
    // Populate inputs from server config
    fetch('/api/config').then(r => r.json()).then(cfg => {
      if (cfg.num_agents) document.getElementById('num_agents').value = cfg.num_agents;
      if (cfg.steps)      document.getElementById('steps').value = cfg.steps;
      if (cfg.seed != null) document.getElementById('seed').value = cfg.seed;
    });
    drawLegend();
    fetchStatic().then(pollState);
  </script>
</body>
</html>
"""


# ── Session (background-thread simulation runner) ────────────────────


class SimulationSession:
    """Manages a running simulation with background thread advancement."""

    def __init__(
        self,
        config: SimConfig,
        interval_ms: int = 250,
        scenario_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.interval_ms = max(40, int(interval_ms))
        self.scenario_path = scenario_path

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._running = False

        self.model = SwarmModel(config=config, scenario_path=scenario_path)
        # Sync config to reflect what the model *actually* used (scenario
        # loading may override width/height/num_agents/goal/scenario).
        self.config = self.model.config

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.interval_ms / 1000.0)
            with self._lock:
                if not self._running:
                    continue
                self._advance_once()

    def _advance_once(self) -> None:
        if not self.model.running:
            self._running = False
            return
        self.model.step()
        if not self.model.running:
            self._running = False

    def start(self) -> None:
        with self._lock:
            self._running = True

    def pause(self) -> None:
        with self._lock:
            self._running = False

    def step_once(self) -> None:
        with self._lock:
            self._advance_once()

    def reset(self, config: SimConfig, interval_ms: int | None = None) -> None:
        with self._lock:
            self._running = False
            if interval_ms is not None:
                self.interval_ms = max(40, int(interval_ms))
            self.model = SwarmModel(config=config, scenario_path=self.scenario_path)
            # Sync config to reflect what the model actually used.
            self.config = self.model.config

    def static_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "width": self.model.world.width,
                "height": self.model.world.height,
                "patches": self.model.get_terrain_grid(),
            }

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            agents = self.model.get_agent_list()
            stats = self.model.get_stats_dict()

            # Build hazard map for cells with non-zero hazard
            hazards: dict[str, float] = {}
            for y in range(self.model.world.height):
                for x in range(self.model.world.width):
                    hz = float(self.model.world.hazard_grid[y, x])
                    if hz > 0.001:
                        hazards[f"{x},{y}"] = round(hz, 4)

            return {
                "tick": self.model.tick,
                "running": self._running,
                "stats": stats,
                "agents": agents,
                "hazards": hazards,
            }


# ── Flask app ────────────────────────────────────────────────────────


class AppContainer:
    def __init__(self, session: SimulationSession) -> None:
        self.session = session


def build_sim_config(payload: dict[str, Any], fallback: SimConfig) -> SimConfig:
    return SimConfig(
        data_path=fallback.data_path,
        num_agents=int(payload.get("num_agents", fallback.num_agents)),
        steps=int(payload.get("steps", fallback.steps)),
        seed=(None if payload.get("seed") in (None, "", "null") else int(payload["seed"])),
        scenario=fallback.scenario,
        goal=fallback.goal,
        personality=fallback.personality,
        awareness_radius=fallback.awareness_radius,
        use_llm=fallback.use_llm,
    )


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
            config = build_sim_config(payload, container.session.config)
            interval_ms = int(payload.get("interval_ms", container.session.interval_ms))
            if config.to_dict() != container.session.config.to_dict():
                container.session.reset(config=config, interval_ms=interval_ms)
            elif interval_ms != container.session.interval_ms:
                container.session.interval_ms = max(40, interval_ms)
        container.session.start()
        return jsonify({"ok": True, "tick": container.session.model.tick})

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
        config = build_sim_config(payload, container.session.config)
        interval_ms = int(payload.get("interval_ms", container.session.interval_ms))
        container.session.reset(config=config, interval_ms=interval_ms)
        return jsonify({"ok": True, "config": config.to_dict()})

    @app.get("/api/config")
    def api_config() -> Response:
        return Response(
            json.dumps(container.session.config.to_dict()),
            mimetype="application/json",
        )

    return app


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swarm simulation web viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use real LLM client (requires API_KEY and BASE_URL in .env).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a scenario YAML file (e.g. scenarios/baseline.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve scenario YAML path (look in scenarios/ if bare filename)
    scenario_path: str | None = None
    if args.config:
        p = Path(args.config)
        if not p.exists():
            candidate = Path("scenarios") / p
            if candidate.exists():
                p = candidate
        scenario_path = str(p)

    config = SimConfig(
        num_agents=args.num_agents,
        steps=args.steps,
        seed=args.seed,
        use_llm=args.use_llm,
    )

    session = SimulationSession(
        config=config,
        interval_ms=args.interval_ms,
        scenario_path=scenario_path,
    )
    container = AppContainer(session)
    app = create_app(container)

    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        session.close()


if __name__ == "__main__":
    main()
