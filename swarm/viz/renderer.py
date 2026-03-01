"""
Pygame-based real-time visualization renderer.

Renders the world grid, agents, hazards, pheromones, and flow fields.
Falls back to matplotlib for static frame rendering if Pygame is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from swarm.agents.swarm import Swarm
    from swarm.core.engine import SimulationState
    from swarm.core.world import World

# Color palette
TERRAIN_COLORS = {
    0: (200, 200, 200),  # OPEN — light gray
    1: (180, 180, 180),  # CORRIDOR — slightly darker gray
    2: (80, 80, 80),     # ROAD — asphalt
    3: (160, 160, 160),  # SIDEWALK — medium gray
    4: (100, 100, 100),  # STAIRS — darker gray
    5: (160, 82, 45),    # DOOR — brown
    6: (0, 200, 0),      # EXIT — green
    7: (40, 40, 40),     # WALL — dark gray
    8: (60, 60, 180),    # WATER — blue
    9: (180, 120, 60),   # BUILDING — tan
    10: (50, 50, 50),    # OBSTACLE — near-black
    11: (0, 150, 0),     # GRASS — dark green
}

AGENT_COLORS = {
    "NAVIGATING": (0, 120, 255),   # Blue
    "EVACUATED": (0, 200, 0),      # Green
    "PANIC": (255, 165, 0),        # Orange
    "STUCK": (200, 200, 0),        # Yellow
    "DEAD": (255, 0, 0),           # Red
}


class PygameRenderer:
    """Real-time renderer using Pygame."""

    def __init__(
        self,
        world: "World",
        cell_size: int = 4,
        fps: int = 30,
        show_pheromones: bool = False,
        show_flow: bool = False,
    ):
        import pygame

        self.cell_size = cell_size
        self.fps = fps
        self.show_pheromones = show_pheromones
        self.show_flow = show_flow
        self.width = world.width * cell_size
        self.height = world.height * cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("SWARM Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)

        # Pre-render terrain as a surface
        self._terrain_surface = self._render_terrain(world)

    def _render_terrain(self, world: "World"):
        """Pre-render the static terrain layer."""
        import pygame

        surface = pygame.Surface((self.width, self.height))
        cs = self.cell_size

        for y in range(world.height):
            for x in range(world.width):
                terrain = int(world.terrain_grid[y, x])
                color = TERRAIN_COLORS.get(terrain, (128, 128, 128))
                pygame.draw.rect(surface, color, (x * cs, y * cs, cs, cs))

        return surface

    def draw(self, world: "World", swarm: "Swarm", state: "SimulationState") -> bool:
        """
        Draw one frame.

        Returns False if the window was closed.
        """
        import pygame

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        cs = self.cell_size

        # Base terrain
        self.screen.blit(self._terrain_surface, (0, 0))

        # Hazard overlay
        hazard = world.hazard_grid
        hazard_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        hy, hx = np.where(hazard > 0.0)
        for i in range(len(hx)):
            intensity = min(hazard[hy[i], hx[i]], 1.0)
            alpha = int(intensity * 180)
            pygame.draw.rect(
                hazard_surface,
                (255, 60, 0, alpha),
                (hx[i] * cs, hy[i] * cs, cs, cs),
            )
        self.screen.blit(hazard_surface, (0, 0))

        # Agents
        for agent in swarm.agents.values():
            pos = agent.position
            color = AGENT_COLORS.get(agent.state.name, (128, 128, 128))
            cx = pos.x * cs + cs // 2
            cy = pos.y * cs + cs // 2
            radius = max(cs // 2, 2)
            pygame.draw.circle(self.screen, color, (cx, cy), radius)

        # HUD
        stats = state.swarm_stats
        hud_lines = [
            f"t={state.tick:>5} | {state.time:.1f}s",
            f"active={stats.active} evac={stats.evacuated} dead={stats.dead}",
            f"panic={stats.panicking} hazards={state.active_hazards}",
        ]
        for i, line in enumerate(hud_lines):
            text = self.font.render(line, True, (255, 255, 255))
            bg = pygame.Surface(text.get_size())
            bg.fill((0, 0, 0))
            bg.set_alpha(180)
            self.screen.blit(bg, (4, 4 + i * 18))
            self.screen.blit(text, (4, 4 + i * 18))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self):
        """Shut down Pygame."""
        import pygame

        pygame.quit()


def render_frame_matplotlib(
    world: "World",
    agent_positions: list[tuple[int, int]],
    title: str = "Simulation Frame",
    save_path: str | None = None,
):
    """Render a single static frame using matplotlib (no Pygame needed)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Terrain
    cmap = ListedColormap([
        "#C8C8C8", "#B4B4B4", "#505050", "#A0A0A0", "#646464",
        "#A0522D", "#00C800", "#282828", "#3C3CB4", "#B4783C",
        "#323232", "#009600",
    ])
    ax.imshow(world.terrain_grid, cmap=cmap, origin="upper", vmin=0, vmax=9)

    # Hazards
    hazard_masked = np.ma.masked_where(world.hazard_grid == 0, world.hazard_grid)
    ax.imshow(hazard_masked, cmap="Reds", alpha=0.6, origin="upper", vmin=0, vmax=1)

    # Agents
    if agent_positions:
        xs, ys = zip(*agent_positions)
        ax.scatter(xs, ys, c="blue", s=4, alpha=0.7, marker="o")

    ax.set_title(title)
    ax.set_xlim(0, world.width)
    ax.set_ylim(world.height, 0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
