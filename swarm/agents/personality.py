"""
Stochastic agent personality system.

Each agent draws personality traits from configurable probability distributions,
ensuring population-level diversity. Two agents in the same swarm will behave
differently because they have different risk tolerances, speeds, panic thresholds, etc.

This is a key innovation: realistic behavioral heterogeneity without hand-coding
individual agents. The distributions are parameterized per scenario.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PersonalityTraits:
    """
    Concrete personality values sampled for one agent.

    All values are normalized to [0, 1] or reasonable physical ranges.
    """

    speed: float              # Base movement speed (cells per tick), typically 0.5–2.0
    risk_tolerance: float     # 0 = extremely risk-averse, 1 = danger-seeking
    panic_threshold: float    # Hazard level at which agent enters panic mode [0, 1]
    herding_tendency: float   # 0 = independent, 1 = strong follow-the-crowd
    patience: float           # How many ticks before rerouting when stuck
    exploration: float        # 0 = always exploit known paths, 1 = explore new routes
    awareness_radius: float   # How many cells the agent can perceive (vision range)

    def __repr__(self) -> str:
        return (
            f"Personality(spd={self.speed:.2f}, risk={self.risk_tolerance:.2f}, "
            f"panic={self.panic_threshold:.2f}, herd={self.herding_tendency:.2f})"
        )


@dataclass
class PersonalityDistribution:
    """
    Distribution parameters for generating stochastic personalities.

    Each trait has a distribution type and parameters. The swarm manager
    samples from these to create individual agents.
    """

    # (mean, std) for normal distributions, (alpha, beta) for beta distributions
    speed_mean: float = 1.0
    speed_std: float = 0.2
    risk_alpha: float = 2.0      # Beta distribution alpha (lower = more risk-averse)
    risk_beta: float = 5.0       # Beta distribution beta
    panic_low: float = 0.3       # Uniform range for panic threshold
    panic_high: float = 0.8
    herding_alpha: float = 3.0   # Beta distribution for herding
    herding_beta: float = 2.0
    patience_mean: float = 20.0  # Mean ticks before reroute
    patience_std: float = 10.0
    exploration_alpha: float = 2.0
    exploration_beta: float = 8.0
    awareness_mean: float = 5.0  # Cells of awareness radius
    awareness_std: float = 1.5

    def sample(self, rng: np.random.Generator | None = None) -> PersonalityTraits:
        """Sample a concrete personality from the distributions."""
        if rng is None:
            rng = np.random.default_rng()

        return PersonalityTraits(
            speed=max(0.1, rng.normal(self.speed_mean, self.speed_std)),
            risk_tolerance=float(rng.beta(self.risk_alpha, self.risk_beta)),
            panic_threshold=float(rng.uniform(self.panic_low, self.panic_high)),
            herding_tendency=float(rng.beta(self.herding_alpha, self.herding_beta)),
            patience=max(1.0, rng.normal(self.patience_mean, self.patience_std)),
            exploration=float(rng.beta(self.exploration_alpha, self.exploration_beta)),
            awareness_radius=max(1.0, rng.normal(self.awareness_mean, self.awareness_std)),
        )

    def sample_batch(
        self, n: int, rng: np.random.Generator | None = None
    ) -> list[PersonalityTraits]:
        """Sample n personalities."""
        if rng is None:
            rng = np.random.default_rng()
        return [self.sample(rng) for _ in range(n)]


# ── Preset personality distributions for common scenarios ────────────

CALM_POPULATION = PersonalityDistribution(
    speed_mean=1.0, speed_std=0.15,
    risk_alpha=3.0, risk_beta=3.0,
    panic_low=0.5, panic_high=0.9,
    herding_alpha=2.0, herding_beta=3.0,
    patience_mean=30.0, patience_std=10.0,
)

PANICKY_POPULATION = PersonalityDistribution(
    speed_mean=1.5, speed_std=0.4,
    risk_alpha=1.5, risk_beta=6.0,
    panic_low=0.1, panic_high=0.4,
    herding_alpha=5.0, herding_beta=2.0,
    patience_mean=5.0, patience_std=3.0,
)

MIXED_POPULATION = PersonalityDistribution(
    speed_mean=1.0, speed_std=0.3,
    risk_alpha=2.0, risk_beta=5.0,
    panic_low=0.3, panic_high=0.8,
    herding_alpha=3.0, herding_beta=2.0,
    patience_mean=20.0, patience_std=15.0,
)
