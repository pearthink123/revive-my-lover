"""
Stochastic Differential Equations for engagement dynamics.

Three models for different engagement patterns:
1. Ornstein-Uhlenbeck: mean-reverting (engagement returns to baseline)
2. Geometric Brownian: trending (engagement has momentum)
3. Heston: volatility clustering (calm/stormy periods alternate)
"""

from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class SDEModel(ABC):
    """Abstract base for SDE models."""

    @abstractmethod
    def step(self, dt: float = 1.0) -> float:
        """Take one step of the SDE. Returns new engagement level."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset to initial state."""
        ...


@dataclass
class OrnsteinUhlenbeck(SDEModel):
    """
    Ornstein-Uhlenbeck process: dX = θ(μ - X)dt + σdW

    Mean-reverting: engagement drifts toward μ (baseline) at speed θ.
    σ controls noise intensity.

    Good for: stable relationships where engagement has a natural baseline.

    Args:
        theta: Mean reversion speed (0-1). Higher = faster return to baseline.
        mu: Long-term mean (baseline engagement, 0-1).
        sigma: Volatility (noise intensity, 0-1).
        x0: Initial engagement level.
        seed: Random seed for reproducibility.
    """

    theta: float = 0.3       # Mean reversion speed
    mu: float = 0.5          # Long-term mean
    sigma: float = 0.1       # Volatility
    x0: float = 0.5          # Initial value
    seed: Optional[int] = None

    def __post_init__(self):
        self._x = self.x0
        self._rng = random.Random(self.seed)
        self._history = [self._x]

    def step(self, dt: float = 1.0) -> float:
        """One step of OU process."""
        # Drift: θ(μ - X)dt
        drift = self.theta * (self.mu - self._x) * dt
        
        # Diffusion: σ√dt * N(0,1)
        diffusion = self.sigma * math.sqrt(dt) * self._rng.gauss(0, 1)
        
        # Update
        self._x += drift + diffusion
        
        # Clamp to [0, 1]
        self._x = max(0.0, min(1.0, self._x))
        
        self._history.append(self._x)
        return self._x

    def reset(self) -> None:
        """Reset to initial state."""
        self._x = self.x0
        self._history = [self._x]

    @property
    def current(self) -> float:
        """Current engagement level."""
        return self._x

    @property
    def history(self) -> list[float]:
        """Engagement history."""
        return list(self._history)


@dataclass
class GeometricBrownian(SDEModel):
    """
    Geometric Brownian Motion: dX = μXdt + σXdW

    Trending: engagement grows/decays exponentially with noise.
    μ = drift (growth rate), σ = volatility.

    Good for: relationships with momentum (honeymoon phase, cooling off).

    Args:
        mu: Drift (growth rate, can be negative).
        sigma: Volatility.
        x0: Initial engagement level.
        seed: Random seed.
    """

    mu: float = 0.01         # Growth rate
    sigma: float = 0.1       # Volatility
    x0: float = 0.5          # Initial value
    seed: Optional[int] = None

    def __post_init__(self):
        self._x = self.x0
        self._rng = random.Random(self.seed)
        self._history = [self._x]

    def step(self, dt: float = 1.0) -> float:
        """One step of GBM."""
        # dX = μXdt + σX√dt * N(0,1)
        drift = self.mu * self._x * dt
        diffusion = self.sigma * self._x * math.sqrt(dt) * self._rng.gauss(0, 1)
        
        self._x += drift + diffusion
        
        # Clamp to [0, 1]
        self._x = max(0.01, min(1.0, self._x))  # Avoid zero
        
        self._history.append(self._x)
        return self._x

    def reset(self) -> None:
        """Reset to initial state."""
        self._x = self.x0
        self._history = [self._x]

    @property
    def current(self) -> float:
        return self._x

    @property
    def history(self) -> list[float]:
        return list(self._history)


@dataclass
class Heston(SDEModel):
    """
    Heston model: stochastic volatility.

    dX = μXdt + √V X dW₁
    dV = κ(θ - V)dt + σ√V dW₂

    Where:
    X = engagement level
    V = variance (volatility squared)
    κ = mean reversion speed for variance
    θ = long-term variance
    σ = volatility of volatility

    Good for: relationships with alternating calm/stormy periods.

    Args:
        mu: Drift for engagement.
        kappa: Mean reversion speed for variance.
        theta: Long-term variance.
        sigma_v: Volatility of volatility.
        x0: Initial engagement.
        v0: Initial variance.
        seed: Random seed.
    """

    mu: float = 0.01         # Engagement drift
    kappa: float = 0.3       # Variance mean reversion
    theta: float = 0.01      # Long-term variance
    sigma_v: float = 0.1     # Vol of vol
    x0: float = 0.5          # Initial engagement
    v0: float = 0.01         # Initial variance
    seed: Optional[int] = None

    def __post_init__(self):
        self._x = self.x0
        self._v = self.v0
        self._rng = random.Random(self.seed)
        self._history_x = [self._x]
        self._history_v = [self._v]

    def step(self, dt: float = 1.0) -> float:
        """One step of Heston model."""
        # Ensure variance stays positive
        sqrt_v = math.sqrt(max(0, self._v))
        
        # Correlated Brownian motions (ρ = -0.5 for leverage effect)
        z1 = self._rng.gauss(0, 1)
        z2 = -0.5 * z1 + math.sqrt(0.75) * self._rng.gauss(0, 1)
        
        # dX = μXdt + √V X dW₁
        drift_x = self.mu * self._x * dt
        diffusion_x = sqrt_v * self._x * math.sqrt(dt) * z1
        self._x += drift_x + diffusion_x
        
        # dV = κ(θ - V)dt + σ√V dW₂
        drift_v = self.kappa * (self.theta - self._v) * dt
        diffusion_v = self.sigma_v * sqrt_v * math.sqrt(dt) * z2
        self._v += drift_v + diffusion_v
        
        # Clamp
        self._x = max(0.01, min(1.0, self._x))
        self._v = max(0.0001, self._v)  # Variance must be positive
        
        self._history_x.append(self._x)
        self._history_v.append(self._v)
        return self._x

    def reset(self) -> None:
        """Reset to initial state."""
        self._x = self.x0
        self._v = self.v0
        self._history_x = [self._x]
        self._history_v = [self._v]

    @property
    def current(self) -> float:
        return self._x

    @property
    def current_variance(self) -> float:
        return self._v

    @property
    def history(self) -> list[float]:
        return list(self._history_x)

    @property
    def variance_history(self) -> list[float]:
        return list(self._history_v)


@dataclass
class EngagementDynamics:
    """
    Unified engagement dynamics using SDE models.

    Combines SDE with Bayesian state inference to model how
    user engagement evolves over time.

    Args:
        model: SDE model to use.
        state_estimator: Bayesian state estimator (optional).
        observation_noise: Noise added to observations (0-1).
        seed: Random seed.
    """

    model: SDEModel = field(default_factory=OrnsteinUhlenbeck)
    observation_noise: float = 0.05
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._time = 0.0
        self._observations = []

    def observe(self, dt: float = 1.0) -> float:
        """
        Observe engagement with noise.

        Returns:
            Noisy observation of current engagement level.
        """
        true_engagement = self.model.step(dt)
        
        # Add observation noise
        noise = self._rng.gauss(0, self.observation_noise)
        observed = true_engagement + noise
        
        # Clamp
        observed = max(0.0, min(1.0, observed))
        
        self._time += dt
        self._observations.append({
            "time": self._time,
            "true": true_engagement,
            "observed": observed,
        })
        
        return observed

    def predict(self, steps: int = 10, dt: float = 1.0) -> list[float]:
        """
        Predict future engagement levels.

        Args:
            steps: Number of steps to predict.
            dt: Time step size.

        Returns:
            List of predicted engagement levels.
        """
        predictions = []
        
        # Save current state
        if hasattr(self.model, '_x'):
            x_save = self.model._x
        if hasattr(self.model, '_v'):
            v_save = self.model._v
        
        # Predict forward
        for _ in range(steps):
            pred = self.model.step(dt)
            predictions.append(pred)
        
        # Restore state
        if hasattr(self.model, '_x'):
            self.model._x = x_save
        if hasattr(self.model, '_v'):
            self.model._v = v_save
        
        return predictions

    def reset(self) -> None:
        """Reset model and observations."""
        self.model.reset()
        self._time = 0.0
        self._observations = []

    @property
    def current_engagement(self) -> float:
        """Current true engagement level."""
        return self.model.current

    @property
    def observations(self) -> list[dict]:
        """Observation history."""
        return list(self._observations)
