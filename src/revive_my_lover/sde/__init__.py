"""
revive_my_lover.sde — Stochastic Differential Equations for engagement dynamics.

Models user engagement as a continuous-time stochastic process:
  dX = μ(X, t)dt + σ(X, t)dW

Where:
  X = engagement level (0-1)
  μ = drift (trend)
  σ = diffusion (volatility)
  dW = Wiener process (random noise)

Three models:
1. OrnsteinUhlenbeck — mean-reverting engagement
2. GeometricBrownian — trending engagement
3. Heston — volatility clustering

Example:
    from revive_my_lover.sde import OrnsteinUhlenbeck

    ou = OrnsteinUhlenbeck(theta=0.5, mu=0.5, sigma=0.1)
    for t in range(100):
        engagement = ou.step(dt=0.1)
        print(f"t={t}: engagement={engagement:.3f}")
"""

from .core import (
    SDEModel,
    OrnsteinUhlenbeck,
    GeometricBrownian,
    Heston,
    EngagementDynamics,
)

__all__ = [
    "SDEModel",
    "OrnsteinUhlenbeck",
    "GeometricBrownian",
    "Heston",
    "EngagementDynamics",
]
