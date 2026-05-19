"""
SDE Demo — Stochastic Differential Equations for engagement dynamics.

Shows how different SDE models capture different engagement patterns:
1. Ornstein-Uhlenbeck: mean-reverting (stable relationship)
2. Geometric Brownian: trending (honeymoon/cooling off)
3. Heston: volatility clustering (calm/stormy periods)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from revive_my_lover.sde import (
    OrnsteinUhlenbeck,
    GeometricBrownian,
    Heston,
    EngagementDynamics,
)


def demo_ou():
    """Ornstein-Uhlenbeck: mean-reverting engagement."""
    print("=" * 60)
    print("🎲 Ornstein-Uhlenbeck: Mean-Reverting Engagement")
    print("=" * 60)
    print()
    print("Models stable relationships where engagement returns to baseline.")
    print("θ (theta) = mean reversion speed")
    print("μ (mu) = baseline engagement level")
    print("σ (sigma) = volatility")
    print()

    # Create model
    ou = OrnsteinUhlenbeck(
        theta=0.3,      # Reverts to baseline at 30% per step
        mu=0.5,         # Baseline engagement = 50%
        sigma=0.1,      # 10% volatility
        x0=0.8,         # Start at 80% (high engagement)
        seed=42,
    )

    print(f"Initial engagement: {ou.current:.1%}")
    print(f"Baseline: {ou.mu:.1%}")
    print()

    # Simulate 50 steps
    print("Step  | Engagement | Bar")
    print("-" * 40)
    for i in range(50):
        engagement = ou.step(dt=0.1)
        bar = "█" * int(engagement * 30)
        if i % 5 == 0 or i == 49:
            print(f"{i:4d}  | {engagement:6.1%}    | {bar}")

    print()
    print(f"Final engagement: {ou.current:.1%}")
    print(f"Mean of history: {sum(ou.history)/len(ou.history):.1%}")
    print()
    print("💡 Insight: Engagement drifts toward baseline (50%) over time.")
    print("   High engagement → decreases, Low engagement → increases.")


def demo_gbm():
    """Geometric Brownian Motion: trending engagement."""
    print()
    print("=" * 60)
    print("📈 Geometric Brownian Motion: Trending Engagement")
    print("=" * 60)
    print()
    print("Models relationships with momentum (honeymoon phase, cooling off).")
    print("μ (mu) = growth rate (positive = growing, negative = decaying)")
    print("σ (sigma) = volatility")
    print()

    # Create model with positive drift (honeymoon phase)
    gbm = GeometricBrownian(
        mu=0.02,        # 2% growth per step
        sigma=0.15,     # 15% volatility
        x0=0.3,         # Start at 30%
        seed=42,
    )

    print(f"Initial engagement: {gbm.current:.1%}")
    print(f"Growth rate: {gbm.mu:.1%} per step")
    print()

    # Simulate 50 steps
    print("Step  | Engagement | Bar")
    print("-" * 40)
    for i in range(50):
        engagement = gbm.step(dt=0.1)
        bar = "█" * int(engagement * 30)
        if i % 5 == 0 or i == 49:
            print(f"{i:4d}  | {engagement:6.1%}    | {bar}")

    print()
    print(f"Final engagement: {gbm.current:.1%}")
    print()
    print("💡 Insight: With positive drift, engagement grows over time.")
    print("   With negative drift, engagement decays (cooling off).")


def demo_heston():
    """Heston model: volatility clustering."""
    print()
    print("=" * 60)
    print("🌊 Heston Model: Volatility Clustering")
    print("=" * 60)
    print()
    print("Models alternating calm/stormy periods.")
    print("κ (kappa) = variance mean reversion speed")
    print("θ (theta) = long-term variance")
    print("σ_v = volatility of volatility")
    print()

    # Create model
    heston = Heston(
        mu=0.01,        # Slight upward drift
        kappa=0.3,      # Variance reverts at 30% per step
        theta=0.01,     # Long-term variance = 1%
        sigma_v=0.2,    # High vol-of-vol
        x0=0.5,         # Start at 50%
        v0=0.001,       # Start with low variance (calm)
        seed=42,
    )

    print(f"Initial engagement: {heston.current:.1%}")
    print(f"Initial variance: {heston.current_variance:.4f}")
    print()

    # Simulate 50 steps
    print("Step  | Engagement | Variance | Bar")
    print("-" * 50)
    for i in range(50):
        engagement = heston.step(dt=0.1)
        variance = heston.current_variance
        bar = "█" * int(engagement * 30)
        vol_indicator = "🌪️" if variance > 0.02 else "  "
        if i % 5 == 0 or i == 49:
            print(f"{i:4d}  | {engagement:6.1%}    | {variance:.4f} {vol_indicator} | {bar}")

    print()
    print(f"Final engagement: {heston.current:.1%}")
    print(f"Final variance: {heston.current_variance:.4f}")
    print()
    print("💡 Insight: Variance itself changes over time.")
    print("   Calm periods → low variance, Stormy periods → high variance.")


def demo_dynamics():
    """EngagementDynamics: unified interface."""
    print()
    print("=" * 60)
    print("🎯 EngagementDynamics: Unified Interface")
    print("=" * 60)
    print()
    print("Combines SDE models with observation noise.")
    print()

    # Create dynamics with OU model
    dynamics = EngagementDynamics(
        model=OrnsteinUhlenbeck(theta=0.3, mu=0.5, sigma=0.1, seed=42),
        observation_noise=0.05,
        seed=42,
    )

    print("Step  | True      | Observed  | Bar")
    print("-" * 50)
    for i in range(20):
        observed = dynamics.observe(dt=0.1)
        true_val = dynamics.current_engagement
        bar = "█" * int(true_val * 30)
        if i % 2 == 0 or i == 19:
            print(f"{i:4d}  | {true_val:6.1%}    | {observed:6.1%}    | {bar}")

    print()
    print("Predicting next 10 steps:")
    predictions = dynamics.predict(steps=10, dt=0.1)
    for i, pred in enumerate(predictions):
        bar = "█" * int(pred * 30)
        print(f"  +{i+1:2d}  | {pred:6.1%}    | {bar}")

    print()
    print("💡 Insight: Observations have noise, predictions use true model.")


def demo_comparison():
    """Compare all three models."""
    print()
    print("=" * 60)
    print("📊 Model Comparison")
    print("=" * 60)
    print()

    # All start at 50%
    ou = OrnsteinUhlenbeck(theta=0.3, mu=0.5, sigma=0.1, x0=0.5, seed=42)
    gbm = GeometricBrownian(mu=0.02, sigma=0.1, x0=0.5, seed=42)
    heston = Heston(mu=0.01, kappa=0.3, theta=0.01, sigma_v=0.2, x0=0.5, seed=42)

    print("Step  | OU        | GBM       | Heston")
    print("-" * 50)
    for i in range(30):
        ou.step(dt=0.1)
        gbm.step(dt=0.1)
        heston.step(dt=0.1)
        if i % 3 == 0 or i == 29:
            print(f"{i:4d}  | {ou.current:6.1%}    | {gbm.current:6.1%}    | {heston.current:6.1%}")

    print()
    print("💡 Insights:")
    print("  • OU: Stays near baseline (50%)")
    print("  • GBM: Grows exponentially (with noise)")
    print("  • Heston: Similar to GBM but with varying volatility")


if __name__ == "__main__":
    demo_ou()
    demo_gbm()
    demo_heston()
    demo_dynamics()
    demo_comparison()
    print()
    print("=" * 60)
    print("✅ All SDE demos completed!")
    print("=" * 60)
