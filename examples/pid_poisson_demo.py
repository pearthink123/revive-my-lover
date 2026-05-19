"""
PID × Poisson — integrated demo.

Simulates a real chat scenario where:
1. Poisson engine decides when to send
2. User responds with varying engagement (speed, quality, reaction)
3. PID adjusts the engine's lambda rate based on engagement
4. Watch the system adapt in real-time

Run: PYTHONPATH=src python examples/pid_poisson_demo.py
"""

import math
import random
from datetime import datetime, timedelta

from revive_my_lover.core.engine import PoissonEngine
from revive_my_lover.core.config import Config
from revive_my_lover.core.models import Action
from revive_my_lover.control import PIDController, Signal, CombinedSignal


# ─── Simulated user behavior ───

class SimUser:
    """Simulates a user with mood cycles and engagement patterns."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._mood = 0.6  # 0=bored, 1=engaged
        self._mood_direction = 1
        self._days_since_last_send = 0

    def respond(self, hour: float, longing: float) -> dict:
        """
        Simulate user response to a message.

        Returns dict with reply_speed, reply_quality, has_reaction.
        """
        # Mood oscillates more realistically
        delta = self._rng.uniform(0.08, 0.2)
        self._mood += self._mood_direction * delta
        if self._mood > 0.95:
            self._mood_direction = -1
            self._mood = 0.95
        elif self._mood < 0.15:
            self._mood_direction = 1
            self._mood = 0.15

        # Time-of-day effect (stronger variation)
        if 0 <= hour < 8:
            time_factor = 0.1
        elif 8 <= hour < 10:
            time_factor = 0.6
        elif 10 <= hour < 12:
            time_factor = 0.8
        elif 12 <= hour < 14:
            time_factor = 0.5
        elif 14 <= hour < 17:
            time_factor = 0.85
        elif 17 <= hour < 20:
            time_factor = 0.95
        elif 20 <= hour < 22:
            time_factor = 0.9
        else:
            time_factor = 0.3

        # Engagement = mood × time_factor × randomness
        base = self._mood * time_factor
        noise = self._rng.uniform(0.85, 1.15)
        engagement = base * noise

        # Longing sweet spot: moderate longing (0.3-0.6) boosts engagement
        if 0.3 <= longing <= 0.6:
            engagement *= 1.15
        elif longing > 0.8:
            engagement *= 0.7  # too much → overwhelmed

        engagement = max(0.05, min(0.95, engagement))

        return {
            "reply_speed": max(0.05, min(0.95, engagement + self._rng.uniform(-0.1, 0.1))),
            "reply_quality": max(0.05, min(0.95, engagement * 0.9 + self._rng.uniform(-0.08, 0.08))),
            "has_reaction": 1.0 if self._rng.random() < engagement * 0.8 else 0.0,
        }


# ─── Signal implementations ───

class LastReplySpeedSignal(Signal):
    """Returns reply speed from the last interaction."""

    def __init__(self):
        self.value = 0.5

    def measure(self) -> float:
        return self.value


class LastReplyQualitySignal(Signal):
    """Returns reply quality from the last interaction."""

    def __init__(self):
        self.value = 0.5

    def measure(self) -> float:
        return self.value


class LastReactionSignal(Signal):
    """Returns whether user reacted to the last message."""

    def __init__(self):
        self.value = 0.0

    def measure(self) -> float:
        return self.value


# ─── Main simulation ───

def run_simulation(days: int = 3):
    print("=" * 70)
    print("PID × Poisson — Integrated Simulation")
    print("=" * 70)
    print()

    # Config: base poisson engine
    config = Config.from_dict({
        "engagement": {
            "lambda_rate": 0.15,
            "check_interval_minutes": 30,
            "growth_factor": 0.08,
            "max_probability": 0.95,
            "min_interval_hours": 1.0,
            "adjudication": {
                "quiet_hours": {"start": "00:00", "end": "08:00"},
                "normal_send_probability": 0.7,
            },
        },
        "persona": {
            "name": "Companion",
            "tone": "warm-brief",
            "context": "You are a caring companion.",
        },
    })

    engine = PoissonEngine(config, seed=42)
    user = SimUser(seed=123)

    # PID: target engagement = 0.5, sweet zone 0.35~0.65
    # Below 0.35 → increase lambda (user needs more attention)
    # Above 0.65 → decrease lambda (give user space)
    # 0.35~0.65 → no adjustment (comfortable distance)
    pid = PIDController(kp=0.15, ki=0.03, kd=0.08, setpoint=0.5, dead_band=0.15)

    # Signals (updated after each send)
    speed_signal = LastReplySpeedSignal()
    quality_signal = LastReplyQualitySignal()
    reaction_signal = LastReactionSignal()
    combined = CombinedSignal(
        (speed_signal, 0.4),
        (quality_signal, 0.4),
        (reaction_signal, 0.2),
    )

    # Simulation state
    base_lambda = 0.15
    start_time = datetime(2026, 5, 19, 8, 0)  # Start at 8am
    tick_minutes = 30
    total_ticks = days * 24 * 60 // tick_minutes

    # Tracking
    sends = []
    holds = []
    misses = []
    lambda_history = []

    print(f"{'Day':<4} {'Time':<6} {'Action':<10} {'Prob':<6} {'Score':<6} {'PID Adj':<8} {'λ':<6} {'Reason'}")
    print("-" * 70)

    for i in range(total_ticks):
        now = start_time + timedelta(minutes=i * tick_minutes)
        hour = now.hour + now.minute / 60.0

        result = engine.tick(now)

        if result.action == Action.HIT_SEND:
            # User responds
            response = user.respond(hour, result.probability)
            speed_signal.value = response["reply_speed"]
            quality_signal.value = response["reply_quality"]
            reaction_signal.value = response["has_reaction"]

            # PID adjusts lambda
            score = combined.measure()
            adj = pid.update(score)
            base_lambda = max(0.05, min(0.4, base_lambda + adj))

            # Update engine config (simulated)
            engine.config.engagement.lambda_rate = base_lambda

            sends.append(now)
            lambda_history.append((now, base_lambda))

            # Print interesting events
            if i % 8 == 0 or abs(adj) > 0.02:  # Print every 4 hours or big changes
                print(f"{now.day:<4} {now.strftime('%H:%M'):<6} {'✅ SEND':<10} {result.probability:<6.0%} {score:<6.2f} {adj:<+8.3f} {base_lambda:<6.3f} {result.reason or ''}")

        elif result.action == Action.HIT_HOLD:
            holds.append(now)
            if i % 16 == 0:
                print(f"{now.day:<4} {now.strftime('%H:%M'):<6} {'⏸️  HOLD':<10} {result.probability:<6.0%} {'':6} {'':8} {'':6} {result.reason or ''}")

        elif result.action == Action.MISS:
            misses.append(now)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Duration:        {days} days")
    print(f"Total ticks:     {total_ticks}")
    print(f"Messages sent:   {len(sends)}")
    print(f"Held back:       {len(holds)}")
    print(f"Missed:          {len(misses)}")
    print()
    print("Lambda trajectory (start → end):")
    if lambda_history:
        print(f"  Start: λ = {lambda_history[0][1]:.3f}")
        print(f"  End:   λ = {lambda_history[-1][1]:.3f}")
        print(f"  Range: {min(l for _, l in lambda_history):.3f} — {max(l for _, l in lambda_history):.3f}")
    print()
    print("What happened:")
    print("  - User engagement fluctuated (mood cycles)")
    print("  - Score in sweet zone (0.35~0.65) → PID does nothing (comfortable)")
    print("  - Score too high (>0.65) → PID reduces λ (give user space)")
    print("  - Score too low (<0.35) → PID increases λ (reach out more)")
    print("  - System self-balances around the sweet zone")
    print()
    print("Without PID: λ stays fixed at 0.15 regardless of user response.")
    print("With PID:    λ adapts — respects the sweet spot between too much and too little.")


if __name__ == "__main__":
    run_simulation(days=3)
