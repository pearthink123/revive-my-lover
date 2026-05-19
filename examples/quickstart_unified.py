"""
Quick Start — 5 lines to use PoissonLove (Bayesian version).

Run: PYTHONPATH=src python examples/quickstart_unified.py
"""

from poisson_love import PoissonLove

# Create engine — no complex config needed
love = PoissonLove()

# Simulate a day
from datetime import datetime, timedelta
import random

rng = random.Random(42)
now = datetime(2026, 5, 19, 8, 0)

print("PoissonLove v0.4 — Bayesian State Estimation")
print("=" * 55)

for i in range(48):  # 24 hours, 30-min ticks
    result = love.tick(now)

    if result.should_send:
        # Simulate user response
        reply_speed = rng.uniform(0.2, 0.9)
        reply_length = rng.uniform(0.1, 0.8)

        love.record_send()

        if rng.random() < 0.6:  # 60% chance user replies
            love.record_reply(reply_speed=reply_speed, reply_length=reply_length)

        print(f"  {now.strftime('%m/%d %H:%M')} → SEND "
              f"state={result.user_state} utility={result.send_utility:.2f} "
              f"confidence={result.state_confidence:.0%}")

    now += timedelta(minutes=30)

print(f"\nTotal sends: done.")
