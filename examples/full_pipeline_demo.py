"""
Full Pipeline: Poisson + PID + Information Gain

Three-stage decision:
  Stage 1: Poisson dice → when to consider sending
  Stage 2: Info Gain   → is this interaction worth it?
  Stage 3: PID         → adjust frequency based on user response

Each stage filters the previous:
  Poisson (timing) → InfoGain (value) → Send → PID (adapt)

Run: PYTHONPATH=src python examples/full_pipeline_demo.py
"""

import math
import random
from datetime import datetime, timedelta

from revive_my_lover.core.engine import PoissonEngine
from revive_my_lover.core.config import Config
from revive_my_lover.core.models import Action
from revive_my_lover.control import PIDController, Signal, CombinedSignal, UserPreference, Style, Response
from revive_my_lover.info_gain import InformationGain, SilenceDuration, MessageNovelty, ConversationFlow


# ─── Simulated User ───

class SimUser:
    def __init__(self, seed=42):
        self._rng = random.Random(seed)
        self._mood = 0.6
        self._mood_direction = 1

    def respond(self, hour, longing):
        delta = self._rng.uniform(0.08, 0.2)
        self._mood += self._mood_direction * delta
        if self._mood > 0.95: self._mood_direction = -1; self._mood = 0.95
        elif self._mood < 0.15: self._mood_direction = 1; self._mood = 0.15

        if 0 <= hour < 8: tf = 0.1
        elif hour < 10: tf = 0.6
        elif hour < 12: tf = 0.8
        elif hour < 14: tf = 0.5
        elif hour < 17: tf = 0.85
        elif hour < 20: tf = 0.95
        elif hour < 22: tf = 0.9
        else: tf = 0.3

        base = self._mood * tf
        eng = base * self._rng.uniform(0.85, 1.15)
        if 0.3 <= longing <= 0.6: eng *= 1.15
        elif longing > 0.8: eng *= 0.7
        eng = max(0.05, min(0.95, eng))

        return {
            "speed": max(0.05, min(0.95, eng + self._rng.uniform(-0.1, 0.1))),
            "quality": max(0.05, min(0.95, eng * 0.9 + self._rng.uniform(-0.08, 0.08))),
            "reaction": 1.0 if self._rng.random() < eng * 0.8 else 0.0,
        }


class ValSignal(Signal):
    def __init__(self): self.value = 0.5
    def measure(self): return self.value


# ─── Main Simulation ───

def run():
    print("=" * 72)
    print("  FULL PIPELINE: Poisson → InfoGain → PID")
    print("=" * 72)
    print()
    print("  Stage 1: Poisson dice → 'should I consider sending now?'")
    print("  Stage 2: Info Gain   → 'is this interaction worth it?'")
    print("  Stage 3: PID         → 'how should I adjust my frequency?'")
    print()

    # ── Config ──
    config = Config.from_dict({
        "engagement": {
            "lambda_rate": 0.15, "check_interval_minutes": 30,
            "growth_factor": 0.08, "max_probability": 0.95,
            "min_interval_hours": 1.0,
            "adjudication": {
                "quiet_hours": {"start": "00:00", "end": "08:00"},
                "normal_send_probability": 0.7,
            },
        },
        "persona": {"name": "AI", "tone": "warm", "context": "Caring companion."},
    })

    engine = PoissonEngine(config, seed=42)
    user = SimUser(seed=42)

    # PID: respectful style
    pref = UserPreference(
        style=Style.RESPECTFUL,
        on_engaged=Response.MORE,
        on_disengaged=Response.LESS,
        sweet_zone=(0.35, 0.65),
    )
    pid = PIDController.from_preference(pref)

    # Signals for PID
    spd = ValSignal(); qlt = ValSignal(); rct = ValSignal()
    combined_signal = CombinedSignal((spd, 0.4), (qlt, 0.4), (rct, 0.2))

    # Info Gain
    info_gain = InformationGain(threshold=0.20, decay=0.8)

    # State tracking
    base_lambda = 0.15
    start = datetime(2026, 5, 19, 8, 0)
    total_ticks = 3 * 24 * 60 // 30  # 3 days

    last_user_reply = start - timedelta(hours=2)  # Assume replied 2h ago
    my_unanswered = 0
    recent_messages = []
    sends = 0
    skips_poisson = 0
    skips_infogain = 0
    actual_sends = 0

    print(f"{'Day':<4} {'Time':<6} {'Stage':<28} {'λ':<6} {'Detail'}")
    print("-" * 72)

    for i in range(total_ticks):
        now = start + timedelta(minutes=i * 30)
        hour = now.hour + now.minute / 60.0

        # ── Stage 1: Poisson Dice ──
        result = engine.tick(now)

        if result.action == Action.HIT_HOLD:
            skips_poisson += 1
            continue
        elif result.action == Action.MISS:
            skips_poisson += 1
            continue
        elif result.action == Action.SKIP:
            skips_poisson += 1
            continue

        # HIT_SEND → consider sending, but check info gain first

        # ── Stage 2: Information Gain ──
        silence_src = SilenceDuration(last_reply_time=last_user_reply, now=now)
        flow_src = ConversationFlow(
            my_unanswered_messages=my_unanswered,
            user_replied_in_last_hour=(now - last_user_reply).total_seconds() < 3600,
        )

        info_gain.sources = [silence_src, flow_src]
        info_gain._send_count = my_unanswered  # Track consecutive sends

        ig_result = info_gain.evaluate()

        if not ig_result.worth_sending:
            skips_infogain += 1
            if i % 16 == 0:
                print(f"{now.day:<4} {now.strftime('%H:%M'):<6} "
                      f"{'🎲→📊 SKIP (info)':<28} {base_lambda:<6.3f} "
                      f"{ig_result.reason}")
            continue

        # ── Stage 3: Send & PID Update ──
        resp = user.respond(hour, result.probability)
        spd.value = resp["speed"]; qlt.value = resp["quality"]; rct.value = resp["reaction"]
        raw_score = combined_signal.measure()

        # Transform score based on preference (B: match user's energy)
        low, high = pref.sweet_zone
        setpoint = (low + high) / 2
        if raw_score < low:
            pid_input = setpoint + 0.2  # User quiet → decrease λ
        elif raw_score > high:
            pid_input = setpoint - 0.2  # User engaged → increase λ
        else:
            pid_input = setpoint  # Sweet zone

        adj = pid.update(pid_input)
        base_lambda = max(0.05, min(0.4, base_lambda + adj))
        engine.config.engagement.lambda_rate = base_lambda

        # Update state
        last_user_reply = now  # Simulate user responding
        my_unanswered = 0
        actual_sends += 1
        info_gain.on_send()

        # Emoji
        if adj == 0:
            tag = "😌甜区"
        elif adj > 0:
            tag = "❤️加触达"
        else:
            tag = "🫧给空间"

        print(f"{now.day:<4} {now.strftime('%H:%M'):<6} "
              f"{'🎲→📊→🎯 SEND ' + tag:<28} {base_lambda:<6.3f} "
              f"score={raw_score:.2f} adj={adj:+.3f}")

    # ── Summary ──
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Duration:          3 days ({total_ticks} ticks)")
    print(f"  Poisson hits:      {actual_sends + skips_infogain}")
    print(f"  Skipped (Poisson): {skips_poisson} (dice didn't hit / night)")
    print(f"  Skipped (InfoGain):{skips_infogain} (not worth sending)")
    print(f"  Actual sends:      {actual_sends}")
    print(f"  Final λ:           {base_lambda:.3f}")
    print()
    print("  Pipeline: 144 ticks → ~45 Poisson hits → ~20 pass InfoGain → ~18 sent")
    print()
    print("  Each module filters:")
    print("    🎲 Poisson:  'Is it time?' (randomized timing)")
    print("    📊 InfoGain: 'Is it worth it?' (entropy × resolution)")
    print("    🎯 PID:      'How should I adapt?' (engagement feedback)")


if __name__ == "__main__":
    run()
