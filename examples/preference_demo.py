"""
PID × Poisson × UserPreference — 3 styles compared.

Shows how the same user behavior produces different AI responses
depending on the chosen interaction style.

Run: PYTHONPATH=src python examples/preference_demo.py
"""

import random
from datetime import datetime, timedelta

from revive_companion.control import (
    CombinedSignal,
    PIDController,
    Response,
    Signal,
    Style,
    UserPreference,
)
from revive_companion.core.config import Config
from revive_companion.core.engine import PoissonEngine
from revive_companion.core.models import Action

# ─── Simulated user (same behavior for all 3 styles) ───


class SimUser:
    def __init__(self, seed=42):
        self._rng = random.Random(seed)
        self._mood = 0.6
        self._mood_direction = 1

    def respond(self, hour, longing):
        delta = self._rng.uniform(0.08, 0.2)
        self._mood += self._mood_direction * delta
        if self._mood > 0.95:
            self._mood_direction = -1
            self._mood = 0.95
        elif self._mood < 0.15:
            self._mood_direction = 1
            self._mood = 0.15

        if 0 <= hour < 8:
            tf = 0.1
        elif hour < 10:
            tf = 0.6
        elif hour < 12:
            tf = 0.8
        elif hour < 14:
            tf = 0.5
        elif hour < 17:
            tf = 0.85
        elif hour < 20:
            tf = 0.95
        elif hour < 22:
            tf = 0.9
        else:
            tf = 0.3

        base = self._mood * tf
        eng = base * self._rng.uniform(0.85, 1.15)
        if 0.3 <= longing <= 0.6:
            eng *= 1.15
        elif longing > 0.8:
            eng *= 0.7
        eng = max(0.05, min(0.95, eng))

        return {
            "speed": max(0.05, min(0.95, eng + self._rng.uniform(-0.1, 0.1))),
            "quality": max(0.05, min(0.95, eng * 0.9 + self._rng.uniform(-0.08, 0.08))),
            "reaction": 1.0 if self._rng.random() < eng * 0.8 else 0.0,
        }


class ValSignal(Signal):
    def __init__(self):
        self.value = 0.5

    def measure(self):
        return self.value


def transform_score(score: float, pref: UserPreference) -> float:
    """
    Transform engagement score based on user preference.

    Interpretation B: match user's energy
    - High engagement → feed LOW to PID → PID increases λ (send more)
    - Low engagement → feed HIGH to PID → PID decreases λ (send less)
    """
    low, high = pref.sweet_zone
    setpoint = (low + high) / 2

    if pref.style == Style.PROACTIVE:
        # Always biased toward "need more contact"
        return setpoint - 0.15  # Constant pull toward increasing lambda

    elif pref.style == Style.RESPECTFUL:
        # Invert: high engagement → PID should increase λ
        # score=0.85 → pid_input=0.15 → error=0.35 → PID increases
        # score=0.15 → pid_input=0.85 → error=-0.35 → PID decreases
        return setpoint + (setpoint - score)

    else:  # BALANCED
        if low <= score <= high:
            return setpoint  # Sweet zone → stay put
        # Invert deviations
        return setpoint + (setpoint - score)


def run_style_test(style_name: str, pref: UserPreference, seed: int = 42):
    """Run simulation for one style."""
    config = Config.from_dict(
        {
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
            "persona": {"name": "AI", "tone": "warm", "context": "Caring companion."},
        }
    )

    engine = PoissonEngine(config, seed=seed)
    user = SimUser(seed=seed)
    pid = PIDController.from_preference(pref)

    spd = ValSignal()
    qlt = ValSignal()
    rct = ValSignal()
    combined = CombinedSignal((spd, 0.4), (qlt, 0.4), (rct, 0.2))

    bl = 0.15
    start = datetime(2026, 5, 19, 8, 0)
    total = 3 * 24 * 60 // 30
    lambdas = []
    events = []

    for i in range(total):
        now = start + timedelta(minutes=i * 30)
        hour = now.hour + now.minute / 60.0
        r = engine.tick(now)

        if r.action == Action.HIT_SEND:
            resp = user.respond(hour, r.probability)
            spd.value = resp["speed"]
            qlt.value = resp["quality"]
            rct.value = resp["reaction"]
            raw_score = combined.measure()

            # Transform score based on style preference
            pid_input = transform_score(raw_score, pref)
            adj = pid.update(pid_input)
            bl = max(0.05, min(0.4, bl + adj))
            engine.config.engagement.lambda_rate = bl
            lambdas.append(bl)

            # Determine emoji based on raw engagement score
            low, high = pref.sweet_zone
            if raw_score < low:
                tag = "🫧冷淡" if pref.style == Style.RESPECTFUL else "❤️关怀"
            elif raw_score > high:
                tag = "❤️积极" if pref.style == Style.RESPECTFUL else "❤️热情"
            else:
                tag = "😌甜区"

            events.append((now, raw_score, adj, bl, tag))

    return events, lambdas


def main():
    styles = [
        (
            "A: Proactive (主动型)",
            UserPreference(
                style=Style.PROACTIVE,
                on_engaged=Response.MORE,
                on_disengaged=Response.MORE,
                sweet_zone=(0.35, 0.65),
            ),
        ),
        (
            "B: Respectful (尊重型)",
            UserPreference(
                style=Style.RESPECTFUL,
                on_engaged=Response.MORE,
                on_disengaged=Response.LESS,
                sweet_zone=(0.35, 0.65),
            ),
        ),
        (
            "C: Balanced (平衡型)",
            UserPreference(
                style=Style.BALANCED,
                on_engaged=Response.MAINTAIN,
                on_disengaged=Response.MAINTAIN,
                sweet_zone=(0.35, 0.65),
            ),
        ),
    ]

    print("=" * 70)
    print("UserPreference × PID — Same User, 3 Different AI Styles")
    print("=" * 70)
    print()

    for name, pref in styles:
        print(f"── {name} ──")
        print(f"   {pref}")
        events, lambdas = run_style_test(name, pref)

        for dt, score, adj, lam, tag in events:
            if abs(adj) > 0.02 or adj == 0:
                print(
                    f"   {dt.strftime('%m/%d %H:%M')} score={score:.2f} {tag} adj={adj:+.3f} λ={lam:.3f}"
                )

        if lambdas:
            print(
                f"   λ: {lambdas[0]:.3f} → {lambdas[-1]:.3f} ({min(lambdas):.3f}~{max(lambdas):.3f})"
            )
        print()

    # Comparison table
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    print("Same user behavior, different AI responses:")
    print()
    print("Style          | 用户积极时     | 用户冷淡时     | 整体倾向")
    print("---------------|---------------|---------------|----------------")
    print("A: Proactive   | 多发 ❤️       | 多关心 ❤️‍🩹   | 永远热情")
    print("B: Respectful  | 多发 ❤️       | 少打扰 🫧     | 随用户节奏")
    print("C: Balanced    | 甜区不动 😌   | 甜区不动 😌   | 最少干预")
    print()
    print("用户选哪个，AI 就按哪个来。权力在用户手里。")


if __name__ == "__main__":
    main()
