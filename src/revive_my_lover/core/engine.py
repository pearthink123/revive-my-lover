"""
Poisson Longing Engine — the mathematical heart.

Applies a Poisson process to model "missing someone":
- Each tick, a random roll determines if the dice "hits"
- Hit probability grows with each miss/hit-hold (longing accumulates)
- On hit, an adjudication layer decides: send or hold back
- On send, probability resets (longing satisfied)
- On hold, probability keeps growing (longing suppressed)
"""

from __future__ import annotations
import math
import random
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .config import Config
from .models import Action, LogEntry, TickResult


class PoissonEngine:
    """
    Core engine that computes one tick at a time.

    Usage:
        engine = PoissonEngine(config)
        result = engine.tick()  # call every check_interval_minutes
        if result.should_send:
            send_message(result.prompt)
    """

    def __init__(self, config: Config, seed: Optional[int] = None):
        self.config = config
        self.probability = self._base_probability()
        self.last_send_time: Optional[datetime] = None
        self.miss_streak = 0
        self.log: list[LogEntry] = []

        self._rng = random.Random(seed)

    def _base_probability(self) -> float:
        """Base hit probability from Poisson process: P(≥1 event) = 1 - e^(-λt)"""
        lam = self.config.engagement.lambda_rate
        t = self.config.engagement.check_interval_minutes / 60.0
        return 1 - math.exp(-lam * t)

    def tick(self, now: Optional[datetime] = None) -> TickResult:
        """
        Run one tick. Call this every check_interval_minutes.

        Args:
            now: Current time. Defaults to datetime.now().

        Returns:
            TickResult with action, probability, and optional prompt.
        """
        if now is None:
            now = datetime.now()

        hour = now.hour + now.minute / 60.0

        # ── Phase 1: Min interval check ──
        if self.last_send_time is not None:
            elapsed = (now - self.last_send_time).total_seconds() / 3600.0
            if elapsed < self.config.engagement.min_interval_hours:
                result = TickResult(
                    action=Action.SKIP,
                    probability=self.probability,
                    roll=0.0,
                    hour_of_day=hour,
                    reason=f"Too early (t={elapsed:.2f}h < {self.config.engagement.min_interval_hours}h)",
                )
                self._log(now, result)
                return result

        # ── Phase 2: Poisson dice roll ──
        roll = self._rng.random()
        hit = roll < self.probability

        if not hit:
            # Miss — longing grows
            self._grow()
            result = TickResult(
                action=Action.MISS,
                probability=self.probability,
                roll=roll,
                hour_of_day=hour,
            )
            self._log(now, result)
            return result

        # ── Phase 3: Adjudication ──
        should_send, reason = self._adjudicate(hour)

        if should_send:
            # Hit candidate — do NOT mutate state yet.
            # Call confirm_send(now) after all downstream gates pass.
            result = TickResult(
                action=Action.HIT_SEND,
                probability=self.probability,
                roll=roll,
                hour_of_day=hour,
                reason=reason,
                prompt=self._build_prompt(now, self.probability),
            )
        else:
            # Hold — longing suppressed, grows more
            self._grow()
            result = TickResult(
                action=Action.HIT_HOLD,
                probability=self.probability,
                roll=roll,
                hour_of_day=hour,
                reason=reason,
            )

        self._log(now, result)
        return result

    def confirm_send(self, now: Optional[datetime] = None) -> None:
        """Call after all downstream gates approve a HIT_SEND candidate.

        Resets probability and records the send time.
        Without this call, the engine won't know a message was sent.
        """
        if now is None:
            now = datetime.now()
        self.last_send_time = now
        self.miss_streak = 0
        self.probability = self._base_probability()

    def _grow(self):
        """Increase hit probability (longing accumulates)."""
        cfg = self.config.engagement
        self.probability = min(
            self.probability + cfg.growth_factor,
            cfg.max_probability,
        )
        self.miss_streak += 1

    def _adjudicate(self, hour: float) -> tuple[bool, str]:
        """
        Decide whether to actually send on a hit.

        Default logic: time-based rules.
        Override this method for custom adjudication (LLM-based, etc).
        """
        adj = self.config.engagement.adjudication

        # Check quiet hours (night)
        quiet_hours = adj.get("quiet_hours", {})
        night_start = _parse_hour(quiet_hours.get("start", "00:00"))
        night_end = _parse_hour(quiet_hours.get("end", "08:00"))

        if night_start <= hour < night_end:
            return False, f"Night quiet hours ({_fmt_hour(hour)})"

        # Check lunch hours
        lunch = adj.get("lunch_hours", {})
        if lunch:
            lunch_start = _parse_hour(lunch.get("start", "12:00"))
            lunch_end = _parse_hour(lunch.get("end", "14:00"))
            if lunch_start <= hour < lunch_end:
                # Probabilistic — 30% chance to send during lunch
                if self._rng.random() < 0.3:
                    return True, f"Lunch time, user might be free ({_fmt_hour(hour)})"
                return False, f"Lunch time, don't disturb ({_fmt_hour(hour)})"

        # Check late night
        late = adj.get("late_hours", {})
        if late:
            late_start = _parse_hour(late.get("start", "23:00"))
            late_end = _parse_hour(late.get("end", "24:00"))
            if late_start <= hour < late_end:
                if self._rng.random() < 0.3:
                    return True, f"Late night, but user might be up ({_fmt_hour(hour)})"
                return False, f"Late night, let them sleep ({_fmt_hour(hour)})"

        # Normal hours — send probability from config
        send_prob = adj.get("normal_send_probability", 0.7)
        if self._rng.random() < send_prob:
            return True, f"Normal hours, sending ({_fmt_hour(hour)})"
        return False, f"User might be busy ({_fmt_hour(hour)})"

    def _build_prompt(self, now: datetime, hit_probability: float) -> str:
        """Build the prompt/message to send."""
        persona = self.config.persona
        hour = now.hour

        # Context-aware prompt fragments
        if 6 <= hour < 10:
            time_context = "morning"
        elif 11 <= hour < 14:
            time_context = "midday"
        elif 14 <= hour < 18:
            time_context = "afternoon"
        elif 18 <= hour < 22:
            time_context = "evening"
        else:
            time_context = "late night"

        return (
            f"[Poisson Longing Trigger] "
            f"Time: {now.strftime('%H:%M')} ({time_context}), "
            f"Longing: {hit_probability:.0%}, "
            f"Miss streak: {self.miss_streak}"
        )

    def _log(self, now: datetime, result: TickResult):
        """Append to internal log."""
        self.log.append(LogEntry(
            timestamp=now,
            action=result.action,
            probability=result.probability,
            roll=result.roll,
            reason=result.reason,
        ))

    def save_log(self, path: str | Path) -> None:
        """Save log to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([entry.to_dict() for entry in self.log], f,
                      ensure_ascii=False, indent=2)

    def load_log(self, path: str | Path) -> None:
        """Load log from JSON file (for state recovery)."""
        path = Path(path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.log = [LogEntry.from_dict(d) for d in data]
        # Restore last state
        if self.log:
            last = self.log[-1]
            self.probability = last.probability
            if last.action == Action.HIT_SEND:
                self.last_send_time = last.timestamp

    def get_curve(self) -> list[dict]:
        """Export the longing curve for visualization."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "probability": e.probability,
                "action": e.action.value,
            }
            for e in self.log
        ]


def _parse_hour(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if ":" in value:
            h, m = value.split(":")
            return int(h) + int(m) / 60.0
        return float(value)
    return 0.0


def _fmt_hour(h: float) -> str:
    hours = int(h)
    minutes = int((h % 1) * 60)
    return f"{hours:02d}:{minutes:02d}"
