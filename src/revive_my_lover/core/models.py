"""Core data structures for revive-companion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Action(Enum):
    """What the engine decided to do."""

    SKIP = "skip"  # Too early, t < min_interval
    HIT_SEND = "send"  # Hit + adjudication says send
    HIT_HOLD = "hold"  # Hit + adjudication says hold back
    MISS = "miss"  # Dice didn't hit


@dataclass
class TickResult:
    """Result of one engine tick."""

    action: Action
    probability: float  # Current hit probability
    roll: float  # Random roll value
    hour_of_day: float  # Current hour (0-24)
    reason: str | None = None  # Adjudication reason
    prompt: str | None = None  # Prompt to send (if HIT_SEND)
    metadata: dict = field(default_factory=dict)

    @property
    def should_send(self) -> bool:
        return self.action == Action.HIT_SEND


@dataclass
class LogEntry:
    """A single log entry for persistence."""

    timestamp: datetime
    action: Action
    probability: float
    roll: float
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "probability": self.probability,
            "roll": self.roll,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LogEntry:
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=Action(data["action"]),
            probability=data["probability"],
            roll=data["roll"],
            reason=data.get("reason"),
        )
