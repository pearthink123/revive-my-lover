"""Core data structures for revive-companion (pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Action(Enum):
    """What the engine decided to do."""

    SKIP = "skip"  # Too early, t < min_interval
    HIT_SEND = "send"  # Hit + adjudication says send
    HIT_HOLD = "hold"  # Hit + adjudication says hold back
    MISS = "miss"  # Dice didn't hit


class TickResult(BaseModel):
    """Result of one engine tick."""

    action: Action
    probability: float = Field(description="Current hit probability")
    roll: float = Field(description="Random roll value")
    hour_of_day: float = Field(ge=0, le=24, description="Current hour (0-24)")
    reason: str | None = None
    prompt: str | None = None
    metadata: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @property
    def should_send(self) -> bool:
        return self.action == Action.HIT_SEND


class LogEntry(BaseModel):
    """A single log entry for persistence."""

    timestamp: datetime
    action: Action
    probability: float
    roll: float
    reason: str | None = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> LogEntry:
        return cls.model_validate(data)
