"""
Common signal sources for optimal stopping in engagement scenarios.

Each returns a signal value (0-1) representing "how good is this moment to act?"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class UserActivitySignal:
    """
    Signal based on user's recent activity.

    Higher = user is more likely to be responsive.
    """

    last_seen: Optional[datetime] = None
    last_seen_minutes_ago: Optional[float] = None  # Alternative to last_seen
    messages_today: int = 0
    hour: Optional[float] = None
    now: Optional[datetime] = None

    def value(self) -> float:
        current = self.now or datetime.now()
        h = self.hour or (current.hour + current.minute / 60)

        # Time of day factor
        if 0 <= h < 7:
            time_factor = 0.1
        elif 7 <= h < 9:
            time_factor = 0.5
        elif 9 <= h < 12:
            time_factor = 0.7
        elif 12 <= h < 14:
            time_factor = 0.6
        elif 14 <= h < 17:
            time_factor = 0.7
        elif 17 <= h < 22:
            time_factor = 0.9
        else:
            time_factor = 0.4

        # Recency factor
        if self.last_seen_minutes_ago is not None:
            minutes_ago = self.last_seen_minutes_ago
        elif self.last_seen:
            minutes_ago = (current - self.last_seen).total_seconds() / 60
        else:
            minutes_ago = 999

        if minutes_ago < 5:
            recency = 1.0
        elif minutes_ago < 30:
            recency = 0.8
        elif minutes_ago < 120:
            recency = 0.5
        else:
            recency = 0.2

        # Activity factor
        activity = min(1.0, self.messages_today / 10)

        return time_factor * 0.4 + recency * 0.4 + activity * 0.2


@dataclass
class ConversationPotential:
    """
    Signal based on conversation potential.

    Higher = this moment has more potential for good interaction.

    Factors:
    - How long since last real conversation
    - Whether there's a new topic to discuss
    - User's typical availability at this time
    """

    hours_since_conversation: float = 24.0
    has_new_topic: bool = False
    typical_availability: float = 0.5  # 0-1, based on historical data

    def value(self) -> float:
        # Longer silence → higher potential (up to a point)
        silence_factor = min(1.0, self.hours_since_conversation / 12) * 0.8
        if self.hours_since_conversation > 48:
            silence_factor *= 0.7  # Too long, might be awkward

        # New topic bonus
        topic_bonus = 0.2 if self.has_new_topic else 0.0

        # Availability
        avail = self.typical_availability

        return min(1.0, silence_factor * 0.4 + avail * 0.4 + topic_bonus)


@dataclass
class UrgencySignal:
    """
    Signal based on urgency/importance.

    Higher = more important to act now.

    Examples:
    - User posted something (time-sensitive reaction)
    - User's birthday/special day
    - Breaking news user cares about
    """

    time_sensitive_event: bool = False
    event_freshness_hours: float = 24.0  # How fresh is the event
    emotional_weight: float = 0.5  # 0=low, 1=high emotional significance

    def value(self) -> float:
        if not self.time_sensitive_event:
            return 0.2  # Baseline

        # Events lose value over time
        freshness = max(0, 1.0 - self.event_freshness_hours / 24)

        return min(1.0, freshness * 0.6 + self.emotional_weight * 0.4)
