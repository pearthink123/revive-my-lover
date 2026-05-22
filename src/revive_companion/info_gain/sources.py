"""
Common InfoSource implementations.

Each source returns:
- entropy: how uncertain are we? (0-1)
- resolution_potential: how much can a message resolve? (0-1)

Key insight: High entropy + low resolution = don't send (know little, message won't help)
             High entropy + high resolution = send! (uncertain, message will help)
             Low entropy + any resolution = don't send (already know)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .core import InfoSource


@dataclass
class SilenceDuration(InfoSource):
    """
    How long since user last engaged.

    Short silence: Low entropy (user was just here), low resolution (don't need to check)
    Long silence: High entropy (don't know state), high resolution (message will reveal)
    """

    last_reply_time: datetime | None = None
    now: datetime | None = None

    def _hours(self) -> float:
        if self.last_reply_time is None:
            return 48.0  # Default: assume long silence
        current = self.now or datetime.now()
        return (current - self.last_reply_time).total_seconds() / 3600

    def entropy(self) -> float:
        h = self._hours()
        # Short silence → low uncertainty; long → high uncertainty
        return min(1.0, h / 12.0)  # Saturates at 12 hours

    def resolution_potential(self) -> float:
        h = self._hours()
        if h < 0.5:
            return 0.1  # Just talked, message adds nothing
        elif h < 2:
            return 0.3
        elif h < 8:
            return 0.6
        else:
            return 0.8  # Long silence, message will definitely reveal state


@dataclass
class MessageNovelty(InfoSource):
    """
    Is this message new or repetitive?

    Novel message: High resolution (might get engagement)
    Repeated message: Low resolution (won't get new info)
    """

    recent_messages: list[str] = field(default_factory=list)
    current_message: str = ""

    def entropy(self) -> float:
        # Conversation always has some uncertainty
        return 0.5

    def resolution_potential(self) -> float:
        if not self.recent_messages:
            return 0.8  # First message, high novelty

        is_repeat = any(
            self._similarity(self.current_message, msg) > 0.6 for msg in self.recent_messages[-5:]
        )
        return 0.1 if is_repeat else 0.7

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / max(len(wa | wb), 1)


@dataclass
class ConversationFlow(InfoSource):
    """
    Is the conversation active or dormant?

    Active (user just replied): Low entropy (know they're engaged), low resolution (already chatting)  # noqa: E501
    Dormant (no replies): High entropy (don't know if they'll respond), moderate resolution
    Sending without reply: Low resolution (they're not responding, more messages won't help)
    """

    messages_in_last_hour: int = 0
    user_replied_in_last_hour: bool = False
    my_unanswered_messages: int = 0

    def entropy(self) -> float:
        if self.user_replied_in_last_hour:
            return 0.2  # User is engaged, low uncertainty
        elif self.my_unanswered_messages > 0:
            return 0.4  # Sent messages, waiting
        else:
            return 0.7  # Dormant, high uncertainty

    def resolution_potential(self) -> float:
        if self.user_replied_in_last_hour:
            return 0.3  # Already chatting, just continue
        elif self.my_unanswered_messages >= 3:
            return 0.1  # Sent 3+ without reply, more won't help
        elif self.my_unanswered_messages >= 1:
            return 0.3  # Sent 1-2, might still get reply
        else:
            return 0.7  # Haven't sent anything, message will reveal engagement level


@dataclass
class TimeOfDaySource(InfoSource):
    """
    Time-based uncertainty.

    Night: Low entropy (user is probably sleeping), low resolution
    Evening: Higher resolution (user is more likely free)
    Work hours: Moderate
    """

    hour: float | None = None

    def _get_hour(self) -> float:
        return self.hour or datetime.now().hour + datetime.now().minute / 60

    def entropy(self) -> float:
        h = self._get_hour()
        if 0 <= h < 7:
            return 0.1  # Night, very predictable
        elif 9 <= h < 18:
            return 0.4  # Work hours, somewhat predictable
        else:
            return 0.5  # Evening/late, less predictable

    def resolution_potential(self) -> float:
        h = self._get_hour()
        if 0 <= h < 7:
            return 0.1  # Don't message at night
        elif 18 <= h < 22:
            return 0.8  # Evening, user is free
        elif 7 <= h < 9 or 12 <= h < 14:
            return 0.6  # Morning/lunch, might be free
        else:
            return 0.4  # Work hours, probably busy
