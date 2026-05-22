"""
revive_companion.info_gain — Information theory for AI engagement.

Uses entropy × resolution_potential to answer: "Is this interaction worth it?"

Example:
    from revive_companion.info_gain import InformationGain, SilenceDuration

    gain = InformationGain(
        sources=[SilenceDuration(last_reply_time=some_time)],
        threshold=0.25,
    )
    result = gain.evaluate()
    if result.worth_sending:
        send_message()
"""

from .core import GainResult, InformationGain, InfoSource
from .sources import (
    ConversationFlow,
    MessageNovelty,
    SilenceDuration,
    TimeOfDaySource,
)

__all__ = [
    "InfoSource",
    "InformationGain",
    "GainResult",
    "SilenceDuration",
    "MessageNovelty",
    "ConversationFlow",
    "TimeOfDaySource",
]
