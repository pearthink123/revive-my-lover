"""
revive_my_lover.info_gain — Information theory for AI engagement.

Uses entropy × resolution_potential to answer: "Is this interaction worth it?"

Example:
    from revive_my_lover.info_gain import InformationGain, SilenceDuration

    gain = InformationGain(
        sources=[SilenceDuration(last_reply_time=some_time)],
        threshold=0.25,
    )
    result = gain.evaluate()
    if result.worth_sending:
        send_message()
"""

from .core import InfoSource, InformationGain, GainResult
from .sources import (
    SilenceDuration,
    MessageNovelty,
    ConversationFlow,
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
