"""
revive_my_lover.optimal_stop — Best moment to intervene.

Uses optimal stopping theory to decide when to act.

Example:
    from revive_my_lover.optimal_stop import OptimalStop, ThresholdRule

    stop = OptimalStop(rule=ThresholdRule(horizon=8, value_range=(0, 1)))
    for t in range(8):
        signal = observe_activity()
        result = stop.decide(signal, step=t)
        if result.should_stop:
            send_message()
            break
"""

from .core import OptimalStop, StopResult, Decision, StoppingRule, ThresholdRule, SecretaryRule
from .signals import UserActivitySignal, ConversationPotential, UrgencySignal

__all__ = [
    "OptimalStop",
    "StopResult",
    "Decision",
    "StoppingRule",
    "ThresholdRule",
    "SecretaryRule",
    "UserActivitySignal",
    "ConversationPotential",
    "UrgencySignal",
]
