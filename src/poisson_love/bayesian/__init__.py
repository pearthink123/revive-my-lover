"""
poisson_love.bayesian — Infer user's hidden state, then decide.

Replaces PID with state-based reasoning:
  "What is the user probably doing?" → "Should I send?"

Example:
    from poisson_love.bayesian import StateEstimator

    est = StateEstimator()
    est.update(reply_speed=0.8, hour=14, silence=0.5)
    should, reason = est.should_send()
"""

from .core import StateEstimator, State, SEND_UTILITY

__all__ = ["StateEstimator", "State", "SEND_UTILITY"]
