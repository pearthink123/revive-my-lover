"""
revive_my_lover.bayesian — Infer user's hidden state, then decide.

Instead of "engagement high → send more", we ask:
"What is the user probably doing right now?" → then decide accordingly.

Hidden states:
  CHATTING    — user is in active conversation
  IDLE_ONLINE — user is online but not chatting
  BUSY        — user is occupied (work, eating, etc.)
  SLEEPING    — user is asleep
  AWAY        — user is away (no recent activity)
  NEEDING     — user might need a check-in (long silence + uncertain)

Observations:
  reply_speed  — how fast user replied (0-1)
  reply_length — how long the reply was (0-1)
  time_of_day  — hour (0-24)
  silence_hours — hours since last interaction
  has_reaction — did user react to last message

Usage:
    from revive_my_lover.bayesian import StateEstimator, BayesianLearner

    est = StateEstimator()
    learner = BayesianLearner()
    
    # Update with observations
    est.update(reply_speed=0.8, hour=14, silence=0.5)
    state, probs = est.most_likely()
    
    # Learn from observations
    learner.record(state, reply_speed=0.8, hour=14)
    if learner.should_update():
        new_params = learner.learn()
        est.update_params(new_params)
"""

from .core import StateEstimator, State, SEND_UTILITY, TRANSITIONS
from .learner import BayesianLearner, ObservationRecord

__all__ = [
    "StateEstimator",
    "State",
    "SEND_UTILITY",
    "TRANSITIONS",
    "BayesianLearner",
    "ObservationRecord",
]
