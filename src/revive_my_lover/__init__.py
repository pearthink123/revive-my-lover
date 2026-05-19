"""
revive-my-lover: A probabilistic engagement engine for AI companions.

Three core modules, one pipeline:
1. Poisson process — randomized timing (like "thinking about you")
2. Information gain — is this interaction worth it?
3. Bayesian inference — what's the user doing?

Plus optional modules:
- Optimal stopping — best moment to intervene (standalone use)
- Control — PID controller for adaptive frequency (standalone use)

Quick start:
    from revive_my_lover import PoissonLove

    love = PoissonLove()
    result = love.tick()

    if result.should_send:
        send_message(result.prompt)
"""

from .love import PoissonLove, LoveResult
from .core.engine import PoissonEngine
from .core.config import Config
from .core.models import TickResult, Action, LogEntry
from .control import Signal, CombinedSignal
from .info_gain import InformationGain, SilenceDuration, ConversationFlow
from .optimal_stop import OptimalStop, ThresholdRule
from .bayesian import StateEstimator, State

__version__ = "0.9.0"
__all__ = [
    # Unified API
    "PoissonLove", "LoveResult",
    # Core
    "PoissonEngine", "Config", "TickResult", "Action", "LogEntry",
    # Info Gain
    "InformationGain", "SilenceDuration", "ConversationFlow",
    # Optimal Stop (optional)
    "OptimalStop", "ThresholdRule",
    # Bayesian
    "StateEstimator", "State",
    # Control (optional)
    "Signal", "CombinedSignal",
]
