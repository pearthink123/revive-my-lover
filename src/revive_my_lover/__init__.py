"""
revive-companion: A probabilistic engagement engine for AI companions.

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

from .bayesian import State, StateEstimator
from .control import CombinedSignal, Signal
from .core.config import Config
from .core.engine import PoissonEngine
from .core.models import Action, LogEntry, TickResult
from .info_gain import ConversationFlow, InformationGain, SilenceDuration
from .love import LoveResult, PoissonLove
from .optimal_stop import OptimalStop, ThresholdRule

__version__ = "0.9.0"
__all__ = [
    # Unified API
    "PoissonLove",
    "LoveResult",
    # Core
    "PoissonEngine",
    "Config",
    "TickResult",
    "Action",
    "LogEntry",
    # Info Gain
    "InformationGain",
    "SilenceDuration",
    "ConversationFlow",
    # Optimal Stop (optional)
    "OptimalStop",
    "ThresholdRule",
    # Bayesian
    "StateEstimator",
    "State",
    # Control (optional)
    "Signal",
    "CombinedSignal",
]
