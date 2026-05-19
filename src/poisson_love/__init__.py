"""
poisson-love: Math models that make AI engagement feel human.

Four models, one pipeline:
1. Poisson process — randomized timing (like "thinking about you")
2. Information gain — is this interaction worth it?
3. PID controller — adaptive frequency based on user feedback
4. Optimal stopping — best moment to intervene

Quick start:
    from poisson_love import PoissonLove, UserPreference, Style

    love = PoissonLove(preference=UserPreference(style=Style.RESPECTFUL))
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

__version__ = "0.4.0"
__all__ = [
    # Unified API
    "PoissonLove", "LoveResult",
    # Core
    "PoissonEngine", "Config", "TickResult", "Action", "LogEntry",
    # Info Gain
    "InformationGain", "SilenceDuration", "ConversationFlow",
    # Optimal Stop
    "OptimalStop", "ThresholdRule",
    # Bayesian
    "StateEstimator", "State",
    # Control (kept for standalone use)
    "Signal", "CombinedSignal",
]
