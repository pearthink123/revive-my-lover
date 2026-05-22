"""
revive_companion.control — PID controller + pluggable signal framework.

This module is independent of the Poisson engine.
Use it for any adaptive control scenario.

Example:
    from revive_companion.control import PIDController, Signal, CombinedSignal, UserPreference

    # Option 1: Manual PID setup
    pid = PIDController(kp=0.1, ki=0.01, kd=0.05, setpoint=0.5)

    # Option 2: User preference → automatic PID setup
    pref = UserPreference(style=Style.RESPECTFUL, on_engaged=Response.MORE)
    pid = PIDController.from_preference(pref)
"""

from .pid import PIDController
from .preference import Response, Style, UserPreference
from .signal import BufferedSignal, CombinedSignal, ConstantSignal, Signal

__all__ = [
    "PIDController",
    "Signal",
    "CombinedSignal",
    "ConstantSignal",
    "BufferedSignal",
    "UserPreference",
    "Style",
    "Response",
]
