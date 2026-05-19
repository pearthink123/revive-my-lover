"""
poisson_love.control — PID controller + pluggable signal framework.

This module is independent of the Poisson engine.
Use it for any adaptive control scenario.

Example:
    from poisson_love.control import PIDController, Signal, CombinedSignal

    pid = PIDController(kp=0.1, ki=0.01, kd=0.05, setpoint=0.5)
    adjustment = pid.update(current=0.3)
"""

from .pid import PIDController
from .signal import Signal, CombinedSignal, ConstantSignal, BufferedSignal

__all__ = [
    "PIDController",
    "Signal",
    "CombinedSignal",
    "ConstantSignal",
    "BufferedSignal",
]
