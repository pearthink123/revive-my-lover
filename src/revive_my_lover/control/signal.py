"""
Signal — abstract base for pluggable measurement signals.

Users implement their own signals (reply speed, temperature, game score...)
and combine them with CombinedSignal to feed into PIDController.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class Signal(ABC):
    """
    Abstract signal source.

    Subclass this and implement measure() to return a float value.
    The value should be in a range that makes sense for your PID setpoint.

    Example:
        >>> class TemperatureSignal(Signal):
        ...     def measure(self) -> float:
        ...         return read_sensor()
    """

    @abstractmethod
    def measure(self) -> float:
        """Return the current signal value."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class CombinedSignal(Signal):
    """
    Weighted combination of multiple signals.

    Args:
        *signals: Signal instances with optional weights.
        normalize: If True, divide by total weight to get 0-1 range.

    Example:
        >>> combined = CombinedSignal(
        ...     (ReplySpeedSignal(), 0.5),
        ...     (SentimentSignal(), 0.3),
        ...     (ReactionSignal(), 0.2),
        ... )
        >>> combined.measure()
        0.65
    """

    _signals: list[tuple[Signal, float]] = field(default_factory=list)

    def __init__(self, *signals: Signal | tuple[Signal, float], normalize: bool = True):
        """
        Args:
            *signals: Either bare Signal instances (weight=1.0) or (Signal, weight) tuples.
            normalize: Divide by total weight (default True).
        """
        self._signals = []
        for item in signals:
            if isinstance(item, tuple):
                sig, weight = item
            else:
                sig, weight = item, 1.0
            self._signals.append((sig, weight))
        self.normalize = normalize

    def measure(self) -> float:
        """Compute weighted sum of all signals."""
        if not self._signals:
            return 0.0

        total = 0.0
        weight_sum = 0.0
        for sig, weight in self._signals:
            total += sig.measure() * weight
            weight_sum += weight

        if self.normalize and weight_sum > 0:
            return total / weight_sum
        return total

    def add(self, signal: Signal, weight: float = 1.0) -> "CombinedSignal":
        """Add a signal and return self for chaining."""
        self._signals.append((signal, weight))
        return self

    def __repr__(self) -> str:
        names = [f"{sig.__class__.__name__}(w={w})" for sig, w in self._signals]
        return f"CombinedSignal({', '.join(names)})"


@dataclass
class ConstantSignal(Signal):
    """A signal that always returns the same value. Useful for testing."""

    value: float = 0.0

    def measure(self) -> float:
        return self.value


@dataclass
class BufferedSignal(Signal):
    """
    Wraps a signal and caches its value for a number of ticks.
    Useful when measurement is expensive (API call, sensor read).
    """

    source: Signal
    buffer_ticks: int = 1
    _buffer: list[float] = field(default_factory=list, repr=False)
    _tick: int = field(default=0, repr=False)

    def measure(self) -> float:
        self._tick += 1
        if self._tick >= self.buffer_ticks or not self._buffer:
            self._buffer.append(self.source.measure())
            self._tick = 0
        return self._buffer[-1]
