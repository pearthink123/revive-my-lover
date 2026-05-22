"""
Optimal Stop — "Is NOW the best moment to act?"

Implements optimal stopping theory for timing interventions.

Core problem: You observe a signal at each step. Should you ACT now
or WAIT for a potentially better moment?

Three strategies:
1. Threshold Rule: Act when signal > threshold (threshold decreases over time)
2. Secretary Rule: Observe first 37%, then pick the first best
3. Custom: User-defined stopping rule

Usage:
    from revive_companion.optimal_stop import OptimalStop, ThresholdRule

    stop = OptimalStop(rule=ThresholdRule(horizon=10, value_range=(0, 1)))
    for t in range(10):
        signal = observe()
        decision = stop.decide(signal, step=t)
        if decision.should_stop:
            act()
            break
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class Decision(Enum):
    """Stop or continue?"""

    STOP = "stop"
    CONTINUE = "continue"


@dataclass
class StopResult:
    """Result of a stopping decision."""

    decision: Decision
    signal: float  # Current signal value
    threshold: float  # Current threshold
    step: int  # Current step (0-indexed)
    steps_remaining: int  # Steps left
    reason: str = ""

    @property
    def should_stop(self) -> bool:
        return self.decision == Decision.STOP

    def __repr__(self) -> str:
        icon = "🛑 STOP" if self.should_stop else "⏳ WAIT"
        return f"{icon} signal={self.signal:.3f} threshold={self.threshold:.3f} step={self.step}"


class StoppingRule(ABC):
    """Abstract stopping rule."""

    @abstractmethod
    def threshold(self, step: int, horizon: int) -> float:
        """Compute threshold at given step."""
        ...

    @abstractmethod
    def decide(self, signal: float, step: int, horizon: int) -> Decision:
        """Decide whether to stop or continue."""
        ...


@dataclass
class ThresholdRule(StoppingRule):
    """
    Simple threshold rule.

    Threshold decreases linearly from max_value to min_value as steps pass.
    When signal > threshold → STOP.

    This models increasing urgency: as time runs out, you become more willing
    to accept lower signal values.

    Attributes:
        horizon: Total number of steps.
        value_range: (min, max) possible signal values.
        urgency: How fast threshold drops (0=linear, 1=aggressive).
        observe_steps: Number of initial steps to just observe (don't stop).
    """

    horizon: int = 10
    value_range: tuple[float, float] = (0.0, 1.0)
    urgency: float = 0.5
    observe_steps: int = 0

    def threshold(self, step: int, horizon: int) -> float:
        """Threshold decreases as time runs out."""
        lo, hi = self.value_range
        progress = step / max(horizon - 1, 1)  # 0 to 1

        # Urgency curve: threshold drops faster with higher urgency
        if self.urgency <= 0.5:
            # Linear
            drop = progress
        else:
            # Exponential-ish: drops slowly at first, then fast
            drop = progress ** (1 / (2 * self.urgency))

        return hi - (hi - lo) * drop

    def decide(self, signal: float, step: int, horizon: int) -> Decision:
        if step < self.observe_steps:
            return Decision.CONTINUE
        thresh = self.threshold(step, horizon)
        return Decision.STOP if signal >= thresh else Decision.CONTINUE


@dataclass
class SecretaryRule(StoppingRule):
    """
    Classic secretary problem (37% rule).

    Observe the first ~37% of candidates without stopping.
    After that, stop at the first candidate better than all observed.

    Works best when values are ordinal (ranking matters more than magnitude).

    Attributes:
        horizon: Total number of steps.
    """

    horizon: int = 10

    def threshold(self, step: int, horizon: int) -> float:
        """Not applicable for secretary rule — uses best-so-far."""
        return 0.0  # Placeholder

    def decide(self, signal: float, step: int, horizon: int) -> Decision:
        # Phase 1: Just observe
        observe_until = max(1, int(horizon * 0.37))

        if step < observe_until:
            if not hasattr(self, "_best_observed"):
                self._best_observed = signal
            else:
                self._best_observed = max(self._best_observed, signal)
            return Decision.CONTINUE

        # Phase 2: Stop at first better than all observed
        if signal > self._best_observed:
            return Decision.STOP

        # Last step: must stop
        if step >= horizon - 1:
            return Decision.STOP

        return Decision.CONTINUE

    def reason(self, signal: float, step: int, horizon: int) -> str:
        """Human-readable reason for the decision."""
        observe_until = max(1, int(horizon * 0.37))
        if step < observe_until:
            return f"观察阶段（{step + 1}/{observe_until}），记录最佳={self._best_observed:.3f}"
        elif signal > self._best_observed:
            return f"信号 {signal:.3f} > 之前最佳 {self._best_observed:.3f}，行动！"
        else:
            return f"信号 {signal:.3f} ≤ 之前最佳 {self._best_observed:.3f}，继续等"

    def reset(self):
        """Reset for a new run."""
        self._best_observed = float("-inf")


@dataclass
class OptimalStop:
    """
    Optimal stopping controller.

    Wraps a stopping rule and tracks state across observations.

    Attributes:
        rule: The stopping strategy to use.
        history: Past signal values.
        stopped: Whether we've already stopped.

    Example:
        >>> stop = OptimalStop(rule=ThresholdRule(horizon=8, value_range=(0, 1)))
        >>> for t in range(8):
        ...     signal = get_user_activity()
        ...     result = stop.decide(signal, step=t)
        ...     if result.should_stop:
        ...         send_message()
        ...         break
    """

    rule: StoppingRule = field(default_factory=lambda: ThresholdRule())
    history: list[float] = field(default_factory=list)
    stopped: bool = False
    stopped_at: int | None = None
    stopped_signal: float | None = None

    def decide(self, signal: float, step: int, horizon: int | None = None) -> StopResult:
        """
        Make a stopping decision.

        Args:
            signal: Current observation value.
            step: Current step (0-indexed).
            horizon: Total steps (overrides rule's horizon if provided).

        Returns:
            StopResult with decision and context.
        """
        if self.stopped:
            return StopResult(
                decision=Decision.STOP,
                signal=signal,
                threshold=0,
                step=step,
                steps_remaining=0,
                reason="Already stopped",
            )

        h = horizon or getattr(self.rule, "horizon", 10)
        self.history.append(signal)

        decision = self.rule.decide(signal, step, h)
        thresh = self.rule.threshold(step, h)

        # Use rule's custom reason if available
        if hasattr(self.rule, "reason"):
            reason = self.rule.reason(signal, step, h)
        elif decision == Decision.STOP:
            self.stopped = True
            self.stopped_at = step
            self.stopped_signal = signal
            reason = f"Signal {signal:.3f} ≥ threshold {thresh:.3f}"
        else:
            remaining = h - step - 1
            reason = f"Signal {signal:.3f} < threshold {thresh:.3f}, {remaining} steps left"

        return StopResult(
            decision=decision,
            signal=signal,
            threshold=thresh,
            step=step,
            steps_remaining=h - step - 1,
            reason=reason,
        )

    def reset(self) -> None:
        """Reset for a new run."""
        self.history = []
        self.stopped = False
        self.stopped_at = None
        self.stopped_signal = None
        if hasattr(self.rule, "reset"):
            self.rule.reset()

    @property
    def best_observed(self) -> float | None:
        """Best signal observed so far."""
        return max(self.history) if self.history else None
