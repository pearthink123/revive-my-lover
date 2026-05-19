"""
Information Gain — "Is this interaction worth it?"

Core insight: Information gain = uncertainty reduction.
- If you already know user's state → low gain → don't send
- If you're uncertain → high gain → send to learn

Key: resolution_rate depends on how much NEW information a message can provide,
not just the entropy of the current state.

Usage:
    from revive_my_lover.info_gain import InformationGain, InfoSource

    gain = InformationGain(sources=[...], threshold=0.25)
    result = gain.evaluate()
    if result.worth_sending:
        send_message()
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InfoSource(ABC):
    """
    Abstract information source.

    Returns two values:
    - entropy: How uncertain are we about the current state? (0-1 normalized)
    - resolution_potential: How much can a message resolve this uncertainty? (0-1)

    Example:
        >>> class MySource(InfoSource):
        ...     def entropy(self) -> float:
        ...         return 0.3  # 30% uncertain
        ...     def resolution_potential(self) -> float:
        ...         return 0.1  # Message can only resolve 10% of that
    """

    @abstractmethod
    def entropy(self) -> float:
        """How uncertain are we? (0 = fully certain, 1 = max uncertainty)."""
        ...

    @abstractmethod
    def resolution_potential(self) -> float:
        """
        How much can a message resolve this uncertainty? (0-1)

        High = message will likely get useful response
        Low = message won't tell us much (we already know, or user won't respond)
        """
        ...


@dataclass
class GainResult:
    """Result of an information gain evaluation."""

    total_entropy: float           # Total uncertainty across all sources
    total_resolution: float        # How much can be resolved
    gain: float                    # Bits of information gained
    gain_ratio: float              # gain / total_entropy (0-1)
    worth_sending: bool            # True if gain > threshold
    reason: str = ""

    def __repr__(self) -> str:
        return f"GainResult(gain={self.gain:.3f}, ratio={self.gain_ratio:.1%}, send={self.worth_sending})"


@dataclass
class InformationGain:
    """
    Evaluate whether a message is worth sending.

    Combines multiple information sources, each providing:
    - entropy: current uncertainty
    - resolution_potential: how much a message can resolve

    Total gain = Σ(entropy_i × resolution_i)

    Attributes:
        sources: List of InfoSource instances.
        threshold: Minimum gain ratio to send (0-1).
        min_gain: Minimum absolute gain in bits.
        decay: Decay factor per consecutive unsolicited message.

    Example:
        >>> gain = InformationGain(sources=[...], threshold=0.25)
        >>> result = gain.evaluate()
        >>> if result.worth_sending:
        ...     send()
    """

    sources: list[InfoSource] = field(default_factory=list)
    threshold: float = 0.25       # Min gain ratio (25%)
    min_gain: float = 0.1         # Min absolute gain
    decay: float = 0.85           # Decay per consecutive message

    # Internal state
    _send_count: int = field(default=0, repr=False)

    def evaluate(self) -> GainResult:
        """Evaluate whether to send a message."""
        if not self.sources:
            return GainResult(0, 0, 0, 0, False, "No sources configured")

        total_entropy = 0.0
        total_resolution = 0.0

        for source in self.sources:
            e = source.entropy()
            r = source.resolution_potential()
            total_entropy += e
            total_resolution += e * r  # Weighted: entropy × how much can be resolved

        # Apply decay for consecutive messages
        decay_factor = self.decay ** self._send_count
        gain = total_resolution * decay_factor

        # Gain ratio
        gain_ratio = gain / total_entropy if total_entropy > 0 else 0.0

        # Decision
        worth_sending = gain_ratio >= self.threshold and gain >= self.min_gain

        # Reason
        if total_entropy < 0.05:
            reason = "Very low uncertainty — already know user's state"
        elif gain < self.min_gain:
            reason = f"Gain too low ({gain:.3f} < {self.min_gain} bits)"
        elif gain_ratio < self.threshold:
            reason = f"Gain ratio too low ({gain_ratio:.1%} < {self.threshold:.0%})"
        else:
            reason = f"Good gain ({gain:.3f} bits, {gain_ratio:.1%})"

        return GainResult(
            total_entropy=total_entropy,
            total_resolution=total_resolution,
            gain=gain,
            gain_ratio=gain_ratio,
            worth_sending=worth_sending,
            reason=reason,
        )

    def on_send(self) -> None:
        """Call when a message is sent."""
        self._send_count += 1

    def on_receive(self) -> None:
        """Call when user responds (resets decay)."""
        self._send_count = 0

    def reset(self) -> None:
        """Reset state."""
        self._send_count = 0

    def add_source(self, source: InfoSource) -> "InformationGain":
        self.sources.append(source)
        return self
