"""
UserPreference — let users choose their interaction style.

Instead of hardcoding PID behavior, users define their preferences
and the system adapts accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Style(Enum):
    """Interaction style."""

    PROACTIVE = "proactive"  # Always reaching out, warm and persistent
    RESPECTFUL = "respectful"  # Match user's energy, give space when needed
    BALANCED = "balanced"  # Sweet zone based, only adjust when out of range


class Response(Enum):
    """What to do when engagement is high/low."""

    MORE = "more"  # Increase contact
    LESS = "less"  # Decrease contact
    MAINTAIN = "maintain"  # Keep current rate


@dataclass
class UserPreference:
    """
    User's interaction preference configuration.

    Attributes:
        style: Overall interaction style (proactive/respectful/balanced).
        on_engaged: What to do when user is highly engaged.
        on_disengaged: What to do when user is disengaged.
        sweet_zone: (low, high) engagement range where system stays put.
        max_daily: Maximum messages per day (None = unlimited).
        quiet_hours: (start, end) time range for no messages.
        engagement_threshold_low: Below this = disengaged (default 0.35).
        engagement_threshold_high: Above this = engaged (default 0.65).

    Example:
        >>> pref = UserPreference(
        ...     style=Style.RESPECTFUL,
        ...     on_engaged=Response.MORE,
        ...     on_disengaged=Response.LESS,
        ...     sweet_zone=(0.35, 0.65),
        ...     max_daily=8,
        ...     quiet_hours=("00:00", "08:00"),
        ... )
    """

    style: Style = Style.RESPECTFUL
    on_engaged: Response = Response.MORE
    on_disengaged: Response = Response.LESS
    sweet_zone: tuple[float, float] = (0.35, 0.65)
    max_daily: int | None = None
    quiet_hours: tuple[str, str] | None = None
    engagement_threshold_low: float = 0.35
    engagement_threshold_high: float = 0.65

    def to_pid_params(self) -> dict:
        """
        Convert preference to PID parameters.

        Returns dict with kp, ki, kd, setpoint, dead_band based on style.
        """
        low, high = self.sweet_zone
        setpoint = (low + high) / 2
        dead_band = (high - low) / 2

        if self.style == Style.PROACTIVE:
            # Aggressive: always try to increase engagement
            return {
                "kp": 0.20,
                "ki": 0.05,
                "kd": 0.10,
                "setpoint": setpoint,
                "dead_band": None,  # No dead band, always adjusting
                "output_min": -0.15,
                "output_max": 0.20,  # Bias toward increasing
            }

        elif self.style == Style.RESPECTFUL:
            # Responsive: match user's energy
            return {
                "kp": 0.15,
                "ki": 0.03,
                "kd": 0.08,
                "setpoint": setpoint,
                "dead_band": dead_band,
                "output_min": -0.15,
                "output_max": 0.15,
            }

        else:  # BALANCED
            # Conservative: only adjust when clearly out of range
            return {
                "kp": 0.10,
                "ki": 0.02,
                "kd": 0.05,
                "setpoint": setpoint,
                "dead_band": dead_band * 1.2,  # Wider dead band
                "output_min": -0.10,
                "output_max": 0.10,
            }

    def adjust_direction(self, engagement: float) -> str:
        """
        Given an engagement score, what should the system do?

        Returns: "increase", "decrease", or "maintain"
        """
        low, high = self.sweet_zone

        if engagement < low:
            # Disengaged
            return (
                "increase"
                if self.on_disengaged == Response.MORE
                else "decrease"
                if self.on_disengaged == Response.LESS
                else "maintain"
            )

        elif engagement > high:
            # Highly engaged
            return (
                "increase"
                if self.on_engaged == Response.MORE
                else "decrease"
                if self.on_engaged == Response.LESS
                else "maintain"
            )

        else:
            # Sweet zone
            return "maintain"

    @classmethod
    def from_dict(cls, data: dict) -> UserPreference:
        """Create from dict (e.g., from YAML config)."""
        style = Style(data.get("style", "respectful"))
        on_engaged = Response(data.get("on_engaged", "more"))
        on_disengaged = Response(data.get("on_disengaged", "less"))

        sweet = data.get("sweet_zone", [0.35, 0.65])
        quiet = data.get("quiet_hours")
        if quiet:
            quiet = tuple(quiet)

        return cls(
            style=style,
            on_engaged=on_engaged,
            on_disengaged=on_disengaged,
            sweet_zone=tuple(sweet),
            max_daily=data.get("max_daily"),
            quiet_hours=quiet,
            engagement_threshold_low=data.get("engagement_threshold_low", sweet[0]),
            engagement_threshold_high=data.get("engagement_threshold_high", sweet[1]),
        )

    def __repr__(self) -> str:
        return (
            f"UserPreference("
            f"style={self.style.value}, "
            f"on_engaged={self.on_engaged.value}, "
            f"on_disengaged={self.on_disengaged.value}, "
            f"sweet_zone={self.sweet_zone})"
        )
