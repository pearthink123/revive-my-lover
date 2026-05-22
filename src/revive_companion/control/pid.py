"""
PID Controller — pure math, zero dependencies.

A generic Proportional-Integral-Derivative controller.
Takes a measured value, compares to a setpoint, returns an adjustment.

Usage:
    pid = PIDController(kp=0.1, ki=0.01, kd=0.05, setpoint=0.5)
    adjustment = pid.update(current=0.3)  # adjustment to reduce error
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PIDController:
    """
    Generic PID controller.

    Attributes:
        kp: Proportional gain — how aggressively to respond to current error.
        ki: Integral gain — how much to accumulate past errors.
        kd: Derivative gain — how much to dampen rapid changes.
        setpoint: Target value to converge toward.
        output_min: Optional lower bound for output.
        output_max: Optional upper bound for output.
        integral_limit: Optional anti-windup cap on the integral term.

    Example:
        >>> pid = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=22.0)
        >>> pid.update(20.0)  # 2°C below target
        2.15  # adjustment toward target
    """

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    setpoint: float = 0.0
    output_min: float | None = None
    output_max: float | None = None
    integral_limit: float | None = None
    dead_band: float | None = None  # If set, |error| < dead_band → no adjustment

    # Internal state (auto-managed)
    _integral: float = field(default=0.0, repr=False)
    _prev_error: float | None = field(default=None, repr=False)

    def update(self, current: float, dt: float = 1.0) -> float:
        """
        Compute one PID update.

        Args:
            current: Current measured value.
            dt: Time delta since last update (default 1.0).

        Returns:
            Adjustment value. Positive = move toward setpoint.
        """
        error = self.setpoint - current

        # Dead band: if error is small enough, don't adjust
        if self.dead_band is not None and abs(error) < self.dead_band:
            self._prev_error = error
            return 0.0

        # Proportional
        p = self.kp * error

        # Integral (with anti-windup)
        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        i = self.ki * self._integral

        # Derivative
        if self._prev_error is not None and dt > 0:
            d = self.kd * (error - self._prev_error) / dt
        else:
            d = 0.0
        self._prev_error = error

        output = p + i + d

        # Clamp output
        if self.output_min is not None:
            output = max(self.output_min, output)
        if self.output_max is not None:
            output = min(self.output_max, output)

        return output

    def reset(self) -> None:
        """Reset internal state (integral, previous error)."""
        self._integral = 0.0
        self._prev_error = None

    @property
    def error(self) -> float | None:
        """Last computed error, or None if update() hasn't been called."""
        return self._prev_error

    def tune(self, kp: float, ki: float, kd: float) -> None:
        """Update gains without resetting state."""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    @classmethod
    def from_preference(cls, pref) -> PIDController:
        """
        Create PID controller from a UserPreference.

        Args:
            pref: UserPreference instance.

        Returns:
            PIDController configured according to user's style.
        """
        params = pref.to_pid_params()
        return cls(**params)
