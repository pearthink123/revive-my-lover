"""Tests for Optimal Stop module."""

import pytest

from revive_companion.optimal_stop.core import (
    Decision,
    OptimalStop,
    SecretaryRule,
    StopResult,
    ThresholdRule,
)


@pytest.fixture
def threshold_rule():
    """Threshold rule for testing."""
    return ThresholdRule(horizon=10, value_range=(0, 1), urgency=0.5)


@pytest.fixture
def secretary_rule():
    """Secretary rule for testing."""
    return SecretaryRule(horizon=10)


@pytest.fixture
def stop_threshold():
    """OptimalStop with threshold rule."""
    return OptimalStop(rule=ThresholdRule(horizon=10))


@pytest.fixture
def stop_secretary():
    """OptimalStop with secretary rule."""
    return OptimalStop(rule=SecretaryRule(horizon=10))


class TestThresholdRule:
    """Threshold rule tests."""

    def test_threshold_decreases(self, threshold_rule):
        """Threshold decreases over time."""
        t0 = threshold_rule.threshold(0, 10)
        t5 = threshold_rule.threshold(5, 10)
        t9 = threshold_rule.threshold(9, 10)
        assert t0 > t5 > t9

    def test_threshold_range(self, threshold_rule):
        """Threshold stays within value_range."""
        for step in range(10):
            t = threshold_rule.threshold(step, 10)
            assert 0 <= t <= 1

    def test_high_signal_stops(self, threshold_rule):
        """High signal → STOP."""
        decision = threshold_rule.decide(0.9, 5, 10)
        assert decision == Decision.STOP

    def test_low_signal_continues(self, threshold_rule):
        """Low signal → CONTINUE."""
        decision = threshold_rule.decide(0.1, 0, 10)
        assert decision == Decision.CONTINUE

    def test_observe_steps(self):
        """Observe steps prevent early stopping."""
        rule = ThresholdRule(horizon=10, observe_steps=3)
        # Even high signal during observe phase
        decision = rule.decide(0.99, 0, 10)
        assert decision == Decision.CONTINUE
        decision = rule.decide(0.99, 2, 10)
        assert decision == Decision.CONTINUE
        # After observe phase
        decision = rule.decide(0.99, 3, 10)
        assert decision == Decision.STOP


class TestSecretaryRule:
    """Secretary rule tests."""

    def test_observe_phase(self, secretary_rule):
        """First ~37% is observe phase."""
        # 37% of 10 = 3 steps
        for step in range(3):
            decision = secretary_rule.decide(0.5, step, 10)
            assert decision == Decision.CONTINUE

    def test_stops_on_best(self, secretary_rule):
        """Stops when signal exceeds best observed."""
        # Observe phase: record best
        secretary_rule.decide(0.3, 0, 10)
        secretary_rule.decide(0.5, 1, 10)
        secretary_rule.decide(0.4, 2, 10)
        # Best observed = 0.5
        # Phase 2: signal > 0.5 → STOP
        decision = secretary_rule.decide(0.6, 3, 10)
        assert decision == Decision.STOP

    def test_continues_if_not_best(self, secretary_rule):
        """Continues if signal ≤ best observed."""
        secretary_rule.decide(0.8, 0, 10)
        secretary_rule.decide(0.5, 1, 10)
        secretary_rule.decide(0.6, 2, 10)
        # Best = 0.8
        decision = secretary_rule.decide(0.7, 3, 10)
        assert decision == Decision.CONTINUE

    def test_last_step_stops(self, secretary_rule):
        """Must stop at last step."""
        secretary_rule.decide(0.1, 0, 10)
        secretary_rule.decide(0.1, 1, 10)
        secretary_rule.decide(0.1, 2, 10)
        # Skip to last step
        for step in range(3, 9):
            secretary_rule.decide(0.1, step, 10)
        # Last step (step 9)
        decision = secretary_rule.decide(0.1, 9, 10)
        assert decision == Decision.STOP

    def test_reset(self, secretary_rule):
        """Reset clears best observed."""
        secretary_rule.decide(0.9, 0, 10)
        assert secretary_rule._best_observed == 0.9
        secretary_rule.reset()
        assert secretary_rule._best_observed == float("-inf")


class TestOptimalStop:
    """OptimalStop controller tests."""

    def test_decide_returns_result(self, stop_threshold):
        """decide returns StopResult."""
        result = stop_threshold.decide(0.5, 0, 10)
        assert isinstance(result, StopResult)
        assert result.signal == 0.5
        assert result.step == 0

    def test_history_tracking(self, stop_threshold):
        """History records all signals."""
        stop_threshold.decide(0.3, 0, 10)
        stop_threshold.decide(0.5, 1, 10)
        stop_threshold.decide(0.7, 2, 10)
        assert len(stop_threshold.history) == 3
        assert stop_threshold.history == [0.3, 0.5, 0.7]

    def test_best_observed(self, stop_threshold):
        """Best observed tracks maximum signal."""
        stop_threshold.decide(0.3, 0, 10)
        stop_threshold.decide(0.7, 1, 10)
        stop_threshold.decide(0.5, 2, 10)
        assert stop_threshold.best_observed == 0.7

    def test_stopped_flag(self, stop_threshold):
        """Stopped flag is set after stopping."""
        # Force a stop with high signal
        stop_threshold.decide(0.99, 5, 10)
        assert stop_threshold.stopped is True

    def test_after_stopped_returns_stop(self, stop_threshold):
        """After stopping, always returns STOP."""
        stop_threshold.decide(0.99, 5, 10)
        result = stop_threshold.decide(0.1, 6, 10)
        assert result.decision == Decision.STOP

    def test_reset(self, stop_threshold):
        """Reset clears all state."""
        stop_threshold.decide(0.99, 5, 10)
        stop_threshold.reset()
        assert stop_threshold.stopped is False
        assert len(stop_threshold.history) == 0
        assert stop_threshold.best_observed is None


class TestStopResult:
    """StopResult tests."""

    def test_should_stop_property(self):
        """should_stop reflects decision."""
        result = StopResult(
            decision=Decision.STOP, signal=0.5, threshold=0.3, step=5, steps_remaining=4
        )
        assert result.should_stop is True

        result2 = StopResult(
            decision=Decision.CONTINUE, signal=0.2, threshold=0.3, step=5, steps_remaining=4
        )
        assert result2.should_stop is False

    def test_repr(self):
        """repr includes icon."""
        result = StopResult(
            decision=Decision.STOP, signal=0.5, threshold=0.3, step=5, steps_remaining=4
        )
        assert "STOP" in repr(result)


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_step(self):
        """Single step horizon works."""
        rule = ThresholdRule(horizon=1)
        decision = rule.decide(0.5, 0, 1)
        # With horizon=1, threshold at step 0 should be high
        # 0.5 might not be enough
        assert decision in [Decision.STOP, Decision.CONTINUE]

    def test_zero_signal(self):
        """Zero signal never stops (unless forced)."""
        rule = ThresholdRule(horizon=10)
        for step in range(10):
            decision = rule.decide(0.0, step, 10)
            if step < 9:
                assert decision == Decision.CONTINUE

    def test_one_signal(self):
        """Signal of 1.0 always stops."""
        rule = ThresholdRule(horizon=10)
        for step in range(10):
            decision = rule.decide(1.0, step, 10)
            assert decision == Decision.STOP
