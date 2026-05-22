"""Tests for Bayesian state estimation."""

import pytest

from revive_companion.bayesian.core import TRANSITIONS, State, StateEstimator


@pytest.fixture
def estimator():
    """Fresh estimator for each test."""
    return StateEstimator()


class TestStateEstimatorInit:
    """Initialization tests."""

    def test_default_prior(self, estimator):
        """Default prior sums to 1."""
        total = sum(estimator.belief.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_states_present(self, estimator):
        """All 6 states are present."""
        assert len(estimator.belief) == 6
        for state in State:
            assert state in estimator.belief

    def test_custom_prior(self):
        """Custom prior is normalized."""
        custom = {
            State.CHATTING: 0.5,
            State.IDLE_ONLINE: 0.5,
            State.BUSY: 0.0,
            State.SLEEPING: 0.0,
            State.AWAY: 0.0,
            State.NEEDING: 0.0,
        }
        est = StateEstimator(prior=custom)
        assert abs(est.belief[State.CHATTING] - 0.5) < 1e-6
        assert abs(est.belief[State.IDLE_ONLINE] - 0.5) < 1e-6


class TestBayesianUpdate:
    """Bayesian update mechanics."""

    def test_update_returns_distribution(self, estimator):
        """Update returns a valid probability distribution."""
        belief = estimator.update(reply_speed=0.8, hour=14, silence_hours=0.5)
        total = sum(belief.values())
        assert abs(total - 1.0) < 1e-6

    def test_fast_reply_increases_chatting(self, estimator):
        """Fast reply shifts belief toward CHATTING."""
        initial_chatting = estimator.belief[State.CHATTING]
        estimator.update(reply_speed=0.9, reply_length=0.8)
        assert estimator.belief[State.CHATTING] > initial_chatting

    def test_slow_reply_decreases_chatting(self, estimator):
        """Slow reply reduces CHATTING probability."""
        # First, make CHATTING likely
        estimator._belief = {s: 0.1 for s in State}
        estimator._belief[State.CHATTING] = 0.8
        initial = estimator.belief[State.CHATTING]
        estimator.update(reply_speed=0.1)
        assert estimator.belief[State.CHATTING] < initial

    def test_night_hour_increases_sleeping(self, estimator):
        """Late night hours increase SLEEPING probability."""
        initial_sleep = estimator.belief[State.SLEEPING]
        estimator.update(hour=3.0)  # 3 AM
        assert estimator.belief[State.SLEEPING] > initial_sleep

    def test_long_silence_increases_needing(self, estimator):
        """Very long silence increases NEEDING probability."""
        initial_need = estimator.belief[State.NEEDING]
        estimator.update(silence_hours=48.0)  # 2 days
        assert estimator.belief[State.NEEDING] > initial_need


class TestMostLikely:
    """Most likely state detection."""

    def test_returns_correct_state(self, estimator):
        """Returns the state with highest probability."""
        estimator._belief = {s: 0.1 for s in State}
        estimator._belief[State.BUSY] = 0.7
        state, probs = estimator.most_likely()
        assert state == State.BUSY

    def test_returns_full_distribution(self, estimator):
        """Returns complete probability distribution."""
        state, probs = estimator.most_likely()
        assert len(probs) == 6
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6


class TestSendUtility:
    """Send utility calculation."""

    def test_utility_range(self, estimator):
        """Utility is always between 0 and 1."""
        for _ in range(10):
            estimator.update(reply_speed=0.5, hour=12, silence_hours=1)
            utility = estimator.send_utility()
            assert 0 <= utility <= 1

    def test_chatting_low_utility(self, estimator):
        """When CHATTING dominates, utility should be low."""
        estimator._belief = {s: 0.05 for s in State}
        estimator._belief[State.CHATTING] = 0.7
        utility = estimator.send_utility()
        assert utility < 0.3

    def test_needing_high_utility(self, estimator):
        """When NEEDING dominates, utility should be high."""
        estimator._belief = {s: 0.05 for s in State}
        estimator._belief[State.NEEDING] = 0.7
        utility = estimator.send_utility()
        assert utility > 0.6


class TestShouldSend:
    """Decision logic."""

    def test_high_utility_sends(self, estimator):
        """High utility → should send."""
        estimator._belief = {s: 0.05 for s in State}
        estimator._belief[State.IDLE_ONLINE] = 0.7
        should, reason = estimator.should_send(threshold=0.5)
        assert should is True

    def test_low_utility_blocks(self, estimator):
        """Low utility → should not send."""
        estimator._belief = {s: 0.05 for s in State}
        estimator._belief[State.SLEEPING] = 0.7
        should, reason = estimator.should_send(threshold=0.5)
        assert should is False

    def test_reason_contains_state(self, estimator):
        """Reason includes the most likely state."""
        estimator._belief = {s: 0.05 for s in State}
        estimator._belief[State.BUSY] = 0.7
        should, reason = estimator.should_send()
        assert "busy" in reason.lower()


class TestTransitionMatrix:
    """Transition matrix properties."""

    def test_transitions_sum_to_one(self):
        """Each row of transition matrix sums to 1."""
        for state, transitions in TRANSITIONS.items():
            total = sum(transitions.values())
            assert abs(total - 1.0) < 1e-6, f"Row {state} sums to {total}"

    def test_all_states_in_transitions(self):
        """All states appear in transition matrix."""
        for state in State:
            assert state in TRANSITIONS
            for next_state in State:
                assert next_state in TRANSITIONS[state]


class TestLikelihoodFunctions:
    """Likelihood calculation functions."""

    def test_gaussian_likelihood(self, estimator):
        """Gaussian likelihood is symmetric around mean."""
        # CHATTING has mean=0.8, std=0.15 for reply_speed
        p1 = estimator._likelihood_reply_speed(0.8, State.CHATTING)
        p2 = estimator._likelihood_reply_speed(0.8, State.CHATTING)
        assert abs(p1 - p2) < 1e-6

    def test_hour_likelihood(self, estimator):
        """Night hours give low likelihood for BUSY."""
        p_day = estimator._likelihood_hour(14.0, State.BUSY)
        p_night = estimator._likelihood_hour(3.0, State.BUSY)
        assert p_day > p_night

    def test_silence_likelihood(self, estimator):
        """Silence near expected hours gives higher likelihood."""
        # For NEEDING: expected 24h, std 12h
        p_near = estimator._likelihood_silence(24.0, State.NEEDING)  # At mean
        p_far = estimator._likelihood_silence(48.0, State.NEEDING)  # 2 std away
        assert p_near > p_far


class TestReset:
    """Reset functionality."""

    def test_reset_restores_prior(self, estimator):
        """Reset returns to prior distribution."""
        estimator.update(reply_speed=0.9, hour=3, silence_hours=48)
        estimator.reset()
        for state in State:
            assert abs(estimator.belief[state] - estimator.prior[state]) < 1e-6

    def test_reset_custom_prior(self, estimator):
        """Reset with custom prior."""
        custom = {s: 1 / 6 for s in State}
        estimator.reset(prior=custom)
        for state in State:
            assert abs(estimator.belief[state] - 1 / 6) < 1e-6
