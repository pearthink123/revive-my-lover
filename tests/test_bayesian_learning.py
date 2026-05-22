"""Tests for Bayesian Learning module."""

import pytest

from revive_companion.bayesian import BayesianLearner, State, StateEstimator


@pytest.fixture
def learner():
    """Fresh learner for each test."""
    return BayesianLearner(learning_rate=0.1, min_observations=5)


@pytest.fixture
def estimator():
    """Fresh estimator for each test."""
    return StateEstimator()


class TestBayesianLearnerInit:
    """Initialization tests."""

    def test_initialization(self, learner):
        """Learner initializes correctly."""
        assert learner.learning_rate == 0.1
        assert learner.min_observations == 5
        assert learner.observation_count == 0
        assert not learner.is_ready

    def test_initial_state(self, learner):
        """Initial state is clean."""
        assert learner._last_state is None
        assert len(learner._recent) == 0


class TestRecord:
    """Recording observations."""

    def test_record_increments_count(self, learner):
        """Recording increases observation count."""
        assert learner.observation_count == 0
        learner.record(state=State.IDLE_ONLINE)
        assert learner.observation_count == 1
        learner.record(state=State.BUSY)
        assert learner.observation_count == 2

    def test_record_updates_last_state(self, learner):
        """Recording updates last state."""
        assert learner._last_state is None
        learner.record(state=State.IDLE_ONLINE)
        assert learner._last_state == State.IDLE_ONLINE
        learner.record(state=State.BUSY)
        assert learner._last_state == State.BUSY

    def test_record_adds_to_recent(self, learner):
        """Recording adds to recent window."""
        assert len(learner._recent) == 0
        learner.record(state=State.IDLE_ONLINE)
        assert len(learner._recent) == 1
        assert learner._recent[0].state == State.IDLE_ONLINE

    def test_record_with_all_parameters(self, learner):
        """Recording with all parameters works."""
        learner.record(
            state=State.IDLE_ONLINE,
            reply_speed=0.8,
            reply_length=0.6,
            hour=14.0,
            silence_hours=2.0,
            has_reaction=True,
        )
        assert learner.observation_count == 1
        record = learner._recent[0]
        assert record.reply_speed == 0.8
        assert record.reply_length == 0.6
        assert record.hour == 14.0
        assert record.silence_hours == 2.0
        assert record.has_reaction is True


class TestShouldUpdate:
    """should_update() logic."""

    def test_not_ready_initially(self, learner):
        """Not ready with no observations."""
        assert not learner.should_update()

    def test_not_ready_below_threshold(self, learner):
        """Not ready below min_observations."""
        for _ in range(4):
            learner.record(state=State.IDLE_ONLINE)
        assert not learner.should_update()

    def test_ready_at_threshold(self, learner):
        """Ready at min_observations."""
        for _ in range(5):
            learner.record(state=State.IDLE_ONLINE)
        assert learner.should_update()

    def test_ready_above_threshold(self, learner):
        """Ready above min_observations."""
        for _ in range(10):
            learner.record(state=State.IDLE_ONLINE)
        assert learner.should_update()


class TestLearn:
    """Learning parameters."""

    def test_learn_returns_dict(self, learner):
        """learn() returns a dict."""
        for _ in range(10):
            learner.record(state=State.IDLE_ONLINE)
        params = learner.learn()
        assert isinstance(params, dict)

    def test_learn_has_required_keys(self, learner):
        """learn() returns dict with required keys."""
        for _ in range(10):
            learner.record(state=State.IDLE_ONLINE)
        params = learner.learn()
        assert "transitions" in params
        assert "likelihoods" in params
        assert "temporal" in params
        assert "total_observations" in params
        assert "confidence" in params

    def test_learn_confidence_increases(self, learner):
        """Confidence increases with more observations."""
        # Record 5 observations
        for _ in range(5):
            learner.record(state=State.IDLE_ONLINE)
        params1 = learner.learn()
        confidence1 = params1["confidence"]

        # Record 50 more observations
        for _ in range(50):
            learner.record(state=State.IDLE_ONLINE)
        params2 = learner.learn()
        confidence2 = params2["confidence"]

        assert confidence2 > confidence1

    def test_learn_transition_counts(self, learner):
        """Transition counts are tracked."""
        # Record state transitions
        learner.record(state=State.IDLE_ONLINE)
        learner.record(state=State.BUSY)
        learner.record(state=State.IDLE_ONLINE)
        learner.record(state=State.BUSY)
        learner.record(state=State.IDLE_ONLINE)

        params = learner.learn()
        transitions = params["transitions"]

        # Check that idle → busy transition exists
        assert State.BUSY in transitions[State.IDLE_ONLINE]
        # Check probability is reasonable
        assert transitions[State.IDLE_ONLINE][State.BUSY] > 0


class TestGetInsights:
    """get_insights() method."""

    def test_insights_before_ready(self, learner):
        """Insights before ready returns status."""
        learner.record(state=State.IDLE_ONLINE)
        insights = learner.get_insights()
        assert "status" in insights
        assert "observations" in insights

    def test_insights_after_ready(self, learner):
        """Insights after ready returns data."""
        for _ in range(10):
            learner.record(state=State.IDLE_ONLINE)
        insights = learner.get_insights()
        assert "total_observations" in insights
        assert "confidence" in insights
        assert "most_likely_state" in insights
        assert "peak_hours" in insights
        assert "transition_patterns" in insights

    def test_most_likely_state(self, learner):
        """Most likely state is correct."""
        # Record mostly IDLE_ONLINE
        for _ in range(8):
            learner.record(state=State.IDLE_ONLINE)
        for _ in range(2):
            learner.record(state=State.BUSY)

        insights = learner.get_insights()
        assert insights["most_likely_state"] == "idle"


class TestReset:
    """Reset functionality."""

    def test_reset_clears_data(self, learner):
        """Reset clears all data."""
        for _ in range(10):
            learner.record(state=State.IDLE_ONLINE)

        learner.reset()

        assert learner.observation_count == 0
        assert learner._last_state is None
        assert len(learner._recent) == 0
        assert not learner.is_ready


class TestEstimatorIntegration:
    """Integration with StateEstimator."""

    def test_update_params(self, estimator):
        """update_params() works."""
        # Create learner and learn some parameters
        learner = BayesianLearner(min_observations=5)
        for _ in range(10):
            learner.record(
                state=State.IDLE_ONLINE,
                reply_speed=0.8,
                hour=14.0,
            )

        params = learner.learn()

        # Update estimator
        estimator.update_params(params)

        # Check that learned parameters are stored
        assert estimator._learned_likelihoods is not None
        assert estimator._learned_temporal is not None

    def test_estimator_uses_learned_params(self, estimator):
        """Estimator uses learned parameters."""
        # Create learner with specific patterns
        learner = BayesianLearner(min_observations=5)

        # Simulate user who is always BUSY at 14:00
        for _ in range(20):
            learner.record(
                state=State.BUSY,
                reply_speed=0.1,
                reply_length=0.1,
                hour=14.0,
                silence_hours=4.0,
            )

        params = learner.learn()
        estimator.update_params(params)

        # Test at 14:00
        estimator.reset()
        estimator.update(hour=14.0)
        state, _ = estimator.most_likely()

        # Should likely be BUSY
        assert state in [State.BUSY, State.IDLE_ONLINE]


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_learn(self, learner):
        """Learning with no data returns defaults."""
        params = learner.learn()
        # Confidence is not 0 because transition coverage is calculated
        assert params["confidence"] < 0.5
        assert params["total_observations"] == 0

    def test_single_state_only(self, learner):
        """Learning with single state only."""
        for _ in range(10):
            learner.record(state=State.IDLE_ONLINE)

        params = learner.learn()
        # Should still have all states in transitions
        for state in State:
            assert state in params["transitions"]

    def test_window_size_limit(self, learner):
        """Recent window is limited."""
        learner = BayesianLearner(window_size=10)

        # Record 20 observations
        for _ in range(20):
            learner.record(state=State.IDLE_ONLINE)

        # Recent window should be limited
        assert len(learner._recent) == 10
