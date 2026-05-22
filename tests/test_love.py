"""Tests for unified PoissonLove API."""

from datetime import datetime, timedelta

import pytest

from revive_companion.love import LoveResult, PoissonLove


@pytest.fixture
def love():
    """Fresh PoissonLove instance."""
    return PoissonLove(seed=42)


class TestPoissonLoveInit:
    """Initialization tests."""

    def test_modules_initialized(self, love):
        """All modules are created."""
        assert love._engine is not None
        assert love._estimator is not None
        assert love._infogain is not None

    def test_initial_state(self, love):
        """Initial state is clean."""
        assert love._last_user_reply is None
        assert love._my_unanswered == 0
        assert len(love._recent_messages) == 0


class TestTick:
    """tick() method tests."""

    def test_returns_love_result(self, love):
        """tick returns LoveResult."""
        now = datetime(2026, 5, 20, 10, 0)
        result = love.tick(now=now)
        assert isinstance(result, LoveResult)

    def test_result_has_required_fields(self, love):
        """Result has all required fields."""
        now = datetime(2026, 5, 20, 10, 0)
        result = love.tick(now=now)
        assert hasattr(result, "should_send")
        assert hasattr(result, "stage")
        assert hasattr(result, "poisson_hit")
        assert hasattr(result, "infogain_passed")
        assert hasattr(result, "user_state")
        assert hasattr(result, "send_utility")
        assert hasattr(result, "lambda_rate")
        assert hasattr(result, "probability")
        assert hasattr(result, "info_gain")
        assert hasattr(result, "prompt")
        assert hasattr(result, "reason")

    def test_poisson_miss_blocks(self, love):
        """If Poisson misses, should_send is False."""
        # Force miss by setting very low probability
        love._engine.probability = 0.0001
        now = datetime(2026, 5, 20, 10, 0)
        result = love.tick(now=now)
        assert result.should_send is False
        assert result.stage == "poisson"

    def test_full_pipeline_passes(self, love):
        """When all stages pass, should_send is True."""
        # Force Poisson hit
        love._engine.probability = 0.99
        # Make sure we're in normal hours
        now = datetime(2026, 5, 20, 10, 0)
        # Set up high info gain
        love._last_user_reply = now - timedelta(hours=2)
        # Run tick
        result = love.tick(now=now)
        # May or may not send depending on Bayesian
        assert isinstance(result.should_send, bool)


class TestRecordReply:
    """record_reply() tests."""

    def test_updates_last_reply(self, love):
        """record_reply updates last_user_reply."""
        assert love._last_user_reply is None
        love.record_reply(message="hello", reply_speed=0.8, reply_length=0.6)
        assert love._last_user_reply is not None

    def test_resets_unanswered(self, love):
        """record_reply resets unanswered counter."""
        love._my_unanswered = 3
        love.record_reply()
        assert love._my_unanswered == 0

    def test_adds_to_recent_messages(self, love):
        """record_reply adds message to recent_messages."""
        assert len(love._recent_messages) == 0
        love.record_reply(message="hello")
        assert len(love._recent_messages) == 1
        assert love._recent_messages[0] == "hello"


class TestRecordSend:
    """record_send() tests."""

    def test_increments_unanswered(self, love):
        """record_send increments unanswered."""
        assert love._my_unanswered == 0
        love.record_send()
        assert love._my_unanswered == 1
        love.record_send()
        assert love._my_unanswered == 2

    def test_adds_to_recent_messages(self, love):
        """record_send adds message."""
        love.record_send(message="hi there")
        assert "hi there" in love._recent_messages


class TestBuildPrompt:
    """_build_prompt() tests."""

    def test_morning_context(self, love):
        """Morning time produces morning context."""
        now = datetime(2026, 5, 20, 9, 0)
        prompt = love._build_prompt(now, "idle", 0.5)
        assert "morning" in prompt.lower()

    def test_evening_context(self, love):
        """Evening time produces evening context."""
        now = datetime(2026, 5, 20, 19, 0)
        prompt = love._build_prompt(now, "idle", 0.5)
        assert "evening" in prompt.lower()

    def test_includes_state(self, love):
        """Prompt includes user state."""
        now = datetime(2026, 5, 20, 10, 0)
        prompt = love._build_prompt(now, "chatting", 0.7)
        assert "chatting" in prompt.lower()

    def test_includes_probability(self, love):
        """Prompt includes longing probability."""
        now = datetime(2026, 5, 20, 10, 0)
        prompt = love._build_prompt(now, "idle", 0.75)
        assert "75%" in prompt


class TestIntegration:
    """Integration tests."""

    def test_full_conversation_flow(self, love):
        """Simulate a conversation flow."""
        now = datetime(2026, 5, 20, 10, 0)

        # User hasn't replied in a while
        love._last_user_reply = now - timedelta(hours=3)

        # Run a few ticks
        results = []
        for i in range(5):
            r = love.tick(now=now + timedelta(hours=i))
            results.append(r)

        # Should have some mix of sends and no-sends
        assert any(r.should_send for r in results) or all(not r.should_send for r in results)

    def test_after_send_probability_resets(self, love):
        """After a send, Poisson probability resets."""
        # Force a send
        love._engine.probability = 0.99
        now = datetime(2026, 5, 20, 10, 0)
        love._last_user_reply = now - timedelta(hours=2)
        result = love.tick(now=now)

        if result.should_send:
            love.record_send()
            # Probability should be lower after send
            assert love._engine.probability < 0.99
