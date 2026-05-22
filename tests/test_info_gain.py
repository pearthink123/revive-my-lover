"""Tests for Information Gain module."""

import pytest

from revive_companion.info_gain import (
    ConversationFlow,
    InformationGain,
    InfoSource,
    MessageNovelty,
    SilenceDuration,
)


class DummySource(InfoSource):
    """Dummy source for testing."""

    def __init__(self, entropy: float = 0.5, resolution: float = 0.5):
        self._entropy = entropy
        self._resolution = resolution

    def entropy(self) -> float:
        return self._entropy

    def resolution_potential(self) -> float:
        return self._resolution


@pytest.fixture
def gain():
    """Fresh InformationGain for testing."""
    return InformationGain(threshold=0.25, min_gain=0.1)


class TestInformationGainBasics:
    """Basic functionality."""

    def test_empty_sources(self, gain):
        """No sources → not worth sending."""
        result = gain.evaluate()
        assert result.worth_sending is False
        assert "No sources" in result.reason

    def test_single_source(self, gain):
        """Single source works."""
        gain.sources = [DummySource(entropy=0.8, resolution=0.6)]
        result = gain.evaluate()
        assert result.total_entropy == 0.8
        assert result.gain > 0

    def test_multiple_sources(self, gain):
        """Multiple sources combine."""
        gain.sources = [
            DummySource(entropy=0.5, resolution=0.5),
            DummySource(entropy=0.3, resolution=0.8),
        ]
        result = gain.evaluate()
        assert result.total_entropy == 0.8  # 0.5 + 0.3
        # gain = 0.5*0.5 + 0.3*0.8 = 0.25 + 0.24 = 0.49
        assert abs(result.gain - 0.49) < 0.01


class TestGainCalculation:
    """Gain calculation logic."""

    def test_gain_formula(self, gain):
        """Gain = Σ(entropy × resolution) × decay."""
        gain.sources = [DummySource(entropy=0.6, resolution=0.5)]
        result = gain.evaluate()
        # gain = 0.6 * 0.5 * 1.0 (no decay) = 0.3
        assert abs(result.gain - 0.3) < 1e-6

    def test_decay_applied(self, gain):
        """Decay reduces gain after consecutive sends."""
        gain.sources = [DummySource(entropy=0.6, resolution=0.5)]
        gain.on_send()  # _send_count = 1
        gain.on_send()  # _send_count = 2
        result = gain.evaluate()
        # decay_factor = 0.85^2 = 0.7225
        # gain = 0.3 * 0.7225 = 0.21675
        assert result.gain < 0.3

    def test_gain_ratio(self, gain):
        """Gain ratio = gain / total_entropy."""
        gain.sources = [DummySource(entropy=0.8, resolution=0.5)]
        result = gain.evaluate()
        # gain = 0.8 * 0.5 = 0.4
        # ratio = 0.4 / 0.8 = 0.5
        assert abs(result.gain_ratio - 0.5) < 1e-6


class TestWorthSending:
    """Decision logic."""

    def test_high_gain_worth_sending(self, gain):
        """High gain → worth sending."""
        gain.sources = [DummySource(entropy=0.9, resolution=0.8)]
        result = gain.evaluate()
        assert result.worth_sending is True

    def test_low_gain_not_worth(self, gain):
        """Low gain → not worth sending."""
        gain.sources = [DummySource(entropy=0.1, resolution=0.1)]
        result = gain.evaluate()
        assert result.worth_sending is False

    def test_threshold_applied(self):
        """Threshold blocks marginal gains."""
        gain = InformationGain(threshold=0.5, min_gain=0.01)
        gain.sources = [DummySource(entropy=0.8, resolution=0.4)]
        # gain = 0.8 * 0.4 = 0.32
        # ratio = 0.32 / 0.8 = 0.4 < 0.5 threshold
        result = gain.evaluate()
        assert result.worth_sending is False

    def test_min_gain_applied(self):
        """min_gain blocks very small gains."""
        gain = InformationGain(threshold=0.01, min_gain=0.5)
        gain.sources = [DummySource(entropy=0.1, resolution=0.1)]
        # gain = 0.01 < 0.5 min_gain
        result = gain.evaluate()
        assert result.worth_sending is False


class TestSendReceive:
    """Send/receive tracking."""

    def test_on_send_increments(self, gain):
        """on_send increments counter."""
        assert gain._send_count == 0
        gain.on_send()
        assert gain._send_count == 1
        gain.on_send()
        assert gain._send_count == 2

    def test_on_receive_resets(self, gain):
        """on_receive resets counter."""
        gain.on_send()
        gain.on_send()
        gain.on_receive()
        assert gain._send_count == 0

    def test_decay_recovery(self, gain):
        """After receive, decay resets."""
        gain.sources = [DummySource(entropy=0.8, resolution=0.6)]
        gain.on_send()
        gain.on_send()
        result1 = gain.evaluate()
        gain.on_receive()
        result2 = gain.evaluate()
        assert result2.gain > result1.gain


class TestReset:
    """Reset functionality."""

    def test_reset_clears_state(self, gain):
        """Reset clears send count."""
        gain.on_send()
        gain.on_send()
        gain.reset()
        assert gain._send_count == 0


class TestAddSource:
    """Source management."""

    def test_add_source(self, gain):
        """add_source appends to list."""
        assert len(gain.sources) == 0
        gain.add_source(DummySource())
        assert len(gain.sources) == 1


class TestBuiltInSources:
    """Test built-in info sources."""

    def test_silence_duration(self):
        """SilenceDuration source works."""
        from datetime import datetime, timedelta

        now = datetime(2026, 5, 20, 10, 0)
        last_reply = now - timedelta(hours=2)
        src = SilenceDuration(last_reply_time=last_reply, now=now)
        assert 0 <= src.entropy() <= 1
        assert 0 <= src.resolution_potential() <= 1

    def test_conversation_flow(self):
        """ConversationFlow source works."""
        src = ConversationFlow(my_unanswered_messages=2, user_replied_in_last_hour=False)
        assert 0 <= src.entropy() <= 1
        assert 0 <= src.resolution_potential() <= 1

    def test_message_novelty(self):
        """MessageNovelty source works."""
        src = MessageNovelty(recent_messages=["hello", "how are you"], current_message="what's up")
        assert 0 <= src.entropy() <= 1
        assert 0 <= src.resolution_potential() <= 1
