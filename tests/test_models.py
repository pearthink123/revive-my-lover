"""Tests for core models and config."""

from datetime import datetime

from revive_companion.core.config import Config
from revive_companion.core.models import Action, LogEntry, TickResult


class TestAction:
    """Action enum tests."""

    def test_action_values(self):
        """All expected actions exist."""
        assert Action.MISS.value == "miss"
        assert Action.HIT_SEND.value == "send"
        assert Action.HIT_HOLD.value == "hold"
        assert Action.SKIP.value == "skip"

    def test_action_comparison(self):
        """Actions can be compared."""
        assert Action.MISS != Action.HIT_SEND
        assert Action.MISS == Action.MISS


class TestTickResult:
    """TickResult tests."""

    def test_creation(self):
        """TickResult can be created."""
        result = TickResult(action=Action.MISS, probability=0.5, roll=0.3, hour_of_day=10.0)
        assert result.action == Action.MISS
        assert result.probability == 0.5

    def test_should_send_property(self):
        """should_send reflects HIT_SEND action."""
        send_result = TickResult(
            action=Action.HIT_SEND, probability=0.5, roll=0.3, hour_of_day=10.0
        )
        assert send_result.should_send is True

        miss_result = TickResult(action=Action.MISS, probability=0.5, roll=0.3, hour_of_day=10.0)
        assert miss_result.should_send is False


class TestLogEntry:
    """LogEntry tests."""

    def test_to_dict(self):
        """LogEntry converts to dict."""
        now = datetime(2026, 5, 20, 10, 0)
        entry = LogEntry(timestamp=now, action=Action.MISS, probability=0.5, roll=0.3)
        d = entry.to_dict()
        assert d["action"] == "miss"
        assert d["probability"] == 0.5

    def test_from_dict(self):
        """LogEntry reconstructs from dict."""
        now = datetime(2026, 5, 20, 10, 0)
        original = LogEntry(timestamp=now, action=Action.MISS, probability=0.5, roll=0.3)
        d = original.to_dict()
        restored = LogEntry.from_dict(d)
        assert restored.action == original.action
        assert restored.probability == original.probability


class TestConfig:
    """Config tests."""

    def test_from_dict(self):
        """Config loads from dict."""
        cfg = Config.from_dict(
            {
                "engagement": {
                    "lambda_rate": 0.2,
                    "check_interval_minutes": 30,
                },
                "persona": {"name": "Test"},
            }
        )
        assert cfg.engagement.lambda_rate == 0.2
        assert cfg.persona.name == "Test"

    def test_default_values(self):
        """Config has sensible defaults."""
        cfg = Config.from_dict({})
        assert cfg.engagement.lambda_rate > 0
        assert cfg.engagement.check_interval_minutes > 0
        assert cfg.engagement.growth_factor > 0

    def test_nested_access(self):
        """Config supports nested attribute access."""
        cfg = Config.from_dict(
            {"engagement": {"adjudication": {"quiet_hours": {"start": "23:00", "end": "07:00"}}}}
        )
        assert cfg.engagement.adjudication["quiet_hours"]["start"] == "23:00"
