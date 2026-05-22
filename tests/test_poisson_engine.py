"""Tests for PoissonEngine — the core math engine."""

import math
from datetime import datetime, timedelta

import pytest

from revive_companion.core.config import Config
from revive_companion.core.engine import PoissonEngine
from revive_companion.core.models import Action


@pytest.fixture
def default_config():
    """Default config for testing."""
    return Config.from_dict(
        {
            "engagement": {
                "lambda_rate": 0.15,
                "check_interval_minutes": 30,
                "growth_factor": 0.08,
                "max_probability": 0.95,
                "min_interval_hours": 1.0,
                "adjudication": {
                    "quiet_hours": {"start": "00:00", "end": "08:00"},
                    "normal_send_probability": 0.7,
                },
            },
            "persona": {"name": "Test", "tone": "test", "context": "Test"},
        }
    )


@pytest.fixture
def seeded_engine(default_config):
    """Engine with fixed seed for deterministic tests."""
    return PoissonEngine(config=default_config, seed=42)


class TestPoissonEngineBasics:
    """Basic engine functionality."""

    def test_initialization(self, seeded_engine):
        """Engine initializes with correct default state."""
        assert seeded_engine.probability > 0
        assert seeded_engine.last_send_time is None
        assert seeded_engine.miss_streak == 0
        assert len(seeded_engine.log) == 0

    def test_base_probability_formula(self, default_config):
        """Base probability follows Poisson formula: P = 1 - e^(-λt)."""
        engine = PoissonEngine(config=default_config, seed=0)
        lam = default_config.engagement.lambda_rate
        t = default_config.engagement.check_interval_minutes / 60.0
        expected = 1 - math.exp(-lam * t)
        assert abs(engine.probability - expected) < 1e-6

    def test_probability_increases_on_miss(self, seeded_engine):
        """Probability grows after a miss."""
        # Force a miss by setting very low probability
        seeded_engine.probability = 0.001
        now = datetime(2026, 5, 20, 10, 0)  # Normal hours
        result = seeded_engine.tick(now=now)
        if result.action == Action.MISS:
            assert seeded_engine.probability > 0.001


class TestPoissonEngineTiming:
    """Time-based logic."""

    def test_min_interval_blocks(self, seeded_engine):
        """Sending too soon is blocked by min_interval."""
        now = datetime(2026, 5, 20, 10, 0)
        # Set last send time to 30 min ago (less than 1h min_interval)
        seeded_engine.last_send_time = now - timedelta(minutes=30)
        result = seeded_engine.tick(now=now)
        assert result.action == Action.SKIP
        assert "Too early" in result.reason

    def test_min_interval_passes(self, seeded_engine):
        """Sending after min_interval is allowed."""
        now = datetime(2026, 5, 20, 10, 0)
        seeded_engine.last_send_time = now - timedelta(hours=2)
        # Force a hit
        seeded_engine.probability = 0.99
        result = seeded_engine.tick(now=now)
        assert result.action in [Action.HIT_SEND, Action.HIT_HOLD]

    def test_quiet_hours_block(self, seeded_engine):
        """Night hours (0-8) are blocked."""
        now = datetime(2026, 5, 20, 3, 0)  # 3 AM
        seeded_engine.probability = 0.99
        result = seeded_engine.tick(now=now)
        # Should hit but be held due to quiet hours
        assert result.action == Action.HIT_HOLD
        assert "Night" in result.reason


class TestPoissonEngineLog:
    """Logging functionality."""

    def test_log_grows_on_tick(self, seeded_engine):
        """Each tick adds to log."""
        now = datetime(2026, 5, 20, 10, 0)
        assert len(seeded_engine.log) == 0
        seeded_engine.tick(now=now)
        assert len(seeded_engine.log) == 1
        seeded_engine.tick(now=now + timedelta(hours=2))
        assert len(seeded_engine.log) == 2

    def test_log_entry_has_required_fields(self, seeded_engine):
        """Log entries contain all required fields."""
        now = datetime(2026, 5, 20, 10, 0)
        seeded_engine.tick(now=now)
        entry = seeded_engine.log[0]
        assert entry.timestamp == now
        assert isinstance(entry.action, Action)
        assert 0 <= entry.probability <= 1
        assert 0 <= entry.roll <= 1


class TestPoissonEngineGrowth:
    """Probability growth mechanics."""

    def test_growth_factor_applied(self, seeded_engine):
        """Growth factor is added to probability on miss."""
        seeded_engine.probability = 0.1
        seeded_engine._grow()
        expected = min(
            0.1 + seeded_engine.config.engagement.growth_factor,
            seeded_engine.config.engagement.max_probability,
        )
        assert abs(seeded_engine.probability - expected) < 1e-6

    def test_max_probability_capped(self, seeded_engine):
        """Probability never exceeds max_probability."""
        seeded_engine.probability = 0.94
        seeded_engine._grow()  # Should cap at 0.95
        assert seeded_engine.probability <= 0.95

    def test_miss_streak_increments(self, seeded_engine):
        """Miss streak increments on miss."""
        assert seeded_engine.miss_streak == 0
        seeded_engine._grow()
        assert seeded_engine.miss_streak == 1
        seeded_engine._grow()
        assert seeded_engine.miss_streak == 2


class TestPoissonEngineDeterministic:
    """Deterministic behavior with fixed seed."""

    def test_same_seed_same_sequence(self, default_config):
        """Same seed produces same sequence."""
        engine1 = PoissonEngine(config=default_config, seed=123)
        engine2 = PoissonEngine(config=default_config, seed=123)
        now = datetime(2026, 5, 20, 10, 0)
        for _ in range(10):
            r1 = engine1.tick(now=now)
            r2 = engine2.tick(now=now)
            assert r1.action == r2.action
            assert abs(r1.roll - r2.roll) < 1e-9
            now += timedelta(hours=1)

    def test_different_seed_different_sequence(self, default_config):
        """Different seeds produce different sequences."""
        engine1 = PoissonEngine(config=default_config, seed=1)
        engine2 = PoissonEngine(config=default_config, seed=2)
        now = datetime(2026, 5, 20, 10, 0)
        different = False
        for _ in range(20):
            r1 = engine1.tick(now=now)
            r2 = engine2.tick(now=now)
            if r1.roll != r2.roll:
                different = True
                break
            now += timedelta(hours=1)
        assert different
