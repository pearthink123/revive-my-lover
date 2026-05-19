"""Tests for SDE (Stochastic Differential Equations) module."""

import pytest
import math

from revive_my_lover.sde import (
    OrnsteinUhlenbeck,
    GeometricBrownian,
    Heston,
    EngagementDynamics,
)


@pytest.fixture
def ou():
    """Ornstein-Uhlenbeck model."""
    return OrnsteinUhlenbeck(theta=0.3, mu=0.5, sigma=0.1, x0=0.8, seed=42)


@pytest.fixture
def gbm():
    """Geometric Brownian Motion model."""
    return GeometricBrownian(mu=0.02, sigma=0.1, x0=0.3, seed=42)


@pytest.fixture
def heston():
    """Heston model."""
    return Heston(mu=0.01, kappa=0.3, theta=0.01, sigma_v=0.2, x0=0.5, seed=42)


class TestOrnsteinUhlenbeck:
    """OU model tests."""

    def test_initialization(self, ou):
        """Model initializes correctly."""
        assert ou.current == 0.8
        assert ou.theta == 0.3
        assert ou.mu == 0.5
        assert ou.sigma == 0.1

    def test_step_returns_float(self, ou):
        """step() returns a float."""
        result = ou.step(dt=0.1)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_mean_reversion(self, ou):
        """Engagement reverts toward mean over many steps."""
        # Start at 80%, should decrease toward 50%
        for _ in range(500):
            ou.step(dt=0.1)
        
        # After many steps, should be closer to 50% than 80%
        assert abs(ou.current - 0.5) < abs(ou.current - 0.8)

    def test_clamping(self, ou):
        """Engagement stays in [0, 1]."""
        for _ in range(100):
            val = ou.step(dt=0.1)
            assert 0 <= val <= 1

    def test_history_tracking(self, ou):
        """History records all steps."""
        assert len(ou.history) == 1  # Initial value
        ou.step(dt=0.1)
        assert len(ou.history) == 2
        ou.step(dt=0.1)
        assert len(ou.history) == 3

    def test_reset(self, ou):
        """Reset restores initial state."""
        ou.step(dt=0.1)
        ou.step(dt=0.1)
        ou.reset()
        assert ou.current == 0.8
        assert len(ou.history) == 1

    def test_deterministic_with_seed(self):
        """Same seed produces same sequence."""
        ou1 = OrnsteinUhlenbeck(seed=123)
        ou2 = OrnsteinUhlenbeck(seed=123)
        for _ in range(10):
            v1 = ou1.step(dt=0.1)
            v2 = ou2.step(dt=0.1)
            assert abs(v1 - v2) < 1e-9


class TestGeometricBrownian:
    """GBM model tests."""

    def test_initialization(self, gbm):
        """Model initializes correctly."""
        assert gbm.current == 0.3
        assert gbm.mu == 0.02
        assert gbm.sigma == 0.1

    def test_step_returns_float(self, gbm):
        """step() returns a float."""
        result = gbm.step(dt=0.1)
        assert isinstance(result, float)
        assert 0.01 <= result <= 1

    def test_clamping(self, gbm):
        """Engagement stays in [0.01, 1]."""
        for _ in range(100):
            val = gbm.step(dt=0.1)
            assert 0.01 <= val <= 1

    def test_history_tracking(self, gbm):
        """History records all steps."""
        assert len(gbm.history) == 1
        gbm.step(dt=0.1)
        assert len(gbm.history) == 2

    def test_reset(self, gbm):
        """Reset restores initial state."""
        gbm.step(dt=0.1)
        gbm.reset()
        assert gbm.current == 0.3
        assert len(gbm.history) == 1


class TestHeston:
    """Heston model tests."""

    def test_initialization(self, heston):
        """Model initializes correctly."""
        assert heston.current == 0.5
        assert heston.current_variance == 0.01
        assert heston.mu == 0.01

    def test_step_returns_float(self, heston):
        """step() returns a float."""
        result = heston.step(dt=0.1)
        assert isinstance(result, float)
        assert 0.01 <= result <= 1

    def test_variance_positive(self, heston):
        """Variance stays positive."""
        for _ in range(100):
            heston.step(dt=0.1)
            assert heston.current_variance > 0

    def test_clamping(self, heston):
        """Engagement stays in [0.01, 1]."""
        for _ in range(100):
            val = heston.step(dt=0.1)
            assert 0.01 <= val <= 1

    def test_history_tracking(self, heston):
        """History records all steps."""
        assert len(heston.history) == 1
        assert len(heston.variance_history) == 1
        heston.step(dt=0.1)
        assert len(heston.history) == 2
        assert len(heston.variance_history) == 2

    def test_reset(self, heston):
        """Reset restores initial state."""
        heston.step(dt=0.1)
        heston.reset()
        assert heston.current == 0.5
        assert heston.current_variance == 0.01
        assert len(heston.history) == 1


class TestEngagementDynamics:
    """EngagementDynamics tests."""

    def test_observe_returns_float(self, ou):
        """observe() returns a float."""
        dynamics = EngagementDynamics(model=ou, seed=42)
        result = dynamics.observe(dt=0.1)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_observations_recorded(self, ou):
        """Observations are recorded."""
        dynamics = EngagementDynamics(model=ou, seed=42)
        dynamics.observe(dt=0.1)
        dynamics.observe(dt=0.1)
        assert len(dynamics.observations) == 2

    def test_predict_returns_list(self, ou):
        """predict() returns a list."""
        dynamics = EngagementDynamics(model=ou, seed=42)
        predictions = dynamics.predict(steps=5, dt=0.1)
        assert len(predictions) == 5
        for p in predictions:
            assert isinstance(p, float)
            assert 0 <= p <= 1

    def test_predict_does_not_change_state(self, ou):
        """predict() doesn't change current state."""
        dynamics = EngagementDynamics(model=ou, seed=42)
        before = dynamics.current_engagement
        dynamics.predict(steps=10, dt=0.1)
        after = dynamics.current_engagement
        assert abs(before - after) < 1e-9

    def test_reset(self, ou):
        """Reset clears observations."""
        dynamics = EngagementDynamics(model=ou, seed=42)
        dynamics.observe(dt=0.1)
        dynamics.observe(dt=0.1)
        dynamics.reset()
        assert len(dynamics.observations) == 0


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_dt(self):
        """Zero dt doesn't crash."""
        ou = OrnsteinUhlenbeck(seed=42)
        val = ou.step(dt=0)
        assert isinstance(val, float)

    def test_large_dt(self):
        """Large dt doesn't crash."""
        ou = OrnsteinUhlenbeck(seed=42)
        val = ou.step(dt=10.0)
        assert isinstance(val, float)
        assert 0 <= val <= 1

    def test_negative_mu_gbm(self):
        """Negative mu in GBM causes decay."""
        gbm = GeometricBrownian(mu=-0.05, sigma=0.1, x0=0.8, seed=42)
        for _ in range(50):
            gbm.step(dt=0.1)
        # Should be lower than initial
        assert gbm.current < 0.8

    def test_zero_sigma(self):
        """Zero sigma gives deterministic behavior."""
        ou = OrnsteinUhlenbeck(theta=0.3, mu=0.5, sigma=0, x0=0.8, seed=42)
        # Should move deterministically toward 0.5
        for _ in range(1000):
            ou.step(dt=0.1)
        # Should be very close to 0.5
        assert abs(ou.current - 0.5) < 0.02
