"""
CompanionEngine — one-line setup for revive-companion.

The simplest way to use revive-companion. Combines PoissonLove + Adapter + Runner
into a single class.

Usage:
    from revive_companion import CompanionEngine

    engine = CompanionEngine(config="config.yaml", adapter="openai")
    result = engine.decide()
    if result.should_send:
        send_message(result.response)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters.base import Adapter
from .core.config import Config
from .love import LoveResult, PoissonLove
from .runner import Runner


# Default configs for quick setup
DEFAULT_CONFIGS: dict[str, dict] = {
    "default": {
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
        "persona": {
            "name": "Companion",
            "tone": "warm-brief",
            "context": "You are a caring companion checking in on your person.",
        },
    },
    "frequent": {
        "engagement": {
            "lambda_rate": 0.30,
            "check_interval_minutes": 15,
            "growth_factor": 0.12,
            "max_probability": 0.98,
            "min_interval_hours": 0.5,
            "adjudication": {
                "quiet_hours": {"start": "01:00", "end": "07:00"},
                "normal_send_probability": 0.8,
            },
        },
        "persona": {
            "name": "Companion",
            "tone": "warm-brief",
            "context": "You are a caring companion checking in on your person.",
        },
    },
    "conservative": {
        "engagement": {
            "lambda_rate": 0.08,
            "check_interval_minutes": 60,
            "growth_factor": 0.05,
            "max_probability": 0.80,
            "min_interval_hours": 3.0,
            "adjudication": {
                "quiet_hours": {"start": "22:00", "end": "10:00"},
                "normal_send_probability": 0.5,
            },
        },
        "persona": {
            "name": "Companion",
            "tone": "warm-brief",
            "context": "You are a caring companion checking in on your person.",
        },
    },
}


def get_default_config(preset: str = "default") -> dict:
    """
    Get a preset default configuration.

    Presets:
        - "default": Balanced engagement (lambda=0.15, 30min checks)
        - "frequent": More aggressive (lambda=0.30, 15min checks)
        - "conservative": Less intrusive (lambda=0.08, 60min checks)
    """
    if preset not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(DEFAULT_CONFIGS.keys())}")
    return DEFAULT_CONFIGS[preset].copy()


class CompanionEngine:
    """
    All-in-one engine: config → decide → send.

    Combines PoissonLove (decision engine) + Adapter (LLM) + Runner (scheduler)
    into a single, easy-to-use class.

    Args:
        config: Engine configuration. Can be:
            - dict: inline config
            - str/Path: path to YAML file
            - "default" / "frequent" / "conservative": preset name
            - Config object
        adapter: LLM adapter. Can be:
            - str: "openai", "anthropic", "ollama" (auto-configured)
            - Adapter instance (custom)
            - None (timing only, no LLM)
        api_key: API key for the adapter (ignored if adapter is an Adapter instance).
        seed: Random seed for reproducibility.

    Example:
        >>> engine = CompanionEngine(adapter="openai", api_key="sk-...")
        >>> result = engine.decide()
        >>> if result.should_send:
        ...     print(result.response)  # LLM-generated message
    """

    def __init__(
        self,
        config: dict | str | Path | Config | None = None,
        adapter: str | Adapter | None = None,
        api_key: str | None = None,
        seed: int | None = None,
    ):
        # Resolve config
        if config is None:
            self._config = Config.from_dict(get_default_config())
        elif isinstance(config, str) and config in DEFAULT_CONFIGS:
            self._config = Config.from_dict(get_default_config(config))
        elif isinstance(config, (str, Path)):
            self._config = Config.from_yaml(config)
        elif isinstance(config, dict):
            self._config = Config.from_dict(config)
        elif isinstance(config, Config):
            self._config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # Build engine
        self._love = PoissonLove(config=self._config, seed=seed)

        # Resolve adapter
        self._adapter = self._build_adapter(adapter, api_key)

        # Build runner
        self._runner = Runner(self._love, self._adapter)

    def _build_adapter(self, adapter, api_key) -> Adapter | None:
        if adapter is None or isinstance(adapter, Adapter):
            return adapter

        if adapter == "openai":
            from .adapters.openai import OpenAIAdapter
            return OpenAIAdapter(self._config, api_key=api_key)
        elif adapter == "anthropic":
            from .adapters.anthropic import AnthropicAdapter
            return AnthropicAdapter(self._config, api_key=api_key)
        elif adapter == "ollama":
            from .adapters.generic import GenericAdapter
            return GenericAdapter(
                self._config,
                mode="openai",
                api_url="http://localhost:11434/v1/chat/completions",
            )
        else:
            raise ValueError(f"Unknown adapter '{adapter}'. Choose: openai, anthropic, ollama, or pass an Adapter instance.")

    def decide(self, now=None) -> LoveResult:
        """
        Run one decision cycle. Returns LoveResult with:
        - should_send: whether to send a message
        - response: LLM-generated message (if adapter configured and should_send)
        - All pipeline details (poisson_hit, infogain_passed, user_state, etc.)
        """
        result = self._runner.tick(now)

        # Convert TickResult to LoveResult-like if using plain engine
        if isinstance(result, LoveResult):
            return result

        # Wrap TickResult for consistent API
        return LoveResult(
            should_send=result.should_send,
            stage="engine",
            poisson_hit=result.action.value in ("send", "hold"),
            infogain_passed=True,
            user_state="unknown",
            state_confidence=0,
            send_utility=result.probability,
            lambda_rate=self._config.engagement.lambda_rate,
            probability=result.probability,
            info_gain=0,
            prompt=result.prompt or "",
            reason=result.reason or "",
            metadata=result.metadata,
        )

    def record_send(self, message: str = "") -> None:
        """Record that we sent a message."""
        self._love.record_send(message)

    def record_reply(
        self,
        message: str = "",
        reply_speed: float = 0.5,
        reply_length: float = 0.5,
    ) -> None:
        """Record that the user replied."""
        self._love.record_reply(message, reply_speed, reply_length)

    def run(self, interval_minutes: int | None = None, max_ticks: int | None = None):
        """Run the engine in a blocking loop."""
        self._runner.run(interval_minutes, max_ticks)

    def simulate(self, hours: int = 48, interval_minutes: int | None = None):
        """Run a fast simulation for testing/visualization."""
        self._runner.run_simulation(hours, interval_minutes)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def engine(self) -> PoissonLove:
        return self._love
