"""Configuration loader for revive-my-lover."""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Window:
    start: float   # hour (0-24)
    end: float
    weight: float  # probability weight


@dataclass
class EngagementConfig:
    strategy: str = "poisson"
    lambda_rate: float = 0.15
    check_interval_minutes: int = 30
    growth_factor: float = 0.08
    max_probability: float = 0.95
    min_interval_hours: float = 1.0
    windows: list[Window] = field(default_factory=list)
    adjudication: dict = field(default_factory=dict)


@dataclass
class PersonaConfig:
    name: str = "AI"
    tone: str = "warm"
    context: str = ""


@dataclass
class Config:
    """Full revive-my-lover configuration."""

    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """Load config from a dictionary."""
        eng_data = data.get("engagement", {})
        persona_data = data.get("persona", {})

        windows = []
        for w in eng_data.get("windows", []):
            windows.append(Window(
                start=_parse_hour(w["start"]),
                end=_parse_hour(w["end"]),
                weight=w.get("weight", 1.0),
            ))

        engagement = EngagementConfig(
            strategy=eng_data.get("strategy", "poisson"),
            lambda_rate=eng_data.get("lambda_rate", 0.15),
            check_interval_minutes=eng_data.get("check_interval_minutes", 30),
            growth_factor=eng_data.get("growth_factor", 0.08),
            max_probability=eng_data.get("max_probability", 0.95),
            min_interval_hours=eng_data.get("min_interval_hours", 1.0),
            windows=windows,
            adjudication=eng_data.get("adjudication", {}),
        )

        persona = PersonaConfig(
            name=persona_data.get("name", "AI"),
            tone=persona_data.get("tone", "warm"),
            context=persona_data.get("context", ""),
        )

        return cls(engagement=engagement, persona=persona, raw=data)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.raw, f, default_flow_style=False, allow_unicode=True)


def _parse_hour(value) -> float:
    """Parse hour from string (e.g. '09:00') or number."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if ":" in value:
            h, m = value.split(":")
            return int(h) + int(m) / 60.0
        return float(value)
    raise ValueError(f"Cannot parse hour: {value}")
