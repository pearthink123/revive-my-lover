"""Configuration models for revive-companion (pydantic v2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class Window(BaseModel):
    """Time window with probability weight."""

    start: float = Field(ge=0, le=24, description="Start hour (0-24)")
    end: float = Field(ge=0, le=24, description="End hour (0-24)")
    weight: float = Field(default=1.0, ge=0, description="Probability weight")


class EngagementConfig(BaseModel):
    """Poisson engagement parameters."""

    strategy: str = "poisson"
    lambda_rate: float = Field(default=0.15, gt=0, description="Base event rate (events/hour)")
    check_interval_minutes: int = Field(default=30, gt=0)
    growth_factor: float = Field(default=0.08, gt=0, le=1)
    max_probability: float = Field(default=0.95, gt=0, le=1)
    min_interval_hours: float = Field(default=1.0, gt=0)
    windows: list[Window] = Field(default_factory=list)
    adjudication: dict[str, Any] = Field(default_factory=dict)


class PersonaConfig(BaseModel):
    """AI persona settings."""

    name: str = "AI"
    tone: str = "warm"
    context: str = ""


class Config(BaseModel):
    """Full revive-companion configuration."""

    engagement: EngagementConfig = Field(default_factory=EngagementConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """Load config from a dictionary with hour-string parsing."""
        eng_data = data.get("engagement", {})

        # Parse windows with hour-string support
        windows = []
        for w in eng_data.get("windows", []):
            windows.append(
                Window(
                    start=_parse_hour(w["start"]),
                    end=_parse_hour(w["end"]),
                    weight=w.get("weight", 1.0),
                )
            )
        if windows:
            eng_data = {**eng_data, "windows": windows}

        return cls(
            engagement=EngagementConfig.model_validate(eng_data) if eng_data else EngagementConfig(),
            persona=PersonaConfig.model_validate(data.get("persona", {})) if data.get("persona") else PersonaConfig(),
            raw=data,
        )

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
