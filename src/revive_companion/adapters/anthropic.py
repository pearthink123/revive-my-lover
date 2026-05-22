"""Anthropic (Claude) adapter for revive-companion."""

from __future__ import annotations

import logging

from ..core.config import Config
from ..core.models import TickResult
from .base import Adapter

logger = logging.getLogger(__name__)


class AnthropicAdapter(Adapter):
    """Connect revive-companion to Anthropic's Claude API."""

    def __init__(
        self, config: Config, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"
    ):
        super().__init__(config)
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = Anthropic(**kwargs)
        return self._client

    def build_system_prompt(self) -> str:
        p = self.config.persona
        parts = [
            f"You are {p.name}.",
            f"Tone: {p.tone}.",
        ]
        if p.context:
            parts.append(f"Context: {p.context}")
        parts.append(
            "You are reaching out to check on someone. "
            "Be natural, warm, and brief. "
            "Don't mention being an AI or a system. "
            "Just genuinely check in."
        )
        return "\n".join(parts)

    def build_engagement_prompt(self, result: TickResult) -> str:
        hour = int(result.hour_of_day)
        minute = int((result.hour_of_day % 1) * 60)
        return (
            f"It's {hour:02d}:{minute:02d}. "
            f"You've been thinking about this person for a while "
            f"(longing level: {result.probability:.0%}). "
            f"Reach out naturally. Keep it short."
        )

    def send(self, system_prompt: str, user_prompt: str) -> str | None:
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=200,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.content[0].text
        except Exception as e:
            logger.error("Anthropic API call failed: %s", e)
            raise
