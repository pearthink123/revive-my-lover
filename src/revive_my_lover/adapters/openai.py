"""OpenAI adapter for revive-companion."""

from __future__ import annotations

import logging

from ..core.config import Config
from ..core.models import TickResult
from .base import Adapter

logger = logging.getLogger(__name__)


class OpenAIAdapter(Adapter):
    """Connect revive-companion to OpenAI's API."""

    def __init__(
        self,
        config: Config,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
    ):
        super().__init__(config)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
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
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.8,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            return None
