"""
Base adapter — interface for all platform adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.config import Config
from ..core.models import TickResult


class Adapter(ABC):
    """
    Base class for platform adapters.

    Subclass this to connect revive-companion to any AI platform.
    """

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def build_system_prompt(self) -> str:
        """
        Build the system prompt from config.
        This defines the AI's persona and context.
        """
        ...

    @abstractmethod
    def build_engagement_prompt(self, result: TickResult) -> str:
        """
        Build the prompt that tells the AI to reach out.

        Args:
            result: The engine tick result with longing probability and context.

        Returns:
            A prompt string to send to the AI.
        """
        ...

    @abstractmethod
    def send(self, system_prompt: str, user_prompt: str) -> str | None:
        """
        Call the AI API and return its response.

        Args:
            system_prompt: The persona/context prompt.
            user_prompt: The engagement trigger prompt.

        Returns:
            The AI's response, or None if failed.
        """
        ...

    def engage(self, result: TickResult) -> str | None:
        """
        Full engagement flow: build prompts → call AI → return response.

        This is the main method most users should call.
        """
        system = self.build_system_prompt()
        user = self.build_engagement_prompt(result)
        return self.send(system, user)
