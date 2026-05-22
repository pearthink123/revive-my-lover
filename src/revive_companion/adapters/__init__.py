"""
Adapters connect revive-companion to different AI platforms.

The engine only computes math. Adapters handle:
- Calling the actual AI API
- Building the system prompt
- Sending the message to the user
"""

from .anthropic import AnthropicAdapter
from .base import Adapter
from .generic import GenericAdapter
from .openai import OpenAIAdapter

__all__ = ["Adapter", "OpenAIAdapter", "AnthropicAdapter", "GenericAdapter"]
