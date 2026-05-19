"""
Adapters connect revive-my-lover to different AI platforms.

The engine only computes math. Adapters handle:
- Calling the actual AI API
- Building the system prompt
- Sending the message to the user
"""

from .base import Adapter
from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter
from .generic import GenericAdapter

__all__ = ["Adapter", "OpenAIAdapter", "AnthropicAdapter", "GenericAdapter"]
