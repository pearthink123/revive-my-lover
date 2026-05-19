"""
Generic adapter — works with any LLM via HTTP API or command.

Supports:
- OpenAI-compatible APIs (Ollama, vLLM, LiteLLM, any proxy)
- Custom HTTP endpoints
- Shell commands (for CLI-based LLM tools)
"""

from __future__ import annotations
import json
import subprocess
import urllib.request
import urllib.error
from typing import Optional
from ..core.config import Config
from ..core.models import TickResult
from .base import Adapter


class GenericAdapter(Adapter):
    """
    Adapter that works with any LLM.

    Modes:
      - "openai"  : Any OpenAI-compatible HTTP API (Ollama, vLLM, etc.)
      - "http"    : Custom HTTP endpoint with template
      - "command" : Shell command (stdin→stdout)
    """

    def __init__(self, config: Config, mode: str = "openai",
                 api_url: str = "http://localhost:11434/v1/chat/completions",
                 model: str = "llama3",
                 command: str = None,
                 headers: dict = None,
                 api_key: str = None):
        super().__init__(config)
        self.mode = mode
        self.api_url = api_url
        self.model = model
        self.command = command
        self.headers = headers or {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

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

    def send(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if self.mode == "command":
            return self._send_command(system_prompt, user_prompt)
        elif self.mode == "http":
            return self._send_http(system_prompt, user_prompt)
        else:
            return self._send_openai_compat(system_prompt, user_prompt)

    def _send_openai_compat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Send via OpenAI-compatible API (Ollama, vLLM, etc)."""
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 200,
            "temperature": 0.8,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception:
            return None

    def _send_http(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Send via custom HTTP endpoint."""
        payload = json.dumps({
            "system": system_prompt,
            "prompt": user_prompt,
            "model": self.model,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            return None

    def _send_command(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Send via shell command (stdin → stdout)."""
        if not self.command:
            return None
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
        try:
            result = subprocess.run(
                self.command,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=60,
                shell=True,
            )
            return result.stdout.strip() or None
        except Exception:
            return None
