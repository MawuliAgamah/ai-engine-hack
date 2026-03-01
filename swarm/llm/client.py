"""
Abstract LLM client and concrete implementations.

Provides:
- ``LLMClient``      — abstract base class
- ``OpenAIClient``    — OpenAI-compatible Chat Completions
- ``MockClient``      — deterministic mock for testing (no API key needed)

All clients expose a single ``complete(messages) -> str`` method so that the
rest of the system never touches provider-specific details.

The expected return format is ``reasoning text | action_index``.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ── Abstract base ────────────────────────────────────────────────────


class LLMClient(ABC):
    """Abstract base for LLM API calls."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        agent_id: int | None = None,
    ) -> str:
        """Send a message list and return the assistant's text reply."""
        ...

# ── OpenAI ───────────────────────────────────────────────────────────


class OpenAIClient(LLMClient):
    """OpenAI-compatible Chat Completions client.

    Requires ``pip install openai`` (or ``pixi add openai``).
    Set the ``API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        temperature: float = 0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK not installed."
            ) from exc

        from openai import OpenAI
        import os

        API_KEY = os.getenv("API_KEY")
        BASE_URL = os.getenv("BASE_URL")
        self._client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        agent_id: int | None = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return resp.choices[0].message.content.strip() or ""


# ── BrevLab ──────────────────────────────────────────────────────────

class BrevLabClient(LLMClient):
    """BrevLab-compatible Chat Completions client (for testing/dev).

    Uses requests to POST to BrevLab endpoint.
    """
    def __init__(
        self,
        url: str = "https://5000-3ioskhaih.brevlab.com/v1/chat/completions",
        model: str = "nemotron",
        max_tokens: int = 512,
        timeout: int = 60,
    ):
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        agent_id: int | None = None,
    ) -> str:
        import requests
        import json
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        try:
            resp = requests.post(
                self.url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"BrevLab request failed: {e}")
            raise SystemExit(f"Request failed: {e}")
        data = resp.json()
        # Expect OpenAI-like response structure
        try:
            return data["choices"][0]["message"]["content"].strip() or ""
        except Exception:
            logger.error(f"Unexpected BrevLab response: {data}")
            return ""


class MockClient(LLMClient):
    """Deterministic mock that returns ``reasoning | index`` responses.

    Strategies:
    - ``"stay"``       — always picks index 0 (STAY).

    Useful for testing the full pipeline without burning API credits.
    """

    def __init__(self, strategy: str = "first_shuffle_move"):
        self.strategy = strategy  # "stay"
        self.call_count: int = 0
        self.last_messages: list[dict[str, str]] | None = None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        agent_id: int | None = None,
    ) -> str:
        self.call_count += 1
        self.last_messages = messages

        user_msg = messages[-1]["content"] if messages else ""
        n_moves = self._count_moves(user_msg)

        if self.strategy == "stay" or n_moves <= 1:
            idx = 0
        else:
            # "first_move": pick index 1 (first MOVE option after STAY)
            idx = 1

        return f"Mock decision (strategy={self.strategy}) | {idx}"

    # ── Internal ─────────────────────────────────────────────────────

    _MOVE_LINE_RE = re.compile(r"^\d+:\s+(?:STAY|MOVE)", re.MULTILINE)

    def _count_moves(self, text: str) -> int:
        """Count numbered action lines in the user message."""
        return len(self._MOVE_LINE_RE.findall(text))
