"""LLM client protocol and built-in implementations."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import anthropic as _anthropic


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface for a text-completion LLM client."""

    def complete(self, prompt: str) -> str: ...


@runtime_checkable
class StructuredLLMClient(Protocol):
    """LLMClient that also supports tool-use structured output."""

    def complete(self, prompt: str) -> str: ...

    def complete_structured(self, prompt: str, tool: dict) -> dict:
        """Call the model and return parsed dict via tool use / structured output.

        ``tool`` is a full Anthropic-style tool definition:
        ``{"name": ..., "description": ..., "input_schema": {...}}``.
        """
        ...


class AnthropicClient:
    """Anthropic Messages API client.

    Implements both :class:`LLMClient` (plain text) and
    :class:`StructuredLLMClient` (tool-use forced structured output).
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        client: _anthropic.Anthropic | None = None,
    ) -> None:
        self._model = model
        self._client = client or _anthropic.Anthropic()

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def complete_structured(self, prompt: str, tool: dict) -> dict:
        """Force the model to populate ``tool`` and return its ``input`` dict."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool["name"]:
                return block.input  # type: ignore[return-value]
        raise ValueError(f"No tool_use block for {tool['name']!r} in response")
