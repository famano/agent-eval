"""LLM client protocol and built-in implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import anthropic as _anthropic
from anthropic.types import TextBlock, ToolUseBlock


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface for a text-completion LLM client."""

    def complete(self, prompt: str) -> str: ...


@runtime_checkable
class StructuredLLMClient(Protocol):
    """LLMClient that also supports tool-use structured output."""

    def complete(self, prompt: str) -> str: ...

    def complete_structured(self, prompt: str, tool: dict[str, Any]) -> dict[str, Any]:
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
        block = response.content[0]
        if not isinstance(block, TextBlock):
            raise ValueError(f"Expected TextBlock, got {type(block).__name__}")
        return block.text.strip()

    def complete_structured(self, prompt: str, tool: dict[str, Any]) -> dict[str, Any]:
        """Force the model to populate ``tool`` and return its ``input`` dict."""
        response = self._client.messages.create(  # type: ignore[call-overload]
            model=self._model,
            max_tokens=512,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if isinstance(block, ToolUseBlock) and block.name == tool["name"]:
                return dict(block.input)  # type: ignore[arg-type,unused-ignore]
        raise ValueError(f"No tool_use block for {tool['name']!r} in response")
