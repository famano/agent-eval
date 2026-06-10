"""Anthropic / Claude wrapper agent."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import anthropic

from ..agent import AgentMetadata
from ..models import RunResult, RunStatus
from .base import WrapperConfig, read_input_dir


class ClaudeWrapperAgent:
    """Calls the Anthropic Messages API and writes the response to *output_dir*.

    Example
    -------
    >>> cfg = WrapperConfig(
    ...     model="claude-sonnet-4-6",
    ...     temperature=0.0,
    ...     system_prompt="You are a helpful assistant.",
    ...     tools=[{
    ...         "name": "search",
    ...         "description": "Search the web.",
    ...         "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    ...     }],
    ... )
    >>> agent = ClaudeWrapperAgent(cfg)
    """

    def __init__(
        self,
        config: WrapperConfig,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._cfg = config
        self._client = client or anthropic.Anthropic()

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            model=self._cfg.model,
            sdk_version=anthropic.__version__,
            tools=self._cfg.tool_names(),
            skills=list(self._cfg.skills),
            temperature=self._cfg.temperature,
            seed=self._cfg.seed,
        )

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        timeout_s: int,
        budget_usd: float,
    ) -> RunResult:
        t0 = time.monotonic()
        prompt = read_input_dir(input_dir)

        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "max_tokens": self._cfg.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._cfg.system_prompt:
            kwargs["system"] = self._cfg.system_prompt
        if self._cfg.temperature is not None:
            kwargs["temperature"] = self._cfg.temperature
        if self._cfg.tools:
            kwargs["tools"] = self._cfg.tools
        kwargs.update(self._cfg.extra_params)

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APITimeoutError:
            return RunResult(
                status=RunStatus.TIMEOUT,
                output_files=[],
                cost_usd=0.0,
                latency_s=time.monotonic() - t0,
                tokens=0,
                tool_calls=0,
            )
        except Exception:
            return RunResult(
                status=RunStatus.CRASHED,
                output_files=[],
                cost_usd=0.0,
                latency_s=time.monotonic() - t0,
                tokens=0,
                tool_calls=0,
            )

        text_parts: list[str] = []
        tool_calls = 0
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls += 1
                text_parts.append(f"[tool_use: {block.name}]\n{block.input}")

        output_file = output_dir / "response.txt"
        output_file.write_text("\n\n".join(text_parts), encoding="utf-8")

        tokens = response.usage.input_tokens + response.usage.output_tokens
        return RunResult(
            status=RunStatus.SUCCESS,
            output_files=[output_file],
            cost_usd=_estimate_cost_claude(self._cfg.model, tokens),
            latency_s=time.monotonic() - t0,
            tokens=tokens,
            tool_calls=tool_calls,
        )


def _estimate_cost_claude(model: str, tokens: int) -> float:
    """Very rough per-token cost estimate for common Claude models."""
    mtok = tokens / 1_000_000
    if "opus" in model:
        return mtok * 75.0
    if "haiku" in model:
        return mtok * 1.25
    return mtok * 15.0  # sonnet default
