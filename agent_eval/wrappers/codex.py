"""OpenAI / Codex wrapper agent."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..agent import AgentMetadata
from ..models import RunResult, RunStatus
from .base import WrapperConfig, read_input_dir

try:
    import openai as _openai

    _OPENAI_AVAILABLE = True
    _OPENAI_VERSION: str = _openai.__version__
except ImportError:
    _openai = None
    _OPENAI_AVAILABLE = False
    _OPENAI_VERSION = "not-installed"


class CodexWrapperAgent:
    """Calls the OpenAI Chat Completions API and writes the response to *output_dir*.

    The ``openai`` package must be installed:

    .. code-block:: shell

        pip install "agent-eval[openai]"

    Example
    -------
    >>> cfg = WrapperConfig(
    ...     model="gpt-4o",
    ...     temperature=0.0,
    ...     seed=42,
    ...     system_prompt="You are a helpful assistant.",
    ...     tools=[{
    ...         "type": "function",
    ...         "function": {
    ...             "name": "search",
    ...             "description": "Search the web.",
    ...             "parameters": {
    ...                 "type": "object",
    ...                 "properties": {"query": {"type": "string"}},
    ...             },
    ...         },
    ...     }],
    ... )
    >>> agent = CodexWrapperAgent(cfg)
    """

    def __init__(
        self,
        config: WrapperConfig,
        client: Any | None = None,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install 'agent-eval[openai]'"
            )
        self._cfg = config
        self._client = client or _openai.OpenAI()

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            model=self._cfg.model,
            sdk_version=_OPENAI_VERSION,
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

        messages: list[dict[str, str]] = []
        if self._cfg.system_prompt:
            messages.append({"role": "system", "content": self._cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "max_tokens": self._cfg.max_tokens,
        }
        if self._cfg.temperature is not None:
            kwargs["temperature"] = self._cfg.temperature
        if self._cfg.seed is not None:
            kwargs["seed"] = self._cfg.seed
        if self._cfg.tools:
            kwargs["tools"] = self._cfg.tools
        kwargs.update(self._cfg.extra_params)

        try:
            response = self._client.chat.completions.create(**kwargs)
        except _openai.APITimeoutError:
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

        choice = response.choices[0]
        text_parts: list[str] = []
        tool_calls = 0

        if choice.message.content:
            text_parts.append(choice.message.content)
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls += 1
                text_parts.append(
                    f"[tool_call: {tc.function.name}]\n{tc.function.arguments}"
                )

        output_file = output_dir / "response.txt"
        output_file.write_text("\n\n".join(text_parts), encoding="utf-8")

        usage = response.usage
        tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
        return RunResult(
            status=RunStatus.SUCCESS,
            output_files=[output_file],
            cost_usd=_estimate_cost_openai(self._cfg.model, tokens),
            latency_s=time.monotonic() - t0,
            tokens=tokens,
            tool_calls=tool_calls,
        )


def _estimate_cost_openai(model: str, tokens: int) -> float:
    """Very rough per-token cost estimate for common OpenAI models."""
    mtok = tokens / 1_000_000
    if "gpt-4o" in model:
        return mtok * 10.0
    if "o1" in model:
        return mtok * 30.0
    return mtok * 5.0  # gpt-3.5 / codex default
