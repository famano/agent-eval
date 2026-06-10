"""Shared config and utilities for model wrapper agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WrapperConfig:
    """Configuration shared by all model wrapper agents.

    Parameters
    ----------
    model:
        Model identifier passed directly to the provider SDK.
    tools:
        Provider-native tool/function definitions.  Each provider expects its
        own schema (Anthropic-style, OpenAI-style, or Gemini-style dicts).
    skills:
        Skill labels stored in :class:`~agent_eval.AgentMetadata` for
        experiment tracking — not forwarded to the API.
    temperature:
        Sampling temperature.  ``None`` leaves the provider default in place.
    seed:
        RNG seed forwarded to providers that support it (OpenAI, Gemini).
        Ignored silently by providers that lack native seed support.
    max_tokens:
        Maximum tokens in the completion.
    system_prompt:
        System / instruction text sent before the user turn.
    extra_params:
        Arbitrary keyword arguments merged into the API call kwargs.
        Use for provider-specific knobs not covered above.
    """

    model: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int = 4096
    system_prompt: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def tool_names(self) -> list[str]:
        return [
            t.get("name", "") for t in self.tools if isinstance(t, dict) and "name" in t
        ]


def read_input_dir(input_dir: Path) -> str:
    """Concatenate all files under *input_dir* into a single prompt string."""
    parts: list[str] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                rel = path.relative_to(input_dir)
                parts.append(f"=== {rel} ===\n{text}")
            except OSError:
                pass
    return "\n\n".join(parts) if parts else "(empty input)"
