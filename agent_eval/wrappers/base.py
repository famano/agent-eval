"""Shared config and utilities for model wrapper agents."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WrapperConfig:
    """Configuration shared by all model wrapper agents.

    Parameters
    ----------
    model:
        Model identifier (passed to the provider SDK or recorded in metadata).
    tools:
        Tool definitions forwarded to the provider, or tool name strings for
        shell-based agents.  Schema is provider-specific.
    skills:
        Skill labels stored in :class:`~agent_eval.AgentMetadata` for
        experiment tracking.
    temperature:
        Sampling temperature.  ``None`` leaves the provider default in place.
    seed:
        RNG seed.  Forwarded to providers that support it; ignored otherwise.
    max_tokens:
        Maximum tokens in the completion.
    system_prompt:
        System / instruction text sent before the user turn.
    extra_params:
        Arbitrary keyword arguments merged into the API call kwargs.
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


# ---------------------------------------------------------------------------
# Prompt reading
# ---------------------------------------------------------------------------


def read_prompt(input_dir: Path) -> str:
    """Read the task prompt from *input_dir*/prompt.md.

    ``prompt.md`` is the single authoritative prompt file by convention.
    All evaluation datasets must place the task description there.

    Raises
    ------
    FileNotFoundError
        When ``prompt.md`` is absent from *input_dir*.
    """
    prompt_file = input_dir / "prompt.md"
    if not prompt_file.is_file():
        raise FileNotFoundError(
            f"prompt.md not found in {input_dir}. "
            "Place the task prompt at input_dir/prompt.md."
        )
    return prompt_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Input-directory protection (Unix only)
# ---------------------------------------------------------------------------


def make_input_readonly(path: Path) -> list[tuple[Path, int]]:
    """Make *path* and all its contents read-only.

    On Windows this is a no-op (returns an empty list).
    On Unix, directories become ``r-xr-xr-x`` (555) and files ``r--r--r--``
    (444), preventing the agent subprocess from writing inside *path*.

    Returns a list of ``(entry, original_mode)`` tuples consumed by
    :func:`restore_permissions`.
    """
    if platform.system() == "Windows":
        return []
    saved: list[tuple[Path, int]] = []
    for p in sorted(path.rglob("*"), reverse=True):
        orig = p.stat().st_mode
        saved.append((p, orig))
        p.chmod(0o555 if p.is_dir() else 0o444)
    orig = path.stat().st_mode
    saved.append((path, orig))
    path.chmod(0o555)
    return saved


def restore_permissions(saved: list[tuple[Path, int]]) -> None:
    """Restore file-system permissions saved by :func:`make_input_readonly`."""
    for p, mode in reversed(saved):
        try:
            p.chmod(mode)
        except OSError:
            pass
