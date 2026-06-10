"""Tests for WrapperConfig and read_prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.wrappers import WrapperConfig, read_prompt

# ---------------------------------------------------------------------------
# read_prompt
# ---------------------------------------------------------------------------


def test_read_prompt_success(tmp_path: Path) -> None:
    (tmp_path / "prompt.md").write_text("Do the task.")
    assert read_prompt(tmp_path) == "Do the task."


def test_read_prompt_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prompt.md"):
        read_prompt(tmp_path)


def test_read_prompt_ignores_other_files(tmp_path: Path) -> None:
    (tmp_path / "prompt.md").write_text("Task here.")
    (tmp_path / "data.txt").write_text("extra data")
    assert read_prompt(tmp_path) == "Task here."


# ---------------------------------------------------------------------------
# WrapperConfig
# ---------------------------------------------------------------------------


def test_wrapper_config_defaults() -> None:
    cfg = WrapperConfig(model="claude-sonnet-4-6")
    assert cfg.tools == []
    assert cfg.skills == []
    assert cfg.temperature is None
    assert cfg.seed is None
    assert cfg.max_tokens == 4096
    assert cfg.system_prompt is None
    assert cfg.extra_params == {}


def test_wrapper_config_tool_names() -> None:
    cfg = WrapperConfig(
        model="m",
        tools=[{"name": "search", "description": "..."}],
    )
    assert cfg.tool_names() == ["search"]


def test_wrapper_config_tool_names_empty() -> None:
    assert WrapperConfig(model="m").tool_names() == []
