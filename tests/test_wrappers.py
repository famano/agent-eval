"""Tests for model wrapper agents (Claude, Codex, Gemini)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent_eval.wrappers.codex as codex_mod
import agent_eval.wrappers.gemini as gemini_mod
from agent_eval.models import RunStatus
from agent_eval.wrappers import (
    ClaudeWrapperAgent,
    CodexWrapperAgent,
    GeminiWrapperAgent,
    WrapperConfig,
)
from agent_eval.wrappers.base import read_input_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_claude_response(
    text: str, input_tokens: int = 10, output_tokens: int = 20
) -> SimpleNamespace:
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], usage=usage)


def _make_openai_response(
    text: str, prompt_tokens: int = 10, completion_tokens: int = 20
) -> SimpleNamespace:
    msg = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_gemini_response(
    text: str, prompt_tokens: int = 10, candidate_tokens: int = 20
) -> SimpleNamespace:
    part = SimpleNamespace(text=text, function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidate_tokens,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


# ---------------------------------------------------------------------------
# read_input_dir
# ---------------------------------------------------------------------------


def test_read_input_dir_empty(tmp_path: Path) -> None:
    assert read_input_dir(tmp_path) == "(empty input)"


def test_read_input_dir_single_file(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello world")
    result = read_input_dir(tmp_path)
    assert "note.txt" in result
    assert "hello world" in result


def test_read_input_dir_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("aaa")
    (tmp_path / "b.txt").write_text("bbb")
    result = read_input_dir(tmp_path)
    assert "a.txt" in result
    assert "b.txt" in result


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


# ---------------------------------------------------------------------------
# ClaudeWrapperAgent
# ---------------------------------------------------------------------------


def test_claude_metadata() -> None:
    cfg = WrapperConfig(
        model="claude-sonnet-4-6",
        skills=["skill-a"],
        temperature=0.5,
        seed=7,
        tools=[
            {"name": "read_file", "description": "reads a file", "input_schema": {}}
        ],
    )
    agent = ClaudeWrapperAgent(cfg, client=MagicMock())
    meta = agent.metadata()
    assert meta.model == "claude-sonnet-4-6"
    assert meta.skills == ["skill-a"]
    assert meta.temperature == 0.5
    assert meta.seed == 7
    assert "read_file" in meta.tools


def test_claude_run_success(tmp_path: Path) -> None:
    cfg = WrapperConfig(model="claude-sonnet-4-6", temperature=0.0)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_claude_response(
        "hello from claude"
    )

    agent = ClaudeWrapperAgent(cfg, client=mock_client)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "task.txt").write_text("do something")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    result = agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    assert result.status == RunStatus.SUCCESS
    assert result.tokens == 30
    assert result.tool_calls == 0
    assert (output_dir / "response.txt").read_text() == "hello from claude"


def test_claude_run_tool_use(tmp_path: Path) -> None:
    cfg = WrapperConfig(model="claude-sonnet-4-6")
    block_text = SimpleNamespace(type="text", text="thinking...")
    block_tool = SimpleNamespace(
        type="tool_use", name="search", input={"query": "test"}
    )
    usage = SimpleNamespace(input_tokens=5, output_tokens=5)
    mock_resp = SimpleNamespace(content=[block_text, block_tool], usage=usage)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp

    agent = ClaudeWrapperAgent(cfg, client=mock_client)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    result = agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    assert result.status == RunStatus.SUCCESS
    assert result.tool_calls == 1


def test_claude_run_crash(tmp_path: Path) -> None:
    cfg = WrapperConfig(model="claude-sonnet-4-6")
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("network error")

    agent = ClaudeWrapperAgent(cfg, client=mock_client)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    result = agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)
    assert result.status == RunStatus.CRASHED
    assert result.output_files == []


def test_claude_run_extra_params_forwarded(tmp_path: Path) -> None:
    cfg = WrapperConfig(model="claude-sonnet-4-6", extra_params={"top_p": 0.9})
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_claude_response("ok")

    agent = ClaudeWrapperAgent(cfg, client=mock_client)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    _, kwargs = mock_client.messages.create.call_args
    assert kwargs.get("top_p") == 0.9


def test_claude_system_prompt_forwarded(tmp_path: Path) -> None:
    cfg = WrapperConfig(model="claude-sonnet-4-6", system_prompt="Be concise.")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_claude_response("ok")

    agent = ClaudeWrapperAgent(cfg, client=mock_client)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    _, kwargs = mock_client.messages.create.call_args
    assert kwargs.get("system") == "Be concise."


# ---------------------------------------------------------------------------
# CodexWrapperAgent
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_openai() -> MagicMock:
    m = MagicMock()
    m.__version__ = "1.0.0"
    m.APITimeoutError = TimeoutError
    return m


def test_codex_metadata(mock_openai: MagicMock) -> None:
    cfg = WrapperConfig(model="gpt-4o", skills=["skill-b"], temperature=0.7, seed=99)
    with (
        patch.object(codex_mod, "_OPENAI_AVAILABLE", True),
        patch.object(codex_mod, "_openai", mock_openai),
        patch.object(codex_mod, "_OPENAI_VERSION", "1.0.0"),
    ):
        agent = CodexWrapperAgent(cfg, client=MagicMock())
        meta = agent.metadata()
    assert meta.model == "gpt-4o"
    assert meta.seed == 99
    assert meta.skills == ["skill-b"]


def test_codex_run_success(tmp_path: Path, mock_openai: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response(
        "hello from gpt"
    )

    cfg = WrapperConfig(model="gpt-4o", temperature=0.0, seed=0)
    with (
        patch.object(codex_mod, "_OPENAI_AVAILABLE", True),
        patch.object(codex_mod, "_openai", mock_openai),
    ):
        agent = CodexWrapperAgent(cfg, client=mock_client)
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    assert result.status == RunStatus.SUCCESS
    assert result.tokens == 30
    assert (output_dir / "response.txt").read_text() == "hello from gpt"


def test_codex_seed_forwarded(tmp_path: Path, mock_openai: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openai_response("ok")

    cfg = WrapperConfig(model="gpt-4o", seed=42)
    with (
        patch.object(codex_mod, "_OPENAI_AVAILABLE", True),
        patch.object(codex_mod, "_openai", mock_openai),
    ):
        agent = CodexWrapperAgent(cfg, client=mock_client)
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs.get("seed") == 42


def test_codex_run_crash(tmp_path: Path, mock_openai: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("api error")

    cfg = WrapperConfig(model="gpt-4o")
    with (
        patch.object(codex_mod, "_OPENAI_AVAILABLE", True),
        patch.object(codex_mod, "_openai", mock_openai),
    ):
        agent = CodexWrapperAgent(cfg, client=mock_client)
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    assert result.status == RunStatus.CRASHED


def test_codex_not_installed() -> None:
    cfg = WrapperConfig(model="gpt-4o")
    with patch.object(codex_mod, "_OPENAI_AVAILABLE", False):
        with pytest.raises(ImportError, match="openai"):
            CodexWrapperAgent(cfg)


# ---------------------------------------------------------------------------
# GeminiWrapperAgent
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_genai() -> MagicMock:
    m = MagicMock()
    m.__version__ = "0.8.0"
    return m


def test_gemini_metadata(mock_genai: MagicMock) -> None:
    cfg = WrapperConfig(model="gemini-1.5-pro", skills=["skill-c"], temperature=0.3)
    with (
        patch.object(gemini_mod, "_GENAI_AVAILABLE", True),
        patch.object(gemini_mod, "genai", mock_genai),
        patch.object(gemini_mod, "_GENAI_VERSION", "0.8.0"),
    ):
        agent = GeminiWrapperAgent(cfg)
        meta = agent.metadata()
    assert meta.model == "gemini-1.5-pro"
    assert meta.skills == ["skill-c"]
    assert meta.temperature == 0.3


def test_gemini_run_success(tmp_path: Path, mock_genai: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.generate_content.return_value = _make_gemini_response(
        "hello from gemini"
    )
    mock_genai.GenerativeModel.return_value = mock_model

    cfg = WrapperConfig(model="gemini-1.5-pro", temperature=0.0)
    with (
        patch.object(gemini_mod, "_GENAI_AVAILABLE", True),
        patch.object(gemini_mod, "genai", mock_genai),
    ):
        agent = GeminiWrapperAgent(cfg)
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    assert result.status == RunStatus.SUCCESS
    assert result.tokens == 30
    assert (output_dir / "response.txt").read_text() == "hello from gemini"


def test_gemini_seed_forwarded(tmp_path: Path, mock_genai: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.generate_content.return_value = _make_gemini_response("ok")
    mock_genai.GenerativeModel.return_value = mock_model

    cfg = WrapperConfig(model="gemini-1.5-pro", seed=7)
    with (
        patch.object(gemini_mod, "_GENAI_AVAILABLE", True),
        patch.object(gemini_mod, "genai", mock_genai),
    ):
        agent = GeminiWrapperAgent(cfg)
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    _, kwargs = mock_genai.GenerativeModel.call_args
    assert kwargs["generation_config"]["seed"] == 7


def test_gemini_run_crash(tmp_path: Path, mock_genai: MagicMock) -> None:
    mock_genai.GenerativeModel.side_effect = RuntimeError("api error")

    cfg = WrapperConfig(model="gemini-1.5-pro")
    with (
        patch.object(gemini_mod, "_GENAI_AVAILABLE", True),
        patch.object(gemini_mod, "genai", mock_genai),
    ):
        agent = GeminiWrapperAgent(cfg)
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    assert result.status == RunStatus.CRASHED


def test_gemini_extra_generation_config(tmp_path: Path, mock_genai: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.generate_content.return_value = _make_gemini_response("ok")
    mock_genai.GenerativeModel.return_value = mock_model

    cfg = WrapperConfig(
        model="gemini-1.5-flash",
        extra_params={"generation_config": {"top_p": 0.8}},
    )
    with (
        patch.object(gemini_mod, "_GENAI_AVAILABLE", True),
        patch.object(gemini_mod, "genai", mock_genai),
    ):
        agent = GeminiWrapperAgent(cfg)
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        agent.run(input_dir, output_dir, timeout_s=60, budget_usd=1.0)

    _, kwargs = mock_genai.GenerativeModel.call_args
    assert kwargs["generation_config"]["top_p"] == 0.8


def test_gemini_not_installed() -> None:
    cfg = WrapperConfig(model="gemini-1.5-pro")
    with patch.object(gemini_mod, "_GENAI_AVAILABLE", False):
        with pytest.raises(ImportError, match="google-generativeai"):
            GeminiWrapperAgent(cfg)
