"""Tests for ShellWrapperAgent and factory functions."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_eval.models import RunStatus
from agent_eval.wrappers import (
    ShellAgentConfig,
    ShellWrapperAgent,
    claude_code_agent,
    codex_agent,
    gemini_agent,
)
from agent_eval.wrappers.base import make_input_readonly, restore_permissions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proc(stdout: str = "", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def _setup_input(tmp_path: Path, prompt: str = "do the task") -> tuple[Path, Path]:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "prompt.md").write_text(prompt)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    return input_dir, output_dir


# ---------------------------------------------------------------------------
# ShellWrapperAgent — print_mode=True (default)
# ---------------------------------------------------------------------------


def test_print_mode_writes_response_txt(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path)
    cfg = ShellAgentConfig(command=["echo"], model="test", readonly_input=False)
    agent = ShellWrapperAgent(cfg)

    with patch("subprocess.run", return_value=_make_proc("hello output")):
        result = agent.run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    assert result.status == RunStatus.SUCCESS
    assert (output_dir / "response.txt").read_text() == "hello output"
    assert result.output_files == [output_dir / "response.txt"]


def test_print_mode_prompt_passed_via_stdin(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path, prompt="my task prompt")
    cfg = ShellAgentConfig(command=["cat"], model="test", readonly_input=False)
    agent = ShellWrapperAgent(cfg)

    with patch("subprocess.run", return_value=_make_proc("ok")) as mock_run:
        agent.run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    _, kwargs = mock_run.call_args
    assert kwargs["input"] == "my task prompt"


def test_extra_args_appended_to_command(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path)
    cfg = ShellAgentConfig(
        command=["claude", "--print"],
        model="claude-sonnet-4-6",
        extra_args=["--allowed-directories", "/tmp"],
        readonly_input=False,
    )
    agent = ShellWrapperAgent(cfg)

    with patch("subprocess.run", return_value=_make_proc()) as mock_run:
        agent.run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    args, _ = mock_run.call_args
    assert args[0] == ["claude", "--print", "--allowed-directories", "/tmp"]


# ---------------------------------------------------------------------------
# ShellWrapperAgent — print_mode=False
# ---------------------------------------------------------------------------


def test_file_mode_collects_output_files(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path)
    (output_dir / "result.md").write_text("done")
    (output_dir / "data.csv").write_text("a,b")

    cfg = ShellAgentConfig(
        command=["true"], model="test", print_mode=False, readonly_input=False
    )
    agent = ShellWrapperAgent(cfg)

    with patch("subprocess.run", return_value=_make_proc()):
        result = agent.run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    assert result.status == RunStatus.SUCCESS
    assert set(result.output_files) == {
        output_dir / "result.md",
        output_dir / "data.csv",
    }


def test_file_mode_appends_output_dir_to_prompt(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path, prompt="original prompt")
    cfg = ShellAgentConfig(
        command=["true"], model="test", print_mode=False, readonly_input=False
    )
    agent = ShellWrapperAgent(cfg)

    with patch("subprocess.run", return_value=_make_proc()) as mock_run:
        agent.run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    _, kwargs = mock_run.call_args
    assert str(output_dir) in kwargs["input"]
    assert "Write all output files to" in kwargs["input"]


def test_output_dir_env_var_is_set(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path)
    cfg = ShellAgentConfig(
        command=["true"],
        model="test",
        output_dir_env_var="MY_OUTPUT",
        readonly_input=False,
    )
    agent = ShellWrapperAgent(cfg)

    with patch("subprocess.run", return_value=_make_proc()) as mock_run:
        agent.run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    _, kwargs = mock_run.call_args
    assert kwargs["env"]["MY_OUTPUT"] == str(output_dir)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_nonzero_returncode_is_crashed(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path)
    cfg = ShellAgentConfig(command=["false"], model="test", readonly_input=False)
    agent = ShellWrapperAgent(cfg)

    with patch("subprocess.run", return_value=_make_proc(returncode=1)):
        result = agent.run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    assert result.status == RunStatus.CRASHED
    assert result.output_files == []


def test_timeout_returns_timeout_status(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path)
    cfg = ShellAgentConfig(command=["sleep"], model="test", readonly_input=False)
    agent = ShellWrapperAgent(cfg)

    with patch(
        "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sleep", timeout=1)
    ):
        result = agent.run(input_dir, output_dir, timeout_s=1, budget_usd=1.0)

    assert result.status == RunStatus.TIMEOUT


def test_missing_prompt_md_is_crashed(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = ShellAgentConfig(command=["echo"], model="test", readonly_input=False)
    agent = ShellWrapperAgent(cfg)

    result = agent.run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)
    assert result.status == RunStatus.CRASHED


def test_subprocess_exception_is_crashed(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path)
    cfg = ShellAgentConfig(command=["bad-cmd"], model="test", readonly_input=False)
    agent = ShellWrapperAgent(cfg)

    with patch("subprocess.run", side_effect=FileNotFoundError("bad-cmd not found")):
        result = agent.run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    assert result.status == RunStatus.CRASHED


# ---------------------------------------------------------------------------
# readonly_input (Unix only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(platform.system() == "Windows", reason="Unix only")
def test_readonly_input_makes_files_unwritable(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path)
    (input_dir / "data.txt").write_text("secret")

    with patch("subprocess.run", return_value=_make_proc()):
        # During the run, input files should be read-only
        original_modes: list[int] = []

        def check_readonly(*args: object, **kwargs: object) -> SimpleNamespace:
            for p in input_dir.rglob("*"):
                if p.is_file():
                    original_modes.append(p.stat().st_mode & 0o777)
            return _make_proc()

        with patch("subprocess.run", side_effect=check_readonly):
            ShellWrapperAgent(
                ShellAgentConfig(command=["true"], model="test", readonly_input=True)
            ).run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    assert all(m == 0o444 for m in original_modes), original_modes


@pytest.mark.skipif(platform.system() == "Windows", reason="Unix only")
def test_readonly_input_restores_permissions(tmp_path: Path) -> None:
    input_dir, output_dir = _setup_input(tmp_path)
    data = input_dir / "data.txt"
    data.write_text("secret")
    original_mode = data.stat().st_mode

    with patch("subprocess.run", return_value=_make_proc()):
        ShellWrapperAgent(
            ShellAgentConfig(command=["true"], model="test", readonly_input=True)
        ).run(input_dir, output_dir, timeout_s=30, budget_usd=1.0)

    assert data.stat().st_mode == original_mode


# ---------------------------------------------------------------------------
# make_input_readonly / restore_permissions utils
# ---------------------------------------------------------------------------


@pytest.mark.skipif(platform.system() == "Windows", reason="Unix only")
def test_make_restore_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hi")
    original = f.stat().st_mode
    saved = make_input_readonly(tmp_path)
    assert f.stat().st_mode & 0o777 == 0o444
    restore_permissions(saved)
    assert f.stat().st_mode == original


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def test_metadata_fields() -> None:
    cfg = ShellAgentConfig(
        command=["claude", "--print"],
        model="claude-sonnet-4-6",
        tools=["read_file", "write_file"],
        skills=["skill-a"],
        sdk_version="1.2.3",
    )
    agent = ShellWrapperAgent(cfg)
    meta = agent.metadata()
    assert meta.model == "claude-sonnet-4-6"
    assert meta.tools == ["read_file", "write_file"]
    assert meta.skills == ["skill-a"]
    assert meta.sdk_version == "1.2.3"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_claude_code_agent_command() -> None:
    agent = claude_code_agent(model="claude-haiku-4-5")
    assert agent._cfg.command == ["claude", "--print", "--model", "claude-haiku-4-5"]
    assert agent._cfg.print_mode is True


def test_claude_code_agent_extra_args() -> None:
    agent = claude_code_agent(extra_args=["--allowed-directories", "/tmp/in,/tmp/out"])
    assert "--allowed-directories" in agent._cfg.extra_args


def test_codex_agent_command() -> None:
    agent = codex_agent(model="gpt-4o")
    assert agent._cfg.command == ["codex", "-m", "gpt-4o"]


def test_gemini_agent_command() -> None:
    agent = gemini_agent(model="gemini-2.0-flash")
    assert agent._cfg.command == ["gemini", "--model", "gemini-2.0-flash"]


def test_factory_kwargs_forwarded() -> None:
    agent = claude_code_agent(
        model="claude-sonnet-4-6",
        skills=["skill-x"],
        print_mode=False,
    )
    assert agent._cfg.skills == ["skill-x"]
    assert agent._cfg.print_mode is False
