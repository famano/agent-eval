"""Shell-based wrapper agents for CLI tools (Claude Code, Codex, Gemini, …)."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent import AgentMetadata
from ..models import RunResult, RunStatus
from .base import make_input_readonly, read_prompt, restore_permissions


@dataclass
class ShellAgentConfig:
    """Configuration for a shell-invoked agent.

    Parameters
    ----------
    command:
        Base command list.  Example: ``["claude", "--print"]``.
        The task prompt is delivered via *stdin*.
    model:
        Model identifier recorded in :class:`~agent_eval.AgentMetadata`.
    tools:
        Tool name strings recorded in metadata (not forwarded to the CLI
        automatically — pass them via *extra_args* if the CLI accepts them).
    skills:
        Skill labels for experiment tracking.
    env:
        Environment variables to add or override for the subprocess.
    readonly_input:
        When ``True`` (default), make *input_dir* read-only before the
        subprocess runs (Unix only; no-op on Windows).  Restored in a
        ``finally`` block regardless of outcome.
    print_mode:
        ``True`` (default): capture stdout and write it to
        ``output_dir/response.txt``.
        ``False``: the agent is expected to write files into *output_dir*
        itself (the path is injected via *output_dir_env_var* and appended
        to the prompt).
    output_dir_env_var:
        Name of the environment variable set to ``str(output_dir)`` before
        the subprocess runs.  Useful when *print_mode* is ``False``.
    extra_args:
        Additional arguments appended to *command* before execution.
    sdk_version:
        Version string recorded in :class:`~agent_eval.AgentMetadata`.
    """

    command: list[str]
    model: str
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    readonly_input: bool = True
    print_mode: bool = True
    output_dir_env_var: str = "OUTPUT_DIR"
    extra_args: list[str] = field(default_factory=list)
    sdk_version: str = "unknown"


class ShellWrapperAgent:
    """Runs an external CLI agent via subprocess and collects its output.

    The task prompt is read from ``input_dir/prompt.md`` (see
    :func:`~agent_eval.wrappers.base.read_prompt`) and delivered to the
    subprocess via *stdin*.

    Sandbox behaviour
    -----------------
    * ``readonly_input=True`` (default): all files and directories inside
      *input_dir* are made read-only before the subprocess runs (Unix only).
      This prevents accidental or deliberate writes that would corrupt the
      evaluation dataset.
    * For CLI tools that support it (e.g. Claude Code), pass
      ``--allowed-directories`` via *extra_args* to restrict the agent's
      tool access to *input_dir* and *output_dir* only:

      .. code-block:: python

          agent = claude_code_agent(
              extra_args=["--allowed-directories", f"{input_dir},{output_dir}"]
          )

    Output collection
    -----------------
    * ``print_mode=True`` (default): the subprocess stdout is written to
      ``output_dir/response.txt``.
    * ``print_mode=False``: the agent is expected to write files directly
      into *output_dir*.  The framework appends an instruction to the prompt
      and sets the ``OUTPUT_DIR`` environment variable so the agent knows
      where to write.
    """

    def __init__(self, config: ShellAgentConfig) -> None:
        self._cfg = config

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            model=self._cfg.model,
            sdk_version=self._cfg.sdk_version,
            tools=list(self._cfg.tools),
            skills=list(self._cfg.skills),
        )

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        timeout_s: int,
        budget_usd: float,
    ) -> RunResult:
        t0 = time.monotonic()

        try:
            prompt = read_prompt(input_dir)
        except FileNotFoundError:
            return RunResult(
                status=RunStatus.CRASHED,
                output_files=[],
                cost_usd=0.0,
                latency_s=time.monotonic() - t0,
                tokens=0,
                tool_calls=0,
            )

        if not self._cfg.print_mode:
            prompt = f"{prompt}\n\n---\nWrite all output files to: {output_dir}"

        merged_env = {**os.environ, **self._cfg.env}
        merged_env[self._cfg.output_dir_env_var] = str(output_dir)

        saved_perms: list[tuple[Path, int]] = []
        if self._cfg.readonly_input:
            saved_perms = make_input_readonly(input_dir)

        try:
            proc = subprocess.run(
                self._cfg.command + self._cfg.extra_args,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=str(input_dir),
                timeout=timeout_s,
                env=merged_env,
            )
        except subprocess.TimeoutExpired:
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
        finally:
            restore_permissions(saved_perms)

        if proc.returncode != 0:
            return RunResult(
                status=RunStatus.CRASHED,
                output_files=[],
                cost_usd=0.0,
                latency_s=time.monotonic() - t0,
                tokens=0,
                tool_calls=0,
            )

        output_files: list[Path] = []
        if self._cfg.print_mode:
            response_file = output_dir / "response.txt"
            response_file.write_text(proc.stdout, encoding="utf-8")
            output_files = [response_file]
        else:
            output_files = [p for p in output_dir.rglob("*") if p.is_file()]

        return RunResult(
            status=RunStatus.SUCCESS,
            output_files=output_files,
            cost_usd=0.0,
            latency_s=time.monotonic() - t0,
            tokens=0,
            tool_calls=0,
        )


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def claude_code_agent(
    model: str = "claude-sonnet-4-6",
    extra_args: list[str] | None = None,
    **kwargs: Any,
) -> ShellWrapperAgent:
    """Return a :class:`ShellWrapperAgent` pre-configured for Claude Code CLI.

    Invokes ``claude --print --model <model>`` with the prompt on stdin.

    To restrict the agent's file-system access to *input_dir* and *output_dir*
    only, pass ``--allowed-directories`` via *extra_args*:

    .. code-block:: python

        agent = claude_code_agent(
            extra_args=["--allowed-directories", f"{input_dir},{output_dir}"]
        )
    """
    return ShellWrapperAgent(
        ShellAgentConfig(
            command=["claude", "--print", "--model", model],
            model=model,
            extra_args=extra_args or [],
            **kwargs,
        )
    )


def codex_agent(
    model: str = "gpt-4o",
    extra_args: list[str] | None = None,
    **kwargs: Any,
) -> ShellWrapperAgent:
    """Return a :class:`ShellWrapperAgent` pre-configured for OpenAI Codex CLI.

    Requires the ``codex`` binary to be available in ``PATH``.
    """
    return ShellWrapperAgent(
        ShellAgentConfig(
            command=["codex", "-m", model],
            model=model,
            extra_args=extra_args or [],
            **kwargs,
        )
    )


def gemini_agent(
    model: str = "gemini-1.5-pro",
    extra_args: list[str] | None = None,
    **kwargs: Any,
) -> ShellWrapperAgent:
    """Return a :class:`ShellWrapperAgent` pre-configured for Google Gemini CLI.

    Requires the ``gemini`` binary to be available in ``PATH``.
    """
    return ShellWrapperAgent(
        ShellAgentConfig(
            command=["gemini", "--model", model],
            model=model,
            extra_args=extra_args or [],
            **kwargs,
        )
    )
