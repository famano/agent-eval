"""Model wrapper agents for shell-based CLI tools."""

from .base import (
    WrapperConfig,
    make_input_readonly,
    read_prompt,
    restore_permissions,
)
from .shell import (
    ShellAgentConfig,
    ShellWrapperAgent,
    claude_code_agent,
    codex_agent,
    gemini_agent,
)

__all__ = [
    "ShellAgentConfig",
    "ShellWrapperAgent",
    "WrapperConfig",
    "claude_code_agent",
    "codex_agent",
    "gemini_agent",
    "make_input_readonly",
    "read_prompt",
    "restore_permissions",
]
