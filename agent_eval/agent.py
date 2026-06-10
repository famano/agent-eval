from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import RunResult


@dataclass
class AgentMetadata:
    model: str
    sdk_version: str
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    temperature: float | None = None
    seed: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


class Agent(Protocol):
    def metadata(self) -> AgentMetadata: ...

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        timeout_s: int,
        budget_usd: float,
    ) -> RunResult: ...
