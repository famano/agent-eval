from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Criterion (観点)
# ---------------------------------------------------------------------------


@dataclass
class Criterion:
    id: str
    description: str
    importance: Literal["must", "should"]
    weight: float = 1.0
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    id: str
    input_dir: Path
    reference_files: list[Path]
    criteria: list[Criterion]
    timeout_s: int = 600
    budget_usd: float = 1.0


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


class RunStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    CRASHED = "crashed"


@dataclass
class RunResult:
    status: RunStatus
    output_files: list[Path]
    cost_usd: float
    latency_s: float
    tokens: int
    tool_calls: int
    trajectory_ref: Path | None = None


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class Verdict(Enum):
    MET = "met"
    PARTIAL = "partial"
    NOT_MET = "not_met"
    CONTRADICTED = "contradicted"


@dataclass
class CriterionResult:
    criterion_id: str
    verdict: Verdict
    rationale: str


@dataclass
class EvalError:
    type: Literal["contradiction", "unsupported", "format", "extra_neutral"]
    severity: Literal["critical", "major", "minor"]
    description: str


@dataclass
class EvaluationResult:
    dataset_id: str
    run_index: int
    criterion_results: list[CriterionResult]
    errors: list[EvalError]
