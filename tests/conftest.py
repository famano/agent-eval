"""Shared fixtures for agent_eval tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.models import (
    Criterion,
    CriterionResult,
    Dataset,
    EvalError,
    EvaluationResult,
    RunResult,
    RunStatus,
    Verdict,
)

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_input_dir(tmp_path: Path) -> Path:
    d = tmp_path / "input"
    d.mkdir()
    (d / "source.txt").write_text(
        "Widget Corp acquired Acme Inc for $50M.", encoding="utf-8"
    )
    return d


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture()
def reference_file(tmp_path: Path) -> Path:
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    p = ref_dir / "report.txt"
    p.write_text(
        "Acquisition: Widget Corp acquired Acme Inc.\n"
        "Deal value: $50M.\n"
        "No regulatory issues identified.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def output_file(tmp_path: Path) -> Path:
    out = tmp_path / "output" / "report.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        "Widget Corp acquired Acme Inc for $50M. No regulatory concerns found.",
        encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


@pytest.fixture()
def must_criterion() -> Criterion:
    return Criterion(
        id="c_must",
        description="Report names both parties: Widget Corp (acquirer) and Acme Inc (target).",
        importance="must",
        weight=2.0,
        tags=["parties"],
    )


@pytest.fixture()
def should_criterion() -> Criterion:
    return Criterion(
        id="c_should",
        description="Report states the deal value of $50M.",
        importance="should",
        weight=1.0,
        tags=["financials"],
    )


@pytest.fixture()
def criteria(must_criterion: Criterion, should_criterion: Criterion) -> list[Criterion]:
    return [must_criterion, should_criterion]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@pytest.fixture()
def dataset(
    tmp_input_dir: Path,
    reference_file: Path,
    criteria: list[Criterion],
) -> Dataset:
    return Dataset(
        id="ds-test-001",
        input_dir=tmp_input_dir,
        reference_files=[reference_file],
        criteria=criteria,
        timeout_s=60,
        budget_usd=0.5,
    )


# ---------------------------------------------------------------------------
# Pre-built EvaluationResult helpers
# ---------------------------------------------------------------------------


def make_eval_result(
    dataset_id: str = "ds-test-001",
    run_index: int = 0,
    verdicts: dict[str, Verdict] | None = None,
    errors: list[EvalError] | None = None,
) -> EvaluationResult:
    verdicts = verdicts or {"c_must": Verdict.MET, "c_should": Verdict.MET}
    return EvaluationResult(
        dataset_id=dataset_id,
        run_index=run_index,
        criterion_results=[
            CriterionResult(criterion_id=cid, verdict=v, rationale="test")
            for cid, v in verdicts.items()
        ],
        errors=errors or [],
    )


def make_run_result(
    status: RunStatus = RunStatus.SUCCESS,
    output_files: list[Path] | None = None,
    cost_usd: float = 0.01,
    latency_s: float = 1.0,
    tokens: int = 100,
    tool_calls: int = 2,
) -> RunResult:
    return RunResult(
        status=status,
        output_files=output_files or [],
        cost_usd=cost_usd,
        latency_s=latency_s,
        tokens=tokens,
        tool_calls=tool_calls,
    )
