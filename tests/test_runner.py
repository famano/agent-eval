"""Tests for Iterator (Runner).

Behavioral specification source: design doc sections 5.4, 7.

What we verify (treating runner as a black box via DatasetReport):
  - Each run gets its own isolated output directory.
  - CRASHED status triggers retry up to max_infra_retries; after that, the run
    is excluded from eval_results and recorded in infra_failures (not as 0-score).
  - TIMEOUT and BUDGET_EXCEEDED are recorded in runs but NOT sent to the evaluator.
  - SUCCESS runs are forwarded to the evaluator.
  - n_repeats controls how many attempts are made.
  - Trajectory is saved per run (not in scoring, but as debug log).
  - An agent that raises an exception is treated as CRASHED.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_eval.agent import AgentMetadata
from agent_eval.models import (
    Criterion,
    CriterionResult,
    Dataset,
    EvaluationResult,
    RunResult,
    RunStatus,
    Verdict,
)
from agent_eval.runner import Iterator

# ---------------------------------------------------------------------------
# Stub agents
# ---------------------------------------------------------------------------


class AlwaysSucceedAgent:
    """Writes a single file and returns SUCCESS."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(model="stub-model", sdk_version="0.0.1", temperature=0.0, seed=42)

    def run(
        self, input_dir: Path, output_dir: Path, timeout_s: int, budget_usd: float
    ) -> RunResult:
        out = output_dir / "result.txt"
        out.write_text("output", encoding="utf-8")
        return RunResult(
            status=RunStatus.SUCCESS,
            output_files=[out],
            cost_usd=0.01,
            latency_s=0.1,
            tokens=50,
            tool_calls=1,
        )


class AlwaysCrashAgent:
    """Always returns CRASHED."""

    def metadata(self) -> MagicMock:
        return MagicMock()

    def run(
        self, input_dir: Path, output_dir: Path, timeout_s: int, budget_usd: float
    ) -> RunResult:
        return RunResult(
            status=RunStatus.CRASHED,
            output_files=[],
            cost_usd=0.0,
            latency_s=0.0,
            tokens=0,
            tool_calls=0,
        )


class AlwaysTimeoutAgent:
    def metadata(self) -> MagicMock:
        return MagicMock()

    def run(
        self, input_dir: Path, output_dir: Path, timeout_s: int, budget_usd: float
    ) -> RunResult:
        return RunResult(
            status=RunStatus.TIMEOUT,
            output_files=[],
            cost_usd=0.0,
            latency_s=float(timeout_s),
            tokens=0,
            tool_calls=0,
        )


class AlwaysBudgetAgent:
    def metadata(self) -> MagicMock:
        return MagicMock()

    def run(
        self, input_dir: Path, output_dir: Path, timeout_s: int, budget_usd: float
    ) -> RunResult:
        return RunResult(
            status=RunStatus.BUDGET_EXCEEDED,
            output_files=[],
            cost_usd=budget_usd + 1.0,
            latency_s=0.1,
            tokens=0,
            tool_calls=0,
        )


class RaisingAgent:
    """run() raises an exception — should be treated like CRASHED."""

    def metadata(self) -> MagicMock:
        return MagicMock()

    def run(
        self, input_dir: Path, output_dir: Path, timeout_s: int, budget_usd: float
    ) -> RunResult:
        raise RuntimeError("Agent exploded")


class CrashThenSucceedAgent:
    """Crashes on the first call, succeeds on the second."""

    def __init__(self) -> None:
        self._calls = 0

    def metadata(self) -> MagicMock:
        return MagicMock()

    def run(
        self, input_dir: Path, output_dir: Path, timeout_s: int, budget_usd: float
    ) -> RunResult:
        self._calls += 1
        if self._calls == 1:
            return RunResult(
                status=RunStatus.CRASHED,
                output_files=[],
                cost_usd=0.0,
                latency_s=0.0,
                tokens=0,
                tool_calls=0,
            )
        out = output_dir / "result.txt"
        out.write_text("output", encoding="utf-8")
        return RunResult(
            status=RunStatus.SUCCESS,
            output_files=[out],
            cost_usd=0.01,
            latency_s=0.1,
            tokens=50,
            tool_calls=1,
        )


# ---------------------------------------------------------------------------
# Stub evaluator
# ---------------------------------------------------------------------------


def _make_stub_evaluator(dataset_id: str = "ds-test-001") -> MagicMock:
    evaluator = MagicMock()
    call_count = {"n": 0}

    def fake_evaluate(output_files, reference_files, criteria, dataset_id, run_index):
        call_count["n"] += 1
        return EvaluationResult(
            dataset_id=dataset_id,
            run_index=run_index,
            criterion_results=[
                CriterionResult(
                    criterion_id=c.id, verdict=Verdict.MET, rationale="stub"
                )
                for c in criteria
            ],
            errors=[],
        )

    evaluator.evaluate.side_effect = fake_evaluate
    return evaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_dataset(tmp_path: Path) -> Dataset:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "doc.txt").write_text("input content", encoding="utf-8")

    ref = tmp_path / "ref" / "ref.txt"
    ref.parent.mkdir()
    ref.write_text("reference content", encoding="utf-8")

    return Dataset(
        id="ds-test-001",
        input_dir=input_dir,
        reference_files=[ref],
        criteria=[
            Criterion(id="c1", description="desc", importance="must", weight=1.0)
        ],
        timeout_s=30,
        budget_usd=1.0,
    )


# ---------------------------------------------------------------------------
# n_repeats
# ---------------------------------------------------------------------------


class TestNRepeats:
    def test_n_repeats_controls_total_runs(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        agent = AlwaysSucceedAgent()
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(simple_dataset, agent, evaluator, n_repeats=4)
        assert len(report.runs) == 4

    def test_zero_repeats_returns_empty_report(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        agent = AlwaysSucceedAgent()
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(simple_dataset, agent, evaluator, n_repeats=0)
        assert len(report.runs) == 0
        assert len(report.eval_results) == 0


# ---------------------------------------------------------------------------
# Run isolation
# ---------------------------------------------------------------------------


class TestRunIsolation:
    def test_each_run_uses_separate_output_directory(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        seen_dirs: list[Path] = []

        class TrackingAgent:
            def metadata(self) -> MagicMock:
                return MagicMock()

            def run(
                self,
                input_dir: Path,
                output_dir: Path,
                timeout_s: int,
                budget_usd: float,
            ) -> RunResult:
                seen_dirs.append(output_dir)
                out = output_dir / "result.txt"
                out.write_text("x", encoding="utf-8")
                return RunResult(
                    status=RunStatus.SUCCESS,
                    output_files=[out],
                    cost_usd=0.0,
                    latency_s=0.0,
                    tokens=0,
                    tool_calls=0,
                )

        runner = Iterator(output_root=tmp_path / "runs")
        runner.run(simple_dataset, TrackingAgent(), _make_stub_evaluator(), n_repeats=3)
        assert len(set(seen_dirs)) == 3, "Each run must have a unique output directory"

    def test_output_dirs_are_disjoint(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """No run's output_dir should be an ancestor of another's."""
        seen_dirs: list[Path] = []

        class TrackingAgent:
            def metadata(self) -> MagicMock:
                return MagicMock()

            def run(
                self,
                input_dir: Path,
                output_dir: Path,
                timeout_s: int,
                budget_usd: float,
            ) -> RunResult:
                seen_dirs.append(output_dir.resolve())
                out = output_dir / "r.txt"
                out.write_text("y", encoding="utf-8")
                return RunResult(
                    status=RunStatus.SUCCESS,
                    output_files=[out],
                    cost_usd=0.0,
                    latency_s=0.0,
                    tokens=0,
                    tool_calls=0,
                )

        runner = Iterator(output_root=tmp_path / "runs")
        runner.run(simple_dataset, TrackingAgent(), _make_stub_evaluator(), n_repeats=3)
        for i, a in enumerate(seen_dirs):
            for j, b in enumerate(seen_dirs):
                if i != j:
                    assert a != b


# ---------------------------------------------------------------------------
# SUCCESS → evaluator called
# ---------------------------------------------------------------------------


class TestSuccessRouting:
    def test_success_run_passed_to_evaluator(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        runner.run(simple_dataset, AlwaysSucceedAgent(), evaluator, n_repeats=3)
        assert evaluator.evaluate.call_count == 3

    def test_eval_results_match_n_repeats_on_all_success(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset, AlwaysSucceedAgent(), evaluator, n_repeats=3
        )
        assert len(report.eval_results) == 3


# ---------------------------------------------------------------------------
# CRASHED handling
# ---------------------------------------------------------------------------


class TestCrashedHandling:
    def test_crash_exhausted_excluded_from_eval(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """After max_infra_retries exhausted, run must NOT appear in eval_results."""
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset,
            AlwaysCrashAgent(),
            evaluator,
            n_repeats=2,
            max_infra_retries=1,
        )
        assert evaluator.evaluate.call_count == 0
        assert len(report.eval_results) == 0

    def test_crash_exhausted_recorded_in_infra_failures(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """Infra failures must be tracked separately — NOT scored as 0."""
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset,
            AlwaysCrashAgent(),
            evaluator,
            n_repeats=2,
            max_infra_retries=0,
        )
        assert len(report.infra_failures) == 2

    def test_crash_then_success_retried_and_evaluated(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """CRASHED on attempt 0 → retry → SUCCESS on attempt 1 → evaluated."""
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset,
            CrashThenSucceedAgent(),
            evaluator,
            n_repeats=1,
            max_infra_retries=1,
        )
        assert evaluator.evaluate.call_count == 1
        assert len(report.infra_failures) == 0

    def test_exception_in_agent_treated_as_crash(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """An unhandled exception from agent.run() is treated like CRASHED."""
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset,
            RaisingAgent(),
            evaluator,
            n_repeats=1,
            max_infra_retries=0,
        )
        assert evaluator.evaluate.call_count == 0
        assert len(report.infra_failures) == 1


# ---------------------------------------------------------------------------
# TIMEOUT / BUDGET_EXCEEDED handling
# ---------------------------------------------------------------------------


class TestNonSuccessStatus:
    def test_timeout_not_sent_to_evaluator(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        runner.run(simple_dataset, AlwaysTimeoutAgent(), evaluator, n_repeats=2)
        assert evaluator.evaluate.call_count == 0

    def test_timeout_recorded_in_runs(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset, AlwaysTimeoutAgent(), evaluator, n_repeats=2
        )
        assert all(r.status == RunStatus.TIMEOUT for r in report.runs)

    def test_budget_exceeded_not_sent_to_evaluator(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        runner.run(simple_dataset, AlwaysBudgetAgent(), evaluator, n_repeats=2)
        assert evaluator.evaluate.call_count == 0

    def test_timeout_not_in_infra_failures(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """TIMEOUT is a task failure, not an infra failure — separate accounting."""
        evaluator = _make_stub_evaluator()
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset, AlwaysTimeoutAgent(), evaluator, n_repeats=2
        )
        # TIMEOUT should NOT bleed into infra_failures
        assert len(report.infra_failures) == 0


# ---------------------------------------------------------------------------
# DatasetReport fields
# ---------------------------------------------------------------------------


class TestDatasetReport:
    def test_report_dataset_id_matches(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset, AlwaysSucceedAgent(), _make_stub_evaluator(), n_repeats=1
        )
        assert report.dataset_id == simple_dataset.id

    def test_report_n_requested_matches(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset, AlwaysSucceedAgent(), _make_stub_evaluator(), n_repeats=5
        )
        assert report.n_requested == 5

    def test_trajectory_saved_for_successful_run(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """Design §7: trajectory (cost/latency/tokens) logged per run for debugging."""
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset, AlwaysSucceedAgent(), _make_stub_evaluator(), n_repeats=1
        )
        successful = [r for r in report.runs if r.status == RunStatus.SUCCESS]
        assert len(successful) == 1
        traj = successful[0].trajectory_ref
        assert traj is not None
        assert traj.exists()

    def test_agent_metadata_recorded_in_report(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """Design §7: AgentMetadata (model/seed/temperature) must be recorded."""
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset, AlwaysSucceedAgent(), _make_stub_evaluator(), n_repeats=1
        )
        assert report.agent_metadata["model"] == "stub-model"
        assert report.agent_metadata["sdk_version"] == "0.0.1"
        assert report.agent_metadata["seed"] == 42
        assert report.agent_metadata["temperature"] == 0.0

    def test_agent_metadata_in_trajectory_json(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """Design §7: trajectory JSON must include agent metadata for reproducibility."""
        import json

        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset, AlwaysSucceedAgent(), _make_stub_evaluator(), n_repeats=1
        )
        traj_path = report.runs[0].trajectory_ref
        assert traj_path is not None
        data = json.loads(traj_path.read_text())
        assert "agent_metadata" in data
        assert data["agent_metadata"]["model"] == "stub-model"

    def test_evaluator_metadata_recorded_in_report(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """Design §7: judge model/sampling settings must be recorded alongside results."""
        evaluator = _make_stub_evaluator()
        evaluator.model = "judge-model"
        evaluator.n_samples = 3
        evaluator.calibration_threshold = 0.9
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(simple_dataset, AlwaysSucceedAgent(), evaluator, n_repeats=1)
        assert report.evaluator_metadata["model"] == "judge-model"
        assert report.evaluator_metadata["n_samples"] == 3
        assert report.evaluator_metadata["calibration_threshold"] == 0.9

    def test_evaluator_metadata_in_trajectory_json(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """Design §7: trajectory JSON must include evaluator metadata."""
        import json

        evaluator = _make_stub_evaluator()
        evaluator.model = "judge-model"
        evaluator.n_samples = 2
        evaluator.calibration_threshold = 0.95
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(simple_dataset, AlwaysSucceedAgent(), evaluator, n_repeats=1)
        traj_path = report.runs[0].trajectory_ref
        assert traj_path is not None
        data = json.loads(traj_path.read_text())
        assert "evaluator_metadata" in data
        assert data["evaluator_metadata"]["model"] == "judge-model"


class TestEvaluateTimeout:
    def test_timeout_not_evaluated_by_default(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """Design §5.4: TIMEOUT runs are recorded but not evaluated by default."""
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset, AlwaysTimeoutAgent(), _make_stub_evaluator(), n_repeats=2
        )
        assert len(report.runs) == 2
        assert all(r.status == RunStatus.TIMEOUT for r in report.runs)
        assert len(report.eval_results) == 0

    def test_timeout_evaluated_when_opted_in(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """Design §5.4: TIMEOUT evaluation is optional (方針で選択)."""
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset,
            AlwaysTimeoutAgent(),
            _make_stub_evaluator(),
            n_repeats=2,
            evaluate_timeout=True,
        )
        assert len(report.runs) == 2
        assert len(report.eval_results) == 2

    def test_budget_exceeded_evaluated_when_opted_in(
        self, tmp_path: Path, simple_dataset: Dataset
    ) -> None:
        """Design §5.4: BUDGET_EXCEEDED evaluation is optional."""
        runner = Iterator(output_root=tmp_path / "runs")
        report = runner.run(
            simple_dataset,
            AlwaysBudgetAgent(),
            _make_stub_evaluator(),
            n_repeats=1,
            evaluate_timeout=True,
        )
        assert len(report.eval_results) == 1
