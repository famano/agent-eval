"""Tests for scoring and statistical aggregation.

Behavioral specification source: design doc sections 6.1, 6.2, 6.3.

What we verify (treating scoring as a black box):
  - Coverage formula: MET=1.0, PARTIAL=0.5, NOT_MET=0.0, CONTRADICTED=0.0.
  - must-importance criteria carry higher weight than should in the weighted sum.
  - Error counts split correctly by severity (critical / major / minor).
  - aggregate_dataset() reports coverage mean, std, and 95% CI.
  - pass@k is a probability in [0, 1]; increases with k.
  - critical_error_rate is the fraction of runs that had ≥1 critical error.
  - Infra failures and TIMEOUT/BUDGET runs are counted separately from evaluated runs.
  - Per-tag coverage slices aggregate only the criteria carrying that tag.
  - Cost / latency summary statistics are populated when RunResult data is present.
  - SuiteReport.print_summary() runs without error and prints dataset_id.
"""
from __future__ import annotations

import math
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from agent_eval.models import (
    Criterion,
    Dataset,
    EvalError,
    EvaluationResult,
    CriterionResult,
    RunResult,
    RunStatus,
    Verdict,
)
from agent_eval.runner import DatasetReport
from agent_eval.scoring import (
    DatasetStats,
    RunMetrics,
    SuiteReport,
    aggregate_dataset,
    compute_run_metrics,
)
from tests.conftest import make_eval_result, make_run_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_criteria() -> list[Criterion]:
    return [
        Criterion(id="c_must", description="must", importance="must", weight=2.0, tags=["legal"]),
        Criterion(id="c_should", description="should", importance="should", weight=1.0, tags=["financial"]),
    ]


@pytest.fixture()
def simple_dataset(tmp_path: Path, two_criteria: list[Criterion]) -> Dataset:
    inp = tmp_path / "input"
    inp.mkdir()
    ref = tmp_path / "ref.txt"
    ref.write_text("reference", encoding="utf-8")
    return Dataset(
        id="ds-score-001",
        input_dir=inp,
        reference_files=[ref],
        criteria=two_criteria,
        timeout_s=60,
        budget_usd=1.0,
    )


# ---------------------------------------------------------------------------
# compute_run_metrics — coverage formula
# ---------------------------------------------------------------------------

class TestCoverageFormula:
    def test_all_met_gives_coverage_one(self, two_criteria: list[Criterion]) -> None:
        er = make_eval_result(verdicts={"c_must": Verdict.MET, "c_should": Verdict.MET})
        m = compute_run_metrics(er, two_criteria)
        assert math.isclose(m.coverage, 1.0)

    def test_all_not_met_gives_coverage_zero(self, two_criteria: list[Criterion]) -> None:
        er = make_eval_result(verdicts={"c_must": Verdict.NOT_MET, "c_should": Verdict.NOT_MET})
        m = compute_run_metrics(er, two_criteria)
        assert math.isclose(m.coverage, 0.0)

    def test_contradicted_counts_as_zero(self, two_criteria: list[Criterion]) -> None:
        er = make_eval_result(verdicts={"c_must": Verdict.CONTRADICTED, "c_should": Verdict.NOT_MET})
        m = compute_run_metrics(er, two_criteria)
        assert math.isclose(m.coverage, 0.0)

    def test_partial_counts_as_half(self, two_criteria: list[Criterion]) -> None:
        """Only the should criterion (weight=1) is PARTIAL; must (weight=2) is MET.
        Expected: (2*1.0 + 1*0.5) / (2+1) = 2.5/3 ≈ 0.8333"""
        er = make_eval_result(verdicts={"c_must": Verdict.MET, "c_should": Verdict.PARTIAL})
        m = compute_run_metrics(er, two_criteria)
        expected = (2 * 1.0 + 1 * 0.5) / 3.0
        assert math.isclose(m.coverage, expected, rel_tol=1e-6)

    def test_must_weight_dominates(self, two_criteria: list[Criterion]) -> None:
        """must(w=2) MET, should(w=1) NOT_MET → coverage = 2/3 > 0.5"""
        er = make_eval_result(verdicts={"c_must": Verdict.MET, "c_should": Verdict.NOT_MET})
        m = compute_run_metrics(er, two_criteria)
        assert m.coverage > 0.5

    def test_should_only_met_gives_less_than_must(self, two_criteria: list[Criterion]) -> None:
        """should(w=1) MET, must(w=2) NOT_MET → coverage = 1/3 < 0.5"""
        er = make_eval_result(verdicts={"c_must": Verdict.NOT_MET, "c_should": Verdict.MET})
        m = compute_run_metrics(er, two_criteria)
        assert m.coverage < 0.5

    def test_coverage_bounded_zero_to_one(self, two_criteria: list[Criterion]) -> None:
        for verdicts in [
            {"c_must": Verdict.MET, "c_should": Verdict.MET},
            {"c_must": Verdict.NOT_MET, "c_should": Verdict.NOT_MET},
            {"c_must": Verdict.PARTIAL, "c_should": Verdict.CONTRADICTED},
        ]:
            er = make_eval_result(verdicts=verdicts)
            m = compute_run_metrics(er, two_criteria)
            assert 0.0 <= m.coverage <= 1.0


# ---------------------------------------------------------------------------
# compute_run_metrics — error severity counts
# ---------------------------------------------------------------------------

class TestErrorSeverityCounts:
    def test_no_errors(self, two_criteria: list[Criterion]) -> None:
        er = make_eval_result(errors=[])
        m = compute_run_metrics(er, two_criteria)
        assert m.critical_error_count == 0
        assert m.major_error_count == 0
        assert m.minor_error_count == 0

    def test_critical_counted_separately(self, two_criteria: list[Criterion]) -> None:
        errors = [
            EvalError(type="contradiction", severity="critical", description="bad"),
            EvalError(type="unsupported", severity="major", description="maybe"),
            EvalError(type="format", severity="minor", description="style"),
        ]
        er = make_eval_result(errors=errors)
        m = compute_run_metrics(er, two_criteria)
        assert m.critical_error_count == 1
        assert m.major_error_count == 1
        assert m.minor_error_count == 1

    def test_multiple_critical_errors(self, two_criteria: list[Criterion]) -> None:
        errors = [
            EvalError(type="contradiction", severity="critical", description="a"),
            EvalError(type="contradiction", severity="critical", description="b"),
        ]
        er = make_eval_result(errors=errors)
        m = compute_run_metrics(er, two_criteria)
        assert m.critical_error_count == 2


# ---------------------------------------------------------------------------
# compute_run_metrics — run cost/latency propagation
# ---------------------------------------------------------------------------

class TestRunMetricsPropagation:
    def test_cost_latency_from_run_result(self, two_criteria: list[Criterion]) -> None:
        er = make_eval_result()
        rr = make_run_result(cost_usd=0.42, latency_s=3.7, tokens=200, tool_calls=5)
        m = compute_run_metrics(er, two_criteria, run_result=rr)
        assert math.isclose(m.cost_usd, 0.42)
        assert math.isclose(m.latency_s, 3.7)
        assert m.tokens == 200
        assert m.tool_calls == 5

    def test_no_run_result_gives_none(self, two_criteria: list[Criterion]) -> None:
        er = make_eval_result()
        m = compute_run_metrics(er, two_criteria, run_result=None)
        assert m.cost_usd is None
        assert m.latency_s is None


# ---------------------------------------------------------------------------
# aggregate_dataset — basic statistics
# ---------------------------------------------------------------------------

def _build_report(
    dataset: Dataset,
    verdicts_per_run: list[dict[str, Verdict]],
    errors_per_run: list[list[EvalError]] | None = None,
    extra_runs: list[RunResult] | None = None,
    infra_failures: list[dict] | None = None,
) -> DatasetReport:
    errors_per_run = errors_per_run or [[] for _ in verdicts_per_run]
    report = DatasetReport(
        dataset_id=dataset.id,
        n_requested=len(verdicts_per_run),
    )
    for i, (vd, errs) in enumerate(zip(verdicts_per_run, errors_per_run)):
        report.runs.append(make_run_result())
        report.eval_results.append(make_eval_result(
            dataset_id=dataset.id,
            run_index=i,
            verdicts=vd,
            errors=errs,
        ))
    if extra_runs:
        report.runs.extend(extra_runs)
    if infra_failures:
        report.infra_failures.extend(infra_failures)
    return report


class TestAggregateDatasetCoverage:
    def test_mean_coverage_all_perfect(
        self, simple_dataset: Dataset, two_criteria: list[Criterion]
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(simple_dataset, [all_met] * 5)
        stats = aggregate_dataset(report, simple_dataset)
        assert math.isclose(stats.coverage.mean, 1.0)

    def test_mean_coverage_all_zero(
        self, simple_dataset: Dataset
    ) -> None:
        all_nm = {"c_must": Verdict.NOT_MET, "c_should": Verdict.NOT_MET}
        report = _build_report(simple_dataset, [all_nm] * 4)
        stats = aggregate_dataset(report, simple_dataset)
        assert math.isclose(stats.coverage.mean, 0.0)

    def test_std_zero_when_all_runs_identical(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(simple_dataset, [all_met] * 4)
        stats = aggregate_dataset(report, simple_dataset)
        assert math.isclose(stats.coverage.std, 0.0, abs_tol=1e-9)

    def test_std_nonzero_with_varying_runs(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        all_nm = {"c_must": Verdict.NOT_MET, "c_should": Verdict.NOT_MET}
        report = _build_report(simple_dataset, [all_met, all_nm, all_met, all_nm])
        stats = aggregate_dataset(report, simple_dataset)
        assert stats.coverage.std > 0.0

    def test_ci_contains_mean(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        all_nm = {"c_must": Verdict.NOT_MET, "c_should": Verdict.NOT_MET}
        report = _build_report(simple_dataset, [all_met, all_nm] * 3)
        stats = aggregate_dataset(report, simple_dataset, n_bootstrap=200)
        assert stats.coverage.ci_low <= stats.coverage.mean <= stats.coverage.ci_high

    def test_n_in_coverage_stats_equals_evaluated_runs(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(simple_dataset, [all_met] * 3)
        stats = aggregate_dataset(report, simple_dataset)
        assert stats.coverage.n == 3


# ---------------------------------------------------------------------------
# aggregate_dataset — pass@k
# ---------------------------------------------------------------------------

class TestPassAtK:
    def test_pass_at_1_equals_fraction_above_threshold(
        self, simple_dataset: Dataset
    ) -> None:
        """pass@1 ≈ fraction of runs above threshold when all runs are independent."""
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        all_nm = {"c_must": Verdict.NOT_MET, "c_should": Verdict.NOT_MET}
        # 3 out of 6 pass (coverage=1.0 > threshold=0.7)
        report = _build_report(simple_dataset, [all_met] * 3 + [all_nm] * 3)
        stats = aggregate_dataset(report, simple_dataset, coverage_threshold=0.7, pass_at_ks=[1])
        assert 0.0 <= stats.pass_at_k[1] <= 1.0

    def test_pass_at_k_increases_with_k(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        all_nm = {"c_must": Verdict.NOT_MET, "c_should": Verdict.NOT_MET}
        report = _build_report(simple_dataset, [all_met] * 2 + [all_nm] * 3)
        stats = aggregate_dataset(
            report, simple_dataset, coverage_threshold=0.7, pass_at_ks=[1, 3, 5]
        )
        if 1 in stats.pass_at_k and 3 in stats.pass_at_k:
            assert stats.pass_at_k[1] <= stats.pass_at_k[3]
        if 3 in stats.pass_at_k and 5 in stats.pass_at_k:
            assert stats.pass_at_k[3] <= stats.pass_at_k[5]

    def test_pass_at_k_one_when_all_pass(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(simple_dataset, [all_met] * 5)
        stats = aggregate_dataset(report, simple_dataset, coverage_threshold=0.7, pass_at_ks=[1, 3])
        for k, v in stats.pass_at_k.items():
            assert math.isclose(v, 1.0), f"pass@{k} should be 1.0"

    def test_pass_at_k_zero_when_none_pass(
        self, simple_dataset: Dataset
    ) -> None:
        all_nm = {"c_must": Verdict.NOT_MET, "c_should": Verdict.NOT_MET}
        report = _build_report(simple_dataset, [all_nm] * 5)
        stats = aggregate_dataset(report, simple_dataset, coverage_threshold=0.7, pass_at_ks=[1, 3])
        for k, v in stats.pass_at_k.items():
            assert math.isclose(v, 0.0), f"pass@{k} should be 0.0"


# ---------------------------------------------------------------------------
# aggregate_dataset — critical error rate
# ---------------------------------------------------------------------------

class TestCriticalErrorRate:
    def test_zero_when_no_critical_errors(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(simple_dataset, [all_met] * 4)
        stats = aggregate_dataset(report, simple_dataset)
        assert math.isclose(stats.critical_error_rate, 0.0)

    def test_one_when_all_runs_have_critical_error(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        critical = [EvalError(type="contradiction", severity="critical", description="c")]
        report = _build_report(
            simple_dataset,
            [all_met] * 3,
            errors_per_run=[critical] * 3,
        )
        stats = aggregate_dataset(report, simple_dataset)
        assert math.isclose(stats.critical_error_rate, 1.0)

    def test_partial_critical_error_rate(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        critical = [EvalError(type="contradiction", severity="critical", description="c")]
        report = _build_report(
            simple_dataset,
            [all_met] * 4,
            errors_per_run=[critical, [], critical, []],
        )
        stats = aggregate_dataset(report, simple_dataset)
        assert math.isclose(stats.critical_error_rate, 0.5)


# ---------------------------------------------------------------------------
# aggregate_dataset — infra failure accounting
# ---------------------------------------------------------------------------

class TestInfraFailureAccounting:
    def test_infra_failures_counted(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(
            simple_dataset, [all_met] * 2,
            infra_failures=[{"run_index": 2, "reason": "crash"}] * 3,
        )
        stats = aggregate_dataset(report, simple_dataset)
        assert stats.n_infra_failures == 3

    def test_infra_failures_not_counted_in_evaluated(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(
            simple_dataset, [all_met] * 2,
            infra_failures=[{"run_index": 2, "reason": "crash"}],
        )
        stats = aggregate_dataset(report, simple_dataset)
        assert stats.n_runs_evaluated == 2

    def test_timeout_counted_separately(
        self, simple_dataset: Dataset
    ) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        timeout_run = make_run_result(status=RunStatus.TIMEOUT)
        report = _build_report(simple_dataset, [all_met] * 2, extra_runs=[timeout_run])
        stats = aggregate_dataset(report, simple_dataset)
        assert stats.n_timeout_or_budget >= 1


# ---------------------------------------------------------------------------
# aggregate_dataset — per-tag coverage
# ---------------------------------------------------------------------------

class TestPerTagCoverage:
    def test_tag_coverage_present_for_tagged_criteria(
        self, simple_dataset: Dataset
    ) -> None:
        """Criteria have tags 'legal' and 'financial'."""
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(simple_dataset, [all_met] * 3)
        stats = aggregate_dataset(report, simple_dataset)
        assert "legal" in stats.per_tag_coverage
        assert "financial" in stats.per_tag_coverage

    def test_tag_coverage_reflects_verdict(
        self, simple_dataset: Dataset
    ) -> None:
        """'legal' tag → c_must (MET=1.0); 'financial' tag → c_should (NOT_MET=0.0)."""
        verdicts = {"c_must": Verdict.MET, "c_should": Verdict.NOT_MET}
        report = _build_report(simple_dataset, [verdicts] * 4)
        stats = aggregate_dataset(report, simple_dataset)
        assert math.isclose(stats.per_tag_coverage["legal"].mean, 1.0)
        assert math.isclose(stats.per_tag_coverage["financial"].mean, 0.0)

    def test_no_tags_gives_empty_per_tag(self, tmp_path: Path) -> None:
        inp = tmp_path / "input"
        inp.mkdir()
        ref = tmp_path / "ref.txt"
        ref.write_text("ref")
        criteria_no_tags = [
            Criterion(id="c1", description="d", importance="must", weight=1.0, tags=[]),
        ]
        ds = Dataset(
            id="ds-notag",
            input_dir=inp,
            reference_files=[ref],
            criteria=criteria_no_tags,
        )
        report = _build_report(ds, [{"c1": Verdict.MET}] * 2)
        stats = aggregate_dataset(report, ds)
        assert stats.per_tag_coverage == {}


# ---------------------------------------------------------------------------
# aggregate_dataset — cost / latency summaries
# ---------------------------------------------------------------------------

class TestCostLatencyStats:
    def test_cost_stats_populated(self, simple_dataset: Dataset) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = DatasetReport(dataset_id=simple_dataset.id, n_requested=3)
        for i in range(3):
            report.runs.append(make_run_result(cost_usd=0.1 * (i + 1), latency_s=float(i + 1)))
            report.eval_results.append(make_eval_result(
                dataset_id=simple_dataset.id, run_index=i, verdicts=all_met
            ))
        stats = aggregate_dataset(report, simple_dataset)
        assert "mean" in stats.cost_stats
        assert stats.cost_stats["mean"] > 0

    def test_latency_stats_populated(self, simple_dataset: Dataset) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = DatasetReport(dataset_id=simple_dataset.id, n_requested=3)
        for i in range(3):
            report.runs.append(make_run_result(latency_s=float(i + 1)))
            report.eval_results.append(make_eval_result(
                dataset_id=simple_dataset.id, run_index=i, verdicts=all_met
            ))
        stats = aggregate_dataset(report, simple_dataset)
        assert "mean" in stats.latency_stats
        assert stats.latency_stats["min"] <= stats.latency_stats["mean"] <= stats.latency_stats["max"]


# ---------------------------------------------------------------------------
# SuiteReport.print_summary()
# ---------------------------------------------------------------------------

class TestSuiteReport:
    def test_print_summary_runs_without_error(self, simple_dataset: Dataset) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(simple_dataset, [all_met] * 3)
        stats = aggregate_dataset(report, simple_dataset, n_bootstrap=100)
        suite = SuiteReport(dataset_stats=[stats])
        buf = io.StringIO()
        with redirect_stdout(buf):
            suite.print_summary()
        output = buf.getvalue()
        assert simple_dataset.id in output

    def test_print_summary_includes_coverage(self, simple_dataset: Dataset) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(simple_dataset, [all_met] * 3)
        stats = aggregate_dataset(report, simple_dataset, n_bootstrap=100)
        suite = SuiteReport(dataset_stats=[stats])
        buf = io.StringIO()
        with redirect_stdout(buf):
            suite.print_summary()
        assert "coverage" in buf.getvalue()

    def test_print_summary_includes_tag_coverage(self, simple_dataset: Dataset) -> None:
        all_met = {"c_must": Verdict.MET, "c_should": Verdict.MET}
        report = _build_report(simple_dataset, [all_met] * 3)
        stats = aggregate_dataset(report, simple_dataset, n_bootstrap=100)
        suite = SuiteReport(dataset_stats=[stats])
        buf = io.StringIO()
        with redirect_stdout(buf):
            suite.print_summary()
        # tags "legal" and "financial" should appear
        assert "legal" in buf.getvalue() or "financial" in buf.getvalue()
