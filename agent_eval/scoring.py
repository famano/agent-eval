"""Scoring and statistical aggregation for evaluation results.

Two reporting axes (as per design):
  - Coverage (recall-like): weighted criterion satisfaction rate
  - Soundness (precision-like): error counts by severity
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .models import (
    Criterion,
    Dataset,
    EvaluationResult,
    RunResult,
    RunStatus,
    Verdict,
)
from .runner import DatasetReport

# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

_VERDICT_SCORE: dict[Verdict, float] = {
    Verdict.MET: 1.0,
    Verdict.PARTIAL: 0.5,
    Verdict.NOT_MET: 0.0,
    Verdict.CONTRADICTED: 0.0,
}


@dataclass
class RunMetrics:
    run_index: int
    coverage: float  # weighted satisfaction rate [0, 1]
    critical_error_count: int
    major_error_count: int
    minor_error_count: int
    cost_usd: float | None = None
    latency_s: float | None = None
    tokens: int | None = None
    tool_calls: int | None = None


def compute_run_metrics(
    eval_result: EvaluationResult,
    criteria: list[Criterion],
    run_result: RunResult | None = None,
) -> RunMetrics:
    criterion_map = {c.id: c for c in criteria}

    total_weight = 0.0
    weighted_score = 0.0
    for cr in eval_result.criterion_results:
        c = criterion_map.get(cr.criterion_id)
        w = c.weight if c else 1.0
        total_weight += w
        weighted_score += w * _VERDICT_SCORE[cr.verdict]

    coverage = weighted_score / total_weight if total_weight > 0 else 0.0

    critical = sum(1 for e in eval_result.errors if e.severity == "critical")
    major = sum(1 for e in eval_result.errors if e.severity == "major")
    minor = sum(1 for e in eval_result.errors if e.severity == "minor")

    return RunMetrics(
        run_index=eval_result.run_index,
        coverage=coverage,
        critical_error_count=critical,
        major_error_count=major,
        minor_error_count=minor,
        cost_usd=run_result.cost_usd if run_result else None,
        latency_s=run_result.latency_s if run_result else None,
        tokens=run_result.tokens if run_result else None,
        tool_calls=run_result.tool_calls if run_result else None,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    lo_p = (1 - ci) / 2
    hi_p = 1 - lo_p
    samples = [
        sum(random.choices(values, k=len(values))) / len(values)
        for _ in range(n_bootstrap)
    ]
    samples.sort()
    lo_idx = int(lo_p * n_bootstrap)
    hi_idx = int(hi_p * n_bootstrap) - 1
    return samples[lo_idx], samples[hi_idx]


# ---------------------------------------------------------------------------
# Dataset-level statistics
# ---------------------------------------------------------------------------


@dataclass
class CoverageStats:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int


@dataclass
class DatasetStats:
    dataset_id: str
    n_runs_total: int
    n_runs_evaluated: int
    n_infra_failures: int
    n_timeout_or_budget: int
    coverage: CoverageStats
    pass_at_k: dict[int, float]  # e.g. {1: 0.7, 3: 0.9}
    all_pass_rate: float  # fraction of runs above threshold
    critical_error_rate: float
    per_tag_coverage: dict[str, CoverageStats] = field(default_factory=dict)
    cost_stats: dict[str, float] = field(default_factory=dict)
    latency_stats: dict[str, float] = field(default_factory=dict)


def aggregate_dataset(
    report: DatasetReport,
    dataset: Dataset,
    coverage_threshold: float = 0.7,
    pass_at_ks: list[int] | None = None,
    n_bootstrap: int = 1000,
) -> DatasetStats:
    if pass_at_ks is None:
        pass_at_ks = [1, 3, 5]

    run_list = report.runs  # ordered; align by eval_result.run_index

    metrics: list[RunMetrics] = []
    for er in report.eval_results:
        # find matching RunResult by run_index order
        rr = run_list[er.run_index] if er.run_index < len(run_list) else None
        m = compute_run_metrics(er, dataset.criteria, rr)
        metrics.append(m)

    coverages = [m.coverage for m in metrics]
    ci_low, ci_high = _bootstrap_ci(coverages, n_bootstrap=n_bootstrap)

    mean_cov = sum(coverages) / len(coverages) if coverages else float("nan")
    std_cov = (
        math.sqrt(sum((x - mean_cov) ** 2 for x in coverages) / len(coverages))
        if len(coverages) > 1
        else 0.0
    )

    n_total = report.n_requested
    n_eval = len(metrics)
    n_infra = len(report.infra_failures)
    n_to_budget = sum(
        1
        for r in run_list
        if r.status in (RunStatus.TIMEOUT, RunStatus.BUDGET_EXCEEDED)
    )

    above = [c >= coverage_threshold for c in coverages]
    all_pass_rate = sum(above) / len(above) if above else 0.0

    # pass@k: probability that at least 1 of k random samples passes
    def _pass_at_k(n: int, c: int, k: int) -> float:
        if n == 0:
            return 0.0
        if n - c < k:
            return 1.0
        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

    n_pass = sum(above)
    pass_at_k_dict = {
        k: _pass_at_k(len(above), n_pass, k) for k in pass_at_ks if k <= len(above)
    }

    critical_error_rate = (
        sum(1 for m in metrics if m.critical_error_count > 0) / len(metrics)
        if metrics
        else 0.0
    )

    # Per-tag coverage
    tag_coverages: dict[str, list[float]] = {}
    criterion_map = {c.id: c for c in dataset.criteria}
    for er in report.eval_results:
        for cr in er.criterion_results:
            c = criterion_map.get(cr.criterion_id)
            if c is None:
                continue
            score = _VERDICT_SCORE[cr.verdict]
            for tag in c.tags:
                tag_coverages.setdefault(tag, []).append(score)

    per_tag: dict[str, CoverageStats] = {}
    for tag, scores in tag_coverages.items():
        tmean = sum(scores) / len(scores)
        tstd = (
            math.sqrt(sum((s - tmean) ** 2 for s in scores) / len(scores))
            if len(scores) > 1
            else 0.0
        )
        tlo, thi = _bootstrap_ci(scores, n_bootstrap=n_bootstrap)
        per_tag[tag] = CoverageStats(
            mean=tmean, std=tstd, ci_low=tlo, ci_high=thi, n=len(scores)
        )

    # Cost / latency distribution summaries
    costs = [m.cost_usd for m in metrics if m.cost_usd is not None]
    latencies = [m.latency_s for m in metrics if m.latency_s is not None]

    def _summary(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {}
        s = sorted(vals)
        n = len(s)
        return {
            "mean": sum(s) / n,
            "min": s[0],
            "p50": s[n // 2],
            "p95": s[int(n * 0.95)],
            "max": s[-1],
        }

    return DatasetStats(
        dataset_id=report.dataset_id,
        n_runs_total=n_total,
        n_runs_evaluated=n_eval,
        n_infra_failures=n_infra,
        n_timeout_or_budget=n_to_budget,
        coverage=CoverageStats(
            mean=mean_cov,
            std=std_cov,
            ci_low=ci_low,
            ci_high=ci_high,
            n=len(coverages),
        ),
        pass_at_k=pass_at_k_dict,
        all_pass_rate=all_pass_rate,
        critical_error_rate=critical_error_rate,
        per_tag_coverage=per_tag,
        cost_stats=_summary(costs),
        latency_stats=_summary(latencies),
    )


# ---------------------------------------------------------------------------
# Multi-dataset summary
# ---------------------------------------------------------------------------


@dataclass
class SuiteReport:
    dataset_stats: list[DatasetStats]

    def print_summary(self) -> None:
        for ds in self.dataset_stats:
            cov = ds.coverage
            print(
                f"[{ds.dataset_id}] "
                f"coverage={cov.mean:.3f} ±{cov.std:.3f} "
                f"(95% CI [{cov.ci_low:.3f}, {cov.ci_high:.3f}], n={cov.n}) | "
                f"critical_error_rate={ds.critical_error_rate:.1%} | "
                f"pass@1={ds.pass_at_k.get(1, float('nan')):.3f} | "
                f"infra_failures={ds.n_infra_failures}"
            )
            if ds.per_tag_coverage:
                for tag, tc in ds.per_tag_coverage.items():
                    print(f"  tag:{tag} coverage={tc.mean:.3f} (n={tc.n})")
