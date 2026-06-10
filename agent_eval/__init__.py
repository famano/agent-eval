"""agent_eval — LLM agent evaluation framework."""
from .agent import Agent, AgentMetadata
from .wrappers import ClaudeWrapperAgent, CodexWrapperAgent, GeminiWrapperAgent, WrapperConfig
from .criteria_generator import generate_criteria
from .evaluator import CalibrationResult, LLMEvaluator
from .models import (
    Criterion,
    CriterionResult,
    Dataset,
    EvalError,
    EvaluationResult,
    RunResult,
    RunStatus,
    Verdict,
)
from .runner import DatasetReport, Iterator
from .scoring import (
    CoverageStats,
    DatasetStats,
    RunMetrics,
    SuiteReport,
    aggregate_dataset,
    compute_run_metrics,
)

__all__ = [
    "Agent",
    "AgentMetadata",
    "ClaudeWrapperAgent",
    "CodexWrapperAgent",
    "GeminiWrapperAgent",
    "WrapperConfig",
    "CalibrationResult",
    "Criterion",
    "CriterionResult",
    "CoverageStats",
    "Dataset",
    "DatasetReport",
    "DatasetStats",
    "EvalError",
    "EvaluationResult",
    "Iterator",
    "LLMEvaluator",
    "RunMetrics",
    "RunResult",
    "RunStatus",
    "SuiteReport",
    "Verdict",
    "aggregate_dataset",
    "compute_run_metrics",
    "generate_criteria",
]
