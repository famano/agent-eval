"""agent_eval — LLM agent evaluation framework."""

from .agent import Agent, AgentMetadata
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
from .wrappers import (
    ShellAgentConfig,
    ShellWrapperAgent,
    WrapperConfig,
    claude_code_agent,
    codex_agent,
    gemini_agent,
    read_prompt,
)

__all__ = [
    "Agent",
    "AgentMetadata",
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
    "ShellAgentConfig",
    "ShellWrapperAgent",
    "SuiteReport",
    "Verdict",
    "WrapperConfig",
    "aggregate_dataset",
    "claude_code_agent",
    "codex_agent",
    "compute_run_metrics",
    "gemini_agent",
    "generate_criteria",
    "read_prompt",
]
