"""LLM-based evaluator with per-criterion scoring and calibration gate."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .llm import AnthropicClient, LLMClient, StructuredLLMClient
from .models import (
    Criterion,
    CriterionResult,
    Dataset,
    EvalError,
    EvaluationResult,
    Verdict,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text(path: Path) -> str:
    """Return plain text from a file. Handles .txt / .md natively; others need
    external libs (docx, pdf) — install as needed."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".rst", ""}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        try:
            import docx  # type: ignore
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            logger.warning("python-docx not installed; reading %s as text", path)
            return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            import pdfminer.high_level as pdf  # type: ignore
            return pdf.extract_text(str(path))
        except ImportError:
            logger.warning("pdfminer.six not installed; reading %s as text", path)
            return path.read_text(encoding="utf-8", errors="replace")
    # fallback
    return path.read_text(encoding="utf-8", errors="replace")


def _load_files(paths: list[Path]) -> str:
    parts: list[str] = []
    for p in paths:
        parts.append(f"=== {p.name} ===\n{_extract_text(p)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    passed: bool
    unmet_criterion_ids: list[str]
    critical_errors: list[EvalError]
    message: str


# ---------------------------------------------------------------------------
# Judge prompt builders
# ---------------------------------------------------------------------------

_CRITERION_PROMPT = """\
You are an expert evaluator assessing whether a document satisfies a specific criterion.

## Criterion
ID: {criterion_id}
Importance: {importance}
Description: {description}

## Reference document (ground truth)
{reference_text}

## Output document (to be evaluated)
{output_text}

## Task
Assess whether the output document satisfies the criterion above.
Return a JSON object with exactly these fields:
{{
  "verdict": "<met|partial|not_met|contradicted>",
  "rationale": "<one or two sentences explaining your judgment>"
}}

Verdict definitions:
- met: criterion is clearly and fully satisfied
- partial: criterion is partly addressed but incomplete
- not_met: criterion is absent or not addressed
- contradicted: output makes a claim that directly contradicts the reference

Reply with raw JSON only, no markdown fences."""

_ERROR_PROMPT = """\
You are an expert evaluator looking for factual errors in a document.

## Reference document (ground truth)
{reference_text}

## Output document (to be evaluated)
{output_text}

## Task
Identify errors in the output document. Focus on:
1. contradiction — a statement that directly contradicts the reference
2. unsupported — a definitive claim not found in the reference or the input data
3. format — output missing required files, wrong format, or wrong location

Do NOT flag:
- extra_neutral: additional correct or harmless information beyond the reference scope

Return a JSON array of error objects (may be empty):
[
  {{
    "type": "<contradiction|unsupported|format>",
    "severity": "<critical|major|minor>",
    "description": "<brief description>"
  }},
  ...
]

Reply with raw JSON only, no markdown fences."""

# Tool definitions for structured output (used when the client supports it)
_VERDICT_TOOL: dict = {
    "name": "report_verdict",
    "description": "Report the evaluation verdict for the criterion.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["met", "partial", "not_met", "contradicted"],
            },
            "rationale": {"type": "string"},
        },
        "required": ["verdict", "rationale"],
    },
}

_ERROR_TOOL: dict = {
    "name": "report_errors",
    "description": "Report factual errors found in the output document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["contradiction", "unsupported", "format"],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor"],
                        },
                        "description": {"type": "string"},
                    },
                    "required": ["type", "severity"],
                },
            }
        },
        "required": ["errors"],
    },
}


# ---------------------------------------------------------------------------
# LLM Evaluator
# ---------------------------------------------------------------------------

class LLMEvaluator:
    """Evaluates agent output against criteria using an LLM as judge.

    Parameters
    ----------
    model:
        Judge model ID. Used only when ``client`` is not provided and an
        :class:`~agent_eval.llm.AnthropicClient` is created automatically.
    n_samples:
        Number of judge samples for majority-vote on must-importance criteria.
    calibration_threshold:
        Fraction of criteria that must be MET when judging the reference itself.
    client:
        Any :class:`~agent_eval.llm.LLMClient` implementation. If the client
        also implements :class:`~agent_eval.llm.StructuredLLMClient`, structured
        output via tool use is used and JSON parse retries are skipped.
    parse_max_retries:
        How many times to retry after a JSON parse failure on the text path.
        Set to 0 to disable retries (useful in tests).
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        n_samples: int = 1,
        calibration_threshold: float = 0.95,
        client: LLMClient | None = None,
        parse_max_retries: int = 2,
    ) -> None:
        self.model = model
        self.n_samples = n_samples
        self.calibration_threshold = calibration_threshold
        self.parse_max_retries = parse_max_retries
        self._client: LLMClient = client or AnthropicClient(model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_judge(self, prompt: str) -> str:
        return self._client.complete(prompt)

    def _call_judge_with_retry(
        self,
        prompt: str,
        parse_fn: Callable[[str], _T],
    ) -> _T | None:
        """Call judge and parse result, retrying on JSON / value parse failures.

        Non-parse exceptions (API errors, network failures) propagate immediately.
        Returns ``None`` only after all retry attempts are exhausted.
        """
        last_exc: Exception | None = None
        raw = ""
        total = 1 + self.parse_max_retries
        for attempt in range(total):
            raw = self._call_judge(prompt)
            try:
                return parse_fn(raw)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < total - 1:
                    logger.warning(
                        "Parse attempt %d/%d failed: %s — retrying",
                        attempt + 1, total, exc,
                    )
        logger.warning(
            "All %d parse attempts failed: %s — raw: %.200s",
            total, last_exc, raw,
        )
        return None

    def _score_criterion(
        self,
        criterion: Criterion,
        output_text: str,
        reference_text: str,
    ) -> CriterionResult:
        prompt = _CRITERION_PROMPT.format(
            criterion_id=criterion.id,
            importance=criterion.importance,
            description=criterion.description,
            reference_text=reference_text,
            output_text=output_text,
        )

        n = self.n_samples if criterion.importance == "must" else 1
        verdicts: list[Verdict] = []
        rationales: list[str] = []

        for _ in range(n):
            if isinstance(self._client, StructuredLLMClient):
                data = self._client.complete_structured(prompt, _VERDICT_TOOL)
                verdict = Verdict(data["verdict"])
                rationale = data.get("rationale", "")
            else:
                def _parse(raw: str) -> tuple[Verdict, str]:
                    d = json.loads(raw)
                    return Verdict(d["verdict"]), d.get("rationale", "")

                result = self._call_judge_with_retry(prompt, _parse)
                if result is None:
                    verdict = Verdict.NOT_MET
                    rationale = "Parse error: all retries exhausted"
                else:
                    verdict, rationale = result

            verdicts.append(verdict)
            rationales.append(rationale)

        # majority vote
        final_verdict = max(set(verdicts), key=verdicts.count)
        # pick rationale matching the majority verdict
        for v, r in zip(verdicts, rationales):
            if v == final_verdict:
                final_rationale = r
                break
        else:
            final_rationale = rationales[0]

        return CriterionResult(
            criterion_id=criterion.id,
            verdict=final_verdict,
            rationale=final_rationale,
        )

    def _detect_errors(
        self,
        output_text: str,
        reference_text: str,
    ) -> list[EvalError]:
        prompt = _ERROR_PROMPT.format(
            reference_text=reference_text,
            output_text=output_text,
        )

        if isinstance(self._client, StructuredLLMClient):
            data = self._client.complete_structured(prompt, _ERROR_TOOL)
            items: list[dict] = data["errors"]
        else:
            result = self._call_judge_with_retry(prompt, json.loads)
            if result is None:
                return []
            items = result

        errors: list[EvalError] = []
        for item in items:
            errors.append(
                EvalError(
                    type=item["type"],
                    severity=item["severity"],
                    description=item.get("description", ""),
                )
            )
        return errors

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        output_files: list[Path],
        reference_files: list[Path],
        criteria: list[Criterion],
        dataset_id: str,
        run_index: int,
    ) -> EvaluationResult:
        output_text = _load_files(output_files)
        reference_text = _load_files(reference_files)

        # (a) Per-criterion independent scoring
        criterion_results: list[CriterionResult] = []
        for criterion in criteria:
            result = self._score_criterion(criterion, output_text, reference_text)
            criterion_results.append(result)
            logger.debug("Criterion %s → %s", criterion.id, result.verdict.value)

        # (b) Error detection pass
        errors = self._detect_errors(output_text, reference_text)

        return EvaluationResult(
            dataset_id=dataset_id,
            run_index=run_index,
            criterion_results=criterion_results,
            errors=errors,
        )

    def calibrate(self, dataset: Dataset) -> CalibrationResult:
        """Judge the reference files against their own criteria.

        All criteria should be MET and no critical errors should appear.
        If not, the criteria or judge are unreliable — abort before real evaluation.
        """
        reference_text = _load_files(dataset.reference_files)

        unmet: list[str] = []
        for criterion in dataset.criteria:
            result = self._score_criterion(criterion, reference_text, reference_text)
            if result.verdict not in (Verdict.MET, Verdict.PARTIAL):
                unmet.append(criterion.id)

        errors = self._detect_errors(reference_text, reference_text)
        critical_errors = [e for e in errors if e.severity == "critical"]

        met_fraction = (len(dataset.criteria) - len(unmet)) / max(len(dataset.criteria), 1)
        passed = met_fraction >= self.calibration_threshold and not critical_errors

        msg = (
            f"Calibration {'PASSED' if passed else 'FAILED'}: "
            f"{met_fraction:.0%} criteria met on reference, "
            f"{len(critical_errors)} critical error(s)."
        )
        if unmet:
            msg += f" Unmet criteria: {unmet}"
        logger.info(msg)

        return CalibrationResult(
            passed=passed,
            unmet_criterion_ids=unmet,
            critical_errors=critical_errors,
            message=msg,
        )
