"""Tests for LLMEvaluator.

Behavioral specification source: design doc sections 5.2, 8.

What we verify (treating evaluator as a black box):
  - evaluate() returns EvaluationResult with one CriterionResult per criterion.
  - Each CriterionResult carries the criterion_id, a valid Verdict, and a non-empty rationale.
  - evaluate() also returns an errors list where each EvalError has valid type/severity.
  - calibrate() on the reference itself PASSES (all criteria MET → calibration succeeds).
  - calibrate() with a corrupted judge (always NOT_MET) FAILS.
  - must-importance criteria trigger multiple judge calls (majority vote).
  - Contradictions in the output surface as errors, not just NOT_MET verdicts.
  - extra_neutral additions are NOT flagged as errors (design: "原則ペナルティなし").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from agent_eval.evaluator import LLMEvaluator
from agent_eval.llm import LLMClient, StructuredLLMClient
from agent_eval.models import (
    Criterion,
    Dataset,
    EvalError,
    EvaluationResult,
    Verdict,
)


# ---------------------------------------------------------------------------
# Helpers: build mock LLMClient
# ---------------------------------------------------------------------------

def _make_client(responses: list[str]) -> MagicMock:
    """Return a LLMClient mock whose complete() cycles through the given responses."""
    client = MagicMock(spec=LLMClient)
    client.complete.side_effect = responses
    return client


def _verdict_response(verdict: str, rationale: str = "test rationale") -> str:
    return json.dumps({"verdict": verdict, "rationale": rationale})


def _error_response(errors: list[dict]) -> str:
    return json.dumps(errors)


# ---------------------------------------------------------------------------
# evaluate() — structure
# ---------------------------------------------------------------------------

class TestEvaluateStructure:
    """evaluate() must return exactly one CriterionResult per criterion."""

    def test_returns_evaluation_result(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        responses = [
            _verdict_response("met"),       # c_must
            _verdict_response("met"),       # c_should
            _error_response([]),            # error-detection pass
        ]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        assert isinstance(result, EvaluationResult)

    def test_one_criterion_result_per_criterion(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        responses = [
            _verdict_response("met"),
            _verdict_response("not_met"),
            _error_response([]),
        ]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        assert len(result.criterion_results) == len(criteria)

    def test_criterion_ids_match(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        responses = [_verdict_response("met")] * len(criteria) + [_error_response([])]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        returned_ids = {cr.criterion_id for cr in result.criterion_results}
        expected_ids = {c.id for c in criteria}
        assert returned_ids == expected_ids

    def test_all_verdicts_are_valid_enum_values(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        responses = [_verdict_response("partial")] * len(criteria) + [_error_response([])]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        valid = {v for v in Verdict}
        for cr in result.criterion_results:
            assert cr.verdict in valid

    def test_rationale_is_non_empty_string(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        responses = [
            _verdict_response("met", "Widget Corp and Acme Inc are both named.")
        ] * len(criteria) + [_error_response([])]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        for cr in result.criterion_results:
            assert isinstance(cr.rationale, str)
            assert len(cr.rationale) > 0

    def test_dataset_id_and_run_index_propagated(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        responses = [_verdict_response("met")] * len(criteria) + [_error_response([])]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="my-ds",
            run_index=7,
        )
        assert result.dataset_id == "my-ds"
        assert result.run_index == 7


# ---------------------------------------------------------------------------
# evaluate() — error detection
# ---------------------------------------------------------------------------

class TestEvaluateErrors:
    """errors list must reflect contradiction / unsupported / format only."""

    def test_no_errors_on_correct_output(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        responses = [_verdict_response("met")] * len(criteria) + [_error_response([])]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        assert result.errors == []

    def test_contradiction_error_returned(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        responses = [_verdict_response("contradicted")] * len(criteria) + [
            _error_response([
                {"type": "contradiction", "severity": "critical",
                 "description": "Price stated as $100M, reference says $50M."}
            ])
        ]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        assert any(e.type == "contradiction" for e in result.errors)

    def test_error_has_valid_type_and_severity(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        valid_types = {"contradiction", "unsupported", "format", "extra_neutral"}
        valid_severities = {"critical", "major", "minor"}
        responses = [_verdict_response("met")] * len(criteria) + [
            _error_response([
                {"type": "unsupported", "severity": "major",
                 "description": "Claimed zero debt with no supporting data."}
            ])
        ]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        for e in result.errors:
            assert e.type in valid_types
            assert e.severity in valid_severities

    def test_extra_neutral_not_flagged_as_critical(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        """Design doc: extra_neutral additions must NOT generate critical/major errors."""
        responses = [_verdict_response("met")] * len(criteria) + [
            _error_response([
                {"type": "extra_neutral", "severity": "minor",
                 "description": "Additional background history not in reference."}
            ])
        ]
        evaluator = LLMEvaluator(client=_make_client(responses))
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        critical_or_major_extra = [
            e for e in result.errors
            if e.type == "extra_neutral" and e.severity in ("critical", "major")
        ]
        assert critical_or_major_extra == [], (
            "extra_neutral must not be flagged critical/major (design: no penalty for thoroughness)"
        )


# ---------------------------------------------------------------------------
# evaluate() — independent scoring (one judge call per criterion)
# ---------------------------------------------------------------------------

class TestPerCriterionIndependentScoring:
    """Design: 'criterion 1個ごとに judge を呼ぶ' — N criteria = N criterion calls."""

    def test_judge_called_once_per_should_criterion(
        self, output_file: Path, reference_file: Path
    ) -> None:
        criteria = [
            Criterion(id=f"c{i}", description=f"criterion {i}", importance="should", weight=1.0)
            for i in range(3)
        ]
        responses = [_verdict_response("met")] * 3 + [_error_response([])]
        client = _make_client(responses)
        evaluator = LLMEvaluator(client=client, n_samples=1)
        evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        # 3 criterion calls + 1 error-detection call = 4 total
        assert client.complete.call_count == 4

    def test_must_criterion_uses_multiple_samples(
        self, output_file: Path, reference_file: Path
    ) -> None:
        """n_samples=3 on a must criterion → 3 judge calls for that criterion."""
        criteria = [
            Criterion(id="c_must", description="must criterion", importance="must", weight=2.0),
        ]
        responses = [_verdict_response("met")] * 3 + [_error_response([])]
        client = _make_client(responses)
        evaluator = LLMEvaluator(client=client, n_samples=3)
        evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        # 3 samples for the one must criterion + 1 error call = 4
        assert client.complete.call_count == 4

    def test_majority_vote_resolves_split(
        self, output_file: Path, reference_file: Path
    ) -> None:
        """With n_samples=3 and 2 MET vs 1 NOT_MET the result must be MET."""
        criteria = [
            Criterion(id="c_must", description="must criterion", importance="must", weight=2.0),
        ]
        responses = [
            _verdict_response("met"),
            _verdict_response("not_met"),
            _verdict_response("met"),
            _error_response([]),
        ]
        evaluator = LLMEvaluator(client=_make_client(responses), n_samples=3)
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        assert result.criterion_results[0].verdict == Verdict.MET

    def test_should_criterion_uses_single_sample(
        self, output_file: Path, reference_file: Path
    ) -> None:
        """should-importance criteria use a single sample even with n_samples>1."""
        criteria = [
            Criterion(id="c_should", description="should criterion", importance="should", weight=1.0),
        ]
        responses = [_verdict_response("partial"), _error_response([])]
        client = _make_client(responses)
        evaluator = LLMEvaluator(client=client, n_samples=3)
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        # 1 criterion call + 1 error call = 2
        assert client.complete.call_count == 2
        assert result.criterion_results[0].verdict == Verdict.PARTIAL


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------

class TestCalibrate:
    """Design: calibrate() judges the reference against itself.
    Expected: all criteria MET, 0 critical errors → passed=True.
    """

    def test_passes_when_reference_fully_satisfies_criteria(
        self, dataset: Dataset
    ) -> None:
        n = len(dataset.criteria)
        responses = [_verdict_response("met")] * n + [_error_response([])]
        evaluator = LLMEvaluator(
            client=_make_client(responses),
            calibration_threshold=0.9,
        )
        cal = evaluator.calibrate(dataset)
        assert cal.passed is True

    def test_fails_when_judge_cannot_confirm_criteria(
        self, dataset: Dataset
    ) -> None:
        n = len(dataset.criteria)
        # All NOT_MET: judge is broken or criteria are wrong
        responses = [_verdict_response("not_met")] * n + [_error_response([])]
        evaluator = LLMEvaluator(
            client=_make_client(responses),
            calibration_threshold=0.9,
        )
        cal = evaluator.calibrate(dataset)
        assert cal.passed is False

    def test_fails_when_critical_error_present_on_reference(
        self, dataset: Dataset
    ) -> None:
        n = len(dataset.criteria)
        responses = [_verdict_response("met")] * n + [
            _error_response([
                {"type": "contradiction", "severity": "critical",
                 "description": "Spurious critical error on reference."}
            ])
        ]
        evaluator = LLMEvaluator(
            client=_make_client(responses),
            calibration_threshold=0.9,
        )
        cal = evaluator.calibrate(dataset)
        assert cal.passed is False

    def test_returns_unmet_criterion_ids(
        self, dataset: Dataset
    ) -> None:
        n = len(dataset.criteria)
        # First criterion fails, rest pass
        responses = (
            [_verdict_response("not_met")]
            + [_verdict_response("met")] * (n - 1)
            + [_error_response([])]
        )
        evaluator = LLMEvaluator(
            client=_make_client(responses),
            calibration_threshold=0.99,  # strict: even 1 failure → fail
        )
        cal = evaluator.calibrate(dataset)
        assert len(cal.unmet_criterion_ids) == 1
        assert dataset.criteria[0].id in cal.unmet_criterion_ids

    def test_calibration_result_has_message(
        self, dataset: Dataset
    ) -> None:
        n = len(dataset.criteria)
        responses = [_verdict_response("met")] * n + [_error_response([])]
        evaluator = LLMEvaluator(client=_make_client(responses))
        cal = evaluator.calibrate(dataset)
        assert isinstance(cal.message, str) and len(cal.message) > 0

    def test_partial_counts_as_met_for_calibration_gate(
        self, dataset: Dataset
    ) -> None:
        """Design: calibration uses reference on reference; PARTIAL should be acceptable."""
        n = len(dataset.criteria)
        responses = [_verdict_response("partial")] * n + [_error_response([])]
        evaluator = LLMEvaluator(
            client=_make_client(responses),
            calibration_threshold=0.9,
        )
        cal = evaluator.calibrate(dataset)
        # partial is acceptable in calibration (criterion is partially in reference)
        assert cal.passed is True


# ---------------------------------------------------------------------------
# Malformed judge response graceful handling
# ---------------------------------------------------------------------------

class TestMalformedJudgeResponse:
    """Judge may produce non-JSON; evaluator must degrade gracefully."""

    def test_malformed_verdict_does_not_raise(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        bad = "Sorry, I cannot assess this."
        # parse_max_retries=0: one attempt per call, no retry — tests graceful degradation
        responses = [bad] * len(criteria) + [_error_response([])]
        evaluator = LLMEvaluator(client=_make_client(responses), parse_max_retries=0)
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        # Must still return a result — degraded to NOT_MET, but no exception
        assert len(result.criterion_results) == len(criteria)

    def test_malformed_error_response_does_not_raise(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        responses = [_verdict_response("met")] * len(criteria) + ["not json at all"]
        evaluator = LLMEvaluator(client=_make_client(responses), parse_max_retries=0)
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        # errors list may be empty but must not raise
        assert isinstance(result.errors, list)

    def test_parse_retried_before_fallback(
        self, output_file: Path, reference_file: Path
    ) -> None:
        """After bad responses, a good response on the retry path yields the correct verdict."""
        criteria = [
            Criterion(id="c1", description="criterion", importance="should", weight=1.0),
        ]
        bad = "not json"
        responses = [
            bad,                           # first attempt: parse failure
            _verdict_response("met"),      # second attempt (retry 1): success
            _error_response([]),
        ]
        client = _make_client(responses)
        evaluator = LLMEvaluator(client=client, parse_max_retries=1)
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        assert result.criterion_results[0].verdict == Verdict.MET
        # 2 calls for the criterion (1 bad + 1 retry) + 1 error-detection = 3
        assert client.complete.call_count == 3

    def test_api_error_propagates(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        """Non-parse errors (API failures) must propagate, not silently default."""
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = RuntimeError("API connection failed")
        evaluator = LLMEvaluator(client=client)
        with pytest.raises(RuntimeError, match="API connection failed"):
            evaluator.evaluate(
                output_files=[output_file],
                reference_files=[reference_file],
                criteria=criteria,
                dataset_id="ds-001",
                run_index=0,
            )


# ---------------------------------------------------------------------------
# Structured output path
# ---------------------------------------------------------------------------

class TestStructuredOutput:
    """When the client implements StructuredLLMClient, complete_structured is used."""

    def _make_structured_client(
        self,
        verdict_responses: list[dict],
        error_response: dict,
    ) -> MagicMock:
        """Mock with both complete and complete_structured (StructuredLLMClient spec)."""
        client = MagicMock(spec=["complete", "complete_structured"])
        verdict_iter = iter(verdict_responses)
        error_iter = iter([error_response])

        def _complete_structured(prompt: str, tool: dict) -> dict:
            if tool["name"] == "report_verdict":
                return next(verdict_iter)
            return next(error_iter)

        client.complete_structured.side_effect = _complete_structured
        return client

    def test_structured_client_uses_complete_structured(
        self, output_file: Path, reference_file: Path, criteria: list[Criterion]
    ) -> None:
        verdict_responses = [
            {"verdict": "met", "rationale": "ok"},
            {"verdict": "partial", "rationale": "almost"},
        ]
        error_resp = {"errors": []}
        client = self._make_structured_client(verdict_responses, error_resp)
        evaluator = LLMEvaluator(client=client)
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        # complete_structured called for each criterion + error detection
        assert client.complete_structured.call_count == len(criteria) + 1
        # complete (text path) must NOT be called
        assert client.complete.call_count == 0

    def test_structured_path_verdict_correct(
        self, output_file: Path, reference_file: Path
    ) -> None:
        criteria = [
            Criterion(id="c1", description="c", importance="should", weight=1.0)
        ]
        verdict_responses = [{"verdict": "contradicted", "rationale": "mismatch"}]
        error_resp = {"errors": [
            {"type": "contradiction", "severity": "critical", "description": "x"}
        ]}
        client = self._make_structured_client(verdict_responses, error_resp)
        evaluator = LLMEvaluator(client=client)
        result = evaluator.evaluate(
            output_files=[output_file],
            reference_files=[reference_file],
            criteria=criteria,
            dataset_id="ds-001",
            run_index=0,
        )
        assert result.criterion_results[0].verdict == Verdict.CONTRADICTED
        assert result.criterion_results[0].rationale == "mismatch"
        assert len(result.errors) == 1
        assert result.errors[0].type == "contradiction"
