"""Generate evaluation criteria from reference documents using an LLM."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import anthropic

from .evaluator import _load_files

logger = logging.getLogger(__name__)

# Errors worth retrying (transient); auth/validation errors are not retried.
_RETRYABLE_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

# Tool schema — forces structured JSON output via the API.
_CRITERIA_TOOL: dict = {
    "name": "submit_criteria",
    "description": (
        "Submit the complete list of evaluation criteria extracted from the reference documents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "description": "Evaluation criteria (観点) derived from the reference documents.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Sequential identifier: c1, c2, c3, ...",
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Clear, atomic statement of what the output must contain "
                                "or address."
                            ),
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["must", "should"],
                            "description": (
                                "must = critical, absence makes output unacceptable. "
                                "should = important but not blocking."
                            ),
                        },
                        "weight": {
                            "type": "number",
                            "description": (
                                "Scoring weight. Default 2.0 for must, 1.0 for should."
                            ),
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1–3 short category labels, e.g. ['parties', 'financials'].",
                        },
                    },
                    "required": ["id", "description", "importance", "weight", "tags"],
                },
            }
        },
        "required": ["criteria"],
    },
}

_GENERATE_PROMPT = """\
You are an expert evaluator designing criteria to assess whether an LLM agent's output \
meets requirements.

Given one or more reference documents (ground-truth output files that represent ideal \
agent output), extract atomic evaluation criteria (観点) that can be used to judge any \
agent's attempt at the same task.

Guiding principles for each criterion:
- Atomic: tests exactly one judgment unit (not compound)
- Independent: no significant overlap with other criteria in this list
- Verifiable: a judge can determine met/not_met by reading the output alone
- Solution-independent: asks "is this point addressed/concluded?" not "does it use \
these exact words?"

## Reference documents (ground truth)
{reference_text}
{input_context}
Aim for 5–15 criteria. Prefer fewer, well-scoped criteria over many overlapping ones. \
Start IDs at "c1" and number sequentially. \
Call submit_criteria with the complete list."""


def generate_criteria(
    reference_files: list[Path],
    input_files: list[Path] | None = None,
    model: str = "claude-opus-4-7",
    max_retries: int = 3,
    client: anthropic.Anthropic | None = None,
) -> list[dict]:
    """Generate evaluation criteria from reference documents.

    Uses structured output (tool use) to guarantee schema-valid JSON.
    Retries on transient API errors with exponential back-off.

    Parameters
    ----------
    reference_files:
        Ground-truth output files to derive criteria from.
    input_files:
        Optional task-context files that give the LLM richer background.
    model:
        Anthropic model to use for generation.
    max_retries:
        Maximum attempts before raising. Must be >= 1.
    client:
        Optional pre-configured Anthropic client (uses ANTHROPIC_API_KEY by default).

    Returns
    -------
    list[dict]
        JSON-compatible dicts with keys: id, description, importance, weight, tags.
    """
    _client = client or anthropic.Anthropic()

    reference_text = _load_files(reference_files)

    input_context = ""
    if input_files:
        input_text = _load_files(input_files)
        input_context = f"\n## Input documents (task context)\n{input_text}\n"

    prompt = _GENERATE_PROMPT.format(
        reference_text=reference_text,
        input_context=input_context,
    )

    delay = 1.0
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = _client.messages.create(
                model=model,
                max_tokens=4096,
                tools=[_CRITERIA_TOOL],
                tool_choice={"type": "tool", "name": "submit_criteria"},
                messages=[{"role": "user", "content": prompt}],
            )

            tool_block = next(
                (b for b in response.content if b.type == "tool_use"),
                None,
            )
            if tool_block is None:
                raise RuntimeError(
                    "Model did not return a tool_use block despite forced tool_choice."
                )

            criteria: list[dict] = tool_block.input["criteria"]
            logger.info(
                "Generated %d criteria from %d reference file(s).",
                len(criteria),
                len(reference_files),
            )
            return criteria

        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                logger.warning(
                    "Attempt %d/%d failed (%s: %s). Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error("All %d attempt(s) failed.", max_retries)

        except RuntimeError as exc:
            # Unexpected response format — worth retrying once.
            last_exc = exc
            if attempt < max_retries - 1:
                logger.warning(
                    "Attempt %d/%d: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error("All %d attempt(s) failed.", max_retries)

    raise last_exc  # type: ignore[misc]
