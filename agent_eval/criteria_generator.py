"""Generate evaluation criteria from reference documents using an LLM."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic

from .evaluator import _load_files

logger = logging.getLogger(__name__)

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
## Task
Return a JSON array where each element is:
{{
  "id": "c1",
  "description": "...",
  "importance": "must" | "should",
  "weight": 2.0,
  "tags": ["tag1", "tag2"]
}}

Rules:
- "must": critical; its absence makes the output unacceptable
- "should": important but not blocking
- weight defaults to 2.0 for "must" and 1.0 for "should"; adjust only when warranted
- tags: 1–3 short labels for grouping (e.g. ["parties", "financials", "regulatory"])
- Aim for 5–15 criteria; prefer fewer, well-scoped criteria over many overlapping ones
- Start IDs at "c1" and number sequentially

Return ONLY a valid JSON array — no markdown fences, no commentary."""


def generate_criteria(
    reference_files: list[Path],
    input_files: list[Path] | None = None,
    model: str = "claude-opus-4-7",
    client: anthropic.Anthropic | None = None,
) -> list[dict]:
    """Generate evaluation criteria from reference documents.

    Calls an LLM to extract atomic, verifiable criteria from the given reference
    files. The result is a list of raw dicts compatible with the Criterion dataclass.
    Human curation is still recommended before use in production evaluation.

    Parameters
    ----------
    reference_files:
        Ground-truth output files to derive criteria from.
    input_files:
        Optional task-context files that give the LLM richer background.
    model:
        Anthropic model to use for generation.
    client:
        Optional pre-configured Anthropic client (uses env ANTHROPIC_API_KEY by default).

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

    response = _client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    try:
        criteria = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse criteria JSON: %s\nRaw response:\n%s", exc, raw)
        raise

    if not isinstance(criteria, list):
        raise ValueError(f"Expected a JSON array, got {type(criteria).__name__}")

    logger.info("Generated %d criteria from %d reference file(s).", len(criteria), len(reference_files))
    return criteria
