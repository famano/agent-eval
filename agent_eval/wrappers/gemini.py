"""Google Gemini wrapper agent."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..agent import AgentMetadata
from ..models import RunResult, RunStatus
from .base import WrapperConfig, read_input_dir

try:
    import google.generativeai as genai

    _GENAI_AVAILABLE = True
    _GENAI_VERSION: str = getattr(genai, "__version__", "genai")
except ImportError:
    genai = None
    _GENAI_AVAILABLE = False
    _GENAI_VERSION = "not-installed"


class GeminiWrapperAgent:
    """Calls the Google Generative AI (Gemini) API and writes the response to *output_dir*.

    The ``google-generativeai`` package must be installed:

    .. code-block:: shell

        pip install "agent-eval[gemini]"

    Pass an *api_key* explicitly, or set the ``GOOGLE_API_KEY`` environment
    variable before creating the agent.

    Example
    -------
    >>> cfg = WrapperConfig(
    ...     model="gemini-1.5-pro",
    ...     temperature=0.0,
    ...     seed=42,
    ...     system_prompt="You are a helpful assistant.",
    ...     tools=[
    ...         genai.protos.Tool(function_declarations=[
    ...             genai.protos.FunctionDeclaration(
    ...                 name="search",
    ...                 description="Search the web.",
    ...                 parameters=genai.protos.Schema(
    ...                     type=genai.protos.Type.OBJECT,
    ...                     properties={"query": genai.protos.Schema(type=genai.protos.Type.STRING)},
    ...                 ),
    ...             )
    ...         ])
    ...     ],
    ... )
    >>> agent = GeminiWrapperAgent(cfg, api_key="YOUR_KEY")
    """

    def __init__(
        self,
        config: WrapperConfig,
        api_key: str | None = None,
    ) -> None:
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai package is required. Install with: pip install 'agent-eval[gemini]'"
            )
        if api_key:
            genai.configure(api_key=api_key)
        self._cfg = config

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            model=self._cfg.model,
            sdk_version=_GENAI_VERSION,
            tools=self._cfg.tool_names(),
            skills=list(self._cfg.skills),
            temperature=self._cfg.temperature,
            seed=self._cfg.seed,
        )

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        timeout_s: int,
        budget_usd: float,
    ) -> RunResult:
        t0 = time.monotonic()
        prompt = read_input_dir(input_dir)

        generation_config: dict[str, Any] = {"max_output_tokens": self._cfg.max_tokens}
        if self._cfg.temperature is not None:
            generation_config["temperature"] = self._cfg.temperature
        if self._cfg.seed is not None:
            generation_config["seed"] = self._cfg.seed
        # Allow fine-grained overrides via extra_params["generation_config"]
        generation_config.update(self._cfg.extra_params.get("generation_config", {}))

        model_kwargs: dict[str, Any] = {"generation_config": generation_config}
        if self._cfg.system_prompt:
            model_kwargs["system_instruction"] = self._cfg.system_prompt
        if self._cfg.tools:
            model_kwargs["tools"] = self._cfg.tools

        try:
            model = genai.GenerativeModel(self._cfg.model, **model_kwargs)
            response = model.generate_content(prompt)
        except Exception:
            return RunResult(
                status=RunStatus.CRASHED,
                output_files=[],
                cost_usd=0.0,
                latency_s=time.monotonic() - t0,
                tokens=0,
                tool_calls=0,
            )

        text_parts: list[str] = []
        tool_calls = 0
        for candidate in getattr(response, "candidates", []):
            for part in getattr(candidate.content, "parts", []):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                elif getattr(part, "function_call", None):
                    tool_calls += 1
                    fc = part.function_call
                    text_parts.append(f"[function_call: {fc.name}]\n{dict(fc.args)}")

        output_file = output_dir / "response.txt"
        output_file.write_text("\n\n".join(text_parts), encoding="utf-8")

        tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            tokens = (getattr(usage, "prompt_token_count", 0) or 0) + (
                getattr(usage, "candidates_token_count", 0) or 0
            )
        return RunResult(
            status=RunStatus.SUCCESS,
            output_files=[output_file],
            cost_usd=_estimate_cost_gemini(self._cfg.model, tokens),
            latency_s=time.monotonic() - t0,
            tokens=tokens,
            tool_calls=tool_calls,
        )


def _estimate_cost_gemini(model: str, tokens: int) -> float:
    """Very rough per-token cost estimate for common Gemini models."""
    mtok = tokens / 1_000_000
    if "ultra" in model:
        return mtok * 28.0
    if "flash" in model:
        return mtok * 0.35
    return mtok * 7.0  # 1.5-pro default
