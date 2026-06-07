"""Iterator / Runner: orchestrates N-repeat evaluation runs for a dataset."""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import Agent
from .evaluator import LLMEvaluator
from .models import Dataset, EvaluationResult, RunResult, RunStatus

logger = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    dataset_id: str
    n_requested: int
    runs: list[RunResult] = field(default_factory=list)
    eval_results: list[EvaluationResult] = field(default_factory=list)
    infra_failures: list[dict] = field(default_factory=list)


class Iterator:
    """Run an agent N times on a dataset, evaluate each successful run, and
    aggregate results into a DatasetReport.

    Parameters
    ----------
    output_root:
        Base directory under which per-run output directories are created.
        Defaults to a ``runs/`` subdirectory in the current working directory.
    """

    def __init__(self, output_root: Path | None = None) -> None:
        self.output_root = output_root or Path("runs")

    def run(
        self,
        dataset: Dataset,
        agent: Agent,
        evaluator: LLMEvaluator,
        n_repeats: int,
        max_infra_retries: int = 2,
    ) -> DatasetReport:
        report = DatasetReport(
            dataset_id=dataset.id,
            n_requested=n_repeats,
        )

        for run_index in range(n_repeats):
            run_result, output_dir = self._execute_with_retries(
                dataset=dataset,
                agent=agent,
                run_index=run_index,
                max_infra_retries=max_infra_retries,
            )

            if run_result is None:
                # all retries exhausted — record as infra failure, skip scoring
                report.infra_failures.append(
                    {"run_index": run_index, "reason": "max_infra_retries exceeded"}
                )
                logger.warning(
                    "Dataset %s run %d: permanently failed (infra). Excluded from scoring.",
                    dataset.id,
                    run_index,
                )
                continue

            report.runs.append(run_result)

            if run_result.status == RunStatus.SUCCESS:
                eval_result = evaluator.evaluate(
                    output_files=run_result.output_files,
                    reference_files=dataset.reference_files,
                    criteria=dataset.criteria,
                    dataset_id=dataset.id,
                    run_index=run_index,
                )
                report.eval_results.append(eval_result)
                logger.info(
                    "Dataset %s run %d: evaluated (%d criteria).",
                    dataset.id,
                    run_index,
                    len(eval_result.criterion_results),
                )
            else:
                logger.info(
                    "Dataset %s run %d: status=%s — recorded but not evaluated.",
                    dataset.id,
                    run_index,
                    run_result.status.value,
                )

        return report

    # ------------------------------------------------------------------

    def _execute_with_retries(
        self,
        dataset: Dataset,
        agent: Agent,
        run_index: int,
        max_infra_retries: int,
    ) -> tuple[RunResult | None, Path | None]:
        for attempt in range(max_infra_retries + 1):
            output_dir = self._make_output_dir(dataset.id, run_index, attempt)
            try:
                t0 = time.monotonic()
                result = agent.run(
                    input_dir=dataset.input_dir,
                    output_dir=output_dir,
                    timeout_s=dataset.timeout_s,
                    budget_usd=dataset.budget_usd,
                )
                elapsed = time.monotonic() - t0
                logger.debug(
                    "Dataset %s run %d attempt %d: status=%s in %.1fs",
                    dataset.id,
                    run_index,
                    attempt,
                    result.status.value,
                    elapsed,
                )

                if result.status == RunStatus.CRASHED:
                    if attempt < max_infra_retries:
                        logger.warning(
                            "Dataset %s run %d attempt %d CRASHED — retrying.",
                            dataset.id,
                            run_index,
                            attempt,
                        )
                        continue
                    else:
                        # Retries exhausted by infra CRASH — exclude from scoring
                        logger.error(
                            "Dataset %s run %d CRASHED on all %d attempt(s) — recording as infra failure.",
                            dataset.id,
                            run_index,
                            max_infra_retries + 1,
                        )
                        return None, None

                self._save_trajectory(result, output_dir)
                return result, output_dir

            except Exception as exc:
                logger.error(
                    "Dataset %s run %d attempt %d raised exception: %s",
                    dataset.id,
                    run_index,
                    attempt,
                    exc,
                    exc_info=True,
                )
                if attempt >= max_infra_retries:
                    return None, None

        return None, None

    def _make_output_dir(self, dataset_id: str, run_index: int, attempt: int) -> Path:
        dir_path = self.output_root / dataset_id / f"run_{run_index:03d}_attempt_{attempt}"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _save_trajectory(self, result: RunResult, output_dir: Path) -> None:
        trajectory_path = output_dir / "_trajectory.json"
        data = {
            "status": result.status.value,
            "cost_usd": result.cost_usd,
            "latency_s": result.latency_s,
            "tokens": result.tokens,
            "tool_calls": result.tool_calls,
            "output_files": [str(p) for p in result.output_files],
        }
        trajectory_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        if result.trajectory_ref is None:
            result.trajectory_ref = trajectory_path
