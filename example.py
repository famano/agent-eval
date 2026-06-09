"""Usage example: wiring up a stub agent and running evaluation.

Replace StubAgent with your real agent implementation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from agent_eval import (
    AgentMetadata,
    Criterion,
    Dataset,
    Iterator,
    LLMEvaluator,
    RunResult,
    RunStatus,
    SuiteReport,
    aggregate_dataset,
)

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Stub agent (replace with real implementation)
# ---------------------------------------------------------------------------


class StubAgent:
    """Minimal example: copies input files to output_dir as-is."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            model="claude-sonnet-4-6",
            sdk_version="0.1.0",
            tools=["read_file", "write_file"],
            temperature=0.0,
            seed=42,
        )

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        timeout_s: int,
        budget_usd: float,
    ) -> RunResult:
        t0 = time.monotonic()
        output_files: list[Path] = []

        for src in input_dir.iterdir():
            if src.is_file():
                dst = output_dir / src.name
                dst.write_bytes(src.read_bytes())
                output_files.append(dst)

        return RunResult(
            status=RunStatus.SUCCESS,
            output_files=output_files,
            cost_usd=0.01,
            latency_s=time.monotonic() - t0,
            tokens=100,
            tool_calls=len(output_files) * 2,
        )


# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------


def build_example_dataset(base: Path) -> Dataset:
    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "source.txt").write_text(
        "Source material: Widget Corp acquired Acme Inc for $50M."
    )

    reference_dir = base / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_file = reference_dir / "expected_report.txt"
    reference_file.write_text(
        "Acquisition: Widget Corp acquired Acme Inc.\n"
        "Deal value: $50M.\n"
        "No regulatory issues identified.\n"
    )

    criteria = [
        Criterion(
            id="c1",
            description="Report mentions the acquiring party (Widget Corp) and target (Acme Inc).",
            importance="must",
            weight=2.0,
            tags=["parties"],
        ),
        Criterion(
            id="c2",
            description="Report states the deal value of $50M.",
            importance="must",
            weight=2.0,
            tags=["financials"],
        ),
        Criterion(
            id="c3",
            description="Report addresses regulatory risk or notes absence of issues.",
            importance="should",
            weight=1.0,
            tags=["regulatory"],
        ),
    ]

    return Dataset(
        id="example-dd-001",
        input_dir=input_dir,
        reference_files=[reference_file],
        criteria=criteria,
        timeout_s=120,
        budget_usd=0.5,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    base = Path("example_data")
    dataset = build_example_dataset(base)

    evaluator = LLMEvaluator(
        model="claude-opus-4-7",
        n_samples=1,
        calibration_threshold=0.9,
    )

    # Calibration gate — abort if reference itself fails its criteria
    print("Running calibration...")
    cal = evaluator.calibrate(dataset)
    print(cal.message)
    if not cal.passed:
        print("Calibration failed — fix criteria or judge before proceeding.")
        return

    agent = StubAgent()
    runner = Iterator(output_root=base / "runs")

    print("\nRunning evaluation (3 repeats)...")
    report = runner.run(
        dataset=dataset,
        agent=agent,
        evaluator=evaluator,
        n_repeats=3,
    )

    stats = aggregate_dataset(report, dataset, coverage_threshold=0.7)
    suite = SuiteReport(dataset_stats=[stats])
    print()
    suite.print_summary()


if __name__ == "__main__":
    main()
