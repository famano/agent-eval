#!/usr/bin/env python3
"""Generate evaluation criteria (観点) from reference files using an LLM.

The generated criteria are a starting point — human curation is recommended
before using them in production evaluation (see design doc §5.3).

Usage examples
--------------
  # From a directory of reference files
  python scripts/generate_criteria.py --reference-dir example_data/reference/

  # From specific files, with input context for richer criteria
  python scripts/generate_criteria.py ref1.txt ref2.md --input-dir example_data/input/

  # Save to a JSON file
  python scripts/generate_criteria.py --reference-dir refs/ --output criteria.json

  # Use a different model
  python scripts/generate_criteria.py --reference-dir refs/ --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_eval.criteria_generator import generate_criteria


def _collect_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation criteria from reference files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "reference_files",
        nargs="*",
        type=Path,
        metavar="FILE",
        help="Reference files to generate criteria from.",
    )
    parser.add_argument(
        "--reference-dir", "-r",
        type=Path,
        metavar="DIR",
        help="Directory of reference files (alternative to positional FILE args).",
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        metavar="DIR",
        help="Directory of input/task-context files for richer criteria.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Individual input/task-context files (alternative to --input-dir).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        metavar="FILE",
        help="Write JSON output to FILE instead of stdout.",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-opus-4-7",
        metavar="MODEL",
        help="Anthropic model to use (default: claude-opus-4-7).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="N",
        help="JSON indentation spaces (default: 2).",
    )

    args = parser.parse_args()

    # --- Resolve reference files ---
    reference_files: list[Path] = list(args.reference_files or [])
    if args.reference_dir:
        reference_files.extend(_collect_files(args.reference_dir))
    if not reference_files:
        parser.error(
            "Provide reference files as positional arguments or via --reference-dir."
        )

    # --- Resolve input/context files ---
    input_files: list[Path] | None = None
    if args.input_dir:
        input_files = _collect_files(args.input_dir)
    elif args.input_files:
        input_files = list(args.input_files)

    print(
        f"Generating criteria from {len(reference_files)} reference file(s) "
        f"using {args.model}...",
        file=sys.stderr,
    )
    if input_files:
        print(f"  + {len(input_files)} input file(s) for context.", file=sys.stderr)

    try:
        criteria = generate_criteria(
            reference_files=reference_files,
            input_files=input_files,
            model=args.model,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output_json = json.dumps(criteria, ensure_ascii=False, indent=args.indent)

    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
        print(f"Wrote {len(criteria)} criteria to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    print(
        f"Done. {len(criteria)} criteria generated. "
        "Review and curate before use in evaluation.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
