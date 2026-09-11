#!/usr/bin/env python3
"""Generate evaluation criteria (観点) from reference files using an LLM.

The generated criteria are a starting point — human curation is recommended
before using them in production evaluation (see design doc §5.3).

Single-directory mode
---------------------
  python scripts/generate_criteria.py --reference-dir example_data/reference/
  python scripts/generate_criteria.py ref1.txt ref2.md --input-dir example_data/input/
  python scripts/generate_criteria.py --reference-dir refs/ --output criteria.json

Batch mode  (one job per immediate subdirectory of --batch-dir)
----------
  # Process each subdir; save criteria.json inside each subdir
  python scripts/generate_criteria.py --batch-dir datasets/

  # Save all outputs to a separate directory
  python scripts/generate_criteria.py --batch-dir datasets/ --output-dir out/criteria/

  # Each subdir also contains an 'input/' folder for task context
  python scripts/generate_criteria.py --batch-dir datasets/ --input-subdir input

Other options
-------------
  python scripts/generate_criteria.py --reference-dir refs/ --model claude-sonnet-4-6
  python scripts/generate_criteria.py --reference-dir refs/ --max-retries 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_eval.criteria_generator import generate_criteria


def _collect_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file())


def _run_single(
    reference_files: list[Path],
    input_files: list[Path] | None,
    output_path: Path | None,
    model: str,
    max_retries: int,
    indent: int,
) -> bool:
    """Run a single criteria-generation job. Returns True on success."""
    label = reference_files[0].parent.name if reference_files else "?"
    print(
        f"[{label}] Generating criteria from {len(reference_files)} file(s)"
        f" using {model}...",
        file=sys.stderr,
    )
    if input_files:
        print(f"  + {len(input_files)} input file(s) for context.", file=sys.stderr)

    try:
        criteria = generate_criteria(
            reference_files=reference_files,
            input_files=input_files,
            model=model,
            max_retries=max_retries,
        )
    except Exception as exc:
        print(f"  [ERROR] {exc}", file=sys.stderr)
        return False

    output_json = json.dumps(criteria, ensure_ascii=False, indent=indent)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(
            f"  -> Wrote {len(criteria)} criteria to {output_path}",
            file=sys.stderr,
        )
    else:
        print(output_json)

    print(
        f"  Done. {len(criteria)} criteria generated."
        " Review and curate before use in evaluation.",
        file=sys.stderr,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation criteria from reference files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input sources (single mode) ---
    parser.add_argument(
        "reference_files",
        nargs="*",
        type=Path,
        metavar="FILE",
        help="Reference files to generate criteria from.",
    )
    parser.add_argument(
        "--reference-dir",
        "-r",
        type=Path,
        metavar="DIR",
        help="Directory of reference files (alternative to positional FILE args).",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
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

    # --- Batch mode ---
    parser.add_argument(
        "--batch-dir",
        "-b",
        type=Path,
        metavar="DIR",
        help=(
            "Process each immediate subdirectory of DIR as a separate job. "
            "Incompatible with --reference-dir and positional FILE args."
        ),
    )
    parser.add_argument(
        "--input-subdir",
        metavar="NAME",
        help=(
            "In batch mode, look for a subdirectory with this name inside each "
            "batch subdir and use its files as input/task context."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help=(
            "In batch mode, write <subdir-name>.json files here instead of "
            "saving criteria.json inside each source subdir."
        ),
    )

    # --- Output (single mode) ---
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="FILE",
        help="Write JSON output to FILE instead of stdout (single mode only).",
    )

    # --- Shared options ---
    parser.add_argument(
        "--model",
        "-m",
        default="claude-opus-4-7",
        metavar="MODEL",
        help="Anthropic model to use (default: claude-opus-4-7).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help="Maximum API attempts before giving up (default: 3).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="N",
        help="JSON indentation spaces (default: 2).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Batch mode                                                           #
    # ------------------------------------------------------------------ #
    if args.batch_dir:
        if args.reference_files or args.reference_dir:
            parser.error(
                "--batch-dir cannot be combined with --reference-dir or FILE args."
            )

        subdirs = sorted(d for d in args.batch_dir.iterdir() if d.is_dir())
        if not subdirs:
            parser.error(f"No subdirectories found in {args.batch_dir}")

        print(
            f"Batch mode: {len(subdirs)} subdirectory/ies found under {args.batch_dir}",
            file=sys.stderr,
        )

        successes = 0
        failures: list[str] = []

        for subdir in subdirs:
            ref_files = _collect_files(subdir)
            if not ref_files:
                print(f"  [SKIP] {subdir.name}: no files found.", file=sys.stderr)
                continue

            inp_files: list[Path] | None = None
            if args.input_subdir:
                inp_dir = subdir / args.input_subdir
                if inp_dir.is_dir():
                    inp_files = _collect_files(inp_dir)
                else:
                    print(
                        f"  [WARN] {subdir.name}: --input-subdir '{args.input_subdir}' "
                        f"not found, skipping context.",
                        file=sys.stderr,
                    )

            if args.output_dir:
                out_path = args.output_dir / f"{subdir.name}.json"
            else:
                out_path = subdir / "criteria.json"

            ok = _run_single(
                reference_files=ref_files,
                input_files=inp_files,
                output_path=out_path,
                model=args.model,
                max_retries=args.max_retries,
                indent=args.indent,
            )
            if ok:
                successes += 1
            else:
                failures.append(subdir.name)

        print(
            f"\nBatch complete: {successes}/{len(subdirs)} succeeded.",
            file=sys.stderr,
        )
        if failures:
            print(f"Failed: {', '.join(failures)}", file=sys.stderr)
            sys.exit(1)
        return

    # ------------------------------------------------------------------ #
    # Single mode                                                          #
    # ------------------------------------------------------------------ #
    reference_files: list[Path] = list(args.reference_files or [])
    if args.reference_dir:
        reference_files.extend(_collect_files(args.reference_dir))
    if not reference_files:
        parser.error(
            "Provide reference files as positional arguments, via --reference-dir, "
            "or use --batch-dir for batch processing."
        )

    input_files: list[Path] | None = None
    if args.input_dir:
        input_files = _collect_files(args.input_dir)
    elif args.input_files:
        input_files = list(args.input_files)

    ok = _run_single(
        reference_files=reference_files,
        input_files=input_files,
        output_path=args.output,
        model=args.model,
        max_retries=args.max_retries,
        indent=args.indent,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
