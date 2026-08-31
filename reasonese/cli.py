"""Command-line entry point for inspecting and planning the design."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from reasonese.axes import Author, Channel, Framing, axis_manifest
from reasonese.config import load_instruction_set
from reasonese.io import write_prompt_specs
from reasonese.planning import build_prompt_specs, specs_per_instruction


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reasonese")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("axes", help="print the four axis definitions as JSON")

    plan = subparsers.add_parser("plan", help="enumerate the unrendered prompt specifications")
    plan.add_argument("--instructions", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "axes":
            print(json.dumps(axis_manifest(), indent=2))
            return 0

        instruction_set = load_instruction_set(args.instructions)
        specs = build_prompt_specs(instruction_set.instructions)
        write_prompt_specs(args.output, specs)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    summary = {
        "authors": len(Author),
        "channels": len(Channel),
        "framings": len(Framing),
        "instructions": len(instruction_set.instructions),
        "output": str(args.output),
        "specs": len(specs),
        "specs_per_instruction": specs_per_instruction(),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0
