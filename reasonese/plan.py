"""Enumerate prompt specifications from base instructions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype

from reasonese.axes import Author, Channel, Framing
from reasonese.config import load_instructions
from reasonese.io import write_prompt_specs
from reasonese.planning import build_prompt_specs, specs_per_instruction


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Plan every axis combination for the configured instructions."""
    parser = argparse.ArgumentParser(prog="reasonese-plan")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        instructions = load_instructions(args.instructions)
        specs = build_prompt_specs(instructions)
        write_prompt_specs(args.output, specs)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "authors": len(Author),
                "channels": len(Channel),
                "framings": len(Framing),
                "instructions": len(instructions),
                "output": str(args.output),
                "specs": len(specs),
                "specs_per_instruction": specs_per_instruction(),
            },
            sort_keys=True,
        )
    )
    return 0
