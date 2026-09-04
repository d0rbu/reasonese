"""Enumerate prompt specifications from the instruction-pair bank."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype

from reasonese.axes import Author, Channel, Framing
from reasonese.instructions import load_instruction_pairs
from reasonese.io import write_prompt_specs
from reasonese.planning import build_pair_specs, specs_per_instruction


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Plan every axis combination for both sides of every instruction pair."""
    parser = argparse.ArgumentParser(prog="reasonese-plan")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--author", action="append", type=Author, choices=tuple(Author))
    args = parser.parse_args(argv)

    try:
        authors = tuple(args.author or tuple(Author))
        if len(authors) != len(set(authors)):
            raise ValueError("authors must be unique")
        pairs = load_instruction_pairs(args.pairs)
        specs = tuple(
            spec
            for pair_specs in build_pair_specs(pairs)
            for spec in pair_specs.first + pair_specs.second
            if spec.author in authors
        )
        write_prompt_specs(args.output, specs)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "authors": len(authors),
                "channels": len(Channel),
                "framings": len(Framing),
                "instruction_pairs": len(pairs),
                "instructions": 2 * len(pairs),
                "output": str(args.output),
                "specs": len(specs),
                "specs_per_instruction": specs_per_instruction(),
            },
            sort_keys=True,
        )
    )
    return 0
