"""Plan a reproducible connected subsample of pairwise studies."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Assistant, Author
from reasonese.config import load_instructions
from reasonese.io import write_study_suite
from reasonese.planning import build_prompt_specs
from reasonese.sampling import (
    build_sampled_studies,
    minimum_connected_pairings,
    pairing_population_size,
)
from reasonese.study import PositiveInteger


def _unique_values[T](values: Sequence[T], label: str) -> tuple[T, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must be unique")
    return result


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Write one sparse, shared-across-assistants comparison design."""
    parser = argparse.ArgumentParser(prog="reasonese-sample-studies")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairings-per-assistant", type=int, required=True)
    parser.add_argument("--rollouts-per-permutation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--author", action="append", type=Author, choices=tuple(Author))
    parser.add_argument("--assistant", action="append", type=Assistant, choices=tuple(Assistant))
    args = parser.parse_args(argv)

    try:
        instructions = load_instructions(args.instructions)
        authors = _unique_values(args.author or tuple(Author), "authors")
        assistants = _unique_values(args.assistant or tuple(Assistant), "assistants")
        specs = tuple(
            spec for spec in build_prompt_specs(instructions) if spec.author in authors
        )
        pairings = PositiveInteger.parse(args.pairings_per_assistant)
        rollouts = PositiveInteger.parse(args.rollouts_per_permutation)
        seed = Natural.parse(args.seed)
        population = pairing_population_size(specs)
        minimum = minimum_connected_pairings(specs)
        studies = build_sampled_studies(specs, assistants, pairings, rollouts, seed)
        write_study_suite(args.output, studies)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "assistants": [str(assistant) for assistant in assistants],
                "authors": [str(author) for author in authors],
                "instructions": len(instructions),
                "minimum_connected_pairings_per_assistant": int(minimum),
                "output": str(args.output),
                "pairing_population_per_assistant": int(population),
                "pairings_per_assistant": int(pairings),
                "rollouts_per_permutation": int(rollouts),
                "seed": int(seed),
                "specs": len(specs),
                "studies": len(studies),
                "trials": 2 * int(rollouts) * len(studies),
            },
            sort_keys=True,
        )
    )
    return 0
