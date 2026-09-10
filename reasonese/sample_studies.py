"""Plan a reproducible connected subsample of within-pair studies."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Assistant, Author
from reasonese.instructions import load_instruction_pairs
from reasonese.io import write_study_suite
from reasonese.planning import PairSpecs, build_pair_specs
from reasonese.sampling import (
    build_sampled_studies,
    default_pairing_count,
    minimum_connected_pairings,
    pairing_population_size,
)
from reasonese.study import PositiveInteger


def _unique_values[T](values: Sequence[T], label: str) -> tuple[T, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must be unique")
    return result


def _single[T](values: tuple[T, ...], label: str) -> T:
    """Return the one value shared by every pair, or reject a ragged design."""
    distinct = set(values)
    if len(distinct) != 1:
        raise ValueError(f"every instruction pair must share the same {label}")
    return values[0]


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Write one sparse within-pair comparison design shared across assistants."""
    parser = argparse.ArgumentParser(prog="reasonese-sample-studies")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pairings-per-pair",
        type=int,
        help="default: 720, capped by the eligible population of each pair",
    )
    parser.add_argument("--rollouts-per-permutation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--author", action="append", type=Author, choices=tuple(Author))
    parser.add_argument("--assistant", action="append", type=Assistant, choices=tuple(Assistant))
    args = parser.parse_args(argv)

    try:
        pairs = load_instruction_pairs(args.pairs)
        authors = _unique_values(args.author or tuple(Author), "authors")
        assistants = _unique_values(args.assistant or tuple(Assistant), "assistants")
        pair_specs = tuple(
            PairSpecs(
                item.pair,
                tuple(spec for spec in item.first if spec.author in authors),
                tuple(spec for spec in item.second if spec.author in authors),
            )
            for item in build_pair_specs(pairs)
        )
        rollouts = PositiveInteger.parse(args.rollouts_per_permutation)
        seed = Natural.parse(args.seed)
        population = _single(
            tuple(int(pairing_population_size(item)) for item in pair_specs),
            "pairing population",
        )
        minimum = _single(
            tuple(int(minimum_connected_pairings(item)) for item in pair_specs),
            "minimum connected pairing count",
        )
        pairings = (
            default_pairing_count(pair_specs[0])
            if args.pairings_per_pair is None
            else PositiveInteger.parse(args.pairings_per_pair)
        )
        studies = build_sampled_studies(pair_specs, assistants, pairings, rollouts, seed)
        write_study_suite(args.output, studies)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "assistants": [str(assistant) for assistant in assistants],
                "authors": [str(author) for author in authors],
                "instruction_pairs": len(pairs),
                "minimum_connected_pairings_per_pair": minimum,
                "output": str(args.output),
                "pairing_population_per_pair": population,
                "pairings_per_pair": int(pairings),
                "rollouts_per_permutation": int(rollouts),
                "seed": int(seed),
                "specs": sum(len(item.first) + len(item.second) for item in pair_specs),
                "studies": len(studies),
                "trials": 2 * int(rollouts) * len(studies),
            },
            sort_keys=True,
        )
    )
    return 0
