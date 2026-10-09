"""Render one pair's scenario offline so its conversation can be read before a run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype

from reasonese.axes import Assistant, Author, Channel, Framing
from reasonese.conversation import GeneratedMessage, GeneratedText
from reasonese.instructions import load_instruction_pairs
from reasonese.matchup import make_matchup
from reasonese.planning import PromptSpec
from reasonese.scenarios import load_scenarios


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Print a scenario conversation with the pair's base instructions in the chosen channels."""
    parser = argparse.ArgumentParser(prog="reasonese-show-scenario")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=Path("configs/instruction_pairs.yaml"))
    parser.add_argument("--pair", required=True, help="instruction pair id")
    parser.add_argument("--first-channel", type=Channel, default=Channel.SYSTEM)
    parser.add_argument("--second-channel", type=Channel, default=Channel.USER)
    parser.add_argument("--assistant", type=Assistant, default=next(iter(Assistant)))
    args = parser.parse_args(argv)

    try:
        pairs = load_instruction_pairs(args.pairs)
        pair = next((pair for pair in pairs if str(pair.pair_id) == args.pair), None)
        if pair is None:
            raise ValueError(f"no instruction pair is named {args.pair}")
        library = load_scenarios(args.scenarios, pairs)
        scenario = library.scenarios.get(pair.pair_id)
        if scenario is None:
            raise ValueError(f"pair {args.pair} has no scenario below {args.scenarios}")
        # The base instructions stand in for authored text: this only shows the shape.
        specs = tuple(
            PromptSpec(instruction, Framing.NORMAL, channel, Author.USER)
            for instruction, channel in zip(
                pair.instructions, (args.first_channel, args.second_channel), strict=True
            )
        )
        setup = scenario.construct(
            make_matchup(specs, args.assistant),
            tuple(GeneratedMessage(spec, GeneratedText.parse(spec.instruction), None) for spec in specs),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    assert setup.placements is not None
    print(
        json.dumps(
            {
                "pair": args.pair,
                "source": scenario.source,
                "adaptation": scenario.adaptation,
                "messages": setup.openrouter_messages(),
                "placements": [placement.message for placement in setup.placements],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
