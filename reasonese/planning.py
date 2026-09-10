"""Planning over the experimental axes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.instructions import InstructionPair, instruction_index


@beartype
@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One instruction x framing x channel x author datapoint."""

    instruction: Instruction
    framing: Framing
    channel: Channel
    author: Author


@beartype
@dataclass(frozen=True, slots=True)
class PairSpecs:
    """Every condition enumerated for both sides of one instruction pair."""

    pair: InstructionPair
    first: tuple[PromptSpec, ...]
    second: tuple[PromptSpec, ...]


@beartype
def specs_per_instruction() -> Natural:
    """Return the number of framing x channel x author combinations."""
    return Natural.parse(len(Framing) * len(Channel) * len(Author))


@beartype
def build_prompt_specs(instructions: tuple[Instruction, ...]) -> tuple[PromptSpec, ...]:
    """Enumerate every condition for each base instruction."""
    if not instructions:
        raise ValueError("at least one instruction is required")
    if len(instructions) != len(set(instructions)):
        raise ValueError("instructions must be unique")

    return tuple(
        PromptSpec(instruction, framing, channel, author)
        for instruction, framing, channel, author in itertools.product(
            instructions,
            Framing,
            Channel,
            Author,
        )
    )


@beartype
def build_pair_specs(pairs: tuple[InstructionPair, ...]) -> tuple[PairSpecs, ...]:
    """Enumerate every condition for both sides of each instruction pair.

    Instruction text alone determines pair and side, so nothing is recorded on
    ``PromptSpec``. Reuse of one instruction across pairs is rejected because it
    would make that mapping ambiguous.
    """
    instruction_index(pairs)
    return tuple(
        PairSpecs(
            pair,
            build_prompt_specs((pair.first,)),
            build_prompt_specs((pair.second,)),
        )
        for pair in pairs
    )
