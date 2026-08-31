"""Planning over the four experimental axes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Author, Channel, Framing, Instruction


@beartype
@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One instruction x framing x channel x author datapoint."""

    instruction: Instruction
    framing: Framing
    channel: Channel
    author: Author


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
