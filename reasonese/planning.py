"""Planning over the four experimental axes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Author, Channel, Framing, Instruction, author_framings


@beartype
@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One instruction x framing x channel x author datapoint."""

    instruction: Instruction
    framing: Framing
    channel: Channel
    author: Author

    def __post_init__(self) -> None:
        """Require the author to be one that writes the selected framing."""
        if self.framing not in author_framings(self.author):
            raise ValueError(
                f"the {self.author} author does not write the {self.framing} framing"
            )


@beartype
def specs_per_instruction() -> Natural:
    """Return the number of framing x channel x author combinations per instruction."""
    return Natural.parse(
        len(Channel) * sum(len(author_framings(author)) for author in Author)
    )


@beartype
def build_prompt_specs(instructions: tuple[Instruction, ...]) -> tuple[PromptSpec, ...]:
    """Enumerate every condition each author can write for each base instruction."""
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
        if framing in author_framings(author)
    )
