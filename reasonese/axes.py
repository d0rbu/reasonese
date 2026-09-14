"""The four experimental axes."""

from __future__ import annotations

from enum import StrEnum

from beartype import beartype
from phantom import Phantom


def is_non_empty_trimmed(value: str) -> bool:
    """Return whether text is non-empty and has no surrounding whitespace."""
    return bool(value) and value == value.strip()


class Instruction(str, Phantom[str], predicate=is_non_empty_trimmed, bound=str):
    """A non-empty base prompt without surrounding whitespace."""


class Framing(StrEnum):
    """The style or representation of an instruction."""

    NORMAL = "normal"
    CASUAL = "casual"
    PERSUASIVE = "persuasive"
    SUBAGENT = "subagent"
    REASONESE_NORMAL = "reasonese-normal"
    REASONESE_PERSUASIVE = "reasonese-persuasive"
    COMPRESSED_NORMAL = "compressed-normal"
    COMPRESSED_PERSUASIVE = "compressed-persuasive"


class Channel(StrEnum):
    """The context in which an instruction is presented."""

    SYSTEM = "system prompt"
    USER = "user message"
    README = "README.md"


class Author(StrEnum):
    """The person or model that authored a framed instruction."""

    USER = "user"
    QWEN3_8_FLASH = "Qwen3.8 Flash"
    QWEN3_8_2_4T = "Qwen3.8 2.4T"
    INKLING = "Inkling"
    INKLING_SMALL = "Inkling Small"
    GEMMA_4_31B_IT = "Gemma 4 31B"
    NEMOTRON_3_5_LIGHTNING = "Nemotron 3.5 Lightning"


MANUAL_FRAMINGS: tuple[Framing, ...] = (Framing.NORMAL, Framing.CASUAL, Framing.PERSUASIVE)


@beartype
def author_framings(author: Author) -> tuple[Framing, ...]:
    """Return the framings an author writes; the user writes only the manual framings."""
    if author is Author.USER:
        return MANUAL_FRAMINGS
    return tuple(Framing)


class Assistant(StrEnum):
    """The model that receives the generated conversation."""

    QWEN3_8_FLASH = "Qwen3.8 Flash"
    QWEN3_8_2_4T = "Qwen3.8 2.4T"
    INKLING = "Inkling"
    INKLING_SMALL = "Inkling Small"
    GEMMA_4_31B_IT = "Gemma 4 31B"
    NEMOTRON_3_5_LIGHTNING = "Nemotron 3.5 Lightning"


# Explicit experiment defaults; the enums above retain every supported model.
DEFAULT_AUTHORS: tuple[Author, ...] = (
    Author.NEMOTRON_3_5_LIGHTNING,
    Author.GEMMA_4_31B_IT,
)
DEFAULT_ASSISTANTS: tuple[Assistant, ...] = (
    Assistant.NEMOTRON_3_5_LIGHTNING,
    Assistant.GEMMA_4_31B_IT,
)


@beartype
def axis_manifest() -> dict[str, str | list[str]]:
    """Return the values of all four entry axes."""
    return {
        "instruction": "configured base prompts",
        "framing": [str(framing) for framing in Framing],
        "channel": [str(channel) for channel in Channel],
        "author": [str(author) for author in Author],
    }
