"""The four experimental axes."""

from __future__ import annotations

from enum import StrEnum

from beartype import beartype
from phantom import Phantom


def _is_instruction(value: str) -> bool:
    return bool(value) and value == value.strip()


class Instruction(str, Phantom[str], predicate=_is_instruction, bound=str):
    """A non-empty base prompt without surrounding whitespace."""


class Framing(StrEnum):
    """The style or representation of an instruction."""

    NORMAL = "normal"
    CASUAL = "casual"
    PERSUASIVE = "persuasive"
    SUBAGENT = "subagent"
    REASONESE_NORMAL = "reasonese-normal"
    REASONESE_PERSUASIVE = "reasonese-persuasive"


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


@beartype
def axis_manifest() -> dict[str, str | list[str]]:
    """Return the values of all four axes."""
    return {
        "instruction": "configured base prompts",
        "framing": [str(framing) for framing in Framing],
        "channel": [str(channel) for channel in Channel],
        "author": [str(author) for author in Author],
    }
