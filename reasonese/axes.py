"""Definitions for the four experimental axes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

_IDENTIFIER: Final = re.compile(r"^[a-z][a-z0-9_]*$")


class Framing(StrEnum):
    """The style or representation of an instruction."""

    NORMAL = "normal"
    CASUAL = "casual"
    PERSUASIVE = "persuasive"
    SUBAGENT = "subagent"
    REASONESE_NORMAL = "reasonese-normal"
    REASONESE_PERSUASIVE = "reasonese-persuasive"


FRAMING_DESCRIPTIONS: Final[dict[Framing, str]] = {
    Framing.NORMAL: (
        "The author's default rendering in clear, standard prose; for the user author, "
        "this may be the original text."
    ),
    Framing.CASUAL: "Conversational prose with lowercase text and reduced punctuation.",
    Framing.PERSUASIVE: "Natural language deliberately written to secure compliance.",
    Framing.SUBAGENT: "A delegation written as if a parent agent were instructing a subagent.",
    Framing.REASONESE_NORMAL: "Compressed reasonese without deliberate persuasive intent.",
    Framing.REASONESE_PERSUASIVE: (
        "Compressed reasonese deliberately written to secure compliance."
    ),
}


class Channel(StrEnum):
    """The context in which an instruction is presented."""

    SYSTEM = "system"
    USER = "user"
    README = "readme"


CHANNEL_LABELS: Final[dict[Channel, str]] = {
    Channel.SYSTEM: "system prompt",
    Channel.USER: "user message",
    Channel.README: "README.md",
}


class Author(StrEnum):
    """The person or model that authored a framed instruction."""

    USER = "user"
    QWEN3_8_FLASH = "qwen3_8_flash"
    QWEN3_8_2_4T = "qwen3_8_2_4t"
    INKLING = "inkling"
    INKLING_SMALL = "inkling_small"


AUTHOR_LABELS: Final[dict[Author, str]] = {
    Author.USER: "user",
    Author.QWEN3_8_FLASH: "Qwen3.8 Flash",
    Author.QWEN3_8_2_4T: "Qwen3.8 2.4T",
    Author.INKLING: "Inkling",
    Author.INKLING_SMALL: "Inkling Small",
}


def validate_identifier(value: str, *, field: str = "id") -> str:
    """Validate and return a stable snake-case identifier."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if _IDENTIFIER.fullmatch(value) is None:
        msg = f"{field} must match {_IDENTIFIER.pattern!r}; got {value!r}"
        raise ValueError(msg)
    return value


@dataclass(frozen=True, slots=True)
class Instruction:
    """An author-independent base task to be expressed under each condition."""

    id: str
    text: str

    def __post_init__(self) -> None:
        validate_identifier(self.id, field="instruction id")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("instruction text must not be empty")
        if self.text != self.text.strip():
            raise ValueError("instruction text must not have surrounding whitespace")


def axis_manifest() -> dict[str, Any]:
    """Return the ordered, JSON-serializable definitions of all four axes."""
    return {
        "instruction": {
            "description": "An author-independent base task.",
            "source": "configured",
        },
        "framing": [
            {"id": framing.value, "description": FRAMING_DESCRIPTIONS[framing]}
            for framing in Framing
        ],
        "channel": [{"id": channel.value, "label": CHANNEL_LABELS[channel]} for channel in Channel],
        "author": [{"id": author.value, "label": AUTHOR_LABELS[author]} for author in Author],
    }
