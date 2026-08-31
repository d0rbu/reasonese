"""Deterministic planning over the four experimental axes."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from reasonese.axes import Author, Channel, Framing, Instruction

SCHEMA_VERSION: Final = 1


def _make_spec_id(
    instruction: Instruction,
    framing: Framing,
    channel: Channel,
    author: Author,
) -> str:
    coordinates = {
        "author": author.value,
        "channel": channel.value,
        "framing": framing.value,
        "instruction": instruction.text,
        "instruction_id": instruction.id,
    }
    canonical = json.dumps(coordinates, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One unrendered cell in the instruction x framing x channel x author design."""

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "spec_id",
            "instruction_id",
            "instruction",
            "framing",
            "channel",
            "author",
        }
    )

    schema_version: int
    spec_id: str
    instruction_id: str
    instruction: str
    framing: Framing
    channel: Channel
    author: Author

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if not isinstance(self.spec_id, str):
            raise ValueError("spec_id must be a string")
        if not isinstance(self.framing, Framing):
            raise ValueError("framing must be a Framing value")
        if not isinstance(self.channel, Channel):
            raise ValueError("channel must be a Channel value")
        if not isinstance(self.author, Author):
            raise ValueError("author must be an Author value")
        base = Instruction(id=self.instruction_id, text=self.instruction)
        expected = _make_spec_id(base, self.framing, self.channel, self.author)
        if self.spec_id != expected:
            raise ValueError(f"spec_id must be {expected!r} for these coordinates")

    @classmethod
    def create(
        cls,
        instruction: Instruction,
        framing: Framing,
        channel: Channel,
        author: Author,
    ) -> PromptSpec:
        """Construct a prompt specification with a content-addressed ID."""
        return cls(
            schema_version=SCHEMA_VERSION,
            spec_id=_make_spec_id(instruction, framing, channel, author),
            instruction_id=instruction.id,
            instruction=instruction.text,
            framing=framing,
            channel=channel,
            author=author,
        )

    def to_dict(self) -> dict[str, str | int]:
        """Serialize this specification to its versioned wire format."""
        return {
            "schema_version": self.schema_version,
            "spec_id": self.spec_id,
            "instruction_id": self.instruction_id,
            "instruction": self.instruction,
            "framing": self.framing.value,
            "channel": self.channel.value,
            "author": self.author.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptSpec:
        """Parse a strict specification from its wire format."""
        if payload.keys() != cls._FIELDS:
            missing = sorted(cls._FIELDS - payload.keys())
            extra = sorted(payload.keys() - cls._FIELDS)
            raise ValueError(f"invalid prompt spec keys; missing={missing}, extra={extra}")
        schema_version = payload["schema_version"]
        if type(schema_version) is not int:
            raise ValueError("schema_version must be an integer")
        string_fields = ("spec_id", "instruction_id", "instruction", "framing", "channel", "author")
        for field in string_fields:
            if not isinstance(payload[field], str):
                raise ValueError(f"{field} must be a string")
        return cls(
            schema_version=schema_version,
            spec_id=payload["spec_id"],
            instruction_id=payload["instruction_id"],
            instruction=payload["instruction"],
            framing=Framing(payload["framing"]),
            channel=Channel(payload["channel"]),
            author=Author(payload["author"]),
        )


def specs_per_instruction() -> int:
    """Return the size of the full framing x channel x author product."""
    return len(Framing) * len(Channel) * len(Author)


def build_prompt_specs(instructions: tuple[Instruction, ...]) -> tuple[PromptSpec, ...]:
    """Enumerate the complete four-axis design in stable order."""
    if not instructions:
        raise ValueError("at least one instruction is required")
    instruction_ids = [instruction.id for instruction in instructions]
    if len(instruction_ids) != len(set(instruction_ids)):
        raise ValueError("instruction ids must be unique")

    specs = tuple(
        PromptSpec.create(instruction, framing, channel, author)
        for instruction, framing, channel, author in itertools.product(
            instructions,
            Framing,
            Channel,
            Author,
        )
    )
    expected = len(instructions) * specs_per_instruction()
    if len(specs) != expected or len({spec.spec_id for spec in specs}) != expected:
        raise RuntimeError("prompt specification design is incomplete or contains duplicate ids")
    return specs
