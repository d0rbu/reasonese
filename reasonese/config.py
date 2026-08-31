"""Strict TOML input for base instructions."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from reasonese.axes import Instruction


@dataclass(frozen=True, slots=True)
class InstructionSet:
    """A validated collection of base instructions."""

    instructions: tuple[Instruction, ...]

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ValueError("at least one instruction is required")
        ids = [instruction.id for instruction in self.instructions]
        if len(ids) != len(set(ids)):
            raise ValueError("instruction ids must be unique")


class _ConfigSchema:
    TOP_LEVEL: ClassVar[frozenset[str]] = frozenset({"schema_version", "instructions"})
    INSTRUCTION: ClassVar[frozenset[str]] = frozenset({"id", "text"})


def _require_exact_keys(payload: dict[str, Any], expected: frozenset[str], context: str) -> None:
    if payload.keys() != expected:
        missing = sorted(expected - payload.keys())
        extra = sorted(payload.keys() - expected)
        raise ValueError(f"invalid {context} keys; missing={missing}, extra={extra}")


def load_instruction_set(path: Path) -> InstructionSet:
    """Load and validate an instruction set from TOML."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    _require_exact_keys(payload, _ConfigSchema.TOP_LEVEL, "top-level")
    if payload["schema_version"] != 1:
        raise ValueError(f"unsupported schema version: {payload['schema_version']}")

    raw_instructions = payload["instructions"]
    if not isinstance(raw_instructions, list):
        raise ValueError("instructions must be an array of tables")

    instructions: list[Instruction] = []
    for index, raw_instruction in enumerate(raw_instructions):
        if not isinstance(raw_instruction, dict):
            raise ValueError(f"instruction {index} must be a table")
        raw_table = cast(dict[str, Any], raw_instruction)
        _require_exact_keys(raw_table, _ConfigSchema.INSTRUCTION, f"instruction {index}")
        identifier = raw_table["id"]
        text = raw_table["text"]
        if not isinstance(identifier, str) or not isinstance(text, str):
            raise ValueError(f"instruction {index} id and text must be strings")
        instructions.append(Instruction(id=identifier, text=text))
    return InstructionSet(instructions=tuple(instructions))
