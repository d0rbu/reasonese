"""Load base instructions from TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml
from beartype import beartype

from reasonese.axes import Instruction
from reasonese.matchup import Matchup, matchup_from_dict


@beartype
def load_instructions(path: Path) -> tuple[Instruction, ...]:
    """Load a non-empty list of base prompts."""
    with path.open("rb") as handle:
        raw_instructions = tomllib.load(handle).get("instructions")
    if not isinstance(raw_instructions, list):
        raise ValueError("instructions must be a TOML array")

    instructions = tuple(Instruction.parse(value) for value in raw_instructions)
    if not instructions:
        raise ValueError("at least one instruction is required")
    return instructions


@beartype
def load_matchup(path: Path) -> Matchup:
    """Load one matchup from YAML."""
    with path.open(encoding="utf-8") as handle:
        return matchup_from_dict(yaml.safe_load(handle))
